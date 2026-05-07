import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from cpr_qpr_killswitch import apply_qpr
from prototype_refinement_conservative import (
    baseline_semantic_proto,
    compose_full_probs,
    conservative_refine_once,
    normalize_rows,
)
from query_graph_router_killswitch import accuracy_stats
from semantic_option2_killswitch import (
    build_model,
    compute_cov_logits,
    compute_knn_logits,
    merge_states,
    to_builtin,
)
from datasets.data_manager import DatasetManager
from utils.util import set_gpu, set_seed


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"sessionwise_cpr_qpr_ablation_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@torch.no_grad()
def extract_state_for_task(cfg, upto_task_id):
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    data_manager = DatasetManager(cfg)
    model = build_model(cfg, data_manager)
    model.eval()

    states = []
    for task_id in range(upto_task_id + 1):
        class_names = np.array(data_manager.class_names)[data_manager.class_index_in_task[task_id]]
        train_loader = data_manager.get_dataloader(task_id, source="train", mode="test", accumulate_past=False)
        state = model.build_task_statistics(
            class_names,
            train_loader,
            class_index=data_manager.class_index_in_task[task_id],
            calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION,
        )
        states.append(state)

    merged = merge_states(states)
    return data_manager, model, merged


@torch.no_grad()
def collect_session_queries(cfg, data_manager, model, task_id):
    model_ref = model.module if hasattr(model, "module") else model
    loader = data_manager.get_dataloader(task_id, source="test", mode="test")
    all_features = []
    all_targets = []
    for batch in loader:
        images = batch["image"].to(cfg.DEVICE.DEVICE_NAME)
        targets = batch["label"].to(cfg.DEVICE.DEVICE_NAME)
        image_features = model_ref.extract_img_feature(images)
        image_features = F.normalize(image_features, dim=-1)
        all_features.append(image_features)
        all_targets.append(targets)
    return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)


def evaluate_one_session(cfg, data_manager, merged_state, model, task_id, alpha, mass_thr, min_count, t_base, t_novel):
    device = cfg.DEVICE.DEVICE_NAME
    query_features, query_targets = collect_session_queries(cfg, data_manager, model, task_id)
    query_features = query_features.to(device).float()
    query_targets = query_targets.to(device)

    num_cls = int(max(data_manager.class_index_in_task[task_id]) + 1)
    num_base_cls = int(len(data_manager.class_index_in_task[0]))
    num_novel_cls = int(num_cls - num_base_cls)
    target_novel_mass = float(num_novel_cls / max(num_cls, 1))
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    image_proto = normalize_rows(merged_state["image_proto"][:num_cls].to(device))
    semantic_proto = baseline_semantic_proto(merged_state, num_cls, lambda_t, device)
    description_features = normalize_rows(merged_state["description_features"].to(device))
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device).float()

    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )
    baseline_probs = compose_full_probs(
        query_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    baseline_full, baseline_base, baseline_novel = accuracy_stats(baseline_probs, query_targets, num_base_cls)

    if num_novel_cls == 0:
        return {
            "task_id": int(task_id),
            "num_seen_classes": num_cls,
            "num_novel_classes": num_novel_cls,
            "target_novel_mass": round(target_novel_mass, 3),
            "baseline_full_acc": round(float(baseline_full), 3),
            "baseline_base_acc": round(float(baseline_base), 3),
            "baseline_novel_acc": None,
            "qpr_only_full_acc": round(float(baseline_full), 3),
            "qpr_only_gain": 0.0,
            "cpr_full_acc": round(float(baseline_full), 3),
            "cpr_gain": 0.0,
            "final_full_acc": round(float(baseline_full), 3),
            "final_gain": 0.0,
            "gain_over_cpr": 0.0,
            "cpr_stats": {
                "updated_class_count": 0,
                "updated_class_rate": 0.0,
                "gated_query_count": 0,
                "gated_base_count": 0,
                "gated_novel_count": 0,
            },
        }

    qpr_only_probs, delta_qpr, achieved_mass_qpr = apply_qpr(
        baseline_probs,
        num_base_cls=num_base_cls,
        t_base=t_base,
        t_novel=t_novel,
        target_novel_mass=target_novel_mass,
    )
    qpr_only_full, qpr_only_base, qpr_only_novel = accuracy_stats(qpr_only_probs, query_targets, num_base_cls)

    refined_proto, cpr_stats = conservative_refine_once(
        query_features=query_features,
        query_targets=query_targets,
        baseline_probs=baseline_probs,
        orig_image_proto=image_proto,
        num_base_cls=num_base_cls,
        num_cls=num_cls,
        alpha=alpha,
        mass_thr=mass_thr,
        min_count=min_count,
        topk_per_class=None,
    )
    cpr_probs = compose_full_probs(
        query_features,
        semantic_proto,
        refined_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    cpr_full, cpr_base, cpr_novel = accuracy_stats(cpr_probs, query_targets, num_base_cls)

    final_probs, delta, achieved_mass = apply_qpr(
        cpr_probs,
        num_base_cls=num_base_cls,
        t_base=t_base,
        t_novel=t_novel,
        target_novel_mass=target_novel_mass,
    )
    final_full, final_base, final_novel = accuracy_stats(final_probs, query_targets, num_base_cls)

    return {
        "task_id": int(task_id),
        "num_seen_classes": num_cls,
        "num_novel_classes": num_novel_cls,
        "target_novel_mass": round(target_novel_mass, 3),
        "baseline_full_acc": round(float(baseline_full), 3),
        "baseline_base_acc": round(float(baseline_base), 3),
        "baseline_novel_acc": round(float(baseline_novel), 3),
        "qpr_only_full_acc": round(float(qpr_only_full), 3),
        "qpr_only_base_acc": round(float(qpr_only_base), 3),
        "qpr_only_novel_acc": round(float(qpr_only_novel), 3),
        "qpr_only_gain": round(float(qpr_only_full - baseline_full), 3),
        "qpr_only_delta": round(float(delta_qpr), 4),
        "qpr_only_achieved_novel_mass": round(float(achieved_mass_qpr), 3),
        "cpr_full_acc": round(float(cpr_full), 3),
        "cpr_base_acc": round(float(cpr_base), 3),
        "cpr_novel_acc": round(float(cpr_novel), 3),
        "cpr_gain": round(float(cpr_full - baseline_full), 3),
        "final_full_acc": round(float(final_full), 3),
        "final_base_acc": round(float(final_base), 3),
        "final_novel_acc": round(float(final_novel), 3),
        "final_gain": round(float(final_full - baseline_full), 3),
        "gain_over_cpr": round(float(final_full - cpr_full), 3),
        "delta": round(float(delta), 4),
        "achieved_novel_mass": round(float(achieved_mass), 3),
        "cpr_stats": {
            "updated_class_count": int(cpr_stats["updated_class_count"]),
            "updated_class_rate": round(float(cpr_stats["updated_class_rate"]), 4),
            "gated_query_count": int(cpr_stats["gated_query_count"]),
            "gated_base_count": int(cpr_stats["gated_base_count"]),
            "gated_novel_count": int(cpr_stats["gated_novel_count"]),
        },
    }


def run_dataset(data_cfg, train_cfg, alpha, mass_thr, min_count, t_base, t_novel, seed_override=None):
    cfg = setup_cfg(data_cfg, train_cfg)
    if seed_override is not None:
        cfg.defrost()
        cfg.SEED = int(seed_override)
        cfg.freeze()

    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    data_manager, _, _ = extract_state_for_task(cfg, 0)
    session_rows = []
    for task_id in range(data_manager.num_tasks):
        data_manager_t, model_t, merged_state_t = extract_state_for_task(cfg, task_id)
        row = evaluate_one_session(
            cfg=cfg,
            data_manager=data_manager_t,
            merged_state=merged_state_t,
            model=model_t,
            task_id=task_id,
            alpha=alpha,
            mass_thr=mass_thr,
            min_count=min_count,
            t_base=t_base,
            t_novel=t_novel,
        )
        session_rows.append(row)

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "alpha": float(alpha),
        "mass_thr": float(mass_thr),
        "min_count": int(min_count),
        "t_base": float(t_base),
        "t_novel": float(t_novel),
        "sessions": to_builtin(session_rows),
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [f"dataset: {cfg.DATASET.NAME}"]
    for row in session_rows:
        lines.append(
            "session {task_id}: baseline={baseline_full_acc:.3f} qpr={qpr_only_full_acc:.3f} "
            "cpr={cpr_full_acc:.3f} cpr_qpr={final_full_acc:.3f}".format(**row)
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Session-wise CPR/QPR ablation for BiMC.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mass_thr", type=float, default=0.3)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--t_base", type=float, default=0.75)
    parser.add_argument("--t_novel", type=float, default=0.75)
    parser.add_argument("--seed_override", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        alpha=args.alpha,
        mass_thr=args.mass_thr,
        min_count=args.min_count,
        t_base=args.t_base,
        t_novel=args.t_novel,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
