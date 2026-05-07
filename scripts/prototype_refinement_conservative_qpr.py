import argparse
import json
import sys
import time
from pathlib import Path

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
    collect_final_queries,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    to_builtin,
)

import torch
import torch.nn.functional as F


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"prototype_refinement_conservative_qpr_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_dataset(
    data_cfg,
    train_cfg,
    alpha,
    mass_thr,
    min_count,
    t_base,
    t_novel,
    topk_per_class=None,
    seed_override=None,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    if seed_override is not None:
        cfg.defrost()
        cfg.SEED = int(seed_override)
        cfg.freeze()

    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)

    device = cfg.DEVICE.DEVICE_NAME
    query_features = query_features.to(device).float()
    query_targets = query_targets.to(device)

    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    num_novel_cls = int(num_cls - num_base_cls)
    target_novel_mass = float(num_novel_cls / max(num_cls, 1))
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    orig_image_proto = normalize_rows(merged_state["image_proto"][:num_cls].to(device))
    semantic_proto = baseline_semantic_proto(merged_state, num_cls, lambda_t, device)
    description_features = normalize_rows(merged_state["description_features"].to(device))
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device).float()

    prob_cov = F.softmax(compute_cov_logits(query_features, orig_image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )
    baseline_probs = compose_full_probs(
        query_features,
        semantic_proto,
        orig_image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    baseline_full, baseline_base, baseline_novel = accuracy_stats(baseline_probs, query_targets, num_base_cls)

    refined_proto, cpr_stats = conservative_refine_once(
        query_features=query_features,
        query_targets=query_targets,
        baseline_probs=baseline_probs,
        orig_image_proto=orig_image_proto,
        num_base_cls=num_base_cls,
        num_cls=num_cls,
        alpha=alpha,
        mass_thr=mass_thr,
        min_count=min_count,
        topk_per_class=topk_per_class,
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

    qpr_probs, delta, achieved_mass = apply_qpr(
        cpr_probs,
        num_base_cls=num_base_cls,
        t_base=t_base,
        t_novel=t_novel,
        target_novel_mass=target_novel_mass,
    )
    final_full, final_base, final_novel = accuracy_stats(qpr_probs, query_targets, num_base_cls)

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "num_seen_classes": int(num_cls),
        "num_base_classes": int(num_base_cls),
        "num_novel_classes": int(num_novel_cls),
        "target_novel_mass": round(float(target_novel_mass), 3),
        "alpha": float(alpha),
        "mass_thr": float(mass_thr),
        "min_count": int(min_count),
        "topk_per_class": None if topk_per_class is None else int(topk_per_class),
        "t_base": float(t_base),
        "t_novel": float(t_novel),
        "delta": round(float(delta), 4),
        "achieved_novel_mass": round(float(achieved_mass), 3),
        "baseline_full_acc": round(float(baseline_full), 3),
        "baseline_base_acc": round(float(baseline_base), 3),
        "baseline_novel_acc": round(float(baseline_novel), 3),
        "cpr_full_acc": round(float(cpr_full), 3),
        "cpr_base_acc": round(float(cpr_base), 3),
        "cpr_novel_acc": round(float(cpr_novel), 3),
        "cpr_gain": round(float(cpr_full - baseline_full), 3),
        "final_full_acc": round(float(final_full), 3),
        "final_base_acc": round(float(final_base), 3),
        "final_novel_acc": round(float(final_novel), 3),
        "final_gain": round(float(final_full - baseline_full), 3),
        "gain_over_cpr": round(float(final_full - cpr_full), 3),
        "cpr_stats": {
            "updated_class_count": int(cpr_stats["updated_class_count"]),
            "updated_class_rate": round(float(cpr_stats["updated_class_rate"]), 4),
            "gated_query_count": int(cpr_stats["gated_query_count"]),
            "gated_base_count": int(cpr_stats["gated_base_count"]),
            "gated_novel_count": int(cpr_stats["gated_novel_count"]),
        },
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"cpr_full_acc: {payload['cpr_full_acc']}",
        f"final_full_acc: {payload['final_full_acc']}",
        f"final_gain: {payload['final_gain']}",
        f"gain_over_cpr: {payload['gain_over_cpr']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed CPR + QPR method for BiMC.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mass_thr", type=float, default=0.3)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--t_base", type=float, default=0.75)
    parser.add_argument("--t_novel", type=float, default=0.75)
    parser.add_argument("--topk_per_class", type=int, default=None)
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
        topk_per_class=args.topk_per_class,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
