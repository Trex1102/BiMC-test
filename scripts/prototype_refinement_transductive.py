import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from query_graph_router_killswitch import accuracy_stats
from semantic_option2_killswitch import (
    collect_final_queries,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    to_builtin,
)


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"prototype_refinement_transductive_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_rows(tensor):
    return F.normalize(tensor.float(), dim=-1)


def baseline_semantic_proto(merged_state, num_cls, lambda_t, device):
    text_features = normalize_rows(merged_state["text_features"][:num_cls].to(device))
    description_proto = normalize_rows(merged_state["description_proto"][:num_cls].to(device))
    return normalize_rows((1.0 - lambda_t) * text_features + lambda_t * description_proto)


def compose_full_probs(
    query_features,
    semantic_proto,
    image_proto,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
):
    probs, logits = compose_probs_from_semantic(
        query_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    return probs, logits


def sinkhorn_balanced(novel_cond_probs, n_iters=50, tau=0.5):
    kernel = torch.clamp(novel_cond_probs, min=1e-12).pow(1.0 / max(float(tau), 1e-6))
    m, k = kernel.shape
    r = torch.full((m,), 1.0 / float(m), device=kernel.device, dtype=kernel.dtype)
    c = torch.full((k,), 1.0 / float(k), device=kernel.device, dtype=kernel.dtype)
    u = torch.ones_like(r)
    v = torch.ones_like(c)

    for _ in range(int(n_iters)):
        kv = torch.clamp(torch.matmul(kernel, v.unsqueeze(1)).squeeze(1), min=1e-12)
        u = r / kv
        ktu = torch.clamp(torch.matmul(kernel.T, u.unsqueeze(1)).squeeze(1), min=1e-12)
        v = c / ktu

    q = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    q = q / torch.clamp(q.sum(dim=1, keepdim=True), min=1e-12)
    return q


def refine_novel_prototypes(
    query_features,
    probs,
    orig_image_proto,
    current_image_proto,
    num_base_cls,
    num_cls,
    quota_multiplier,
    proto_alpha,
    sinkhorn_tau,
    sinkhorn_iters,
):
    num_novel = num_cls - num_base_cls
    n_query = query_features.shape[0]
    per_class_quota = max(1, int(round(float(n_query) / float(num_cls))))
    novel_quota = max(num_novel, int(round(float(per_class_quota * num_novel) * float(quota_multiplier))))

    novel_mass = probs[:, num_base_cls:].sum(dim=1)
    top_idx = torch.topk(novel_mass, k=min(novel_quota, n_query)).indices
    selected_features = query_features[top_idx]
    selected_probs = probs[top_idx, num_base_cls:]
    selected_novel_mass = torch.clamp(novel_mass[top_idx].unsqueeze(1), min=1e-12)
    novel_cond = selected_probs / selected_novel_mass
    assign = sinkhorn_balanced(novel_cond, n_iters=sinkhorn_iters, tau=sinkhorn_tau)

    new_proto = current_image_proto.clone()
    target_proto = current_image_proto.clone()
    class_masses = assign.sum(dim=0)
    for local_idx, cls_id in enumerate(range(num_base_cls, num_cls)):
        weights = assign[:, local_idx]
        if float(weights.sum().item()) <= 1e-12:
            continue
        query_mean = F.normalize(torch.matmul(weights.unsqueeze(0), selected_features), dim=-1).squeeze(0)
        target_proto[cls_id] = query_mean
        mixed = (1.0 - float(proto_alpha)) * orig_image_proto[cls_id] + float(proto_alpha) * query_mean
        new_proto[cls_id] = F.normalize(mixed.unsqueeze(0), dim=-1).squeeze(0)

    return {
        "image_proto": new_proto,
        "target_proto": target_proto,
        "selected_idx": top_idx,
        "class_masses": class_masses,
        "per_class_quota": per_class_quota,
        "novel_quota": novel_quota,
    }


def latest_rectification_oracle_gain(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    oracle_root = output_root / cfg.DATASET.NAME / f"prototype_rectification_killswitch_{Path(train_cfg).stem}"
    if not oracle_root.exists():
        return None, None
    results_files = sorted(oracle_root.glob("*/results.json"))
    if not results_files:
        return None, None
    latest = results_files[-1]
    with latest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload.get("oracle_gain", 0.0)), latest.parent


def run_dataset(
    data_cfg,
    train_cfg,
    num_iters,
    quota_multiplier,
    proto_alpha,
    sinkhorn_tau,
    sinkhorn_iters,
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
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    orig_image_proto = normalize_rows(merged_state["image_proto"][:num_cls].to(device))
    current_image_proto = orig_image_proto.clone()
    semantic_proto = baseline_semantic_proto(merged_state, num_cls, lambda_t, device)
    description_features = normalize_rows(merged_state["description_features"].to(device))
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device).float()

    prob_cov = F.softmax(compute_cov_logits(query_features, orig_image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )

    baseline_probs, _ = compose_full_probs(
        query_features,
        semantic_proto,
        current_image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    baseline_full, baseline_base, baseline_novel = accuracy_stats(baseline_probs, query_targets, num_base_cls)

    history = []
    current_probs = baseline_probs
    for step in range(int(num_iters)):
        update = refine_novel_prototypes(
            query_features=query_features,
            probs=current_probs,
            orig_image_proto=orig_image_proto,
            current_image_proto=current_image_proto,
            num_base_cls=num_base_cls,
            num_cls=num_cls,
            quota_multiplier=quota_multiplier,
            proto_alpha=proto_alpha,
            sinkhorn_tau=sinkhorn_tau,
            sinkhorn_iters=sinkhorn_iters,
        )
        current_image_proto = update["image_proto"]
        current_probs, _ = compose_full_probs(
            query_features,
            semantic_proto,
            current_image_proto,
            num_base_cls,
            beta,
            ensemble_alpha,
            prob_cov,
            prob_knn,
        )
        full_acc, base_acc, novel_acc = accuracy_stats(current_probs, query_targets, num_base_cls)
        history.append(
            {
                "iter": step + 1,
                "full_acc": round(full_acc, 3),
                "base_acc": round(base_acc, 3),
                "novel_acc": round(novel_acc, 3),
                "novel_quota": int(update["novel_quota"]),
                "per_class_quota": int(update["per_class_quota"]),
                "mean_class_mass": round(float(update["class_masses"].mean().item()), 4),
                "std_class_mass": round(float(update["class_masses"].std().item()), 4),
            }
        )

    final_full, final_base, final_novel = accuracy_stats(current_probs, query_targets, num_base_cls)
    oracle_gain, oracle_path = latest_rectification_oracle_gain(cfg, train_cfg)

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "num_seen_classes": int(num_cls),
        "num_base_classes": int(num_base_cls),
        "beta": beta,
        "lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "num_iters": int(num_iters),
        "quota_multiplier": float(quota_multiplier),
        "proto_alpha": float(proto_alpha),
        "sinkhorn_tau": float(sinkhorn_tau),
        "sinkhorn_iters": int(sinkhorn_iters),
        "baseline_full_acc": round(baseline_full, 3),
        "baseline_base_acc": round(baseline_base, 3),
        "baseline_novel_acc": round(baseline_novel, 3),
        "final_full_acc": round(final_full, 3),
        "final_base_acc": round(final_base, 3),
        "final_novel_acc": round(final_novel, 3),
        "final_gain": round(final_full - baseline_full, 3),
        "history": history,
        "rectification_oracle_gain": None if oracle_gain is None else round(float(oracle_gain), 3),
        "rectification_oracle_path": None if oracle_path is None else str(oracle_path),
    }
    if oracle_gain is not None and abs(oracle_gain) > 1e-8:
        payload["recovery_vs_oracle"] = round(float((final_full - baseline_full) / oracle_gain * 100.0), 1)
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"final_full_acc: {payload['final_full_acc']}",
        f"final_gain: {payload['final_gain']}",
        f"baseline_base_acc: {payload['baseline_base_acc']}",
        f"baseline_novel_acc: {payload['baseline_novel_acc']}",
        f"final_base_acc: {payload['final_base_acc']}",
        f"final_novel_acc: {payload['final_novel_acc']}",
        f"recovery_vs_oracle: {payload.get('recovery_vs_oracle')}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Training-free transductive prototype refinement for BiMC.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--quota_multiplier", type=float, default=1.0)
    parser.add_argument("--proto_alpha", type=float, default=0.75)
    parser.add_argument("--sinkhorn_tau", type=float, default=0.5)
    parser.add_argument("--sinkhorn_iters", type=int, default=50)
    parser.add_argument("--seed_override", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        num_iters=args.num_iters,
        quota_multiplier=args.quota_multiplier,
        proto_alpha=args.proto_alpha,
        sinkhorn_tau=args.sinkhorn_tau,
        sinkhorn_iters=args.sinkhorn_iters,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
