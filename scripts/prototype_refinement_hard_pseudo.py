import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from prototype_refinement_conservative import (
    baseline_semantic_proto,
    compose_full_probs,
    create_output_dir,
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


def create_hard_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"prototype_refinement_hard_pseudo_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def hard_pseudo_refine_once(
    query_features,
    query_targets,
    baseline_probs,
    orig_image_proto,
    num_base_cls,
    num_cls,
    alpha,
    min_count,
):
    image_proto = orig_image_proto.clone()
    preds = torch.argmax(baseline_probs, dim=1)
    gate = preds >= int(num_base_cls)

    per_class_counts = {}
    updated_classes = []
    for cls_id in range(num_base_cls, num_cls):
        mask = gate & (preds == int(cls_id))
        count = int(mask.sum().item())
        per_class_counts[int(cls_id)] = count
        if count < int(min_count):
            continue
        proto = query_features[mask].mean(dim=0)
        proto = F.normalize(proto.unsqueeze(0), dim=-1).squeeze(0)
        mixed = (1.0 - float(alpha)) * orig_image_proto[cls_id] + float(alpha) * proto
        image_proto[cls_id] = F.normalize(mixed.unsqueeze(0), dim=-1).squeeze(0)
        updated_classes.append(int(cls_id))

    stats = {
        "gated_query_count": int(gate.sum().item()),
        "gated_base_count": int((gate & (query_targets < num_base_cls)).sum().item()),
        "gated_novel_count": int((gate & (query_targets >= num_base_cls)).sum().item()),
        "updated_class_count": len(updated_classes),
        "updated_class_rate": float(len(updated_classes) / max(num_cls - num_base_cls, 1)),
        "mean_queries_per_updated_class": float(sum(per_class_counts[c] for c in updated_classes) / max(len(updated_classes), 1)) if updated_classes else 0.0,
        "per_class_counts": per_class_counts,
    }
    return image_proto, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Naive hard pseudo-label prototype refinement baseline.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--seed_override", type=int, default=None)
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=[0.5])
    parser.add_argument("--min_count", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = setup_cfg(args.data_cfg, args.train_cfg)
    if args.seed_override is not None:
        cfg.defrost()
        cfg.SEED = int(args.seed_override)
        cfg.freeze()
    out_dir = create_hard_output_dir(cfg, args.train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start = time.time()
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

    sweep = []
    best = None
    for alpha in args.alpha_grid:
        refined_proto, stats = hard_pseudo_refine_once(
            query_features=query_features,
            query_targets=query_targets,
            baseline_probs=baseline_probs,
            orig_image_proto=orig_image_proto,
            num_base_cls=num_base_cls,
            num_cls=num_cls,
            alpha=float(alpha),
            min_count=int(args.min_count),
        )
        probs = compose_full_probs(
            query_features,
            semantic_proto,
            refined_proto,
            num_base_cls,
            beta,
            ensemble_alpha,
            prob_cov,
            prob_knn,
        )
        full_acc, base_acc, novel_acc = accuracy_stats(probs, query_targets, num_base_cls)
        record = {
            "alpha": float(alpha),
            "full_acc": round(full_acc, 3),
            "base_acc": round(base_acc, 3),
            "novel_acc": round(novel_acc, 3),
            "gain": round(full_acc - baseline_full, 3),
            "updated_class_count": int(stats["updated_class_count"]),
            "updated_class_rate": round(float(stats["updated_class_rate"]), 4),
            "mean_queries_per_updated_class": round(float(stats["mean_queries_per_updated_class"]), 3),
            "gated_query_count": int(stats["gated_query_count"]),
            "gated_novel_count": int(stats["gated_novel_count"]),
        }
        sweep.append(record)
        if best is None or full_acc > best["full_acc"]:
            best = {
                "alpha": float(alpha),
                "full_acc": full_acc,
                "base_acc": base_acc,
                "novel_acc": novel_acc,
                "stats": stats,
            }

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": args.train_cfg,
        "runtime_sec": round(time.time() - start, 3),
        "num_seen_classes": int(num_cls),
        "num_base_classes": int(num_base_cls),
        "beta": beta,
        "lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "alpha_grid": [float(x) for x in args.alpha_grid],
        "min_count": int(args.min_count),
        "baseline_full_acc": round(baseline_full, 3),
        "baseline_base_acc": round(baseline_base, 3),
        "baseline_novel_acc": round(baseline_novel, 3),
        "best_alpha": round(float(best["alpha"]), 3),
        "final_full_acc": round(float(best["full_acc"]), 3),
        "final_base_acc": round(float(best["base_acc"]), 3),
        "final_novel_acc": round(float(best["novel_acc"]), 3),
        "final_gain": round(float(best["full_acc"] - baseline_full), 3),
        "updated_class_count": int(best["stats"]["updated_class_count"]),
        "updated_class_rate": round(float(best["stats"]["updated_class_rate"]), 4),
        "gated_query_count": int(best["stats"]["gated_query_count"]),
        "gated_base_count": int(best["stats"]["gated_base_count"]),
        "gated_novel_count": int(best["stats"]["gated_novel_count"]),
        "mean_queries_per_updated_class": round(float(best["stats"]["mean_queries_per_updated_class"]), 3),
        "alpha_sweep": sweep,
    }
    payload = to_builtin(payload)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    (out_dir / "summary.txt").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
