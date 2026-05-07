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
    trainer_name = f"prototype_rectification_killswitch_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_rows(tensor):
    return F.normalize(tensor.float(), dim=-1)


def compute_baseline_semantic(merged_state, num_cls, lambda_t, device):
    text_features = normalize_rows(merged_state["text_features"][:num_cls].to(device))
    description_proto = normalize_rows(merged_state["description_proto"][:num_cls].to(device))
    return normalize_rows((1.0 - lambda_t) * text_features + lambda_t * description_proto)


def novel_query_means(query_features, query_targets, num_base_cls, num_cls):
    means = {}
    for cls_id in range(num_base_cls, num_cls):
        mask = query_targets == int(cls_id)
        if mask.any():
            means[int(cls_id)] = F.normalize(query_features[mask].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
    return means


def probs_for_image_proto(
    query_features,
    semantic_proto,
    image_proto,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
):
    return compose_probs_from_semantic(
        query_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )[0]


def sweep_global_alpha(
    query_features,
    query_targets,
    semantic_proto,
    image_proto,
    novel_means,
    alpha_grid,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
):
    results = []
    num_cls = image_proto.shape[0]
    for alpha in alpha_grid:
        candidate = image_proto.clone()
        for cls_id in range(num_base_cls, num_cls):
            if cls_id not in novel_means:
                continue
            mixed = (1.0 - float(alpha)) * image_proto[cls_id] + float(alpha) * novel_means[cls_id]
            candidate[cls_id] = F.normalize(mixed.unsqueeze(0), dim=-1).squeeze(0)
        probs = probs_for_image_proto(
            query_features,
            semantic_proto,
            candidate,
            num_base_cls,
            beta,
            ensemble_alpha,
            prob_cov,
            prob_knn,
        )
        full_acc, base_acc, novel_acc = accuracy_stats(probs, query_targets, num_base_cls)
        results.append(
            {
                "alpha": float(alpha),
                "full_acc": round(full_acc, 3),
                "base_acc": round(base_acc, 3),
                "novel_acc": round(novel_acc, 3),
            }
        )
    return results


def greedy_oracle_rectification(
    query_features,
    query_targets,
    semantic_proto,
    image_proto,
    novel_means,
    alpha_grid,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
    max_passes,
):
    current_proto = image_proto.clone()
    current_probs = probs_for_image_proto(
        query_features,
        semantic_proto,
        current_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    current_full, current_base, current_novel = accuracy_stats(current_probs, query_targets, num_base_cls)
    chosen_alpha = {int(cls_id): 0.0 for cls_id in range(num_base_cls, image_proto.shape[0])}

    for _ in range(max_passes):
        improved = False
        for cls_id in range(num_base_cls, image_proto.shape[0]):
            if cls_id not in novel_means:
                continue
            best_alpha = chosen_alpha[int(cls_id)]
            best_full = current_full
            best_proto = None
            best_probs = None
            best_base = current_base
            best_novel = current_novel

            for alpha in alpha_grid:
                candidate_proto = current_proto.clone()
                mixed = (1.0 - float(alpha)) * image_proto[cls_id] + float(alpha) * novel_means[cls_id]
                candidate_proto[cls_id] = F.normalize(mixed.unsqueeze(0), dim=-1).squeeze(0)
                probs = probs_for_image_proto(
                    query_features,
                    semantic_proto,
                    candidate_proto,
                    num_base_cls,
                    beta,
                    ensemble_alpha,
                    prob_cov,
                    prob_knn,
                )
                full_acc, base_acc, novel_acc = accuracy_stats(probs, query_targets, num_base_cls)
                if full_acc > best_full + 1e-6:
                    best_full = full_acc
                    best_alpha = float(alpha)
                    best_proto = candidate_proto
                    best_probs = probs
                    best_base = base_acc
                    best_novel = novel_acc

            if best_proto is not None:
                current_proto = best_proto
                current_probs = best_probs
                current_full = best_full
                current_base = best_base
                current_novel = best_novel
                chosen_alpha[int(cls_id)] = best_alpha
                improved = True
        if not improved:
            break

    changed = [cls_id for cls_id, alpha in chosen_alpha.items() if abs(alpha) > 1e-8]
    return {
        "image_proto": current_proto,
        "probs": current_probs,
        "full_acc": current_full,
        "base_acc": current_base,
        "novel_acc": current_novel,
        "chosen_alpha": chosen_alpha,
        "changed_class_count": len(changed),
        "changed_class_rate": float(len(changed) / max(len(chosen_alpha), 1)),
        "mean_alpha_changed": float(sum(chosen_alpha[c] for c in changed) / max(len(changed), 1)) if changed else 0.0,
    }


def run_dataset(data_cfg, train_cfg, alpha_grid, max_passes):
    cfg = setup_cfg(data_cfg, train_cfg)
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

    image_proto = normalize_rows(merged_state["image_proto"][:num_cls].to(device))
    semantic_proto = compute_baseline_semantic(merged_state, num_cls, lambda_t, device)
    description_features = normalize_rows(merged_state["description_features"].to(device))
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device).float()

    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )

    baseline_probs = probs_for_image_proto(
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

    means = novel_query_means(query_features, query_targets, num_base_cls, num_cls)
    global_sweep = sweep_global_alpha(
        query_features,
        query_targets,
        semantic_proto,
        image_proto,
        means,
        alpha_grid,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    best_global = max(global_sweep, key=lambda x: x["full_acc"])

    oracle = greedy_oracle_rectification(
        query_features,
        query_targets,
        semantic_proto,
        image_proto,
        means,
        alpha_grid,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
        max_passes=max_passes,
    )

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "num_seen_classes": int(num_cls),
        "num_base_classes": int(num_base_cls),
        "beta": beta,
        "lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "alpha_grid": [float(a) for a in alpha_grid],
        "max_passes": int(max_passes),
        "baseline_full_acc": round(baseline_full, 3),
        "baseline_base_acc": round(baseline_base, 3),
        "baseline_novel_acc": round(baseline_novel, 3),
        "global_best_alpha": float(best_global["alpha"]),
        "global_best_full_acc": round(float(best_global["full_acc"]), 3),
        "global_best_base_acc": round(float(best_global["base_acc"]), 3),
        "global_best_novel_acc": round(float(best_global["novel_acc"]), 3),
        "global_best_gain": round(float(best_global["full_acc"] - baseline_full), 3),
        "oracle_full_acc": round(float(oracle["full_acc"]), 3),
        "oracle_base_acc": round(float(oracle["base_acc"]), 3),
        "oracle_novel_acc": round(float(oracle["novel_acc"]), 3),
        "oracle_gain": round(float(oracle["full_acc"] - baseline_full), 3),
        "changed_class_count": int(oracle["changed_class_count"]),
        "changed_class_rate": round(float(oracle["changed_class_rate"]), 4),
        "mean_alpha_changed": round(float(oracle["mean_alpha_changed"]), 4),
        "num_novel_means": int(len(means)),
        "global_sweep": global_sweep,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"global_best_alpha: {payload['global_best_alpha']}",
        f"global_best_gain: {payload['global_best_gain']}",
        f"oracle_full_acc: {payload['oracle_full_acc']}",
        f"oracle_gain: {payload['oracle_gain']}",
        f"baseline_base_acc: {payload['baseline_base_acc']}",
        f"baseline_novel_acc: {payload['baseline_novel_acc']}",
        f"oracle_base_acc: {payload['oracle_base_acc']}",
        f"oracle_novel_acc: {payload['oracle_novel_acc']}",
        f"changed_class_rate: {payload['changed_class_rate']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Novel prototype rectification kill switch for BiMC.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--max_passes", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        alpha_grid=args.alpha_grid,
        max_passes=args.max_passes,
    )


if __name__ == "__main__":
    main()
