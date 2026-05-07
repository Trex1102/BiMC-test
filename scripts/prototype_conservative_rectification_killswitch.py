import argparse
import itertools
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
    trainer_name = f"prototype_conservative_rectification_killswitch_{Path(train_cfg).stem}"
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
    probs, _ = compose_probs_from_semantic(
        query_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    return probs


def build_gate_masks(probs, num_base_cls, mass_thr, top_novel_thr, margin_thr):
    top_idx = torch.argmax(probs, dim=1)
    pred_novel = top_idx >= int(num_base_cls)
    novel_probs = probs[:, num_base_cls:]
    novel_mass = novel_probs.sum(dim=1)
    top_novel_prob = novel_probs.max(dim=1).values
    top_base_prob = probs[:, :num_base_cls].max(dim=1).values
    margin = top_novel_prob - top_base_prob
    gate = pred_novel & (novel_mass >= float(mass_thr)) & (top_novel_prob >= float(top_novel_thr)) & (margin >= float(margin_thr))
    return {
        "gate": gate,
        "pred_novel": pred_novel,
        "novel_mass": novel_mass,
        "top_novel_prob": top_novel_prob,
        "top_base_prob": top_base_prob,
        "margin": margin,
    }


def gated_true_means(query_features, query_targets, gate_mask, num_base_cls, num_cls):
    means = {}
    counts = {}
    for cls_id in range(num_base_cls, num_cls):
        mask = gate_mask & (query_targets == int(cls_id))
        count = int(mask.sum().item())
        counts[int(cls_id)] = count
        if count > 0:
            means[int(cls_id)] = F.normalize(query_features[mask].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
    return means, counts


def probs_with_alpha(
    query_features,
    semantic_proto,
    orig_image_proto,
    means,
    alpha,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
):
    image_proto = orig_image_proto.clone()
    for cls_id, mean_vec in means.items():
        mixed = (1.0 - float(alpha)) * orig_image_proto[cls_id] + float(alpha) * mean_vec
        image_proto[cls_id] = F.normalize(mixed.unsqueeze(0), dim=-1).squeeze(0)
    return compose_full_probs(
        query_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )


def global_sweep_for_gate(
    query_features,
    query_targets,
    semantic_proto,
    orig_image_proto,
    means,
    alpha_grid,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
):
    sweep = []
    for alpha in alpha_grid:
        probs = probs_with_alpha(
            query_features,
            semantic_proto,
            orig_image_proto,
            means,
            alpha,
            num_base_cls,
            beta,
            ensemble_alpha,
            prob_cov,
            prob_knn,
        )
        full_acc, base_acc, novel_acc = accuracy_stats(probs, query_targets, num_base_cls)
        sweep.append(
            {
                "alpha": float(alpha),
                "full_acc": round(full_acc, 3),
                "base_acc": round(base_acc, 3),
                "novel_acc": round(novel_acc, 3),
            }
        )
    return sweep


def greedy_oracle_for_best_gate(
    query_features,
    query_targets,
    semantic_proto,
    orig_image_proto,
    means,
    alpha_grid,
    num_base_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
    max_passes,
):
    current_proto = orig_image_proto.clone()
    current_probs = compose_full_probs(
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
    chosen_alpha = {int(cls_id): 0.0 for cls_id in range(num_base_cls, orig_image_proto.shape[0])}

    for _ in range(int(max_passes)):
        improved = False
        for cls_id in range(num_base_cls, orig_image_proto.shape[0]):
            if cls_id not in means:
                continue
            best_full = current_full
            best_alpha = chosen_alpha[int(cls_id)]
            best_proto = None
            best_probs = None
            best_base = current_base
            best_novel = current_novel

            for alpha in alpha_grid:
                candidate_proto = current_proto.clone()
                mixed = (1.0 - float(alpha)) * orig_image_proto[cls_id] + float(alpha) * means[cls_id]
                candidate_proto[cls_id] = F.normalize(mixed.unsqueeze(0), dim=-1).squeeze(0)
                probs = compose_full_probs(
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
        "full_acc": current_full,
        "base_acc": current_base,
        "novel_acc": current_novel,
        "changed_class_count": len(changed),
        "changed_class_rate": float(len(changed) / max(len(chosen_alpha), 1)),
        "mean_alpha_changed": float(sum(chosen_alpha[c] for c in changed) / max(len(changed), 1)) if changed else 0.0,
        "chosen_alpha": chosen_alpha,
    }


def evaluate_gate_candidates(
    query_features,
    query_targets,
    baseline_probs,
    semantic_proto,
    orig_image_proto,
    num_base_cls,
    num_cls,
    beta,
    ensemble_alpha,
    prob_cov,
    prob_knn,
    alpha_grid,
    mass_grid,
    top_novel_grid,
    margin_grid,
):
    candidates = []
    for mass_thr, top_novel_thr, margin_thr in itertools.product(mass_grid, top_novel_grid, margin_grid):
        gate_info = build_gate_masks(
            baseline_probs,
            num_base_cls=num_base_cls,
            mass_thr=mass_thr,
            top_novel_thr=top_novel_thr,
            margin_thr=margin_thr,
        )
        means, counts = gated_true_means(
            query_features,
            query_targets,
            gate_info["gate"],
            num_base_cls,
            num_cls,
        )
        sweep = global_sweep_for_gate(
            query_features,
            query_targets,
            semantic_proto,
            orig_image_proto,
            means,
            alpha_grid,
            num_base_cls,
            beta,
            ensemble_alpha,
            prob_cov,
            prob_knn,
        )
        best = max(sweep, key=lambda x: x["full_acc"])
        novel_mask = query_targets >= int(num_base_cls)
        base_mask = ~novel_mask
        candidates.append(
            {
                "mass_thr": float(mass_thr),
                "top_novel_thr": float(top_novel_thr),
                "margin_thr": float(margin_thr),
                "gated_query_count": int(gate_info["gate"].sum().item()),
                "gated_base_count": int((gate_info["gate"] & base_mask).sum().item()),
                "gated_novel_count": int((gate_info["gate"] & novel_mask).sum().item()),
                "novel_gate_recall": round(
                    float((gate_info["gate"] & novel_mask).sum().item() / max(int(novel_mask.sum().item()), 1)),
                    4,
                ),
                "covered_novel_classes": int(sum(1 for c in range(num_base_cls, num_cls) if counts.get(int(c), 0) > 0)),
                "best_alpha": float(best["alpha"]),
                "best_full_acc": float(best["full_acc"]),
                "best_base_acc": float(best["base_acc"]),
                "best_novel_acc": float(best["novel_acc"]),
                "sweep": sweep,
                "counts": counts,
                "means": means,
            }
        )
    return candidates


def run_dataset(
    data_cfg,
    train_cfg,
    alpha_grid,
    mass_grid,
    top_novel_grid,
    margin_grid,
    max_passes,
):
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

    candidates = evaluate_gate_candidates(
        query_features,
        query_targets,
        baseline_probs,
        semantic_proto,
        orig_image_proto,
        num_base_cls,
        num_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
        alpha_grid,
        mass_grid,
        top_novel_grid,
        margin_grid,
    )
    best_gate = max(candidates, key=lambda x: x["best_full_acc"])

    oracle = greedy_oracle_for_best_gate(
        query_features,
        query_targets,
        semantic_proto,
        orig_image_proto,
        best_gate["means"],
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
        "alpha_grid": [float(x) for x in alpha_grid],
        "mass_grid": [float(x) for x in mass_grid],
        "top_novel_grid": [float(x) for x in top_novel_grid],
        "margin_grid": [float(x) for x in margin_grid],
        "max_passes": int(max_passes),
        "baseline_full_acc": round(baseline_full, 3),
        "baseline_base_acc": round(baseline_base, 3),
        "baseline_novel_acc": round(baseline_novel, 3),
        "best_gate_mass_thr": best_gate["mass_thr"],
        "best_gate_top_novel_thr": best_gate["top_novel_thr"],
        "best_gate_margin_thr": best_gate["margin_thr"],
        "best_gate_gated_query_count": best_gate["gated_query_count"],
        "best_gate_gated_base_count": best_gate["gated_base_count"],
        "best_gate_gated_novel_count": best_gate["gated_novel_count"],
        "best_gate_novel_recall": best_gate["novel_gate_recall"],
        "best_gate_covered_novel_classes": best_gate["covered_novel_classes"],
        "best_global_alpha": round(float(best_gate["best_alpha"]), 3),
        "best_global_full_acc": round(float(best_gate["best_full_acc"]), 3),
        "best_global_base_acc": round(float(best_gate["best_base_acc"]), 3),
        "best_global_novel_acc": round(float(best_gate["best_novel_acc"]), 3),
        "best_global_gain": round(float(best_gate["best_full_acc"] - baseline_full), 3),
        "oracle_full_acc": round(float(oracle["full_acc"]), 3),
        "oracle_base_acc": round(float(oracle["base_acc"]), 3),
        "oracle_novel_acc": round(float(oracle["novel_acc"]), 3),
        "oracle_gain": round(float(oracle["full_acc"] - baseline_full), 3),
        "changed_class_count": int(oracle["changed_class_count"]),
        "changed_class_rate": round(float(oracle["changed_class_rate"]), 4),
        "mean_alpha_changed": round(float(oracle["mean_alpha_changed"]), 4),
        "gate_candidates": [
            {
                "mass_thr": cand["mass_thr"],
                "top_novel_thr": cand["top_novel_thr"],
                "margin_thr": cand["margin_thr"],
                "gated_query_count": cand["gated_query_count"],
                "gated_base_count": cand["gated_base_count"],
                "gated_novel_count": cand["gated_novel_count"],
                "novel_gate_recall": cand["novel_gate_recall"],
                "covered_novel_classes": cand["covered_novel_classes"],
                "best_alpha": cand["best_alpha"],
                "best_full_acc": round(float(cand["best_full_acc"]), 3),
                "best_base_acc": round(float(cand["best_base_acc"]), 3),
                "best_novel_acc": round(float(cand["best_novel_acc"]), 3),
            }
            for cand in candidates
        ],
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"best_global_gain: {payload['best_global_gain']}",
        f"oracle_gain: {payload['oracle_gain']}",
        f"best_gate_mass_thr: {payload['best_gate_mass_thr']}",
        f"best_gate_top_novel_thr: {payload['best_gate_top_novel_thr']}",
        f"best_gate_margin_thr: {payload['best_gate_margin_thr']}",
        f"best_gate_gated_query_count: {payload['best_gate_gated_query_count']}",
        f"best_gate_gated_base_count: {payload['best_gate_gated_base_count']}",
        f"best_gate_gated_novel_count: {payload['best_gate_gated_novel_count']}",
        f"best_gate_novel_recall: {payload['best_gate_novel_recall']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Base-aware conservative prototype rectification kill switch for BiMC.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--mass_grid", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument("--top_novel_grid", type=float, nargs="+", default=[0.15, 0.25, 0.35])
    parser.add_argument("--margin_grid", type=float, nargs="+", default=[0.0, 0.05, 0.1])
    parser.add_argument("--max_passes", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        alpha_grid=args.alpha_grid,
        mass_grid=args.mass_grid,
        top_novel_grid=args.top_novel_grid,
        margin_grid=args.margin_grid,
        max_passes=args.max_passes,
    )


if __name__ == "__main__":
    main()
