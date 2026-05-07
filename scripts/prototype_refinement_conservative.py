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
    trainer_name = f"prototype_refinement_conservative_{Path(train_cfg).stem}"
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


def conservative_refine_once(
    query_features,
    query_targets,
    baseline_probs,
    orig_image_proto,
    num_base_cls,
    num_cls,
    alpha,
    mass_thr,
    min_count,
    topk_per_class,
):
    image_proto = orig_image_proto.clone()
    preds = torch.argmax(baseline_probs, dim=1)
    pred_novel = preds >= int(num_base_cls)
    novel_probs = baseline_probs[:, num_base_cls:]
    novel_mass = novel_probs.sum(dim=1)
    top_novel_local = torch.argmax(novel_probs, dim=1)
    top_novel_global = top_novel_local + int(num_base_cls)
    gate = pred_novel & (novel_mass >= float(mass_thr))

    per_class_counts = {}
    selected_per_class = {}
    updated_classes = []
    for cls_id in range(num_base_cls, num_cls):
        mask = gate & (top_novel_global == int(cls_id))
        count = int(mask.sum().item())
        per_class_counts[int(cls_id)] = count
        if count < int(min_count):
            selected_per_class[int(cls_id)] = 0
            continue
        cls_query_features = query_features[mask]
        weights = baseline_probs[mask, cls_id]
        if topk_per_class is not None and count > int(topk_per_class):
            topk = min(int(topk_per_class), count)
            top_vals, top_idx = torch.topk(weights, k=topk, largest=True, sorted=False)
            cls_query_features = cls_query_features[top_idx]
            weights = top_vals
        selected_per_class[int(cls_id)] = int(weights.numel())
        weights = weights / torch.clamp(weights.sum(), min=1e-12)
        proto = torch.matmul(weights.unsqueeze(0), cls_query_features).squeeze(0)
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
        "mean_selected_per_updated_class": float(sum(selected_per_class[c] for c in updated_classes) / max(len(updated_classes), 1)) if updated_classes else 0.0,
        "per_class_counts": per_class_counts,
        "selected_per_class": selected_per_class,
    }
    return image_proto, stats


def latest_conservative_oracle_gain(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    oracle_root = output_root / cfg.DATASET.NAME / f"prototype_conservative_rectification_killswitch_{Path(train_cfg).stem}"
    if not oracle_root.exists():
        return None, None
    results_files = sorted(oracle_root.glob("*/results.json"))
    if not results_files:
        return None, None
    latest = results_files[-1]
    with latest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload.get("oracle_gain", 0.0)), latest.parent


def run_dataset(data_cfg, train_cfg, alpha_grid, mass_thr, min_count, topk_per_class=None, seed_override=None):
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
    for alpha in alpha_grid:
        refined_proto, stats = conservative_refine_once(
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
            "mean_selected_per_updated_class": round(float(stats["mean_selected_per_updated_class"]), 3),
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

    oracle_gain, oracle_path = latest_conservative_oracle_gain(cfg, train_cfg)
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
        "mass_thr": float(mass_thr),
        "min_count": int(min_count),
        "topk_per_class": None if topk_per_class is None else int(topk_per_class),
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
        "mean_queries_per_updated_class": round(float(best["stats"]["mean_queries_per_updated_class"]), 3),
        "mean_selected_per_updated_class": round(float(best["stats"]["mean_selected_per_updated_class"]), 3),
        "gated_query_count": int(best["stats"]["gated_query_count"]),
        "gated_base_count": int(best["stats"]["gated_base_count"]),
        "gated_novel_count": int(best["stats"]["gated_novel_count"]),
        "conservative_oracle_gain": None if oracle_gain is None else round(float(oracle_gain), 3),
        "conservative_oracle_path": None if oracle_path is None else str(oracle_path),
        "alpha_sweep": sweep,
    }
    if oracle_gain is not None and abs(oracle_gain) > 1e-8:
        payload["recovery_vs_oracle"] = round(float((best["full_acc"] - baseline_full) / oracle_gain * 100.0), 1)
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"best_alpha: {payload['best_alpha']}",
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
    parser = argparse.ArgumentParser(description="Conservative one-step prototype refinement for BiMC.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--mass_thr", type=float, default=0.3)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--topk_per_class", type=int, default=None)
    parser.add_argument("--seed_override", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        alpha_grid=args.alpha_grid,
        mass_thr=args.mass_thr,
        min_count=args.min_count,
        topk_per_class=args.topk_per_class,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
