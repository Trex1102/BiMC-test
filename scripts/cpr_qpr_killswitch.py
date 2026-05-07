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
from prototype_refinement_conservative import (
    baseline_semantic_proto,
    compose_full_probs,
    conservative_refine_once,
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


def create_qpr_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"cpr_qpr_killswitch_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_base_novel_logits(probs, num_base_cls, t_base=1.0, t_novel=1.0, eps=1e-12):
    log_probs = torch.log(torch.clamp(probs, min=eps))
    base_logits = log_probs[:, :num_base_cls] / float(t_base)
    novel_logits = log_probs[:, num_base_cls:] / float(t_novel)
    return base_logits, novel_logits


def mean_novel_mass_with_delta(base_logits, novel_logits, delta):
    base_score = torch.logsumexp(base_logits, dim=1)
    novel_score = torch.logsumexp(novel_logits + float(delta), dim=1)
    return torch.sigmoid(novel_score - base_score).mean()


def solve_group_delta(base_logits, novel_logits, target_novel_mass, low=-20.0, high=20.0, steps=60):
    target = float(target_novel_mass)
    target = max(min(target, 1.0 - 1e-6), 1e-6)
    lo = float(low)
    hi = float(high)
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        mass = float(mean_novel_mass_with_delta(base_logits, novel_logits, mid).item())
        if mass < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def apply_qpr(probs, num_base_cls, t_base, t_novel, target_novel_mass):
    base_logits, novel_logits = build_base_novel_logits(
        probs,
        num_base_cls=num_base_cls,
        t_base=t_base,
        t_novel=t_novel,
    )
    delta = solve_group_delta(base_logits, novel_logits, target_novel_mass)
    logits = torch.cat([base_logits, novel_logits + float(delta)], dim=1)
    calibrated = F.softmax(logits, dim=1)
    achieved_mass = float(calibrated[:, num_base_cls:].sum(dim=1).mean().item())
    return calibrated, delta, achieved_mass


def default_target_grid(known_novel_ratio):
    lo = max(0.1, known_novel_ratio - 0.25)
    hi = min(0.9, known_novel_ratio + 0.25)
    values = []
    cur = lo
    while cur <= hi + 1e-9:
        values.append(round(cur, 3))
        cur += 0.05
    if round(known_novel_ratio, 3) not in values:
        values.append(round(known_novel_ratio, 3))
        values = sorted(set(values))
    return values


def run_dataset(
    data_cfg,
    train_cfg,
    alpha,
    mass_thr,
    min_count,
    topk_per_class=None,
    t_base_grid=None,
    t_novel_grid=None,
    target_novel_grid=None,
    seed_override=None,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    if seed_override is not None:
        cfg.defrost()
        cfg.SEED = int(seed_override)
        cfg.freeze()
    out_dir = create_qpr_output_dir(cfg, train_cfg)
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
    known_novel_ratio = float(num_novel_cls / max(num_cls, 1))
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    if t_base_grid is None:
        t_base_grid = [0.75, 1.0, 1.25, 1.5]
    if t_novel_grid is None:
        t_novel_grid = [0.75, 1.0, 1.25, 1.5]
    if target_novel_grid is None:
        target_novel_grid = default_target_grid(known_novel_ratio)

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

    known_ratio_results = []
    best_known = None
    for t_base in t_base_grid:
        for t_novel in t_novel_grid:
            calibrated, delta, achieved_mass = apply_qpr(
                cpr_probs,
                num_base_cls=num_base_cls,
                t_base=t_base,
                t_novel=t_novel,
                target_novel_mass=known_novel_ratio,
            )
            full_acc, base_acc, novel_acc = accuracy_stats(calibrated, query_targets, num_base_cls)
            rec = {
                "t_base": float(t_base),
                "t_novel": float(t_novel),
                "target_novel_mass": round(float(known_novel_ratio), 3),
                "achieved_novel_mass": round(float(achieved_mass), 3),
                "delta": round(float(delta), 4),
                "full_acc": round(full_acc, 3),
                "base_acc": round(base_acc, 3),
                "novel_acc": round(novel_acc, 3),
                "gain_over_baseline": round(full_acc - baseline_full, 3),
                "gain_over_cpr": round(full_acc - cpr_full, 3),
            }
            known_ratio_results.append(rec)
            if best_known is None or full_acc > best_known["full_acc"]:
                best_known = {
                    "params": rec,
                    "full_acc": full_acc,
                    "base_acc": base_acc,
                    "novel_acc": novel_acc,
                }

    oracle_results = []
    best_oracle = None
    for target_novel_mass in target_novel_grid:
        for t_base in t_base_grid:
            for t_novel in t_novel_grid:
                calibrated, delta, achieved_mass = apply_qpr(
                    cpr_probs,
                    num_base_cls=num_base_cls,
                    t_base=t_base,
                    t_novel=t_novel,
                    target_novel_mass=target_novel_mass,
                )
                full_acc, base_acc, novel_acc = accuracy_stats(calibrated, query_targets, num_base_cls)
                rec = {
                    "target_novel_mass": round(float(target_novel_mass), 3),
                    "t_base": float(t_base),
                    "t_novel": float(t_novel),
                    "achieved_novel_mass": round(float(achieved_mass), 3),
                    "delta": round(float(delta), 4),
                    "full_acc": round(full_acc, 3),
                    "base_acc": round(base_acc, 3),
                    "novel_acc": round(novel_acc, 3),
                    "gain_over_baseline": round(full_acc - baseline_full, 3),
                    "gain_over_cpr": round(full_acc - cpr_full, 3),
                }
                oracle_results.append(rec)
                if best_oracle is None or full_acc > best_oracle["full_acc"]:
                    best_oracle = {
                        "params": rec,
                        "full_acc": full_acc,
                        "base_acc": base_acc,
                        "novel_acc": novel_acc,
                    }

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "num_seen_classes": int(num_cls),
        "num_base_classes": int(num_base_cls),
        "num_novel_classes": int(num_novel_cls),
        "known_novel_ratio": round(float(known_novel_ratio), 3),
        "alpha": float(alpha),
        "mass_thr": float(mass_thr),
        "min_count": int(min_count),
        "topk_per_class": None if topk_per_class is None else int(topk_per_class),
        "t_base_grid": [float(x) for x in t_base_grid],
        "t_novel_grid": [float(x) for x in t_novel_grid],
        "target_novel_grid": [float(x) for x in target_novel_grid],
        "baseline_full_acc": round(baseline_full, 3),
        "baseline_base_acc": round(baseline_base, 3),
        "baseline_novel_acc": round(baseline_novel, 3),
        "cpr_full_acc": round(cpr_full, 3),
        "cpr_base_acc": round(cpr_base, 3),
        "cpr_novel_acc": round(cpr_novel, 3),
        "cpr_gain": round(cpr_full - baseline_full, 3),
        "cpr_stats": {
            "updated_class_count": int(cpr_stats["updated_class_count"]),
            "updated_class_rate": round(float(cpr_stats["updated_class_rate"]), 4),
            "gated_query_count": int(cpr_stats["gated_query_count"]),
            "gated_base_count": int(cpr_stats["gated_base_count"]),
            "gated_novel_count": int(cpr_stats["gated_novel_count"]),
        },
        "known_ratio_best": {
            **best_known["params"],
            "full_acc": round(float(best_known["full_acc"]), 3),
            "base_acc": round(float(best_known["base_acc"]), 3),
            "novel_acc": round(float(best_known["novel_acc"]), 3),
        },
        "oracle_best": {
            **best_oracle["params"],
            "full_acc": round(float(best_oracle["full_acc"]), 3),
            "base_acc": round(float(best_oracle["base_acc"]), 3),
            "novel_acc": round(float(best_oracle["novel_acc"]), 3),
        },
        "known_ratio_sweep": known_ratio_results,
        "oracle_sweep": oracle_results,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"baseline_full_acc: {payload['baseline_full_acc']}",
        f"cpr_full_acc: {payload['cpr_full_acc']}",
        f"cpr_gain: {payload['cpr_gain']}",
        f"known_ratio_best_gain_over_cpr: {payload['known_ratio_best']['gain_over_cpr']}",
        f"oracle_best_gain_over_cpr: {payload['oracle_best']['gain_over_cpr']}",
        f"oracle_best_gain_over_baseline: {payload['oracle_best']['gain_over_baseline']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Kill switch for CPR + query-batch prior rectification.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mass_thr", type=float, default=0.3)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--topk_per_class", type=int, default=None)
    parser.add_argument("--t_base_grid", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--t_novel_grid", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--target_novel_grid", type=float, nargs="+", default=None)
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
        topk_per_class=args.topk_per_class,
        t_base_grid=args.t_base_grid,
        t_novel_grid=args.t_novel_grid,
        target_novel_grid=args.target_novel_grid,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
