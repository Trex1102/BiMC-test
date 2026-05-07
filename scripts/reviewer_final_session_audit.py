import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cpr_qpr_killswitch import apply_qpr
from main import setup_cfg
from prototype_refinement_conservative import (
    baseline_semantic_proto,
    compose_full_probs,
    conservative_refine_once,
    normalize_rows,
)
from prototype_refinement_transductive import sinkhorn_balanced
from query_graph_router_killswitch import accuracy_stats
from semantic_option2_killswitch import (
    collect_final_queries,
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
    trainer_name = f"reviewer_final_session_audit_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_common_state(data_cfg, train_cfg, seed_override=None):
    cfg = setup_cfg(data_cfg, train_cfg)
    if seed_override is not None:
        cfg.defrost()
        cfg.SEED = int(seed_override)
        cfg.freeze()

    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)

    device = cfg.DEVICE.DEVICE_NAME
    query_features = query_features.to(device).float()
    query_targets = query_targets.to(device)

    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    num_novel_cls = int(num_cls - num_base_cls)

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

    return {
        "cfg": cfg,
        "query_features": query_features,
        "query_targets": query_targets,
        "num_cls": int(num_cls),
        "num_base_cls": int(num_base_cls),
        "num_novel_cls": int(num_novel_cls),
        "beta": beta,
        "lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "orig_image_proto": orig_image_proto,
        "semantic_proto": semantic_proto,
        "description_features": description_features,
        "description_targets": description_targets,
        "cov_image": cov_image,
        "prob_cov": prob_cov,
        "prob_knn": prob_knn,
        "baseline_probs": baseline_probs,
        "baseline_full": float(baseline_full),
        "baseline_base": float(baseline_base),
        "baseline_novel": float(baseline_novel),
    }


def aggressive_selection_stats(state, quota_multiplier=1.0, sinkhorn_tau=0.5, sinkhorn_iters=50):
    probs = state["baseline_probs"]
    targets = state["query_targets"]
    num_base_cls = state["num_base_cls"]
    num_cls = state["num_cls"]
    num_novel_cls = state["num_novel_cls"]
    n_query = int(targets.numel())

    per_class_quota = max(1, int(round(float(n_query) / float(num_cls))))
    novel_quota = max(num_novel_cls, int(round(float(per_class_quota * num_novel_cls) * float(quota_multiplier))))

    novel_mass = probs[:, num_base_cls:].sum(dim=1)
    top_idx = torch.topk(novel_mass, k=min(novel_quota, n_query)).indices
    selected_targets = targets[top_idx]
    selected_base = int((selected_targets < num_base_cls).sum().item())
    selected_novel = int((selected_targets >= num_base_cls).sum().item())

    selected_probs = probs[top_idx, num_base_cls:]
    selected_novel_mass = torch.clamp(novel_mass[top_idx].unsqueeze(1), min=1e-12)
    novel_cond = selected_probs / selected_novel_mass
    assign = sinkhorn_balanced(novel_cond, n_iters=sinkhorn_iters, tau=sinkhorn_tau)
    hard_local = torch.argmax(assign, dim=1)

    per_class = []
    for local_idx in range(num_novel_cls):
        mask = hard_local == int(local_idx)
        count = int(mask.sum().item())
        if count == 0:
            continue
        cls_targets = selected_targets[mask]
        base_count = int((cls_targets < num_base_cls).sum().item())
        novel_count = int((cls_targets >= num_base_cls).sum().item())
        per_class.append(
            {
                "local_class_idx": int(local_idx),
                "count": count,
                "base_count": base_count,
                "novel_count": novel_count,
                "base_frac": float(base_count / max(count, 1)),
                "novel_frac": float(novel_count / max(count, 1)),
            }
        )

    return {
        "per_class_quota": int(per_class_quota),
        "novel_quota": int(novel_quota),
        "selected_query_count": int(top_idx.numel()),
        "selected_base_count": int(selected_base),
        "selected_novel_count": int(selected_novel),
        "selected_base_frac": float(selected_base / max(top_idx.numel(), 1)),
        "selected_novel_frac": float(selected_novel / max(top_idx.numel(), 1)),
        "assigned_class_count": int(len(per_class)),
        "per_class_assignment": per_class,
    }


def conservative_stats(state, alpha=0.5, mass_thr=0.3, min_count=5):
    refined_proto, stats = conservative_refine_once(
        query_features=state["query_features"],
        query_targets=state["query_targets"],
        baseline_probs=state["baseline_probs"],
        orig_image_proto=state["orig_image_proto"],
        num_base_cls=state["num_base_cls"],
        num_cls=state["num_cls"],
        alpha=alpha,
        mass_thr=mass_thr,
        min_count=min_count,
        topk_per_class=None,
    )
    cpr_probs = compose_full_probs(
        state["query_features"],
        state["semantic_proto"],
        refined_proto,
        state["num_base_cls"],
        state["beta"],
        state["ensemble_alpha"],
        state["prob_cov"],
        state["prob_knn"],
    )
    cpr_full, cpr_base, cpr_novel = accuracy_stats(cpr_probs, state["query_targets"], state["num_base_cls"])
    stats = dict(stats)
    stats.update(
        {
            "gated_base_frac": float(stats["gated_base_count"] / max(stats["gated_query_count"], 1)),
            "gated_novel_frac": float(stats["gated_novel_count"] / max(stats["gated_query_count"], 1)),
            "cpr_full_acc": float(cpr_full),
            "cpr_base_acc": float(cpr_base),
            "cpr_novel_acc": float(cpr_novel),
            "cpr_gain": float(cpr_full - state["baseline_full"]),
        }
    )
    return refined_proto, cpr_probs, stats


def prior_mismatch_stats(cpr_probs, query_targets, num_base_cls, num_cls, t_base=0.75, t_novel=0.75):
    known_ratio = float((num_cls - num_base_cls) / max(num_cls, 1))
    offsets = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    rows = []
    for off in offsets:
        target = max(0.05, min(0.95, known_ratio + off))
        calibrated, delta, achieved = apply_qpr(
            cpr_probs,
            num_base_cls=num_base_cls,
            t_base=t_base,
            t_novel=t_novel,
            target_novel_mass=target,
        )
        full, base, novel = accuracy_stats(calibrated, query_targets, num_base_cls)
        rows.append(
            {
                "target_novel_mass": float(target),
                "target_offset": float(off),
                "delta": float(delta),
                "achieved_novel_mass": float(achieved),
                "full_acc": float(full),
                "base_acc": float(base),
                "novel_acc": float(novel),
            }
        )

    rho_top1 = float((torch.argmax(cpr_probs, dim=1) >= num_base_cls).float().mean().item())
    calibrated_top1, delta_top1, achieved_top1 = apply_qpr(
        cpr_probs,
        num_base_cls=num_base_cls,
        t_base=t_base,
        t_novel=t_novel,
        target_novel_mass=rho_top1,
    )
    top1_full, top1_base, top1_novel = accuracy_stats(calibrated_top1, query_targets, num_base_cls)

    return {
        "known_ratio": float(known_ratio),
        "sweep": rows,
        "top1_estimated_prior": {
            "target_novel_mass": float(rho_top1),
            "delta": float(delta_top1),
            "achieved_novel_mass": float(achieved_top1),
            "full_acc": float(top1_full),
            "base_acc": float(top1_base),
            "novel_acc": float(top1_novel),
        },
    }


def sync_if_needed(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(device=tensor.device)


def time_op(fn, sync_tensor, warmup=5, repeats=20):
    for _ in range(int(warmup)):
        fn()
    sync_if_needed(sync_tensor)
    times = []
    for _ in range(int(repeats)):
        sync_if_needed(sync_tensor)
        t0 = time.perf_counter()
        fn()
        sync_if_needed(sync_tensor)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
        "repeats": int(repeats),
        "warmup": int(warmup),
    }


def runtime_stats(state, refined_proto, cpr_probs, t_base=0.75, t_novel=0.75, warmup=5, repeats=30):
    sync_tensor = state["query_features"]
    num_base_cls = state["num_base_cls"]
    known_ratio = float(state["num_novel_cls"] / max(state["num_cls"], 1))

    def baseline_forward():
        compose_full_probs(
            state["query_features"],
            state["semantic_proto"],
            state["orig_image_proto"],
            num_base_cls,
            state["beta"],
            state["ensemble_alpha"],
            state["prob_cov"],
            state["prob_knn"],
        )

    def cpr_update():
        conservative_refine_once(
            query_features=state["query_features"],
            query_targets=state["query_targets"],
            baseline_probs=state["baseline_probs"],
            orig_image_proto=state["orig_image_proto"],
            num_base_cls=num_base_cls,
            num_cls=state["num_cls"],
            alpha=0.5,
            mass_thr=0.3,
            min_count=5,
            topk_per_class=None,
        )

    def cpr_second_pass():
        compose_full_probs(
            state["query_features"],
            state["semantic_proto"],
            refined_proto,
            num_base_cls,
            state["beta"],
            state["ensemble_alpha"],
            state["prob_cov"],
            state["prob_knn"],
        )

    def qpr_only():
        apply_qpr(
            cpr_probs,
            num_base_cls=num_base_cls,
            t_base=t_base,
            t_novel=t_novel,
            target_novel_mass=known_ratio,
        )

    baseline_t = time_op(baseline_forward, sync_tensor, warmup=warmup, repeats=repeats)
    cpr_t = time_op(cpr_update, sync_tensor, warmup=warmup, repeats=repeats)
    second_t = time_op(cpr_second_pass, sync_tensor, warmup=warmup, repeats=repeats)
    qpr_t = time_op(qpr_only, sync_tensor, warmup=warmup, repeats=repeats)

    return {
        "baseline_final_pass": baseline_t,
        "cpr_update": cpr_t,
        "cpr_second_pass": second_t,
        "qpr": qpr_t,
        "total_posthoc_overhead_ms": float(cpr_t["mean_ms"] + second_t["mean_ms"] + qpr_t["mean_ms"]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Reviewer-focused final-session audit for BiMC CPR/QPR.")
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--seed_override", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mass_thr", type=float, default=0.3)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--t_base", type=float, default=0.75)
    parser.add_argument("--t_novel", type=float, default=0.75)
    parser.add_argument("--runtime_warmup", type=int, default=5)
    parser.add_argument("--runtime_repeats", type=int, default=20)
    parser.add_argument("--skip_runtime", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    state = load_common_state(args.data_cfg, args.train_cfg, seed_override=args.seed_override)
    out_dir = create_output_dir(state["cfg"], args.train_cfg)
    (out_dir / "config.yaml").write_text(state["cfg"].dump(), encoding="utf-8")

    refined_proto, cpr_probs, cpr_stats = conservative_stats(
        state,
        alpha=args.alpha,
        mass_thr=args.mass_thr,
        min_count=args.min_count,
    )

    payload = {
        "dataset": state["cfg"].DATASET.NAME,
        "seed": int(state["cfg"].SEED),
        "baseline": {
            "full_acc": float(state["baseline_full"]),
            "base_acc": float(state["baseline_base"]),
            "novel_acc": float(state["baseline_novel"]),
        },
        "conservative": cpr_stats,
        "aggressive": aggressive_selection_stats(state),
        "qpr_prior_mismatch": prior_mismatch_stats(
            cpr_probs,
            state["query_targets"],
            state["num_base_cls"],
            state["num_cls"],
            t_base=args.t_base,
            t_novel=args.t_novel,
        ),
    }
    if not args.skip_runtime:
        payload["runtime"] = runtime_stats(
            state,
            refined_proto,
            cpr_probs,
            t_base=args.t_base,
            t_novel=args.t_novel,
            warmup=args.runtime_warmup,
            repeats=args.runtime_repeats,
        )
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    (out_dir / "summary.txt").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
