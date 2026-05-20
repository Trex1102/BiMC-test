import argparse
import contextlib
import json
import socket
import time
from datetime import datetime
from pathlib import Path

from tric_followup_variant_suite import (
    DEFAULT_DATA_CFGS,
    DEFAULT_SUBSPACE_DIM,
    DEFAULT_SUBSPACE_TEMP,
    DEFAULT_TRAIN_CFG,
    accuracy_stats_from_probs,
    build_final_problem,
    build_subspace_branch,
    conservative_router,
    dataset_output_name,
    selective_semantic_repair,
    selective_subspace_repair,
)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_output_root(base_output_root):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(base_output_root) / f"tric_nonqpr_followup_suite_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def summarize_variant_set(variant_probs, targets, num_base_cls, reference_name):
    summary = {}
    ref_acc = accuracy_stats_from_probs(variant_probs[reference_name], targets, num_base_cls)["full_acc"]
    for name, probs in variant_probs.items():
        stats = accuracy_stats_from_probs(probs, targets, num_base_cls)
        stats["gain_vs_reference"] = round(float(stats["full_acc"] - ref_acc), 3)
        summary[name] = stats
    return summary


def write_summary(out_dir, payload):
    lines = [
        f"dataset: {payload['dataset']}",
        f"data_cfg: {payload['data_cfg']}",
        f"train_cfg: {payload['train_cfg']}",
        f"seed: {payload['seed']}",
        f"started_at: {payload['started_at']}",
        f"completed_at: {payload['completed_at']}",
        f"runtime_sec: {payload['runtime_sec']}",
        f"num_queries: {payload['num_queries']}",
        "",
    ]
    for name, stats in payload["variant_summary"].items():
        lines.append(
            f"{name}: full={stats['full_acc']} base={stats['base_acc']} "
            f"novel={stats['novel_acc']} gain_vs_tric={stats['gain_vs_reference']}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_variants(problem, subspace_dim, subspace_temp):
    base = problem["base_probs"]
    _, subspace_probs = build_subspace_branch(
        problem,
        problem["dino_image_proto"],
        subspace_dim=subspace_dim,
        subspace_temp=subspace_temp,
    )

    semantic_repair_probs, semantic_repair_meta = selective_semantic_repair(
        default_probs=base["tric"],
        semantic_probs=base["semantic_branch_only"],
        knn_probs=base["description_knn_branch"],
    )
    subspace_repair_probs, subspace_repair_meta = selective_subspace_repair(
        default_probs=base["tric"],
        visual_probs=base["clip_dino_visual_only"],
        cov_probs=base["covariance_branch"],
        subspace_probs=subspace_probs,
    )
    safe_router_probs, safe_router_meta = conservative_router(
        default_probs=base["tric"],
        cov_probs=base["covariance_branch"],
        knn_probs=base["description_knn_branch"],
        visual_probs=base["clip_dino_visual_only"],
        text_probs=base["semantic_branch_only"],
        subspace_probs=subspace_probs,
        num_base_cls=problem["num_base_cls"],
    )

    variant_probs = {
        "semantic_branch_only": base["semantic_branch_only"],
        "clip_visual_only": base["clip_visual_only"],
        "dino_visual_only": base["dino_visual_only"],
        "clip_dino_visual_only": base["clip_dino_visual_only"],
        "tric_no_mask": base["tric_no_mask"],
        "tric": base["tric"],
        "dino_subspace_only": subspace_probs,
        "tric_semantic_repair_noqpr": semantic_repair_probs,
        "tric_subspace_repair_noqpr": subspace_repair_probs,
        "tric_safe_router_noqpr": safe_router_probs,
    }
    metadata = {
        "dino_subspace_only": {
            "subspace_dim": int(subspace_dim),
            "subspace_temp": float(subspace_temp),
        },
        "tric_semantic_repair_noqpr": semantic_repair_meta,
        "tric_subspace_repair_noqpr": subspace_repair_meta,
        "tric_safe_router_noqpr": safe_router_meta,
    }
    return variant_probs, metadata


def run_dataset(data_cfg, train_cfg, seed, output_root, subspace_dim, subspace_temp):
    dataset_label = dataset_output_name(data_cfg)
    out_dir = output_root / dataset_label / f"seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    driver_log = out_dir / "driver.log"

    started_at = now_iso()
    start_time = time.time()
    with driver_log.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            problem = build_final_problem(data_cfg, train_cfg, seed)

    variant_probs, variant_meta = evaluate_variants(
        problem,
        subspace_dim=subspace_dim,
        subspace_temp=subspace_temp,
    )
    variant_summary = summarize_variant_set(
        variant_probs,
        problem["query_targets"],
        problem["num_base_cls"],
        reference_name="tric",
    )

    completed_at = now_iso()
    runtime_sec = round(time.time() - start_time, 3)
    payload = {
        "analysis": "tric_nonqpr_followup_suite_final_session",
        "dataset": problem["dataset_name"],
        "data_cfg": data_cfg,
        "train_cfg": train_cfg,
        "seed": int(seed),
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime_sec": runtime_sec,
        "hostname": socket.gethostname(),
        "num_queries": int(problem["query_targets"].numel()),
        "num_seen_classes": int(problem["num_cls"]),
        "num_base_classes": int(problem["num_base_cls"]),
        "beta": problem["beta"],
        "lambda_t": problem["lambda_t"],
        "ensemble_alpha": problem["ensemble_alpha"],
        "omega": problem["omega"],
        "variant_summary": variant_summary,
        "variant_metadata": variant_meta,
        "files": {
            "results_json": str(out_dir / "results.json"),
            "summary_txt": str(out_dir / "summary.txt"),
            "driver_log": str(driver_log),
        },
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_summary(out_dir, payload)
    (out_dir / "completion.txt").write_text(
        f"started_at={started_at}\ncompleted_at={completed_at}\nruntime_sec={runtime_sec}\nexit_status=0\n",
        encoding="utf-8",
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Final-session non-QPR TriC follow-up suite.")
    parser.add_argument("--data_cfgs", nargs="+", default=DEFAULT_DATA_CFGS)
    parser.add_argument("--train_cfg", default=DEFAULT_TRAIN_CFG)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_root", default="experiments")
    parser.add_argument("--subspace_dim", type=int, default=DEFAULT_SUBSPACE_DIM)
    parser.add_argument("--subspace_temp", type=float, default=DEFAULT_SUBSPACE_TEMP)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = make_output_root(args.output_root)
    suite_started_at = now_iso()
    suite_start = time.time()

    runs = []
    for data_cfg in args.data_cfgs:
        payload = run_dataset(
            data_cfg=data_cfg,
            train_cfg=args.train_cfg,
            seed=args.seed,
            output_root=output_root,
            subspace_dim=args.subspace_dim,
            subspace_temp=args.subspace_temp,
        )
        runs.append(
            {
                "dataset": payload["dataset"],
                "completed_at": payload["completed_at"],
                "runtime_sec": payload["runtime_sec"],
                "variant_summary": payload["variant_summary"],
                "files": payload["files"],
            }
        )

    suite_payload = {
        "analysis": "tric_nonqpr_followup_suite_final_session",
        "started_at": suite_started_at,
        "completed_at": now_iso(),
        "runtime_sec": round(time.time() - suite_start, 3),
        "seed": int(args.seed),
        "train_cfg": args.train_cfg,
        "data_cfgs": args.data_cfgs,
        "output_root": str(output_root),
        "runs": runs,
    }
    with (output_root / "suite_summary.json").open("w", encoding="utf-8") as f:
        json.dump(suite_payload, f, indent=2)
    print(json.dumps(suite_payload, indent=2))


if __name__ == "__main__":
    main()
