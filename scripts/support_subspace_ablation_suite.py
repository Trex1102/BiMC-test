import argparse
import contextlib
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from support_subspace_prototype_suite import (
    BIMC_DATA_CFGS,
    BIMC_TRAIN_CFG,
    TRIC_DATA_CFGS,
    TRIC_TRAIN_CFG,
    build_bimc_problem,
    build_tric_problem,
    dataset_output_name,
    evaluate_problem,
    output_root_from_arg,
)


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def framework_spec(framework_name):
    if framework_name == "bimc_clip":
        return {
            "framework": framework_name,
            "builder": build_bimc_problem,
            "train_cfg": BIMC_TRAIN_CFG,
            "data_cfgs": BIMC_DATA_CFGS,
        }
    if framework_name == "tric_dinov2":
        return {
            "framework": framework_name,
            "builder": build_tric_problem,
            "train_cfg": TRIC_TRAIN_CFG,
            "data_cfgs": TRIC_DATA_CFGS,
        }
    raise ValueError(f"unknown framework: {framework_name}")


def build_variant_specs(shared_rank, lambda_unique, beta_residual, novel_only, weighted_unique, eps, subspace_rank):
    common = {
        "novel_only": bool(novel_only),
        "weighted_unique": bool(weighted_unique),
        "eps": float(eps),
        "subspace_rank": int(subspace_rank),
    }
    return [
        {
            "name": "shared_removal_only",
            **common,
            "shared_rank": int(shared_rank),
            "lambda_unique": 0.0,
            "beta_residual": 0.0,
        },
        {
            "name": "subspace_term_only",
            **common,
            "shared_rank": int(shared_rank),
            "lambda_unique": float(lambda_unique),
            "beta_residual": 0.0,
        },
        {
            "name": "competitor_residual_only",
            **common,
            "shared_rank": int(shared_rank),
            "lambda_unique": 0.0,
            "beta_residual": float(beta_residual),
        },
        {
            "name": "full_refined",
            **common,
            "shared_rank": int(shared_rank),
            "lambda_unique": float(lambda_unique),
            "beta_residual": float(beta_residual),
        },
        {
            "name": "shared_rank0_control",
            **common,
            "shared_rank": 0,
            "lambda_unique": float(lambda_unique),
            "beta_residual": float(beta_residual),
        },
    ]


def round3(value):
    return round(float(value), 3)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def variant_payload(evaluation, params):
    return {
        "params": {
            "shared_rank": int(params["shared_rank"]),
            "subspace_rank": int(params["subspace_rank"]),
            "lambda_unique": float(params["lambda_unique"]),
            "beta_residual": float(params["beta_residual"]),
            "novel_only": bool(params["novel_only"]),
            "weighted_unique": bool(params["weighted_unique"]),
            "eps": float(params["eps"]),
        },
        "full": evaluation["refined"]["full"],
        "visual": evaluation["refined"]["visual"],
        "gains": {
            "full_gain": round3(evaluation["gains"]["refined_full_gain"]),
            "visual_gain": round3(evaluation["gains"]["refined_visual_gain"]),
        },
        "geometry": evaluation["geometry"],
    }


def compare_variants(variants):
    shared = variants["shared_removal_only"]
    subspace = variants["subspace_term_only"]
    residual = variants["competitor_residual_only"]
    full = variants["full_refined"]
    no_shared = variants["shared_rank0_control"]
    return {
        "full_acc": {
            "shared_removal_effect_vs_baseline": round3(shared["gains"]["full_gain"]),
            "subspace_increment_over_shared": round3(subspace["full"]["full_acc"] - shared["full"]["full_acc"]),
            "residual_increment_over_shared": round3(residual["full"]["full_acc"] - shared["full"]["full_acc"]),
            "full_increment_over_shared": round3(full["full"]["full_acc"] - shared["full"]["full_acc"]),
            "shared_rank0_minus_full": round3(no_shared["full"]["full_acc"] - full["full"]["full_acc"]),
        },
        "visual_acc": {
            "shared_removal_effect_vs_baseline": round3(shared["gains"]["visual_gain"]),
            "subspace_increment_over_shared": round3(subspace["visual"]["full_acc"] - shared["visual"]["full_acc"]),
            "residual_increment_over_shared": round3(residual["visual"]["full_acc"] - shared["visual"]["full_acc"]),
            "full_increment_over_shared": round3(full["visual"]["full_acc"] - shared["visual"]["full_acc"]),
            "shared_rank0_minus_full": round3(no_shared["visual"]["full_acc"] - full["visual"]["full_acc"]),
        },
    }


def summary_lines(result):
    lines = [
        f"framework: {result['framework']}",
        f"dataset: {result['dataset_key']}",
        f"seed: {result['seed']}",
        f"started_at: {result['started_at']}",
        f"completed_at: {result['completed_at']}",
        f"runtime_sec: {result['runtime_sec']}",
        "",
        f"baseline_full: {result['baseline']['full']['full_acc']}",
        f"baseline_visual: {result['baseline']['visual']['full_acc']}",
        f"mean_rebuild_full: {result['mean_rebuild']['full']['full_acc']}",
        f"mean_rebuild_visual: {result['mean_rebuild']['visual']['full_acc']}",
        "",
    ]
    for name, payload in result["variants"].items():
        lines.extend(
            [
                f"{name}:",
                f"  full_acc: {payload['full']['full_acc']}",
                f"  visual_acc: {payload['visual']['full_acc']}",
                f"  full_gain: {payload['gains']['full_gain']}",
                f"  visual_gain: {payload['gains']['visual_gain']}",
                f"  shared_rank: {payload['params']['shared_rank']}",
                f"  lambda_unique: {payload['params']['lambda_unique']}",
                f"  beta_residual: {payload['params']['beta_residual']}",
                "",
            ]
        )
    lines.extend(
        [
            "comparisons:",
            f"  full_shared_removal_effect_vs_baseline: {result['comparisons']['full_acc']['shared_removal_effect_vs_baseline']}",
            f"  full_subspace_increment_over_shared: {result['comparisons']['full_acc']['subspace_increment_over_shared']}",
            f"  full_residual_increment_over_shared: {result['comparisons']['full_acc']['residual_increment_over_shared']}",
            f"  full_shared_rank0_minus_full: {result['comparisons']['full_acc']['shared_rank0_minus_full']}",
            f"  visual_shared_rank0_minus_full: {result['comparisons']['visual_acc']['shared_rank0_minus_full']}",
        ]
    )
    return lines


def run_single_dataset(spec, data_cfg, seed, output_root, variant_specs):
    dataset_key = dataset_output_name(data_cfg)
    out_dir = output_root / spec["framework"] / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = now_iso()
    start_time = time.time()
    problem = spec["builder"](data_cfg, spec["train_cfg"], seed)

    variants = {}
    baseline = None
    mean_rebuild = None
    for params in variant_specs:
        evaluation = evaluate_problem(problem, params)
        if baseline is None:
            baseline = evaluation["baseline"]
            mean_rebuild = evaluation["mean_rebuild"]
        variants[params["name"]] = variant_payload(evaluation, params)

    completed_at = now_iso()
    result = {
        "framework": spec["framework"],
        "dataset_key": dataset_key,
        "dataset_name": problem["dataset_name"],
        "data_cfg": data_cfg,
        "train_cfg": spec["train_cfg"],
        "seed": int(seed),
        "host": socket.gethostname(),
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime_sec": round(time.time() - start_time, 3),
        "baseline": baseline,
        "mean_rebuild": mean_rebuild,
        "variants": variants,
        "comparisons": compare_variants(variants),
        "num_seen_classes": int(problem["num_cls"]),
        "num_base_classes": int(problem["num_base_cls"]),
    }

    write_json(out_dir / "results.json", result)
    (out_dir / "summary.txt").write_text("\n".join(summary_lines(result)) + "\n", encoding="utf-8")
    (out_dir / "completion.txt").write_text(f"{completed_at}\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Ablation suite for support-only subspace prototype refinement.")
    parser.add_argument("--frameworks", nargs="+", default=["bimc_clip", "tric_dinov2"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_root", default="experiments")
    parser.add_argument("--shared_rank", type=int, default=1)
    parser.add_argument("--subspace_rank", type=int, default=3)
    parser.add_argument("--lambda_unique", type=float, default=0.35)
    parser.add_argument("--beta_residual", type=float, default=0.15)
    parser.add_argument("--all_seen", action="store_true")
    parser.add_argument("--no_weighted_unique", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    suite_root = output_root_from_arg(args.output_root, "support_subspace_ablation_suite")
    suite_started_at = now_iso()
    suite_start = time.time()
    suite_results = []

    variant_specs = build_variant_specs(
        shared_rank=args.shared_rank,
        lambda_unique=args.lambda_unique,
        beta_residual=args.beta_residual,
        novel_only=not args.all_seen,
        weighted_unique=not args.no_weighted_unique,
        eps=args.eps,
        subspace_rank=args.subspace_rank,
    )

    for framework_name in args.frameworks:
        spec = framework_spec(framework_name)
        for data_cfg in spec["data_cfgs"]:
            print(
                f"[{now_iso()}] start framework={framework_name} dataset={dataset_output_name(data_cfg)} seed={args.seed}",
                flush=True,
            )
            result = run_single_dataset(
                spec=spec,
                data_cfg=data_cfg,
                seed=args.seed,
                output_root=suite_root,
                variant_specs=variant_specs,
            )
            suite_results.append(result)
            print(
                f"[{now_iso()}] done framework={framework_name} dataset={result['dataset_key']} "
                f"shared_only={result['variants']['shared_removal_only']['full']['full_acc']:.3f} "
                f"full={result['variants']['full_refined']['full']['full_acc']:.3f} "
                f"no_shared={result['variants']['shared_rank0_control']['full']['full_acc']:.3f}",
                flush=True,
            )

    suite_completed_at = now_iso()
    summary = {
        "host": socket.gethostname(),
        "seed": int(args.seed),
        "frameworks": list(args.frameworks),
        "suite_root": str(suite_root),
        "started_at": suite_started_at,
        "completed_at": suite_completed_at,
        "runtime_sec": round(time.time() - suite_start, 3),
        "variant_specs": variant_specs,
        "results": suite_results,
    }
    write_json(suite_root / "suite_summary.json", summary)

    with contextlib.suppress(OSError):
        lines = [
            f"started_at: {suite_started_at}",
            f"completed_at: {suite_completed_at}",
            f"runtime_sec: {summary['runtime_sec']}",
            "",
        ]
        for item in suite_results:
            lines.extend(
                [
                    f"{item['framework']} / {item['dataset_key']}",
                    f"  baseline_full: {item['baseline']['full']['full_acc']}",
                    f"  shared_only: {item['variants']['shared_removal_only']['full']['full_acc']}",
                    f"  subspace_only: {item['variants']['subspace_term_only']['full']['full_acc']}",
                    f"  residual_only: {item['variants']['competitor_residual_only']['full']['full_acc']}",
                    f"  full_refined: {item['variants']['full_refined']['full']['full_acc']}",
                    f"  shared_rank0_control: {item['variants']['shared_rank0_control']['full']['full_acc']}",
                    "",
                ]
            )
        (suite_root / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
