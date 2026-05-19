import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from prototype_refinement_conservative import conservative_refine_once
from prototype_refinement_conservative_dino_fusion_sessionwise import (
    build_session_states,
    choose_alpha_by_pseudo_validation,
    compose_from_components,
    merge_states,
    normalize_rows,
    session_components,
)
from query_graph_router_killswitch import accuracy_stats


DEFAULT_DATASETS = {
    "CIFAR100": "configs/datasets/cifar100.yaml",
    "miniimagenet": "configs/datasets/miniimagenet.yaml",
    "cub200": "configs/datasets/cub200_bimc_dino_fusion.yaml",
}
DEFAULT_TRAIN_CFG = "configs/trainers/bimc_dino_fusion.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed-1 TriC/TriC-PV diagnostics: prototype similarity, misclassification sheets, and session impact."
    )
    parser.add_argument("--train_cfg", default=DEFAULT_TRAIN_CFG)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS.keys()),
        help="Subset of dataset names: CIFAR100 miniimagenet cub200",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--mass_thr", type=float, default=0.3)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--base_val_conf_thr", type=float, default=0.5)
    parser.add_argument("--base_val_max_per_class", type=int, default=50)
    parser.add_argument("--fallback_alpha", type=float, default=0.75)
    parser.add_argument("--topk_per_class", type=int, default=None)
    parser.add_argument("--output_root", default=None)
    return parser.parse_args()


def create_output_root(seed, output_root=None):
    if output_root is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        root = REPO_ROOT / "analysis" / f"tric_seed_diagnostics_seed{seed}_{timestamp}"
    else:
        root = Path(output_root)
        if not root.is_absolute():
            root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_matrix_csv(path, matrix, class_ids, class_names, intro_sessions):
    column_labels = [f"{cls_id}:{name}" for cls_id, name in zip(class_ids, class_names)]
    fieldnames = ["class_id", "class_name", "intro_session"] + column_labels
    rows = []
    matrix = matrix.detach().cpu().numpy()
    for row_idx, (cls_id, name, intro) in enumerate(zip(class_ids, class_names, intro_sessions)):
        row = {
            "class_id": int(cls_id),
            "class_name": name,
            "intro_session": int(intro),
        }
        for col_idx, label in enumerate(column_labels):
            row[label] = f"{float(matrix[row_idx, col_idx]):.6f}"
        rows.append(row)
    write_csv(path, fieldnames, rows)


def safe_corr(xs, ys):
    if len(xs) < 2:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    corr = float(np.corrcoef(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))[0, 1])
    if math.isnan(corr):
        return None
    return corr


def pct_correct(preds, targets, mask):
    if int(mask.sum().item()) == 0:
        return None
    value = (preds[mask] == targets[mask]).float().mean().item() * 100.0
    return round(float(value), 3)


@torch.no_grad()
def collect_session_query_bundle(cfg, data_manager, model, task_id):
    loader = data_manager.get_dataloader(task_id, source="test", mode="test")
    clip_features = []
    dino_features = []
    targets = []
    query_numbers = []
    class_names = []
    intro_sessions = []
    for batch in loader:
        images = batch["image"].to(cfg.DEVICE.DEVICE_NAME)
        labels = batch["label"].to(cfg.DEVICE.DEVICE_NAME)
        clip_features.append(normalize_rows(model.extract_img_feature(images)))
        dino_features.append(normalize_rows(model.extract_dino_img_feature(images)))
        targets.append(labels)
        query_numbers.extend(int(x) for x in batch["idx"].tolist())
        class_names.extend(batch["cls_name"])
        intro_sessions.extend(int(x) for x in batch["task_id"].tolist())
    return {
        "clip_query": torch.cat(clip_features, dim=0),
        "dino_query": torch.cat(dino_features, dim=0),
        "targets": torch.cat(targets, dim=0),
        "query_numbers": query_numbers,
        "class_names": class_names,
        "intro_sessions": intro_sessions,
    }


def build_intro_map(data_manager):
    intro_map = {}
    for session_id, class_ids in enumerate(data_manager.class_index_in_task):
        for class_id in class_ids:
            intro_map[int(class_id)] = int(session_id)
    return intro_map


def class_rows_for_eval_session(eval_session, preds, targets, class_names, intro_map):
    rows = []
    seen_classes = sorted(set(int(x) for x in targets.detach().cpu().tolist()))
    preds_cpu = preds.detach().cpu()
    targets_cpu = targets.detach().cpu()
    for class_id in seen_classes:
        mask = targets_cpu == int(class_id)
        count = int(mask.sum().item())
        acc = 0.0
        if count > 0:
            acc = float((preds_cpu[mask] == targets_cpu[mask]).float().mean().item() * 100.0)
        rows.append(
            {
                "eval_session": int(eval_session),
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "intro_session": int(intro_map[int(class_id)]),
                "query_count": int(count),
                "accuracy": round(acc, 3),
            }
        )
    return rows


def intro_group_rows(eval_session, intro_sessions, preds, targets):
    rows = []
    preds_cpu = preds.detach().cpu()
    targets_cpu = targets.detach().cpu()
    intro_sessions = np.asarray(intro_sessions, dtype=int)
    for intro_session in sorted(set(int(x) for x in intro_sessions.tolist())):
        mask_np = intro_sessions == int(intro_session)
        mask = torch.from_numpy(mask_np)
        count = int(mask.sum().item())
        acc = pct_correct(preds_cpu, targets_cpu, mask)
        rows.append(
            {
                "eval_session": int(eval_session),
                "intro_session": int(intro_session),
                "query_count": int(count),
                "accuracy": acc,
            }
        )
    return rows


def build_misclassification_rows(
    dataset_name,
    eval_session,
    query_numbers,
    targets,
    intro_sessions,
    class_intro_map,
    baseline_probs,
    final_probs,
    clip_sim,
    dino_sim_raw,
    dino_sim_final,
    class_names,
):
    rows = []
    baseline_preds = torch.argmax(baseline_probs, dim=1)
    final_preds = torch.argmax(final_probs, dim=1)
    baseline_conf = torch.max(baseline_probs, dim=1).values
    final_conf = torch.max(final_probs, dim=1).values
    targets_cpu = targets.detach().cpu()
    baseline_preds_cpu = baseline_preds.detach().cpu()
    final_preds_cpu = final_preds.detach().cpu()
    baseline_conf_cpu = baseline_conf.detach().cpu()
    final_conf_cpu = final_conf.detach().cpu()
    baseline_probs_cpu = baseline_probs.detach().cpu()
    final_probs_cpu = final_probs.detach().cpu()
    clip_sim_cpu = clip_sim.detach().cpu()
    dino_sim_raw_cpu = dino_sim_raw.detach().cpu()
    dino_sim_final_cpu = dino_sim_final.detach().cpu()

    for idx in range(targets_cpu.shape[0]):
        true_id = int(targets_cpu[idx].item())
        pred_id = int(final_preds_cpu[idx].item())
        if pred_id == true_id:
            continue
        base_pred_id = int(baseline_preds_cpu[idx].item())
        rows.append(
            {
                "dataset": dataset_name,
                "eval_session": int(eval_session),
                "query_number": int(query_numbers[idx]),
                "true_class_id": int(true_id),
                "true_class_name": class_names[true_id],
                "true_intro_session": int(intro_sessions[idx]),
                "baseline_pred_class_id": int(base_pred_id),
                "baseline_pred_class_name": class_names[base_pred_id],
                "baseline_pred_intro_session": int(class_intro_map[base_pred_id]),
                "baseline_pred_conf": round(float(baseline_conf_cpu[idx].item()), 6),
                "baseline_true_prob": round(float(baseline_probs_cpu[idx, true_id].item()), 6),
                "baseline_pred_prob": round(float(baseline_probs_cpu[idx, base_pred_id].item()), 6),
                "final_pred_class_id": int(pred_id),
                "final_pred_class_name": class_names[pred_id],
                "final_pred_intro_session": int(class_intro_map[pred_id]),
                "final_pred_conf": round(float(final_conf_cpu[idx].item()), 6),
                "final_true_prob": round(float(final_probs_cpu[idx, true_id].item()), 6),
                "final_pred_prob": round(float(final_probs_cpu[idx, pred_id].item()), 6),
                "clip_true_sim": round(float(clip_sim_cpu[idx, true_id].item()), 6),
                "clip_pred_sim": round(float(clip_sim_cpu[idx, pred_id].item()), 6),
                "dino_true_sim_raw": round(float(dino_sim_raw_cpu[idx, true_id].item()), 6),
                "dino_pred_sim_raw": round(float(dino_sim_raw_cpu[idx, pred_id].item()), 6),
                "dino_true_sim_final": round(float(dino_sim_final_cpu[idx, true_id].item()), 6),
                "dino_pred_sim_final": round(float(dino_sim_final_cpu[idx, pred_id].item()), 6),
                "baseline_correct": int(base_pred_id == true_id),
                "final_correct": 0,
            }
        )
    return rows


def summarize_confusions(rows):
    groups = {}
    for row in rows:
        key = (row["eval_session"], row["true_class_id"], row["final_pred_class_id"])
        groups.setdefault(
            key,
            {
                "eval_session": row["eval_session"],
                "true_class_id": row["true_class_id"],
                "true_class_name": row["true_class_name"],
                "final_pred_class_id": row["final_pred_class_id"],
                "final_pred_class_name": row["final_pred_class_name"],
                "count": 0,
                "final_pred_conf_sum": 0.0,
                "final_true_prob_sum": 0.0,
                "final_pred_prob_sum": 0.0,
                "clip_true_sim_sum": 0.0,
                "clip_pred_sim_sum": 0.0,
                "dino_true_sim_raw_sum": 0.0,
                "dino_pred_sim_raw_sum": 0.0,
                "dino_true_sim_final_sum": 0.0,
                "dino_pred_sim_final_sum": 0.0,
                "query_numbers": [],
            },
        )
        bucket = groups[key]
        bucket["count"] += 1
        bucket["final_pred_conf_sum"] += float(row["final_pred_conf"])
        bucket["final_true_prob_sum"] += float(row["final_true_prob"])
        bucket["final_pred_prob_sum"] += float(row["final_pred_prob"])
        bucket["clip_true_sim_sum"] += float(row["clip_true_sim"])
        bucket["clip_pred_sim_sum"] += float(row["clip_pred_sim"])
        bucket["dino_true_sim_raw_sum"] += float(row["dino_true_sim_raw"])
        bucket["dino_pred_sim_raw_sum"] += float(row["dino_pred_sim_raw"])
        bucket["dino_true_sim_final_sum"] += float(row["dino_true_sim_final"])
        bucket["dino_pred_sim_final_sum"] += float(row["dino_pred_sim_final"])
        bucket["query_numbers"].append(int(row["query_number"]))

    summary_rows = []
    for _, bucket in sorted(groups.items()):
        count = max(bucket["count"], 1)
        summary_rows.append(
            {
                "eval_session": bucket["eval_session"],
                "true_class_id": bucket["true_class_id"],
                "true_class_name": bucket["true_class_name"],
                "final_pred_class_id": bucket["final_pred_class_id"],
                "final_pred_class_name": bucket["final_pred_class_name"],
                "count": int(bucket["count"]),
                "mean_final_pred_conf": round(bucket["final_pred_conf_sum"] / count, 6),
                "mean_final_true_prob": round(bucket["final_true_prob_sum"] / count, 6),
                "mean_final_pred_prob": round(bucket["final_pred_prob_sum"] / count, 6),
                "mean_clip_true_sim": round(bucket["clip_true_sim_sum"] / count, 6),
                "mean_clip_pred_sim": round(bucket["clip_pred_sim_sum"] / count, 6),
                "mean_dino_true_sim_raw": round(bucket["dino_true_sim_raw_sum"] / count, 6),
                "mean_dino_pred_sim_raw": round(bucket["dino_pred_sim_raw_sum"] / count, 6),
                "mean_dino_true_sim_final": round(bucket["dino_true_sim_final_sum"] / count, 6),
                "mean_dino_pred_sim_final": round(bucket["dino_pred_sim_final_sum"] / count, 6),
                "query_numbers": " ".join(str(x) for x in bucket["query_numbers"]),
            }
        )
    return summary_rows


def dataset_summary(
    dataset_name,
    session_rows,
    per_class_baseline_final,
    per_class_final_final,
):
    final_session = max(int(row["eval_session"]) for row in session_rows)
    baseline_final_rows = [row for row in per_class_baseline_final if int(row["eval_session"]) == final_session]
    final_final_rows = [row for row in per_class_final_final if int(row["eval_session"]) == final_session]

    intro_to_baseline = defaultdict(list)
    intro_to_final = defaultdict(list)
    for row in baseline_final_rows:
        intro_to_baseline[int(row["intro_session"])].append(float(row["accuracy"]))
    for row in final_final_rows:
        intro_to_final[int(row["intro_session"])].append(float(row["accuracy"]))

    intro_summary_rows = []
    for intro_session in sorted(intro_to_final.keys()):
        base_vals = intro_to_baseline.get(intro_session, [])
        final_vals = intro_to_final.get(intro_session, [])
        intro_summary_rows.append(
            {
                "dataset": dataset_name,
                "intro_session": int(intro_session),
                "baseline_mean_class_acc": round(float(np.mean(base_vals)), 3) if base_vals else None,
                "final_mean_class_acc": round(float(np.mean(final_vals)), 3) if final_vals else None,
                "gain": round(float(np.mean(final_vals) - np.mean(base_vals)), 3) if base_vals and final_vals else None,
                "class_count": int(len(final_vals)),
            }
        )

    baseline_corr = safe_corr(
        [int(row["intro_session"]) for row in baseline_final_rows],
        [float(row["accuracy"]) for row in baseline_final_rows],
    )
    final_corr = safe_corr(
        [int(row["intro_session"]) for row in final_final_rows],
        [float(row["accuracy"]) for row in final_final_rows],
    )

    first_intro = min(intro_to_final.keys())
    last_intro = max(intro_to_final.keys())
    return {
        "dataset": dataset_name,
        "baseline_avg_session_acc": round(float(np.mean([row["baseline_full_acc"] for row in session_rows])), 3),
        "final_avg_session_acc": round(float(np.mean([row["final_full_acc"] for row in session_rows])), 3),
        "baseline_performance_drop": round(float(session_rows[0]["baseline_full_acc"] - session_rows[-1]["baseline_full_acc"]), 3),
        "final_performance_drop": round(float(session_rows[0]["final_full_acc"] - session_rows[-1]["final_full_acc"]), 3),
        "final_session_baseline_acc": round(float(session_rows[-1]["baseline_full_acc"]), 3),
        "final_session_final_acc": round(float(session_rows[-1]["final_full_acc"]), 3),
        "final_session_gain": round(float(session_rows[-1]["final_full_acc"] - session_rows[-1]["baseline_full_acc"]), 3),
        "final_session_intro_corr_baseline": None if baseline_corr is None else round(baseline_corr, 6),
        "final_session_intro_corr_final": None if final_corr is None else round(final_corr, 6),
        "first_intro_session": int(first_intro),
        "last_intro_session": int(last_intro),
        "first_intro_baseline_mean_class_acc": round(float(np.mean(intro_to_baseline[first_intro])), 3),
        "first_intro_final_mean_class_acc": round(float(np.mean(intro_to_final[first_intro])), 3),
        "last_intro_baseline_mean_class_acc": round(float(np.mean(intro_to_baseline[last_intro])), 3),
        "last_intro_final_mean_class_acc": round(float(np.mean(intro_to_final[last_intro])), 3),
        "intro_summary_rows": intro_summary_rows,
    }


def run_one_dataset(args, dataset_name, data_cfg, output_root):
    cfg = setup_cfg(data_cfg, args.train_cfg)
    cfg.defrost()
    cfg.SEED = int(args.seed)
    cfg.freeze()

    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        dataset_dir / "run_config.json",
        {
            "dataset": dataset_name,
            "data_cfg": data_cfg,
            "train_cfg": args.train_cfg,
            "seed": int(args.seed),
            "alpha_grid": args.alpha_grid,
            "mass_thr": args.mass_thr,
            "min_count": args.min_count,
            "val_fraction": args.val_fraction,
            "base_val_conf_thr": args.base_val_conf_thr,
            "base_val_max_per_class": args.base_val_max_per_class,
            "fallback_alpha": args.fallback_alpha,
            "topk_per_class": args.topk_per_class,
        },
    )

    start_time = time.time()
    data_manager, model, states = build_session_states(cfg)
    intro_map = build_intro_map(data_manager)
    class_names = list(data_manager.class_names)

    session_rows = []
    intro_baseline_rows = []
    intro_final_rows = []
    per_class_baseline_rows = []
    per_class_final_rows = []
    misclass_rows = []
    final_clip_proto = None
    final_dino_raw_proto = None
    final_dino_refined_proto = None

    for task_id in range(data_manager.num_tasks):
        merged_state = merge_states(states[: task_id + 1])
        bundle = collect_session_query_bundle(cfg, data_manager, model, task_id)
        clip_query = bundle["clip_query"].to(cfg.DEVICE.DEVICE_NAME)
        dino_query = bundle["dino_query"].to(cfg.DEVICE.DEVICE_NAME)
        targets = bundle["targets"].to(cfg.DEVICE.DEVICE_NAME)

        components = session_components(cfg, data_manager, merged_state, task_id, clip_query, dino_query)
        baseline_probs = compose_from_components(clip_query, dino_query, components, components["dino_image_proto"])
        baseline_full, baseline_base, baseline_novel = accuracy_stats(
            baseline_probs,
            targets,
            components["num_base_cls"],
        )
        baseline_preds = torch.argmax(baseline_probs, dim=1)

        selected_alpha = None
        alpha_mode = "not_applicable_base_session"
        alpha_meta = {"mode": alpha_mode, "selected_alpha": None}
        refined_dino_proto = components["dino_image_proto"]
        final_probs = baseline_probs
        gated_query_count = 0
        gated_base_count = 0
        gated_novel_count = 0
        updated_class_count = 0

        if task_id > 0:
            selected_alpha, alpha_meta = choose_alpha_by_pseudo_validation(
                clip_query=clip_query,
                dino_query=dino_query,
                components=components,
                baseline_probs=baseline_probs,
                alpha_grid=args.alpha_grid,
                mass_thr=args.mass_thr,
                min_count=args.min_count,
                val_fraction=args.val_fraction,
                base_val_conf_thr=args.base_val_conf_thr,
                base_val_max_per_class=args.base_val_max_per_class,
                fallback_alpha=args.fallback_alpha,
                topk_per_class=args.topk_per_class,
            )
            refined_dino_proto, stats = conservative_refine_once(
                query_features=dino_query,
                query_targets=targets,
                baseline_probs=baseline_probs,
                orig_image_proto=components["dino_image_proto"],
                num_base_cls=components["num_base_cls"],
                num_cls=components["num_cls"],
                alpha=selected_alpha,
                mass_thr=args.mass_thr,
                min_count=args.min_count,
                topk_per_class=args.topk_per_class,
            )
            final_probs = compose_from_components(clip_query, dino_query, components, refined_dino_proto)
            gated_query_count = int(stats["gated_query_count"])
            gated_base_count = int(stats["gated_base_count"])
            gated_novel_count = int(stats["gated_novel_count"])
            updated_class_count = int(stats["updated_class_count"])
            alpha_mode = alpha_meta["mode"]

        final_full, final_base, final_novel = accuracy_stats(
            final_probs,
            targets,
            components["num_base_cls"],
        )
        final_preds = torch.argmax(final_probs, dim=1)

        session_rows.append(
            {
                "eval_session": int(task_id),
                "num_seen_classes": int(components["num_cls"]),
                "baseline_full_acc": round(float(baseline_full), 3),
                "baseline_base_acc": round(float(baseline_base), 3),
                "baseline_novel_acc": round(float(baseline_novel), 3),
                "final_full_acc": round(float(final_full), 3),
                "final_base_acc": round(float(final_base), 3),
                "final_novel_acc": round(float(final_novel), 3),
                "gain": round(float(final_full - baseline_full), 3),
                "selected_alpha": "" if selected_alpha is None else round(float(selected_alpha), 3),
                "alpha_mode": alpha_mode,
                "updated_class_count": int(updated_class_count),
                "gated_query_count": int(gated_query_count),
                "gated_base_count": int(gated_base_count),
                "gated_novel_count": int(gated_novel_count),
            }
        )

        intro_baseline_rows.extend(
            {
                "eval_session": row["eval_session"],
                "intro_session": row["intro_session"],
                "query_count": row["query_count"],
                "baseline_accuracy": row["accuracy"],
            }
            for row in intro_group_rows(task_id, bundle["intro_sessions"], baseline_preds, targets)
        )
        intro_final_rows.extend(
            {
                "eval_session": row["eval_session"],
                "intro_session": row["intro_session"],
                "query_count": row["query_count"],
                "final_accuracy": row["accuracy"],
            }
            for row in intro_group_rows(task_id, bundle["intro_sessions"], final_preds, targets)
        )
        per_class_baseline_rows.extend(
            class_rows_for_eval_session(task_id, baseline_preds, targets, class_names, intro_map)
        )
        per_class_final_rows.extend(
            class_rows_for_eval_session(task_id, final_preds, targets, class_names, intro_map)
        )

        clip_sim = torch.matmul(clip_query, components["clip_image_proto"].T)
        dino_sim_raw = torch.matmul(dino_query, components["dino_image_proto"].T)
        dino_sim_final = torch.matmul(dino_query, refined_dino_proto.T)
        misclass_rows.extend(
            build_misclassification_rows(
                dataset_name=dataset_name,
                eval_session=task_id,
                query_numbers=bundle["query_numbers"],
                targets=targets,
                intro_sessions=bundle["intro_sessions"],
                class_intro_map=intro_map,
                baseline_probs=baseline_probs,
                final_probs=final_probs,
                clip_sim=clip_sim,
                dino_sim_raw=dino_sim_raw,
                dino_sim_final=dino_sim_final,
                class_names=class_names,
            )
        )

        if task_id == data_manager.num_tasks - 1:
            seen_class_ids = list(range(int(components["num_cls"])))
            seen_names = [class_names[class_id] for class_id in seen_class_ids]
            seen_intro = [intro_map[class_id] for class_id in seen_class_ids]
            final_clip_proto = (
                torch.matmul(components["clip_image_proto"], components["clip_image_proto"].T),
                seen_class_ids,
                seen_names,
                seen_intro,
            )
            final_dino_raw_proto = (
                torch.matmul(components["dino_image_proto"], components["dino_image_proto"].T),
                seen_class_ids,
                seen_names,
                seen_intro,
            )
            final_dino_refined_proto = (
                torch.matmul(refined_dino_proto, refined_dino_proto.T),
                seen_class_ids,
                seen_names,
                seen_intro,
            )

    summary = dataset_summary(dataset_name, session_rows, per_class_baseline_rows, per_class_final_rows)
    summary["runtime_sec"] = round(time.time() - start_time, 3)

    write_csv(
        dataset_dir / "session_accuracy.csv",
        [
            "eval_session",
            "num_seen_classes",
            "baseline_full_acc",
            "baseline_base_acc",
            "baseline_novel_acc",
            "final_full_acc",
            "final_base_acc",
            "final_novel_acc",
            "gain",
            "selected_alpha",
            "alpha_mode",
            "updated_class_count",
            "gated_query_count",
            "gated_base_count",
            "gated_novel_count",
        ],
        session_rows,
    )
    write_csv(
        dataset_dir / "intro_session_accuracy_baseline.csv",
        ["eval_session", "intro_session", "query_count", "baseline_accuracy"],
        intro_baseline_rows,
    )
    write_csv(
        dataset_dir / "intro_session_accuracy_final.csv",
        ["eval_session", "intro_session", "query_count", "final_accuracy"],
        intro_final_rows,
    )
    write_csv(
        dataset_dir / "per_class_accuracy_baseline.csv",
        ["eval_session", "class_id", "class_name", "intro_session", "query_count", "accuracy"],
        per_class_baseline_rows,
    )
    write_csv(
        dataset_dir / "per_class_accuracy_final.csv",
        ["eval_session", "class_id", "class_name", "intro_session", "query_count", "accuracy"],
        per_class_final_rows,
    )
    misclass_fieldnames = [
        "dataset",
        "eval_session",
        "query_number",
        "true_class_id",
        "true_class_name",
        "true_intro_session",
        "baseline_pred_class_id",
        "baseline_pred_class_name",
        "baseline_pred_intro_session",
        "baseline_pred_conf",
        "baseline_true_prob",
        "baseline_pred_prob",
        "final_pred_class_id",
        "final_pred_class_name",
        "final_pred_intro_session",
        "final_pred_conf",
        "final_true_prob",
        "final_pred_prob",
        "clip_true_sim",
        "clip_pred_sim",
        "dino_true_sim_raw",
        "dino_pred_sim_raw",
        "dino_true_sim_final",
        "dino_pred_sim_final",
        "baseline_correct",
        "final_correct",
    ]
    write_csv(dataset_dir / "misclassifications_all_sessions.csv", misclass_fieldnames, misclass_rows)
    final_session_id = max(int(row["eval_session"]) for row in session_rows)
    write_csv(
        dataset_dir / "misclassifications_final_session.csv",
        misclass_fieldnames,
        [row for row in misclass_rows if int(row["eval_session"]) == final_session_id],
    )
    confusion_fieldnames = [
        "eval_session",
        "true_class_id",
        "true_class_name",
        "final_pred_class_id",
        "final_pred_class_name",
        "count",
        "mean_final_pred_conf",
        "mean_final_true_prob",
        "mean_final_pred_prob",
        "mean_clip_true_sim",
        "mean_clip_pred_sim",
        "mean_dino_true_sim_raw",
        "mean_dino_pred_sim_raw",
        "mean_dino_true_sim_final",
        "mean_dino_pred_sim_final",
        "query_numbers",
    ]
    confusion_rows = summarize_confusions(misclass_rows)
    write_csv(dataset_dir / "confusion_summary_all_sessions.csv", confusion_fieldnames, confusion_rows)
    write_csv(
        dataset_dir / "confusion_summary_final_session.csv",
        confusion_fieldnames,
        [row for row in confusion_rows if int(row["eval_session"]) == final_session_id],
    )
    write_csv(
        dataset_dir / "final_session_intro_summary.csv",
        [
            "dataset",
            "intro_session",
            "baseline_mean_class_acc",
            "final_mean_class_acc",
            "gain",
            "class_count",
        ],
        summary["intro_summary_rows"],
    )

    if final_clip_proto is not None:
        write_matrix_csv(dataset_dir / "clip_visual_prototype_similarity_final.csv", *final_clip_proto)
    if final_dino_raw_proto is not None:
        write_matrix_csv(dataset_dir / "dino_visual_prototype_similarity_final_raw.csv", *final_dino_raw_proto)
    if final_dino_refined_proto is not None:
        write_matrix_csv(dataset_dir / "dino_visual_prototype_similarity_final_refined.csv", *final_dino_refined_proto)

    write_json(dataset_dir / "summary.json", summary)
    return summary


def write_root_summary(output_root, args, dataset_summaries):
    overview_rows = []
    for summary in dataset_summaries:
        overview_rows.append(
            {
                "dataset": summary["dataset"],
                "baseline_avg_session_acc": summary["baseline_avg_session_acc"],
                "final_avg_session_acc": summary["final_avg_session_acc"],
                "baseline_performance_drop": summary["baseline_performance_drop"],
                "final_performance_drop": summary["final_performance_drop"],
                "final_session_baseline_acc": summary["final_session_baseline_acc"],
                "final_session_final_acc": summary["final_session_final_acc"],
                "final_session_gain": summary["final_session_gain"],
                "final_session_intro_corr_baseline": summary["final_session_intro_corr_baseline"],
                "final_session_intro_corr_final": summary["final_session_intro_corr_final"],
                "first_intro_session": summary["first_intro_session"],
                "last_intro_session": summary["last_intro_session"],
                "first_intro_baseline_mean_class_acc": summary["first_intro_baseline_mean_class_acc"],
                "first_intro_final_mean_class_acc": summary["first_intro_final_mean_class_acc"],
                "last_intro_baseline_mean_class_acc": summary["last_intro_baseline_mean_class_acc"],
                "last_intro_final_mean_class_acc": summary["last_intro_final_mean_class_acc"],
                "runtime_sec": summary["runtime_sec"],
            }
        )
    write_csv(
        output_root / "dataset_overview.csv",
        [
            "dataset",
            "baseline_avg_session_acc",
            "final_avg_session_acc",
            "baseline_performance_drop",
            "final_performance_drop",
            "final_session_baseline_acc",
            "final_session_final_acc",
            "final_session_gain",
            "final_session_intro_corr_baseline",
            "final_session_intro_corr_final",
            "first_intro_session",
            "last_intro_session",
            "first_intro_baseline_mean_class_acc",
            "first_intro_final_mean_class_acc",
            "last_intro_baseline_mean_class_acc",
            "last_intro_final_mean_class_acc",
            "runtime_sec",
        ],
        overview_rows,
    )
    write_json(
        output_root / "metadata.json",
        {
            "seed": int(args.seed),
            "train_cfg": args.train_cfg,
            "datasets": args.datasets,
            "alpha_grid": args.alpha_grid,
            "mass_thr": args.mass_thr,
            "min_count": args.min_count,
            "val_fraction": args.val_fraction,
            "base_val_conf_thr": args.base_val_conf_thr,
            "base_val_max_per_class": args.base_val_max_per_class,
            "fallback_alpha": args.fallback_alpha,
            "topk_per_class": args.topk_per_class,
            "dataset_summaries": dataset_summaries,
        },
    )
    lines = [
        f"Seed: {args.seed}",
        f"Train cfg: {args.train_cfg}",
        "",
    ]
    for summary in dataset_summaries:
        lines.extend(
            [
                f"[{summary['dataset']}]",
                f"avg session acc: {summary['baseline_avg_session_acc']} -> {summary['final_avg_session_acc']}",
                f"performance drop: {summary['baseline_performance_drop']} -> {summary['final_performance_drop']}",
                f"final session acc: {summary['final_session_baseline_acc']} -> {summary['final_session_final_acc']}",
                f"intro-session corr: {summary['final_session_intro_corr_baseline']} -> {summary['final_session_intro_corr_final']}",
                f"first vs last intro mean class acc: {summary['first_intro_baseline_mean_class_acc']}->{summary['first_intro_final_mean_class_acc']} vs {summary['last_intro_baseline_mean_class_acc']}->{summary['last_intro_final_mean_class_acc']}",
                "",
            ]
        )
    (output_root / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    invalid = [name for name in args.datasets if name not in DEFAULT_DATASETS]
    if invalid:
        raise ValueError(f"Unknown dataset names: {invalid}. Allowed: {sorted(DEFAULT_DATASETS)}")

    output_root = create_output_root(args.seed, args.output_root)
    dataset_summaries = []
    for dataset_name in args.datasets:
        data_cfg = DEFAULT_DATASETS[dataset_name]
        print(f"Running diagnostics for {dataset_name} with seed={args.seed}")
        summary = run_one_dataset(args, dataset_name, data_cfg, output_root)
        dataset_summaries.append(summary)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_root_summary(output_root, args, dataset_summaries)
    print(f"Diagnostics written to: {output_root}")


if __name__ == "__main__":
    main()
