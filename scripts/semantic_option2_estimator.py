import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from semantic_option2_killswitch import (
    build_model,
    build_semantic_banks,
    collect_final_queries,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    create_output_dir,
    extract_final_state,
    normalize_vector,
    run_semantic_oracle,
    to_builtin,
)


def latest_oracle_gain(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    oracle_root = output_root / cfg.DATASET.NAME / f"semantic_killswitch_{Path(train_cfg).stem}"
    if not oracle_root.exists():
        return None, None
    results_files = sorted(oracle_root.glob("*/results.json"))
    if not results_files:
        return None, None
    latest = results_files[-1]
    with latest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["oracle_gain"]), latest.parent


def spread_indices(num_items, max_items):
    if num_items <= max_items:
        return list(range(num_items))
    positions = np.linspace(0, num_items - 1, max_items)
    return sorted(set(int(round(pos)) for pos in positions))


def select_base_anchors(class_features, class_proto, max_items):
    class_features = F.normalize(class_features, dim=-1)
    sims = torch.matmul(class_features, normalize_vector(class_proto))
    order = torch.argsort(sims).tolist()
    selected = [order[idx] for idx in spread_indices(len(order), max_items)]
    return selected


def build_calibration_set(
    merged_state,
    num_cls,
    num_base_cls,
    image_proto,
    base_calib_per_class,
):
    image_features = F.normalize(merged_state["images_features"][:], dim=-1)
    image_targets = merged_state["images_targets"]

    calib_features = []
    calib_targets = []
    for cls_id in range(num_cls):
        class_feats = image_features[image_targets == cls_id]
        if class_feats.numel() == 0:
            continue
        if cls_id < num_base_cls:
            keep_idx = select_base_anchors(class_feats, image_proto[cls_id], base_calib_per_class)
            selected = class_feats[keep_idx]
        else:
            selected = class_feats
        calib_features.append(selected)
        calib_targets.append(torch.full((selected.size(0),), cls_id, dtype=torch.long, device=selected.device))

    return torch.cat(calib_features, dim=0), torch.cat(calib_targets, dim=0)


def class_balanced_nll(probs, targets, class_ids):
    losses = per_class_mean_nll(probs, targets, class_ids)
    return losses.mean()


def per_class_mean_nll(probs, targets, class_ids):
    losses = []
    idx = torch.arange(targets.size(0), device=targets.device)
    chosen = torch.clamp(probs[idx, targets], min=1e-12)
    nll = -torch.log(chosen)
    for cls_id in class_ids:
        mask = targets == cls_id
        if mask.any():
            losses.append(nll[mask].mean())
    return torch.stack(losses)


def class_balanced_margin(logits, targets, class_ids):
    margins = []
    idx = torch.arange(targets.size(0), device=targets.device)
    true_logits = logits[idx, targets]
    masked_logits = logits.clone()
    masked_logits[idx, targets] = float("-inf")
    competitor = masked_logits.max(dim=1).values
    per_sample = true_logits - competitor
    for cls_id in class_ids:
        mask = targets == cls_id
        if mask.any():
            margins.append(per_sample[mask].mean())
    return torch.stack(margins).mean()


def class_mean_alignment(features, targets, cls_id, semantic_vec):
    mask = targets == cls_id
    if not mask.any():
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    return torch.matmul(features[mask], semantic_vec).mean()


def evaluate_candidate_objective(
    query_features,
    query_targets,
    class_ids,
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
    return (
        class_balanced_nll(probs, query_targets, class_ids),
        class_balanced_margin(logits, query_targets, class_ids),
        per_class_mean_nll(probs, query_targets, class_ids),
        probs,
        logits,
    )


def significant_nll_gain(
    current_per_class_nll,
    cand_per_class_nll,
    eps=1e-6,
):
    improvements = current_per_class_nll - cand_per_class_nll
    mean_improvement = improvements.mean()
    if mean_improvement <= eps:
        return False
    if improvements.numel() == 1:
        return True
    stderr = improvements.std(unbiased=False) / np.sqrt(improvements.numel())
    return mean_improvement > stderr + eps


def better_candidate(
    cand_nll,
    cand_margin,
    cand_align,
    best_nll,
    best_margin,
    best_align,
    eps=1e-6,
):
    if cand_nll < best_nll - eps:
        return True
    if abs(float(cand_nll - best_nll)) <= eps and cand_margin > best_margin + eps:
        return True
    if (
        abs(float(cand_nll - best_nll)) <= eps
        and abs(float(cand_margin - best_margin)) <= eps
        and cand_align > best_align + eps
    ):
        return True
    return False


def run_semantic_estimator(
    merged_state,
    calibration_features,
    calibration_targets,
    test_features,
    test_targets,
    num_cls,
    num_base_cls,
    beta,
    lambda_t,
    ensemble_alpha,
    max_desc_atoms,
    base_atom_topk,
    max_passes,
):
    device = test_features.device
    image_proto = F.normalize(merged_state["image_proto"][:num_cls].to(device), dim=-1)
    description_features = F.normalize(merged_state["description_features"].to(device), dim=-1)
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device)

    calibration_features = calibration_features.to(device)
    calibration_targets = calibration_targets.to(device)
    test_features = test_features.to(device)
    test_targets = test_targets.to(device)

    prob_cov_cal = F.softmax(compute_cov_logits(calibration_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn_cal = F.softmax(
        compute_knn_logits(calibration_features, description_features, description_targets, num_cls),
        dim=-1,
    )
    prob_cov_test = F.softmax(compute_cov_logits(test_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn_test = F.softmax(
        compute_knn_logits(test_features, description_features, description_targets, num_cls),
        dim=-1,
    )

    baseline_sem, banks = build_semantic_banks(
        merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )
    baseline_sem = baseline_sem.to(device)

    class_ids = list(range(num_cls))
    current_sem = baseline_sem.clone()
    current_nll, current_margin, current_per_class_nll, _, _ = evaluate_candidate_objective(
        calibration_features,
        calibration_targets,
        class_ids,
        current_sem,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov_cal,
        prob_knn_cal,
    )
    chosen_meta = {
        cls_id: {
            "name": "baseline_sem",
            "uses_base": False,
            "uses_desc_atom": False,
            "is_baseline": True,
        }
        for cls_id in range(num_cls)
    }

    novel_classes = list(range(num_base_cls, num_cls))
    for _ in range(max_passes):
        improved_any = False
        for cls_id in novel_classes:
            best_nll = current_nll
            best_margin = current_margin
            best_per_class_nll = current_per_class_nll
            best_vec = current_sem[cls_id]
            best_meta = chosen_meta[cls_id]
            best_align = class_mean_alignment(calibration_features, calibration_targets, cls_id, best_vec)
            found_significant = False

            for candidate in banks[cls_id]:
                candidate_vec = candidate["vector"].to(device)
                temp_sem = current_sem.clone()
                temp_sem[cls_id] = candidate_vec
                cand_nll, cand_margin, cand_per_class_nll, _, _ = evaluate_candidate_objective(
                    calibration_features,
                    calibration_targets,
                    class_ids,
                    temp_sem,
                    image_proto,
                    num_base_cls,
                    beta,
                    ensemble_alpha,
                    prob_cov_cal,
                    prob_knn_cal,
                )
                if not significant_nll_gain(current_per_class_nll, cand_per_class_nll):
                    continue
                cand_align = class_mean_alignment(calibration_features, calibration_targets, cls_id, candidate_vec)
                if better_candidate(
                    cand_nll,
                    cand_margin,
                    cand_align,
                    best_nll,
                    best_margin,
                    best_align,
                ):
                    best_nll = cand_nll
                    best_margin = cand_margin
                    best_per_class_nll = cand_per_class_nll
                    best_align = cand_align
                    best_vec = candidate_vec
                    found_significant = True
                    best_meta = {
                        "name": candidate["name"],
                        "uses_base": bool(candidate["uses_base"]),
                        "uses_desc_atom": bool(candidate["uses_desc_atom"]),
                        "is_baseline": bool(candidate["is_baseline"]),
                    }

            if found_significant:
                improved_any = True
                current_nll = best_nll
                current_margin = best_margin
                current_per_class_nll = best_per_class_nll
                current_sem[cls_id] = best_vec
                chosen_meta[cls_id] = best_meta
        if not improved_any:
            break

    baseline_probs, _ = compose_probs_from_semantic(
        test_features,
        baseline_sem,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov_test,
        prob_knn_test,
    )
    estimated_probs, _ = compose_probs_from_semantic(
        test_features,
        current_sem,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov_test,
        prob_knn_test,
    )
    baseline_acc = float((baseline_probs.argmax(dim=1) == test_targets).float().mean().item() * 100.0)
    estimated_acc = float((estimated_probs.argmax(dim=1) == test_targets).float().mean().item() * 100.0)

    changed_classes = [cls_id for cls_id in novel_classes if not chosen_meta[cls_id]["is_baseline"]]
    used_base_classes = [cls_id for cls_id in novel_classes if chosen_meta[cls_id]["uses_base"]]
    used_desc_atom_classes = [cls_id for cls_id in novel_classes if chosen_meta[cls_id]["uses_desc_atom"]]

    baseline_alignment = 1.0 - torch.sum(image_proto[num_base_cls:] * baseline_sem[num_base_cls:], dim=1)
    estimated_alignment = 1.0 - torch.sum(image_proto[num_base_cls:] * current_sem[num_base_cls:], dim=1)

    return {
        "baseline_acc": round(baseline_acc, 3),
        "estimated_acc": round(estimated_acc, 3),
        "estimated_gain": round(estimated_acc - baseline_acc, 3),
        "support_class_balanced_nll": round(float(current_nll.item()), 6),
        "support_class_balanced_margin": round(float(current_margin.item()), 6),
        "changed_class_rate": round(float(len(changed_classes) / max(len(novel_classes), 1)), 4),
        "base_atom_usage_rate": round(float(len(used_base_classes) / max(len(novel_classes), 1)), 4),
        "desc_atom_usage_rate": round(float(len(used_desc_atom_classes) / max(len(novel_classes), 1)), 4),
        "baseline_alignment_gap": round(float(baseline_alignment.mean().item()), 6) if novel_classes else 0.0,
        "estimated_alignment_gap": round(float(estimated_alignment.mean().item()), 6) if novel_classes else 0.0,
        "chosen_candidates": {str(cls_id): chosen_meta[cls_id]["name"] for cls_id in novel_classes},
    }


def continue_flag(oracle_gain, estimated_gain):
    if oracle_gain is None or oracle_gain <= 1e-9:
        return False, None
    recovery = estimated_gain / oracle_gain
    return recovery >= 0.6, recovery


def run_dataset(
    data_cfg,
    train_cfg,
    max_desc_atoms,
    base_atom_topk,
    max_passes,
    base_calib_per_class,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    out_dir = create_output_dir(cfg, f"semantic_estimator_{Path(train_cfg).stem}")
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)

    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)
    image_proto = F.normalize(merged_state["image_proto"][:num_cls], dim=-1)
    calibration_features, calibration_targets = build_calibration_set(
        merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        image_proto=image_proto,
        base_calib_per_class=base_calib_per_class,
    )

    estimated = run_semantic_estimator(
        merged_state=merged_state,
        calibration_features=calibration_features,
        calibration_targets=calibration_targets,
        test_features=query_features,
        test_targets=query_targets,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=ensemble_alpha,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        max_passes=max_passes,
    )

    oracle_gain, oracle_dir = latest_oracle_gain(cfg, train_cfg)
    should_continue, recovery_rate = continue_flag(
        oracle_gain,
        float(estimated["estimated_gain"]),
    )

    runtime_sec = round(time.time() - start_time, 3)
    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": runtime_sec,
        "num_seen_classes": num_cls,
        "num_base_classes": num_base_cls,
        "baseline_beta": beta,
        "baseline_lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "max_desc_atoms": max_desc_atoms,
        "base_atom_topk": base_atom_topk,
        "max_passes": max_passes,
        "base_calib_per_class": base_calib_per_class,
        "oracle_gain": oracle_gain,
        "oracle_results_dir": str(oracle_dir) if oracle_dir is not None else None,
        "recovery_rate": round(float(recovery_rate), 4) if recovery_rate is not None else None,
        "recommendation": "continue_option2" if should_continue else "kill_option2",
        **estimated,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"runtime_sec: {runtime_sec}",
        f"baseline_acc: {estimated['baseline_acc']}",
        f"estimated_acc: {estimated['estimated_acc']}",
        f"estimated_gain: {estimated['estimated_gain']}",
        f"oracle_gain: {oracle_gain}",
        f"recovery_rate: {payload['recovery_rate']}",
        f"recommendation: {payload['recommendation']}",
        f"support_class_balanced_nll: {estimated['support_class_balanced_nll']}",
        f"support_class_balanced_margin: {estimated['support_class_balanced_margin']}",
        f"changed_class_rate: {estimated['changed_class_rate']}",
        f"base_atom_usage_rate: {estimated['base_atom_usage_rate']}",
        f"desc_atom_usage_rate: {estimated['desc_atom_usage_rate']}",
        f"baseline_alignment_gap: {estimated['baseline_alignment_gap']}",
        f"estimated_alignment_gap: {estimated['estimated_alignment_gap']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "baseline_acc": estimated["baseline_acc"],
        "estimated_acc": estimated["estimated_acc"],
        "estimated_gain": estimated["estimated_gain"],
        "oracle_gain": oracle_gain,
        "recovery_rate": payload["recovery_rate"],
        "recommendation": payload["recommendation"],
        "changed_class_rate": estimated["changed_class_rate"],
        "base_atom_usage_rate": estimated["base_atom_usage_rate"],
        "desc_atom_usage_rate": estimated["desc_atom_usage_rate"],
    }, indent=2))
    print(f"Saved semantic estimator diagnostics to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--max_desc_atoms", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    parser.add_argument("--max_passes", type=int, default=2)
    parser.add_argument("--base_calib_per_class", type=int, default=3)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        max_passes=args.max_passes,
        base_calib_per_class=args.base_calib_per_class,
    )


if __name__ == "__main__":
    main()
