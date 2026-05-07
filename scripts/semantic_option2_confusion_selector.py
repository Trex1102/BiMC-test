import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from semantic_option2_killswitch import (
    build_semantic_banks,
    collect_final_queries,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    normalize_vector,
    to_builtin,
)


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"semantic_confusion_selector_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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


def accuracy_from_probs(probs, targets):
    return float((torch.argmax(probs, dim=1) == targets).float().mean().item() * 100.0)


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


def class_balanced_nll(probs, targets, class_ids):
    return per_class_mean_nll(probs, targets, class_ids).mean()


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


def top_wrong_classes(scores, true_cls, topk):
    if scores.numel() <= 1:
        return []
    filtered = scores.clone()
    filtered[int(true_cls)] = float("-inf")
    k = max(1, min(int(topk), filtered.numel() - 1))
    return torch.topk(filtered, k=k).indices.tolist()


def leave_one_out_proto(class_features, holdout_idx, fallback_proto):
    if class_features.shape[0] <= 1:
        return fallback_proto
    keep = torch.ones(class_features.shape[0], dtype=torch.bool, device=class_features.device)
    keep[int(holdout_idx)] = False
    return normalize_vector(class_features[keep].mean(dim=0))


def group_support_features(merged_state, num_cls):
    image_features = F.normalize(merged_state["images_features"], dim=-1)
    image_targets = merged_state["images_targets"]
    groups = {}
    for cls_id in range(num_cls):
        groups[cls_id] = image_features[image_targets == cls_id]
    return groups


def score_candidate_support(support_proto, candidate):
    vec = normalize_vector(candidate["vector"].to(device=support_proto.device, dtype=support_proto.dtype))
    return float(torch.dot(support_proto, vec).item())


def prune_candidate_bank(bank, support_features, max_desc_candidates, max_base_candidates, max_other_candidates):
    support_proto = normalize_vector(support_features.mean(dim=0))
    keep = []
    seen = set()

    def add_candidate(candidate):
        name = candidate["name"]
        if name in seen:
            return
        seen.add(name)
        keep.append(candidate)

    ranked = {"desc": [], "base": [], "other": []}
    for candidate in bank:
        name = candidate["name"]
        if candidate["is_baseline"] or name in {"prompt_only", "desc_mean_only"}:
            add_candidate(candidate)
            continue
        score = score_candidate_support(support_proto, candidate)
        if candidate["uses_base"]:
            ranked["base"].append((score, candidate))
        elif candidate["uses_desc_atom"]:
            ranked["desc"].append((score, candidate))
        else:
            ranked["other"].append((score, candidate))

    for group_name, limit in (
        ("desc", max_desc_candidates),
        ("base", max_base_candidates),
        ("other", max_other_candidates),
    ):
        ranked[group_name].sort(key=lambda item: item[0], reverse=True)
        for _, candidate in ranked[group_name][: max(0, int(limit))]:
            add_candidate(candidate)

    return keep


def compute_sample_probs(
    query_feature,
    semantic_proto,
    image_proto,
    num_cls,
    num_base_cls,
    beta,
    ensemble_alpha,
    description_features,
    description_targets,
    cov_image,
):
    query_feature = query_feature.unsqueeze(0)
    prob_cov = F.softmax(compute_cov_logits(query_feature, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_feature, description_features, description_targets, num_cls),
        dim=-1,
    )
    probs, logits = compose_probs_from_semantic(
        query_feature,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    return probs.squeeze(0), logits.squeeze(0), prob_cov.squeeze(0), prob_knn.squeeze(0)


def probability_margin(probs, true_cls, confusers):
    true_prob = float(probs[int(true_cls)].item())
    if not confusers:
        return true_prob
    conf_scores = probs[torch.tensor(confusers, device=probs.device, dtype=torch.long)]
    return true_prob - float(conf_scores.max().item())


def build_stable_confusers(
    cls_id,
    class_support,
    baseline_sem,
    image_proto,
    num_cls,
    num_base_cls,
    beta,
    ensemble_alpha,
    description_features,
    description_targets,
    cov_image,
    branch_topk,
    max_confusers,
):
    counter = Counter()
    for holdout_idx in range(class_support.shape[0]):
        image_proto_eval = image_proto.clone()
        image_proto_eval[cls_id] = leave_one_out_proto(class_support, holdout_idx, image_proto[cls_id])
        probs, _, prob_cov, prob_knn = compute_sample_probs(
            class_support[holdout_idx],
            baseline_sem,
            image_proto_eval,
            num_cls,
            num_base_cls,
            beta,
            ensemble_alpha,
            description_features,
            description_targets,
            cov_image,
        )
        for branch_scores in (probs, prob_cov, prob_knn):
            for confuser in top_wrong_classes(branch_scores, cls_id, branch_topk):
                counter[int(confuser)] += 1
    confusers = [cls for cls, _ in counter.most_common(max(1, int(max_confusers)))]
    return confusers, counter


def evaluate_candidate(
    cls_id,
    class_support,
    confusers,
    baseline_sem,
    candidate_vec,
    image_proto,
    num_cls,
    num_base_cls,
    beta,
    ensemble_alpha,
    description_features,
    description_targets,
    cov_image,
):
    candidate_sem = baseline_sem.clone()
    candidate_sem[cls_id] = normalize_vector(candidate_vec.to(device=baseline_sem.device, dtype=baseline_sem.dtype))

    gains = []
    baseline_margins = []
    candidate_margins = []
    for holdout_idx in range(class_support.shape[0]):
        image_proto_eval = image_proto.clone()
        image_proto_eval[cls_id] = leave_one_out_proto(class_support, holdout_idx, image_proto[cls_id])
        baseline_probs, _, _, _ = compute_sample_probs(
            class_support[holdout_idx],
            baseline_sem,
            image_proto_eval,
            num_cls,
            num_base_cls,
            beta,
            ensemble_alpha,
            description_features,
            description_targets,
            cov_image,
        )
        candidate_probs, _, _, _ = compute_sample_probs(
            class_support[holdout_idx],
            candidate_sem,
            image_proto_eval,
            num_cls,
            num_base_cls,
            beta,
            ensemble_alpha,
            description_features,
            description_targets,
            cov_image,
        )
        baseline_margin = probability_margin(baseline_probs, cls_id, confusers)
        candidate_margin = probability_margin(candidate_probs, cls_id, confusers)
        baseline_margins.append(baseline_margin)
        candidate_margins.append(candidate_margin)
        gains.append(candidate_margin - baseline_margin)

    gain_tensor = torch.tensor(gains)
    return {
        "mean_gain": float(gain_tensor.mean().item()),
        "std_gain": float(gain_tensor.std(unbiased=False).item()) if gain_tensor.numel() > 1 else 0.0,
        "positive_rate": float((gain_tensor > 0).float().mean().item()),
        "min_gain": float(gain_tensor.min().item()),
        "mean_baseline_margin": float(np.mean(baseline_margins)),
        "mean_candidate_margin": float(np.mean(candidate_margins)),
    }


def select_confusion_aware_semantics(
    merged_state,
    num_cls,
    num_base_cls,
    beta,
    lambda_t,
    ensemble_alpha,
    max_desc_atoms,
    base_atom_topk,
    max_desc_candidates,
    max_base_candidates,
    max_other_candidates,
    branch_topk,
    max_confusers,
    min_margin_gain,
    min_positive_rate,
):
    device = merged_state["image_proto"].device
    image_proto = F.normalize(merged_state["image_proto"][:num_cls].to(device), dim=-1)
    description_features = F.normalize(merged_state["description_features"].to(device), dim=-1)
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device)
    baseline_sem, banks = build_semantic_banks(
        merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )
    baseline_sem = baseline_sem.to(device)
    support_groups = group_support_features(merged_state, num_cls)

    selected_sem = baseline_sem.clone()
    chosen_meta = {}
    diagnostics = {}
    novel_classes = list(range(num_base_cls, num_cls))
    for cls_id in novel_classes:
        class_support = support_groups[cls_id].to(device)
        if class_support.numel() == 0:
            chosen_meta[cls_id] = next(candidate for candidate in banks[cls_id] if candidate["is_baseline"])
            diagnostics[cls_id] = {
                "confusers": [],
                "bank_size": 0,
                "accepted": False,
                "best_mean_gain": 0.0,
                "best_positive_rate": 0.0,
            }
            continue

        pruned_bank = prune_candidate_bank(
            banks[cls_id],
            class_support,
            max_desc_candidates=max_desc_candidates,
            max_base_candidates=max_base_candidates,
            max_other_candidates=max_other_candidates,
        )
        confusers, confuser_counter = build_stable_confusers(
            cls_id=cls_id,
            class_support=class_support,
            baseline_sem=baseline_sem,
            image_proto=image_proto,
            num_cls=num_cls,
            num_base_cls=num_base_cls,
            beta=beta,
            ensemble_alpha=ensemble_alpha,
            description_features=description_features,
            description_targets=description_targets,
            cov_image=cov_image,
            branch_topk=branch_topk,
            max_confusers=max_confusers,
        )
        best_meta = next(candidate for candidate in pruned_bank if candidate["is_baseline"])
        best_stats = {
            "mean_gain": 0.0,
            "positive_rate": 0.0,
            "min_gain": 0.0,
            "std_gain": 0.0,
            "mean_baseline_margin": 0.0,
            "mean_candidate_margin": 0.0,
        }
        accepted = False

        for candidate in pruned_bank:
            if candidate["is_baseline"]:
                continue
            stats = evaluate_candidate(
                cls_id=cls_id,
                class_support=class_support,
                confusers=confusers,
                baseline_sem=baseline_sem,
                candidate_vec=candidate["vector"],
                image_proto=image_proto,
                num_cls=num_cls,
                num_base_cls=num_base_cls,
                beta=beta,
                ensemble_alpha=ensemble_alpha,
                description_features=description_features,
                description_targets=description_targets,
                cov_image=cov_image,
            )
            if stats["mean_gain"] > best_stats["mean_gain"] + 1e-8:
                best_meta = candidate
                best_stats = stats

        if best_stats["mean_gain"] >= float(min_margin_gain) and best_stats["positive_rate"] >= float(min_positive_rate):
            accepted = True
            selected_sem[cls_id] = normalize_vector(best_meta["vector"].to(device))

        chosen_meta[cls_id] = best_meta if accepted else next(candidate for candidate in pruned_bank if candidate["is_baseline"])
        diagnostics[cls_id] = {
            "confusers": confusers,
            "confuser_counts": dict(confuser_counter),
            "bank_size": len(pruned_bank),
            "accepted": accepted,
            "best_mean_gain": round(float(best_stats["mean_gain"]), 6),
            "best_positive_rate": round(float(best_stats["positive_rate"]), 4),
            "best_min_gain": round(float(best_stats["min_gain"]), 6),
            "selected_name": chosen_meta[cls_id]["name"],
        }

    return baseline_sem, selected_sem, chosen_meta, diagnostics


def run_selector(
    merged_state,
    query_features,
    query_targets,
    num_cls,
    num_base_cls,
    beta,
    lambda_t,
    ensemble_alpha,
    max_desc_atoms,
    base_atom_topk,
    max_desc_candidates,
    max_base_candidates,
    max_other_candidates,
    branch_topk,
    max_confusers,
    min_margin_gain,
    min_positive_rate,
):
    device = query_features.device
    image_proto = F.normalize(merged_state["image_proto"][:num_cls].to(device), dim=-1)
    description_features = F.normalize(merged_state["description_features"].to(device), dim=-1)
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device)

    baseline_sem, selected_sem, chosen_meta, diagnostics = select_confusion_aware_semantics(
        merged_state=merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=ensemble_alpha,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        max_desc_candidates=max_desc_candidates,
        max_base_candidates=max_base_candidates,
        max_other_candidates=max_other_candidates,
        branch_topk=branch_topk,
        max_confusers=max_confusers,
        min_margin_gain=min_margin_gain,
        min_positive_rate=min_positive_rate,
    )

    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )
    baseline_probs, baseline_logits = compose_probs_from_semantic(
        query_features,
        baseline_sem,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    method_probs, method_logits = compose_probs_from_semantic(
        query_features,
        selected_sem,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )

    baseline_acc = accuracy_from_probs(baseline_probs, query_targets)
    method_acc = accuracy_from_probs(method_probs, query_targets)
    class_ids = list(range(num_cls))
    novel_classes = list(range(num_base_cls, num_cls))
    changed_classes = [cls_id for cls_id in novel_classes if diagnostics[cls_id]["accepted"]]
    used_base_classes = [cls_id for cls_id in novel_classes if diagnostics[cls_id]["accepted"] and chosen_meta[cls_id]["uses_base"]]
    used_desc_atom_classes = [cls_id for cls_id in novel_classes if diagnostics[cls_id]["accepted"] and chosen_meta[cls_id]["uses_desc_atom"]]
    bank_sizes = [diagnostics[cls_id]["bank_size"] for cls_id in novel_classes]
    confuser_counts = [len(diagnostics[cls_id]["confusers"]) for cls_id in novel_classes]
    selected_gains = [diagnostics[cls_id]["best_mean_gain"] for cls_id in changed_classes]

    baseline_alignment = 1.0 - torch.sum(image_proto[num_base_cls:] * baseline_sem[num_base_cls:], dim=1)
    method_alignment = 1.0 - torch.sum(image_proto[num_base_cls:] * selected_sem[num_base_cls:], dim=1)

    return {
        "baseline_acc": round(float(baseline_acc), 3),
        "method_acc": round(float(method_acc), 3),
        "method_gain": round(float(method_acc - baseline_acc), 3),
        "test_class_balanced_nll": round(float(class_balanced_nll(method_probs, query_targets, class_ids).item()), 6),
        "test_class_balanced_margin": round(float(class_balanced_margin(method_logits, query_targets, class_ids).item()), 6),
        "changed_class_rate": round(float(len(changed_classes) / max(len(novel_classes), 1)), 4),
        "base_atom_usage_rate": round(float(len(used_base_classes) / max(len(novel_classes), 1)), 4),
        "desc_atom_usage_rate": round(float(len(used_desc_atom_classes) / max(len(novel_classes), 1)), 4),
        "baseline_alignment_gap": round(float(baseline_alignment.mean().item()), 6) if novel_classes else 0.0,
        "method_alignment_gap": round(float(method_alignment.mean().item()), 6) if novel_classes else 0.0,
        "mean_pruned_bank_size": round(float(np.mean(bank_sizes)), 4) if bank_sizes else 0.0,
        "mean_confuser_count": round(float(np.mean(confuser_counts)), 4) if confuser_counts else 0.0,
        "mean_selected_margin_gain": round(float(np.mean(selected_gains)), 6) if selected_gains else 0.0,
        "chosen_candidates": {str(cls_id): chosen_meta[cls_id]["name"] for cls_id in novel_classes},
        "class_diagnostics": {str(cls_id): diagnostics[cls_id] for cls_id in novel_classes},
    }


def run_dataset(
    data_cfg,
    train_cfg,
    max_desc_atoms,
    base_atom_topk,
    max_desc_candidates,
    max_base_candidates,
    max_other_candidates,
    branch_topk,
    max_confusers,
    min_margin_gain,
    min_positive_rate,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)

    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    results = run_selector(
        merged_state=merged_state,
        query_features=query_features.to(cfg.DEVICE.DEVICE_NAME),
        query_targets=query_targets.to(cfg.DEVICE.DEVICE_NAME),
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=ensemble_alpha,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        max_desc_candidates=max_desc_candidates,
        max_base_candidates=max_base_candidates,
        max_other_candidates=max_other_candidates,
        branch_topk=branch_topk,
        max_confusers=max_confusers,
        min_margin_gain=min_margin_gain,
        min_positive_rate=min_positive_rate,
    )

    oracle_gain, oracle_dir = latest_oracle_gain(cfg, train_cfg)
    runtime_sec = round(time.time() - start_time, 3)
    recovery_rate = None if oracle_gain is None or abs(oracle_gain) <= 1e-9 else round(float(results["method_gain"] / oracle_gain), 4)
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
        "max_desc_candidates": max_desc_candidates,
        "max_base_candidates": max_base_candidates,
        "max_other_candidates": max_other_candidates,
        "branch_topk": branch_topk,
        "max_confusers": max_confusers,
        "min_margin_gain": min_margin_gain,
        "min_positive_rate": min_positive_rate,
        "oracle_gain": oracle_gain,
        "oracle_results_dir": str(oracle_dir) if oracle_dir is not None else None,
        "recovery_rate": recovery_rate,
        **results,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"runtime_sec: {runtime_sec}",
        f"baseline_acc: {results['baseline_acc']}",
        f"method_acc: {results['method_acc']}",
        f"method_gain: {results['method_gain']}",
        f"oracle_gain: {oracle_gain}",
        f"recovery_rate: {recovery_rate}",
        f"test_class_balanced_nll: {results['test_class_balanced_nll']}",
        f"test_class_balanced_margin: {results['test_class_balanced_margin']}",
        f"changed_class_rate: {results['changed_class_rate']}",
        f"base_atom_usage_rate: {results['base_atom_usage_rate']}",
        f"desc_atom_usage_rate: {results['desc_atom_usage_rate']}",
        f"mean_pruned_bank_size: {results['mean_pruned_bank_size']}",
        f"mean_confuser_count: {results['mean_confuser_count']}",
        f"mean_selected_margin_gain: {results['mean_selected_margin_gain']}",
        f"baseline_alignment_gap: {results['baseline_alignment_gap']}",
        f"method_alignment_gap: {results['method_alignment_gap']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "baseline_acc": results["baseline_acc"],
        "method_acc": results["method_acc"],
        "method_gain": results["method_gain"],
        "oracle_gain": oracle_gain,
        "recovery_rate": recovery_rate,
        "changed_class_rate": results["changed_class_rate"],
        "mean_pruned_bank_size": results["mean_pruned_bank_size"],
        "mean_confuser_count": results["mean_confuser_count"],
    }, indent=2))
    print(f"Saved semantic confusion-selector diagnostics to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--max_desc_atoms", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    parser.add_argument("--max_desc_candidates", type=int, default=2)
    parser.add_argument("--max_base_candidates", type=int, default=1)
    parser.add_argument("--max_other_candidates", type=int, default=1)
    parser.add_argument("--branch_topk", type=int, default=2)
    parser.add_argument("--max_confusers", type=int, default=4)
    parser.add_argument("--min_margin_gain", type=float, default=0.0015)
    parser.add_argument("--min_positive_rate", type=float, default=0.6)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        max_desc_candidates=args.max_desc_candidates,
        max_base_candidates=args.max_base_candidates,
        max_other_candidates=args.max_other_candidates,
        branch_topk=args.branch_topk,
        max_confusers=args.max_confusers,
        min_margin_gain=args.min_margin_gain,
        min_positive_rate=args.min_positive_rate,
    )


if __name__ == "__main__":
    main()
