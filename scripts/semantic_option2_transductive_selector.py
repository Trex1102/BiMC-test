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
    trainer_name = f"semantic_transductive_selector_{Path(train_cfg).stem}"
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


def select_query_pool(cls_id, baseline_probs, query_features, num_base_cls, min_queries, max_queries):
    cls_scores = baseline_probs[:, cls_id]
    preds = torch.argmax(baseline_probs, dim=1)
    idx = torch.where(preds == cls_id)[0]
    novel_candidates = torch.where(preds >= num_base_cls)[0]
    if novel_candidates.numel() == 0:
        novel_candidates = torch.arange(query_features.size(0), device=query_features.device)

    target_count = min(int(max_queries), max(int(min_queries), int(idx.numel())))
    topk = min(target_count, novel_candidates.numel())
    top_idx = novel_candidates[torch.topk(cls_scores[novel_candidates], k=topk).indices]
    idx = torch.unique(torch.cat([idx, top_idx], dim=0))

    if idx.numel() > max_queries:
        idx = idx[torch.topk(cls_scores[idx], k=max_queries).indices]
    else:
        idx = idx[torch.argsort(cls_scores[idx], descending=True)]

    weights = cls_scores[idx]
    if float(weights.sum().item()) <= 1e-12:
        weights = torch.ones_like(weights) / max(weights.numel(), 1)
    else:
        weights = weights / weights.sum()
    centroid = normalize_vector(torch.sum(query_features[idx] * weights.unsqueeze(1), dim=0))
    return idx, weights, centroid


def score_candidate_against_centroid(candidate, centroid):
    vec = normalize_vector(candidate["vector"].to(device=centroid.device, dtype=centroid.dtype))
    return float(torch.dot(vec, centroid).item())


def prune_candidate_bank(bank, centroid, max_desc_candidates, max_base_candidates, max_other_candidates):
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
        score = score_candidate_against_centroid(candidate, centroid)
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


def top_wrong_classes(scores, true_cls, topk):
    if scores.numel() <= 1:
        return []
    filtered = scores.clone()
    filtered[int(true_cls)] = float("-inf")
    k = max(1, min(int(topk), filtered.numel() - 1))
    return torch.topk(filtered, k=k).indices.tolist()


def build_confusers(cls_id, selected_indices, baseline_probs, branch_topk, max_confusers):
    counter = Counter()
    selected_probs = baseline_probs[selected_indices]
    mean_scores = selected_probs.mean(dim=0)
    for confuser in top_wrong_classes(mean_scores, cls_id, branch_topk):
        counter[int(confuser)] += max(1, selected_indices.numel() // 4)
    for probs in selected_probs:
        for confuser in top_wrong_classes(probs, cls_id, branch_topk):
            counter[int(confuser)] += 1
    return [cls for cls, _ in counter.most_common(max(1, int(max_confusers)))], counter


def weighted_log_margin(probs, cls_id, confusers, weights, eps=1e-12):
    cls_prob = torch.clamp(probs[:, cls_id], min=eps)
    if confusers:
        conf_idx = torch.tensor(confusers, device=probs.device, dtype=torch.long)
        conf_prob = torch.clamp(probs[:, conf_idx].max(dim=1).values, min=eps)
    else:
        tmp = probs.clone()
        tmp[:, cls_id] = 0.0
        conf_prob = torch.clamp(tmp.max(dim=1).values, min=eps)
    margin = torch.log(cls_prob) - torch.log(conf_prob)
    return float((margin * weights).sum().item())


def split_pool(indices, num_splits):
    chunks = torch.chunk(indices, max(1, int(num_splits)))
    return [chunk for chunk in chunks if chunk.numel() > 0]


def evaluate_candidate(
    cls_id,
    candidate_vec,
    baseline_sem,
    image_proto,
    num_base_cls,
    beta,
    ensemble_alpha,
    all_query_probs_inputs,
    selected_indices,
    selected_weights,
    confusers,
    stability_splits,
):
    query_features, prob_cov, prob_knn = all_query_probs_inputs
    candidate_sem = baseline_sem.clone()
    candidate_sem[cls_id] = normalize_vector(candidate_vec.to(device=baseline_sem.device, dtype=baseline_sem.dtype))
    candidate_probs, _ = compose_probs_from_semantic(
        query_features,
        candidate_sem,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    selected_probs = candidate_probs[selected_indices]
    score = weighted_log_margin(selected_probs, cls_id, confusers, selected_weights)
    return score, candidate_probs


def select_transductive_semantics(
    merged_state,
    query_features,
    num_cls,
    num_base_cls,
    beta,
    lambda_t,
    ensemble_alpha,
    max_desc_atoms,
    base_atom_topk,
    min_queries,
    max_queries,
    max_desc_candidates,
    max_base_candidates,
    max_other_candidates,
    branch_topk,
    max_confusers,
    min_score_gain,
    min_positive_rate,
    stability_splits,
):
    device = query_features.device
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

    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )
    baseline_probs, _ = compose_probs_from_semantic(
        query_features,
        baseline_sem,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )

    selected_sem = baseline_sem.clone()
    chosen_meta = {}
    diagnostics = {}
    novel_classes = list(range(num_base_cls, num_cls))
    probs_inputs = (query_features, prob_cov, prob_knn)

    for cls_id in novel_classes:
        selected_indices, selected_weights, centroid = select_query_pool(
            cls_id,
            baseline_probs,
            query_features,
            num_base_cls=num_base_cls,
            min_queries=min_queries,
            max_queries=max_queries,
        )
        pruned_bank = prune_candidate_bank(
            banks[cls_id],
            centroid,
            max_desc_candidates=max_desc_candidates,
            max_base_candidates=max_base_candidates,
            max_other_candidates=max_other_candidates,
        )
        confusers, confuser_counter = build_confusers(
            cls_id,
            selected_indices,
            baseline_probs,
            branch_topk=branch_topk,
            max_confusers=max_confusers,
        )

        baseline_selected_probs = baseline_probs[selected_indices]
        baseline_score = weighted_log_margin(baseline_selected_probs, cls_id, confusers, selected_weights)
        best_gain = 0.0
        best_positive_rate = 0.0
        best_candidate = next(candidate for candidate in pruned_bank if candidate["is_baseline"])
        accepted = False

        split_indices = split_pool(torch.arange(selected_indices.numel(), device=device), stability_splits)
        for candidate in pruned_bank:
            if candidate["is_baseline"]:
                continue
            candidate_score, candidate_probs = evaluate_candidate(
                cls_id=cls_id,
                candidate_vec=candidate["vector"],
                baseline_sem=baseline_sem,
                image_proto=image_proto,
                num_base_cls=num_base_cls,
                beta=beta,
                ensemble_alpha=ensemble_alpha,
                all_query_probs_inputs=probs_inputs,
                selected_indices=selected_indices,
                selected_weights=selected_weights,
                confusers=confusers,
                stability_splits=stability_splits,
            )
            gain = candidate_score - baseline_score

            pos = 0
            for split in split_indices:
                local_indices = selected_indices[split]
                local_weights = selected_weights[split]
                local_weights = local_weights / torch.clamp(local_weights.sum(), min=1e-12)
                local_baseline = weighted_log_margin(baseline_probs[local_indices], cls_id, confusers, local_weights)
                local_candidate = weighted_log_margin(candidate_probs[local_indices], cls_id, confusers, local_weights)
                if local_candidate > local_baseline:
                    pos += 1
            positive_rate = pos / max(len(split_indices), 1)

            if gain > best_gain + 1e-8:
                best_gain = float(gain)
                best_positive_rate = float(positive_rate)
                best_candidate = candidate

        if best_gain >= float(min_score_gain) and best_positive_rate >= float(min_positive_rate):
            accepted = True
            selected_sem[cls_id] = normalize_vector(best_candidate["vector"].to(device))

        chosen_meta[cls_id] = best_candidate if accepted else next(candidate for candidate in pruned_bank if candidate["is_baseline"])
        diagnostics[cls_id] = {
            "query_pool_size": int(selected_indices.numel()),
            "confusers": confusers,
            "confuser_counts": dict(confuser_counter),
            "bank_size": len(pruned_bank),
            "accepted": accepted,
            "best_score_gain": round(float(best_gain), 6),
            "best_positive_rate": round(float(best_positive_rate), 4),
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
    min_queries,
    max_queries,
    max_desc_candidates,
    max_base_candidates,
    max_other_candidates,
    branch_topk,
    max_confusers,
    min_score_gain,
    min_positive_rate,
    stability_splits,
):
    device = query_features.device
    image_proto = F.normalize(merged_state["image_proto"][:num_cls].to(device), dim=-1)
    description_features = F.normalize(merged_state["description_features"].to(device), dim=-1)
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device)

    baseline_sem, selected_sem, chosen_meta, diagnostics = select_transductive_semantics(
        merged_state=merged_state,
        query_features=query_features,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=ensemble_alpha,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        min_queries=min_queries,
        max_queries=max_queries,
        max_desc_candidates=max_desc_candidates,
        max_base_candidates=max_base_candidates,
        max_other_candidates=max_other_candidates,
        branch_topk=branch_topk,
        max_confusers=max_confusers,
        min_score_gain=min_score_gain,
        min_positive_rate=min_positive_rate,
        stability_splits=stability_splits,
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
    query_pool_sizes = [diagnostics[cls_id]["query_pool_size"] for cls_id in novel_classes]
    confuser_counts = [len(diagnostics[cls_id]["confusers"]) for cls_id in novel_classes]
    selected_gains = [diagnostics[cls_id]["best_score_gain"] for cls_id in changed_classes]

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
        "mean_query_pool_size": round(float(np.mean(query_pool_sizes)), 4) if query_pool_sizes else 0.0,
        "mean_pruned_bank_size": round(float(np.mean([diagnostics[cls_id]['bank_size'] for cls_id in novel_classes])), 4) if novel_classes else 0.0,
        "mean_confuser_count": round(float(np.mean(confuser_counts)), 4) if confuser_counts else 0.0,
        "mean_selected_score_gain": round(float(np.mean(selected_gains)), 6) if selected_gains else 0.0,
        "chosen_candidates": {str(cls_id): chosen_meta[cls_id]["name"] for cls_id in novel_classes},
        "class_diagnostics": {str(cls_id): diagnostics[cls_id] for cls_id in novel_classes},
    }


def run_dataset(
    data_cfg,
    train_cfg,
    max_desc_atoms,
    base_atom_topk,
    min_queries,
    max_queries,
    max_desc_candidates,
    max_base_candidates,
    max_other_candidates,
    branch_topk,
    max_confusers,
    min_score_gain,
    min_positive_rate,
    stability_splits,
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
        min_queries=min_queries,
        max_queries=max_queries,
        max_desc_candidates=max_desc_candidates,
        max_base_candidates=max_base_candidates,
        max_other_candidates=max_other_candidates,
        branch_topk=branch_topk,
        max_confusers=max_confusers,
        min_score_gain=min_score_gain,
        min_positive_rate=min_positive_rate,
        stability_splits=stability_splits,
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
        "min_queries": min_queries,
        "max_queries": max_queries,
        "max_desc_candidates": max_desc_candidates,
        "max_base_candidates": max_base_candidates,
        "max_other_candidates": max_other_candidates,
        "branch_topk": branch_topk,
        "max_confusers": max_confusers,
        "min_score_gain": min_score_gain,
        "min_positive_rate": min_positive_rate,
        "stability_splits": stability_splits,
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
        f"mean_query_pool_size: {results['mean_query_pool_size']}",
        f"mean_pruned_bank_size: {results['mean_pruned_bank_size']}",
        f"mean_confuser_count: {results['mean_confuser_count']}",
        f"mean_selected_score_gain: {results['mean_selected_score_gain']}",
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
        "mean_query_pool_size": results["mean_query_pool_size"],
        "mean_pruned_bank_size": results["mean_pruned_bank_size"],
    }, indent=2))
    print(f"Saved semantic transductive-selector diagnostics to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--max_desc_atoms", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    parser.add_argument("--min_queries", type=int, default=16)
    parser.add_argument("--max_queries", type=int, default=64)
    parser.add_argument("--max_desc_candidates", type=int, default=2)
    parser.add_argument("--max_base_candidates", type=int, default=1)
    parser.add_argument("--max_other_candidates", type=int, default=1)
    parser.add_argument("--branch_topk", type=int, default=2)
    parser.add_argument("--max_confusers", type=int, default=4)
    parser.add_argument("--min_score_gain", type=float, default=0.002)
    parser.add_argument("--min_positive_rate", type=float, default=0.75)
    parser.add_argument("--stability_splits", type=int, default=4)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        min_queries=args.min_queries,
        max_queries=args.max_queries,
        max_desc_candidates=args.max_desc_candidates,
        max_base_candidates=args.max_base_candidates,
        max_other_candidates=args.max_other_candidates,
        branch_topk=args.branch_topk,
        max_confusers=args.max_confusers,
        min_score_gain=args.min_score_gain,
        min_positive_rate=args.min_positive_rate,
        stability_splits=args.stability_splits,
    )


if __name__ == "__main__":
    main()
