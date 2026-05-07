import argparse
import json
import math
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
    build_semantic_banks,
    collect_final_queries,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    create_output_dir,
    extract_final_state,
    normalize_vector,
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


def spread_middle_indices(num_items, count):
    if count <= 0:
        return []
    if count >= num_items:
        return list(range(num_items))
    positions = np.linspace(0, num_items - 1, count + 2)[1:-1]
    return sorted(set(int(round(pos)) for pos in positions))


def select_split_indices(class_feats, class_proto, train_count, val_count):
    class_feats = F.normalize(class_feats, dim=-1)
    sims = torch.matmul(class_feats, normalize_vector(class_proto))
    order = torch.argsort(sims).tolist()

    val_positions = spread_middle_indices(len(order), val_count)
    val_set = {order[pos] for pos in val_positions}
    remaining = [idx for idx in order if idx not in val_set]

    if train_count >= len(remaining):
        train_set = remaining
    else:
        train_positions = spread_middle_indices(len(remaining), train_count)
        train_set = [remaining[pos] for pos in train_positions]

    val_set = sorted(val_set)
    train_set = sorted(train_set)
    return train_set, val_set


def build_support_splits(
    merged_state,
    num_cls,
    num_base_cls,
    image_proto,
    base_train_per_class,
    base_val_per_class,
    novel_val_per_class,
):
    image_features = F.normalize(merged_state["images_features"], dim=-1)
    image_targets = merged_state["images_targets"]

    train_features = []
    train_targets = []
    val_features = []
    val_targets = []

    for cls_id in range(num_cls):
        class_feats = image_features[image_targets == cls_id]
        if class_feats.numel() == 0:
            continue

        if cls_id < num_base_cls:
            train_count = min(base_train_per_class, class_feats.size(0))
            val_count = min(base_val_per_class, max(class_feats.size(0) - train_count, 0))
        else:
            val_count = min(novel_val_per_class, max(class_feats.size(0) - 1, 0))
            train_count = class_feats.size(0) - val_count

        train_idx, val_idx = select_split_indices(
            class_feats,
            image_proto[cls_id],
            train_count=train_count,
            val_count=val_count,
        )

        if train_idx:
            selected = class_feats[train_idx]
            train_features.append(selected)
            train_targets.append(torch.full((selected.size(0),), cls_id, dtype=torch.long, device=selected.device))
        if val_idx:
            selected = class_feats[val_idx]
            val_features.append(selected)
            val_targets.append(torch.full((selected.size(0),), cls_id, dtype=torch.long, device=selected.device))

    return (
        torch.cat(train_features, dim=0),
        torch.cat(train_targets, dim=0),
        torch.cat(val_features, dim=0),
        torch.cat(val_targets, dim=0),
    )


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


def class_balanced_acc(probs, targets, class_ids):
    preds = probs.argmax(dim=1)
    per_class = []
    for cls_id in class_ids:
        mask = targets == cls_id
        if mask.any():
            per_class.append((preds[mask] == targets[mask]).float().mean())
    return torch.stack(per_class).mean()


def overall_acc(probs, targets):
    return (probs.argmax(dim=1) == targets).float().mean()


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


def candidate_bank_tensors(candidate_banks, novel_classes, device):
    bank_tensors = {}
    meta = {}
    prior_probs = {}
    for cls_id in novel_classes:
        bank = candidate_banks[cls_id]
        vectors = torch.stack([cand["vector"] for cand in bank], dim=0).to(device=device, dtype=torch.float32)
        bank_tensors[cls_id] = F.normalize(vectors, dim=-1)
        meta[cls_id] = bank

        init_logits = torch.full((len(bank),), -6.0, device=device)
        baseline_index = next((i for i, cand in enumerate(bank) if cand["name"] == "baseline_sem"), 0)
        init_logits[baseline_index] = 0.0
        prior_probs[cls_id] = F.softmax(init_logits, dim=0).detach()
    return bank_tensors, meta, prior_probs


def build_semantic_proto(base_semantic, bank_tensors, logits_by_class, novel_classes):
    semantic = base_semantic.clone()
    weight_cache = {}
    for cls_id in novel_classes:
        weights = F.softmax(logits_by_class[cls_id], dim=0)
        semantic_vec = torch.matmul(weights.unsqueeze(0), bank_tensors[cls_id]).squeeze(0)
        semantic[cls_id] = normalize_vector(semantic_vec)
        weight_cache[cls_id] = weights
    return semantic, weight_cache


def evaluate_split(
    split_features,
    split_targets,
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
        split_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    return {
        "acc": class_balanced_acc(probs, split_targets, class_ids),
        "overall_acc": overall_acc(probs, split_targets),
        "nll": class_balanced_nll(probs, split_targets, class_ids),
        "margin": class_balanced_margin(logits, split_targets, class_ids),
        "probs": probs,
        "logits": logits,
    }


def train_semantic_recomposition(
    merged_state,
    train_features,
    train_targets,
    val_features,
    val_targets,
    test_features,
    test_targets,
    num_cls,
    num_base_cls,
    beta,
    lambda_t,
    ensemble_alpha,
    max_desc_atoms,
    base_atom_topk,
    lr,
    max_epochs,
    patience,
    prior_weight,
    margin_weight,
    entropy_weight,
):
    device = test_features.device
    image_proto = F.normalize(merged_state["image_proto"][:num_cls].to(device=device, dtype=torch.float32), dim=-1)
    description_features = F.normalize(merged_state["description_features"].to(device=device, dtype=torch.float32), dim=-1)
    description_targets = merged_state["description_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device=device, dtype=torch.float32)

    train_features = train_features.to(device=device, dtype=torch.float32)
    train_targets = train_targets.to(device)
    val_features = val_features.to(device=device, dtype=torch.float32)
    val_targets = val_targets.to(device)
    test_features = test_features.to(device=device, dtype=torch.float32)
    test_targets = test_targets.to(device)

    prob_cov_train = F.softmax(compute_cov_logits(train_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn_train = F.softmax(compute_knn_logits(train_features, description_features, description_targets, num_cls), dim=-1)
    prob_cov_val = F.softmax(compute_cov_logits(val_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn_val = F.softmax(compute_knn_logits(val_features, description_features, description_targets, num_cls), dim=-1)
    prob_cov_test = F.softmax(compute_cov_logits(test_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn_test = F.softmax(compute_knn_logits(test_features, description_features, description_targets, num_cls), dim=-1)

    base_semantic, candidate_banks = build_semantic_banks(
        merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )
    base_semantic = base_semantic.to(device=device, dtype=torch.float32)

    novel_classes = list(range(num_base_cls, num_cls))
    bank_tensors, bank_meta, prior_probs = candidate_bank_tensors(candidate_banks, novel_classes, device)
    logits_by_class = {
        cls_id: torch.nn.Parameter(torch.log(torch.clamp(prior_probs[cls_id], min=1e-8)))
        for cls_id in novel_classes
    }
    optimizer = torch.optim.Adam(list(logits_by_class.values()), lr=lr, weight_decay=1e-4)
    class_ids = list(range(num_cls))

    baseline_test = evaluate_split(
        test_features,
        test_targets,
        class_ids,
        base_semantic,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov_test,
        prob_knn_test,
    )

    best_state = {cls_id: logits_by_class[cls_id].detach().clone() for cls_id in novel_classes}
    best_val_acc = -1.0
    best_val_nll = float("inf")
    stale_epochs = 0

    for _ in range(max_epochs):
        optimizer.zero_grad()
        semantic_proto, weight_cache = build_semantic_proto(base_semantic, bank_tensors, logits_by_class, novel_classes)
        train_metrics = evaluate_split(
            train_features,
            train_targets,
            class_ids,
            semantic_proto,
            image_proto,
            num_base_cls,
            beta,
            ensemble_alpha,
            prob_cov_train,
            prob_knn_train,
        )

        prior_loss = []
        entropy_loss = []
        for cls_id in novel_classes:
            weights = weight_cache[cls_id]
            prior = prior_probs[cls_id]
            prior_loss.append(F.kl_div(torch.log(torch.clamp(weights, min=1e-8)), prior, reduction="batchmean"))
            entropy_loss.append(-(weights * torch.log(torch.clamp(weights, min=1e-8))).sum())
        prior_loss = torch.stack(prior_loss).mean()
        entropy_loss = torch.stack(entropy_loss).mean()

        loss = train_metrics["nll"] - margin_weight * train_metrics["margin"] + prior_weight * prior_loss + entropy_weight * entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(logits_by_class.values()), 5.0)
        optimizer.step()

        with torch.no_grad():
            semantic_proto, _ = build_semantic_proto(base_semantic, bank_tensors, logits_by_class, novel_classes)
            val_metrics = evaluate_split(
                val_features,
                val_targets,
                class_ids,
                semantic_proto,
                image_proto,
                num_base_cls,
                beta,
                ensemble_alpha,
                prob_cov_val,
                prob_knn_val,
            )
            val_acc = float(val_metrics["acc"].item())
            val_nll = float(val_metrics["nll"].item())
            improved = val_acc > best_val_acc + 1e-6 or (abs(val_acc - best_val_acc) <= 1e-6 and val_nll < best_val_nll - 1e-6)
            if improved:
                best_val_acc = val_acc
                best_val_nll = val_nll
                best_state = {cls_id: logits_by_class[cls_id].detach().clone() for cls_id in novel_classes}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

    with torch.no_grad():
        for cls_id in novel_classes:
            logits_by_class[cls_id].copy_(best_state[cls_id])
        estimated_semantic, final_weights = build_semantic_proto(base_semantic, bank_tensors, logits_by_class, novel_classes)
        test_metrics = evaluate_split(
            test_features,
            test_targets,
            class_ids,
            estimated_semantic,
            image_proto,
            num_base_cls,
            beta,
            ensemble_alpha,
            prob_cov_test,
            prob_knn_test,
        )

    changed_classes = []
    base_atom_usage = 0
    desc_atom_usage = 0
    chosen_candidates = {}
    for cls_id in novel_classes:
        top_idx = int(torch.argmax(final_weights[cls_id]).item())
        top_meta = bank_meta[cls_id][top_idx]
        chosen_candidates[str(cls_id)] = top_meta["name"]
        if top_meta["name"] != "baseline_sem":
            changed_classes.append(cls_id)
        if top_meta["uses_base"]:
            base_atom_usage += 1
        if top_meta["uses_desc_atom"]:
            desc_atom_usage += 1

    baseline_alignment = 1.0 - torch.sum(image_proto[num_base_cls:] * base_semantic[num_base_cls:], dim=1)
    estimated_alignment = 1.0 - torch.sum(image_proto[num_base_cls:] * estimated_semantic[num_base_cls:], dim=1)

    return {
        "baseline_acc": round(float(baseline_test["overall_acc"].item() * 100.0), 3),
        "trainable_acc": round(float(test_metrics["overall_acc"].item() * 100.0), 3),
        "trainable_gain": round(float((test_metrics["overall_acc"] - baseline_test["overall_acc"]).item() * 100.0), 3),
        "best_val_acc": round(float(best_val_acc * 100.0), 3),
        "best_val_nll": round(float(best_val_nll), 6),
        "test_class_balanced_nll": round(float(test_metrics["nll"].item()), 6),
        "test_class_balanced_margin": round(float(test_metrics["margin"].item()), 6),
        "changed_class_rate": round(float(len(changed_classes) / max(len(novel_classes), 1)), 4),
        "base_atom_usage_rate": round(float(base_atom_usage / max(len(novel_classes), 1)), 4),
        "desc_atom_usage_rate": round(float(desc_atom_usage / max(len(novel_classes), 1)), 4),
        "baseline_alignment_gap": round(float(baseline_alignment.mean().item()), 6) if novel_classes else 0.0,
        "trainable_alignment_gap": round(float(estimated_alignment.mean().item()), 6) if novel_classes else 0.0,
        "chosen_candidates": chosen_candidates,
    }


def bmvc_decision(gains):
    non_negative_all = all(gain >= 0.0 for gain in gains.values())
    strong_datasets = sum(gain >= 0.5 for gain in gains.values())
    return non_negative_all and strong_datasets >= 2


def run_dataset(
    data_cfg,
    train_cfg,
    max_desc_atoms,
    base_atom_topk,
    base_train_per_class,
    base_val_per_class,
    novel_val_per_class,
    lr,
    max_epochs,
    patience,
    prior_weight,
    margin_weight,
    entropy_weight,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    out_dir = create_output_dir(cfg, f"semantic_trainable_{Path(train_cfg).stem}")
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

    train_features, train_targets, val_features, val_targets = build_support_splits(
        merged_state=merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        image_proto=image_proto,
        base_train_per_class=base_train_per_class,
        base_val_per_class=base_val_per_class,
        novel_val_per_class=novel_val_per_class,
    )

    results = train_semantic_recomposition(
        merged_state=merged_state,
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        test_features=query_features,
        test_targets=query_targets,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=ensemble_alpha,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        prior_weight=prior_weight,
        margin_weight=margin_weight,
        entropy_weight=entropy_weight,
    )

    oracle_gain, oracle_dir = latest_oracle_gain(cfg, train_cfg)
    runtime_sec = round(time.time() - start_time, 3)
    recovery_rate = None if oracle_gain is None or abs(oracle_gain) <= 1e-9 else round(float(results["trainable_gain"] / oracle_gain), 4)

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
        "base_train_per_class": base_train_per_class,
        "base_val_per_class": base_val_per_class,
        "novel_val_per_class": novel_val_per_class,
        "lr": lr,
        "max_epochs": max_epochs,
        "patience": patience,
        "prior_weight": prior_weight,
        "margin_weight": margin_weight,
        "entropy_weight": entropy_weight,
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
        f"trainable_acc: {results['trainable_acc']}",
        f"trainable_gain: {results['trainable_gain']}",
        f"oracle_gain: {oracle_gain}",
        f"recovery_rate: {recovery_rate}",
        f"best_val_acc: {results['best_val_acc']}",
        f"best_val_nll: {results['best_val_nll']}",
        f"test_class_balanced_nll: {results['test_class_balanced_nll']}",
        f"test_class_balanced_margin: {results['test_class_balanced_margin']}",
        f"changed_class_rate: {results['changed_class_rate']}",
        f"base_atom_usage_rate: {results['base_atom_usage_rate']}",
        f"desc_atom_usage_rate: {results['desc_atom_usage_rate']}",
        f"baseline_alignment_gap: {results['baseline_alignment_gap']}",
        f"trainable_alignment_gap: {results['trainable_alignment_gap']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "baseline_acc": results["baseline_acc"],
        "trainable_acc": results["trainable_acc"],
        "trainable_gain": results["trainable_gain"],
        "oracle_gain": oracle_gain,
        "recovery_rate": recovery_rate,
        "changed_class_rate": results["changed_class_rate"],
        "base_atom_usage_rate": results["base_atom_usage_rate"],
        "desc_atom_usage_rate": results["desc_atom_usage_rate"],
    }, indent=2))
    print(f"Saved trainable semantic recomposition diagnostics to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--max_desc_atoms", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    parser.add_argument("--base_train_per_class", type=int, default=2)
    parser.add_argument("--base_val_per_class", type=int, default=1)
    parser.add_argument("--novel_val_per_class", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--prior_weight", type=float, default=0.2)
    parser.add_argument("--margin_weight", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        base_train_per_class=args.base_train_per_class,
        base_val_per_class=args.base_val_per_class,
        novel_val_per_class=args.novel_val_per_class,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        prior_weight=args.prior_weight,
        margin_weight=args.margin_weight,
        entropy_weight=args.entropy_weight,
    )


if __name__ == "__main__":
    main()
