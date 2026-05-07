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
from semantic_option2_killswitch import (
    average_vectors,
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
    trainer_name = f"semantic_branch_aware_unified_audit_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def setup_cfg_with_override(data_cfg, train_cfg, seed_override=None):
    cfg = setup_cfg(data_cfg, train_cfg)
    if seed_override is None:
        return cfg
    cfg.defrost()
    cfg.SEED = int(seed_override)
    cfg.freeze()
    return cfg


def accuracy_stats(probs, targets, num_base_cls):
    preds = torch.argmax(probs, dim=1)
    overall = float((preds == targets).float().mean().item() * 100.0)
    base_mask = targets < num_base_cls
    novel_mask = targets >= num_base_cls
    base = float((preds[base_mask] == targets[base_mask]).float().mean().item() * 100.0) if base_mask.any() else 0.0
    novel = float((preds[novel_mask] == targets[novel_mask]).float().mean().item() * 100.0) if novel_mask.any() else 0.0
    return round(overall, 3), round(base, 3), round(novel, 3)


def pairwise_diversity(desc_group):
    if desc_group.shape[0] <= 1:
        return 0.0
    sims = torch.matmul(desc_group, desc_group.T)
    triu = torch.triu_indices(desc_group.shape[0], desc_group.shape[0], offset=1)
    return float((1.0 - sims[triu[0], triu[1]]).mean().item())


def select_diverse_indices(desc_group, k):
    desc_group = F.normalize(desc_group, dim=-1)
    n = desc_group.shape[0]
    if n <= k:
        return list(range(n))
    mean_vec = normalize_vector(desc_group.mean(dim=0))
    sims_to_mean = torch.matmul(desc_group, mean_vec)
    selected = [int(torch.argmin(sims_to_mean).item())]
    while len(selected) < k:
        remaining = [idx for idx in range(n) if idx not in selected]
        rem = desc_group[remaining]
        sel = desc_group[selected]
        pair_sims = torch.matmul(rem, sel.T)
        min_dissim = (1.0 - pair_sims).min(dim=1).values
        selected.append(int(remaining[int(torch.argmax(min_dissim).item())]))
    return selected


def reference_confusers(image_proto, baseline_sem, beta, num_cls, topk):
    fused_proto = F.normalize(beta * baseline_sem + (1.0 - beta) * image_proto, dim=-1)
    confuser_proto = {}
    for cls_id in range(num_cls):
        sims = torch.matmul(fused_proto, fused_proto[cls_id])
        sims[cls_id] = float("-inf")
        k = min(int(topk), max(1, num_cls - 1))
        conf_idx = torch.topk(sims, k=k).indices
        confuser_proto[cls_id] = fused_proto[conf_idx]
    return confuser_proto


def select_top_atom(desc_group, own_image_proto, own_text_proto, confuser_bank, strategy):
    if desc_group.shape[0] == 0:
        return None
    if strategy == "prompt_top1":
        score = torch.matmul(desc_group, own_text_proto)
    elif strategy == "image_top1":
        score = torch.matmul(desc_group, own_image_proto)
    elif strategy == "confuser_top1":
        own_score = torch.matmul(desc_group, own_image_proto)
        if confuser_bank is None or confuser_bank.numel() == 0:
            conf_score = torch.zeros_like(own_score)
        else:
            conf_score = torch.matmul(desc_group, confuser_bank.T).max(dim=1).values
        score = own_score - conf_score
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return desc_group[int(torch.argmax(score).item())]


def build_bundle(merged_state, num_cls):
    return {
        "image_proto": F.normalize(merged_state["image_proto"][:num_cls], dim=-1),
        "text_features": F.normalize(merged_state["text_features"][:num_cls], dim=-1),
        "description_proto": F.normalize(merged_state["description_proto"][:num_cls], dim=-1),
        "description_features": F.normalize(merged_state["description_features"], dim=-1),
        "description_targets": merged_state["description_targets"],
        "cov_image": merged_state["cov_image"],
    }


def build_unified_semantic(
    bundle,
    num_cls,
    num_base_cls,
    beta,
    lambda_t,
    threshold,
    k,
    alpha_bank,
    alpha_compose,
    confuser_topk,
    compose_mode,
    strategy,
):
    image_proto = bundle["image_proto"][:num_cls]
    text_features = bundle["text_features"][:num_cls]
    description_proto = bundle["description_proto"][:num_cls]
    description_features = bundle["description_features"]
    description_targets = bundle["description_targets"]

    baseline_sem = F.normalize((1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
    confuser_proto = reference_confusers(image_proto, baseline_sem, beta, num_cls, confuser_topk)

    semantic_proto = baseline_sem.clone()
    gate_counts = {"diverse_branch": 0, "compose_branch": 0}
    novel_diversities = []
    branch_details = {}

    for cls_id in range(num_base_cls, num_cls):
        desc_group = description_features[description_targets == cls_id]
        diversity = pairwise_diversity(desc_group)
        novel_diversities.append(diversity)
        if diversity >= threshold:
            gate_counts["diverse_branch"] += 1
            selected = desc_group[select_diverse_indices(desc_group, k)]
            diverse_proto = normalize_vector(selected.mean(dim=0))
            blended_desc = normalize_vector((1.0 - alpha_bank) * description_proto[cls_id] + alpha_bank * diverse_proto)
            semantic_proto[cls_id] = F.normalize(
                (1.0 - lambda_t) * text_features[cls_id] + lambda_t * blended_desc,
                dim=-1,
            )
            branch_details[str(cls_id)] = {
                "branch": "diverse_branch",
                "diversity": round(diversity, 6),
            }
        else:
            gate_counts["compose_branch"] += 1
            atom = select_top_atom(
                desc_group=desc_group,
                own_image_proto=image_proto[cls_id],
                own_text_proto=text_features[cls_id],
                confuser_bank=confuser_proto[cls_id],
                strategy=strategy,
            )
            if atom is None:
                branch_details[str(cls_id)] = {
                    "branch": "compose_branch_fallback",
                    "diversity": round(diversity, 6),
                }
                continue
            if compose_mode == "prompt_desc_atom":
                composer_sem = average_vectors([text_features[cls_id], description_proto[cls_id], atom])
            elif compose_mode == "prompt_atom":
                composer_sem = average_vectors([text_features[cls_id], atom])
            elif compose_mode == "desc_atom_only":
                composer_sem = atom
            else:
                raise ValueError(f"unknown compose_mode: {compose_mode}")
            semantic_proto[cls_id] = normalize_vector(
                (1.0 - alpha_compose) * baseline_sem[cls_id] + alpha_compose * composer_sem
            )
            branch_details[str(cls_id)] = {
                "branch": "compose_branch",
                "diversity": round(diversity, 6),
            }

    stats = {
        "novel_diversity_mean": round(float(sum(novel_diversities) / max(1, len(novel_diversities))), 6),
        "diversity_threshold": float(threshold),
        "diverse_branch_rate": round(float(gate_counts["diverse_branch"] / max(1, num_cls - num_base_cls)), 4),
        "compose_branch_rate": round(float(gate_counts["compose_branch"] / max(1, num_cls - num_base_cls)), 4),
        "diverse_branch_count": int(gate_counts["diverse_branch"]),
        "compose_branch_count": int(gate_counts["compose_branch"]),
    }
    return semantic_proto, stats, branch_details


def evaluate_semantic(bundle, semantic_proto, query_features, query_targets, num_cls, num_base_cls, beta, ensemble_alpha):
    image_proto = bundle["image_proto"][:num_cls]
    cov_image = bundle["cov_image"]
    knn_description_features = bundle["description_features"]
    knn_description_targets = bundle["description_targets"]

    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(compute_knn_logits(query_features, knn_description_features, knn_description_targets, num_cls), dim=-1)
    full_probs, _ = compose_probs_from_semantic(
        query_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        ensemble_alpha,
        prob_cov,
        prob_knn,
    )
    fused_probs, _ = compose_probs_from_semantic(
        query_features,
        semantic_proto,
        image_proto,
        num_base_cls,
        beta,
        1.0,
        prob_cov,
        prob_knn,
    )
    full_acc, full_base, full_novel = accuracy_stats(full_probs, query_targets, num_base_cls)
    fused_acc, fused_base, fused_novel = accuracy_stats(fused_probs, query_targets, num_base_cls)
    knn_acc, knn_base, knn_novel = accuracy_stats(prob_knn, query_targets, num_base_cls)
    return {
        "full_acc": full_acc,
        "full_base_acc": full_base,
        "full_novel_acc": full_novel,
        "fused_only_acc": fused_acc,
        "fused_only_base_acc": fused_base,
        "fused_only_novel_acc": fused_novel,
        "knn_only_acc": knn_acc,
        "knn_only_base_acc": knn_base,
        "knn_only_novel_acc": knn_novel,
    }


def run_dataset(
    data_cfg,
    train_cfg,
    threshold,
    k,
    alpha_bank,
    alpha_compose,
    confuser_topk,
    compose_mode,
    strategy,
    seed_override=None,
):
    cfg = setup_cfg_with_override(data_cfg, train_cfg, seed_override=seed_override)
    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)
    query_features = query_features.to(cfg.DEVICE.DEVICE_NAME)
    query_targets = query_targets.to(cfg.DEVICE.DEVICE_NAME)

    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    merged_state = {
        key: value.to(cfg.DEVICE.DEVICE_NAME) if isinstance(value, torch.Tensor) else value
        for key, value in merged_state.items()
    }
    bundle = build_bundle(merged_state, num_cls)
    semantic_proto, gate_stats, branch_details = build_unified_semantic(
        bundle=bundle,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        threshold=threshold,
        k=k,
        alpha_bank=alpha_bank,
        alpha_compose=alpha_compose,
        confuser_topk=confuser_topk,
        compose_mode=compose_mode,
        strategy=strategy,
    )
    metrics = evaluate_semantic(
        bundle=bundle,
        semantic_proto=semantic_proto,
        query_features=query_features,
        query_targets=query_targets,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        ensemble_alpha=ensemble_alpha,
    )

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": round(time.time() - start_time, 3),
        "threshold": float(threshold),
        "k": int(k),
        "alpha_bank": float(alpha_bank),
        "alpha_compose": float(alpha_compose),
        "confuser_topk": int(confuser_topk),
        "compose_mode": compose_mode,
        "strategy": strategy,
        "gate_stats": gate_stats,
        "metrics": metrics,
        "branch_details": branch_details,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"runtime_sec: {payload['runtime_sec']}",
        f"threshold: {payload['threshold']}",
        f"alpha_bank: {payload['alpha_bank']}",
        f"alpha_compose: {payload['alpha_compose']}",
        f"diverse_branch_rate: {payload['gate_stats']['diverse_branch_rate']}",
        f"compose_branch_rate: {payload['gate_stats']['compose_branch_rate']}",
        f"full_acc: {payload['metrics']['full_acc']}",
        f"full_base_acc: {payload['metrics']['full_base_acc']}",
        f"full_novel_acc: {payload['metrics']['full_novel_acc']}",
        f"fused_only_novel_acc: {payload['metrics']['fused_only_novel_acc']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": payload["dataset"],
        "gate_stats": payload["gate_stats"],
        "metrics": payload["metrics"],
    }, indent=2))
    print(f"Saved semantic branch-aware unified audit to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha_bank", type=float, default=0.25)
    parser.add_argument("--alpha_compose", type=float, default=0.25)
    parser.add_argument("--confuser_topk", type=int, default=3)
    parser.add_argument("--compose_mode", default="prompt_desc_atom")
    parser.add_argument("--strategy", default="prompt_top1")
    parser.add_argument("--seed_override", type=int, default=None)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        threshold=args.threshold,
        k=args.k,
        alpha_bank=args.alpha_bank,
        alpha_compose=args.alpha_compose,
        confuser_topk=args.confuser_topk,
        compose_mode=args.compose_mode,
        strategy=args.strategy,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
