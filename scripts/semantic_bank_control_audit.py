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
    collect_final_queries,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    normalize_vector,
    run_semantic_oracle,
    to_builtin,
)


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"semantic_bank_control_audit_{Path(train_cfg).stem}"
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
    if base_mask.any():
        base = float((preds[base_mask] == targets[base_mask]).float().mean().item() * 100.0)
    else:
        base = 0.0
    if novel_mask.any():
        novel = float((preds[novel_mask] == targets[novel_mask]).float().mean().item() * 100.0)
    else:
        novel = 0.0
    return round(overall, 3), round(base, 3), round(novel, 3)


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


def select_random_indices(desc_group, k, seed):
    n = desc_group.shape[0]
    if n <= k:
        return list(range(n))
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.randperm(n, generator=g)[:k].tolist()


def select_confuser_aware_indices(desc_group, own_image_proto, confuser_bank, k):
    n = desc_group.shape[0]
    if n <= k:
        return list(range(n))
    own_scores = torch.matmul(desc_group, own_image_proto)
    if confuser_bank is None or confuser_bank.numel() == 0:
        conf_scores = torch.zeros_like(own_scores)
    else:
        conf_scores = torch.matmul(desc_group, confuser_bank.T).max(dim=1).values
    score = own_scores - conf_scores
    return torch.topk(score, k=k).indices.tolist()


def reference_confusers(merged_state, num_cls, num_base_cls, beta, lambda_t, topk):
    image_proto = F.normalize(merged_state["image_proto"][:num_cls], dim=-1)
    text_features = F.normalize(merged_state["text_features"][:num_cls], dim=-1)
    description_proto = F.normalize(merged_state["description_proto"][:num_cls], dim=-1)
    baseline_sem = F.normalize((1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
    fused_proto = F.normalize(beta * baseline_sem + (1.0 - beta) * image_proto, dim=-1)

    confuser_sets = {}
    confuser_proto = {}
    for cls_id in range(num_cls):
        sims = torch.matmul(fused_proto, fused_proto[cls_id])
        sims[cls_id] = float("-inf")
        k = min(int(topk), max(1, num_cls - 1))
        conf_idx = torch.topk(sims, k=k).indices
        confuser_sets[cls_id] = conf_idx.tolist()
        confuser_proto[cls_id] = fused_proto[conf_idx]
    return fused_proto, confuser_sets, confuser_proto


def pairwise_diversity(desc_group):
    if desc_group.shape[0] <= 1:
        return 0.0
    sims = torch.matmul(desc_group, desc_group.T)
    triu = torch.triu_indices(desc_group.shape[0], desc_group.shape[0], offset=1)
    values = 1.0 - sims[triu[0], triu[1]]
    return float(values.mean().item())


def confuser_overlap_stats(desc_group, cls_id, reference_fused_proto, ref_confuser_sets, ref_confuser_proto):
    if desc_group.numel() == 0:
        return 0.0, 0.0
    other_sims = torch.matmul(desc_group, reference_fused_proto.T)
    other_sims[:, cls_id] = float("-inf")
    nearest_other = torch.argmax(other_sims, dim=1)
    ref_set = set(ref_confuser_sets[cls_id])
    overlap = float(sum(int(idx.item()) in ref_set for idx in nearest_other) / max(1, desc_group.shape[0]))
    conf_sims = torch.matmul(desc_group, ref_confuser_proto[cls_id].T).max(dim=1).values
    return overlap, float(conf_sims.mean().item())


def build_variant_state(merged_state, variant, k, seed, reference_fused_proto, ref_confuser_sets, ref_confuser_proto):
    num_cls = merged_state["image_proto"].shape[0]
    description_features = F.normalize(merged_state["description_features"], dim=-1)
    description_targets = merged_state["description_targets"]
    image_proto = F.normalize(merged_state["image_proto"][:num_cls], dim=-1)

    new_desc_features = []
    new_desc_targets = []
    new_desc_proto = []
    stats = {
        "description_count_mean": 0.0,
        "description_count_min": 0,
        "description_count_max": 0,
        "within_class_diversity_mean": 0.0,
        "support_alignment_mean": 0.0,
        "support_alignment_spread_mean": 0.0,
        "confuser_overlap_rate_mean": 0.0,
        "confuser_similarity_mean": 0.0,
    }
    counts = []
    diversity_vals = []
    align_means = []
    align_spreads = []
    overlap_vals = []
    conf_sims = []

    for cls_id in range(num_cls):
        desc_group = description_features[description_targets == cls_id]
        if desc_group.shape[0] == 0:
            continue
        if variant == "original":
            selected_idx = list(range(desc_group.shape[0]))
        elif variant == "random_k":
            selected_idx = select_random_indices(desc_group, k, seed + cls_id * 17)
        elif variant == "diverse_k":
            selected_idx = select_diverse_indices(desc_group, k)
        elif variant == "confuser_k":
            selected_idx = select_confuser_aware_indices(
                desc_group,
                image_proto[cls_id],
                ref_confuser_proto[cls_id],
                k,
            )
        else:
            raise ValueError(f"unknown variant: {variant}")

        selected = desc_group[selected_idx]
        selected = F.normalize(selected, dim=-1)
        new_desc_features.append(selected)
        new_desc_targets.append(torch.full((selected.shape[0],), cls_id, dtype=torch.long, device=selected.device))
        new_desc_proto.append(normalize_vector(selected.mean(dim=0)))

        counts.append(int(selected.shape[0]))
        diversity_vals.append(pairwise_diversity(selected))
        align = torch.matmul(selected, image_proto[cls_id])
        align_means.append(float(align.mean().item()))
        align_spreads.append(float(align.std(unbiased=False).item()))
        overlap, conf_sim = confuser_overlap_stats(selected, cls_id, reference_fused_proto, ref_confuser_sets, ref_confuser_proto)
        overlap_vals.append(overlap)
        conf_sims.append(conf_sim)

    stats["description_count_mean"] = round(float(sum(counts) / max(1, len(counts))), 3)
    stats["description_count_min"] = int(min(counts))
    stats["description_count_max"] = int(max(counts))
    stats["within_class_diversity_mean"] = round(float(sum(diversity_vals) / max(1, len(diversity_vals))), 6)
    stats["support_alignment_mean"] = round(float(sum(align_means) / max(1, len(align_means))), 6)
    stats["support_alignment_spread_mean"] = round(float(sum(align_spreads) / max(1, len(align_spreads))), 6)
    stats["confuser_overlap_rate_mean"] = round(float(sum(overlap_vals) / max(1, len(overlap_vals))), 6)
    stats["confuser_similarity_mean"] = round(float(sum(conf_sims) / max(1, len(conf_sims))), 6)

    variant_state = dict(merged_state)
    variant_state["description_features"] = torch.cat(new_desc_features, dim=0)
    variant_state["description_targets"] = torch.cat(new_desc_targets, dim=0)
    variant_state["description_proto"] = torch.stack(new_desc_proto, dim=0)
    return variant_state, stats


def evaluate_baseline_metrics(variant_state, query_features, query_targets, num_cls, num_base_cls, beta, lambda_t, ensemble_alpha):
    image_proto = F.normalize(variant_state["image_proto"][:num_cls], dim=-1)
    text_features = F.normalize(variant_state["text_features"][:num_cls], dim=-1)
    description_proto = F.normalize(variant_state["description_proto"][:num_cls], dim=-1)
    description_features = F.normalize(variant_state["description_features"], dim=-1)
    description_targets = variant_state["description_targets"]
    cov_image = variant_state["cov_image"]

    semantic_proto = F.normalize((1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)
    logits_cov = compute_cov_logits(query_features, image_proto, cov_image)
    prob_cov = F.softmax(logits_cov / 512.0, dim=-1)
    logits_knn = compute_knn_logits(query_features, description_features, description_targets, num_cls)
    prob_knn = F.softmax(logits_knn, dim=-1)

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


def evaluate_variant(variant_state, query_features, query_targets, num_cls, num_base_cls, beta, lambda_t, ensemble_alpha, max_desc_atoms, base_atom_topk):
    baseline_metrics = evaluate_baseline_metrics(
        variant_state,
        query_features,
        query_targets,
        num_cls,
        num_base_cls,
        beta,
        lambda_t,
        ensemble_alpha,
    )
    oracle_full = run_semantic_oracle(
        merged_state=variant_state,
        query_features=query_features,
        query_targets=query_targets,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=ensemble_alpha,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        max_passes=2,
    )
    oracle_fused = run_semantic_oracle(
        merged_state=variant_state,
        query_features=query_features,
        query_targets=query_targets,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=1.0,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        max_passes=2,
    )
    return {
        **baseline_metrics,
        "full_oracle_gain": oracle_full["oracle_gain"],
        "full_oracle_changed_rate": oracle_full["changed_class_rate"],
        "fused_oracle_gain": oracle_fused["oracle_gain"],
        "fused_oracle_changed_rate": oracle_fused["changed_class_rate"],
    }


def run_dataset(data_cfg, train_cfg, k, confuser_topk, max_desc_atoms, base_atom_topk, seed_override=None):
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
    k_eff = min(
        int(k),
        min(
            int((merged_state["description_targets"] == cls_id).sum().item())
            for cls_id in range(num_cls)
        ),
    )
    reference_fused_proto, ref_confuser_sets, ref_confuser_proto = reference_confusers(
        merged_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        topk=confuser_topk,
    )

    results = {}
    for variant_name in ("original", "random_k", "diverse_k", "confuser_k"):
        variant_state, bank_stats = build_variant_state(
            merged_state=merged_state,
            variant=variant_name,
            k=k_eff,
            seed=int(cfg.SEED),
            reference_fused_proto=reference_fused_proto,
            ref_confuser_sets=ref_confuser_sets,
            ref_confuser_proto=ref_confuser_proto,
        )
        metrics = evaluate_variant(
            variant_state=variant_state,
            query_features=query_features,
            query_targets=query_targets,
            num_cls=num_cls,
            num_base_cls=num_base_cls,
            beta=beta,
            lambda_t=lambda_t,
            ensemble_alpha=ensemble_alpha,
            max_desc_atoms=max_desc_atoms,
            base_atom_topk=base_atom_topk,
        )
        results[variant_name] = {
            "bank_stats": bank_stats,
            "metrics": metrics,
        }

    runtime_sec = round(time.time() - start_time, 3)
    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": runtime_sec,
        "k_requested": int(k),
        "k_effective": int(k_eff),
        "confuser_topk": int(confuser_topk),
        "max_desc_atoms": int(max_desc_atoms),
        "base_atom_topk": int(base_atom_topk),
        "results": results,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"runtime_sec: {runtime_sec}",
        f"k_effective: {k_eff}",
    ]
    for variant_name, variant_payload in results.items():
        m = variant_payload["metrics"]
        s = variant_payload["bank_stats"]
        lines.extend([
            f"[{variant_name}] full_acc={m['full_acc']} full_base_acc={m['full_base_acc']} full_novel_acc={m['full_novel_acc']} full_oracle_gain={m['full_oracle_gain']}",
            f"[{variant_name}] fused_only_acc={m['fused_only_acc']} fused_only_base_acc={m['fused_only_base_acc']} fused_only_novel_acc={m['fused_only_novel_acc']} fused_oracle_gain={m['fused_oracle_gain']}",
            f"[{variant_name}] knn_only_acc={m['knn_only_acc']} knn_only_base_acc={m['knn_only_base_acc']} knn_only_novel_acc={m['knn_only_novel_acc']}",
            f"[{variant_name}] count_mean={s['description_count_mean']} diversity_mean={s['within_class_diversity_mean']} align_spread={s['support_alignment_spread_mean']} confuser_overlap={s['confuser_overlap_rate_mean']}",
        ])
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "runtime_sec": runtime_sec,
        "k_effective": k_eff,
        "results": {
            key: {
                "full_acc": value["metrics"]["full_acc"],
                "full_base_acc": value["metrics"]["full_base_acc"],
                "full_oracle_gain": value["metrics"]["full_oracle_gain"],
                "fused_only_acc": value["metrics"]["fused_only_acc"],
                "fused_only_base_acc": value["metrics"]["fused_only_base_acc"],
                "knn_only_acc": value["metrics"]["knn_only_acc"],
                "knn_only_base_acc": value["metrics"]["knn_only_base_acc"],
            }
            for key, value in results.items()
        },
    }, indent=2))
    print(f"Saved semantic bank control audit to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--confuser_topk", type=int, default=3)
    parser.add_argument("--max_desc_atoms", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    parser.add_argument("--seed_override", type=int, default=None)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        k=args.k,
        confuser_topk=args.confuser_topk,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
