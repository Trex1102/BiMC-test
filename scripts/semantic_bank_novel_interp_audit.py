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
    trainer_name = f"semantic_bank_novel_interp_audit_{Path(train_cfg).stem}"
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
    novel_mask = targets >= num_base_cls
    novel = float((preds[novel_mask] == targets[novel_mask]).float().mean().item() * 100.0) if novel_mask.any() else 0.0
    return round(overall, 3), round(novel, 3)


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


def pairwise_diversity(desc_group):
    if desc_group.shape[0] <= 1:
        return 0.0
    sims = torch.matmul(desc_group, desc_group.T)
    triu = torch.triu_indices(desc_group.shape[0], desc_group.shape[0], offset=1)
    return float((1.0 - sims[triu[0], triu[1]]).mean().item())


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


def confuser_overlap_stats(desc_group, cls_id, reference_fused_proto, ref_confuser_sets, ref_confuser_proto):
    other_sims = torch.matmul(desc_group, reference_fused_proto.T)
    other_sims[:, cls_id] = float("-inf")
    nearest_other = torch.argmax(other_sims, dim=1)
    ref_set = set(ref_confuser_sets[cls_id])
    overlap = float(sum(int(idx.item()) in ref_set for idx in nearest_other) / max(1, desc_group.shape[0]))
    conf_sims = torch.matmul(desc_group, ref_confuser_proto[cls_id].T).max(dim=1).values
    return overlap, float(conf_sims.mean().item())


def build_diverse_novel_bundle(merged_state, num_base_cls, k, reference_fused_proto, ref_confuser_sets, ref_confuser_proto):
    num_cls = merged_state["image_proto"].shape[0]
    full_description_features = F.normalize(merged_state["description_features"], dim=-1)
    description_targets = merged_state["description_targets"]
    image_proto = F.normalize(merged_state["image_proto"][:num_cls], dim=-1)
    original_description_proto = F.normalize(merged_state["description_proto"][:num_cls], dim=-1)

    fused_groups = []
    compact_novel_proto = []
    novel_counts = []
    novel_diversity = []
    novel_align_mean = []
    novel_align_spread = []
    novel_overlap = []
    novel_conf_sims = []

    for cls_id in range(num_cls):
        desc_group = full_description_features[description_targets == cls_id]
        if cls_id < num_base_cls:
            selected = desc_group
            compact_novel_proto.append(original_description_proto[cls_id])
        else:
            selected_idx = select_diverse_indices(desc_group, k)
            selected = F.normalize(desc_group[selected_idx], dim=-1)
            compact_novel_proto.append(normalize_vector(selected.mean(dim=0)))
            novel_counts.append(int(selected.shape[0]))
            novel_diversity.append(pairwise_diversity(selected))
            align = torch.matmul(selected, image_proto[cls_id])
            novel_align_mean.append(float(align.mean().item()))
            novel_align_spread.append(float(align.std(unbiased=False).item()))
            overlap, conf_sim = confuser_overlap_stats(selected, cls_id, reference_fused_proto, ref_confuser_sets, ref_confuser_proto)
            novel_overlap.append(overlap)
            novel_conf_sims.append(conf_sim)
        fused_groups.append(F.normalize(selected, dim=-1))

    stats = {
        "base_fused_count_mean": round(float(sum(group.shape[0] for group in fused_groups[:num_base_cls]) / max(1, num_base_cls)), 3),
        "novel_fused_count_mean": round(float(sum(novel_counts) / max(1, len(novel_counts))), 3),
        "knn_description_count_mean": round(float(full_description_features.shape[0] / num_cls), 3),
        "novel_within_class_diversity_mean": round(float(sum(novel_diversity) / max(1, len(novel_diversity))), 6),
        "novel_support_alignment_mean": round(float(sum(novel_align_mean) / max(1, len(novel_align_mean))), 6),
        "novel_support_alignment_spread_mean": round(float(sum(novel_align_spread) / max(1, len(novel_align_spread))), 6),
        "novel_confuser_overlap_rate_mean": round(float(sum(novel_overlap) / max(1, len(novel_overlap))), 6),
        "novel_confuser_similarity_mean": round(float(sum(novel_conf_sims) / max(1, len(novel_conf_sims))), 6),
    }

    return {
        "image_proto": merged_state["image_proto"][:num_cls],
        "text_features": merged_state["text_features"][:num_cls],
        "original_description_proto": original_description_proto,
        "compact_novel_proto": torch.stack(compact_novel_proto, dim=0),
        "fused_groups": fused_groups,
        "knn_description_features": merged_state["description_features"],
        "knn_description_targets": merged_state["description_targets"],
        "cov_image": merged_state["cov_image"],
        "stats": stats,
    }


def build_candidate_banks(bundle, fused_description_proto, num_cls, num_base_cls, lambda_t, base_atom_topk):
    text_features = F.normalize(bundle["text_features"][:num_cls], dim=-1)
    fused_description_proto = F.normalize(fused_description_proto[:num_cls], dim=-1)
    baseline_sem = F.normalize((1.0 - lambda_t) * text_features + lambda_t * fused_description_proto, dim=-1)

    banks = []
    for cls_id in range(num_cls):
        atoms = [
            {"name": "prompt", "vector": text_features[cls_id], "uses_base": False, "uses_desc_atom": False},
            {"name": "desc_mean", "vector": fused_description_proto[cls_id], "uses_base": False, "uses_desc_atom": False},
        ]
        for atom_idx, atom in enumerate(bundle["fused_groups"][cls_id]):
            atoms.append({
                "name": f"desc_atom_{atom_idx}",
                "vector": atom,
                "uses_base": False,
                "uses_desc_atom": True,
            })

        if cls_id >= num_base_cls and num_base_cls > 0:
            sims = torch.matmul(baseline_sem[cls_id], baseline_sem[:num_base_cls].T)
            topk = min(int(base_atom_topk), num_base_cls)
            top_idx = torch.topk(sims, k=topk).indices.tolist()
            for rank, base_idx in enumerate(top_idx):
                atoms.append({
                    "name": f"base_atom_{rank}",
                    "vector": baseline_sem[base_idx],
                    "uses_base": True,
                    "uses_desc_atom": False,
                })

        candidates = [{
            "name": "baseline_sem",
            "vector": baseline_sem[cls_id],
            "uses_base": False,
            "uses_desc_atom": False,
            "is_baseline": True,
        }]
        candidates.append({
            "name": "prompt_only",
            "vector": text_features[cls_id],
            "uses_base": False,
            "uses_desc_atom": False,
            "is_baseline": False,
        })
        candidates.append({
            "name": "desc_mean_only",
            "vector": fused_description_proto[cls_id],
            "uses_base": False,
            "uses_desc_atom": False,
            "is_baseline": False,
        })

        for atom in atoms[2:]:
            candidates.append({
                "name": atom["name"],
                "vector": atom["vector"],
                "uses_base": atom["uses_base"],
                "uses_desc_atom": atom["uses_desc_atom"],
                "is_baseline": False,
            })
            candidates.append({
                "name": f"prompt_plus_{atom['name']}",
                "vector": average_vectors([text_features[cls_id], atom["vector"]]),
                "uses_base": atom["uses_base"],
                "uses_desc_atom": atom["uses_desc_atom"],
                "is_baseline": False,
            })
            candidates.append({
                "name": f"desc_plus_{atom['name']}",
                "vector": average_vectors([fused_description_proto[cls_id], atom["vector"]]),
                "uses_base": atom["uses_base"],
                "uses_desc_atom": atom["uses_desc_atom"],
                "is_baseline": False,
            })

        desc_atom_vectors = [atom["vector"] for atom in atoms if atom["uses_desc_atom"]]
        if desc_atom_vectors:
            candidates.append({
                "name": "prompt_desc_top_descatom",
                "vector": average_vectors([text_features[cls_id], fused_description_proto[cls_id], desc_atom_vectors[0]]),
                "uses_base": False,
                "uses_desc_atom": True,
                "is_baseline": False,
            })

        for atom in [atom for atom in atoms if atom["uses_base"]]:
            candidates.append({
                "name": f"prompt_desc_{atom['name']}",
                "vector": average_vectors([text_features[cls_id], fused_description_proto[cls_id], atom["vector"]]),
                "uses_base": True,
                "uses_desc_atom": False,
                "is_baseline": False,
            })

        dedup = []
        seen = set()
        for candidate in candidates:
            if candidate["name"] in seen:
                continue
            seen.add(candidate["name"])
            dedup.append(candidate)
        banks.append(dedup)
    return baseline_sem, banks


def interpolate_description_proto(bundle, num_base_cls, alpha):
    alpha = float(alpha)
    original = bundle["original_description_proto"].clone()
    compact = bundle["compact_novel_proto"]
    fused = original.clone()
    if num_base_cls < fused.shape[0]:
        blended = (1.0 - alpha) * original[num_base_cls:] + alpha * compact[num_base_cls:]
        fused[num_base_cls:] = F.normalize(blended, dim=-1)
    return fused


def evaluate_baseline(bundle, fused_description_proto, query_features, query_targets, num_cls, num_base_cls, beta, lambda_t, ensemble_alpha):
    image_proto = F.normalize(bundle["image_proto"][:num_cls], dim=-1)
    text_features = F.normalize(bundle["text_features"][:num_cls], dim=-1)
    knn_description_features = F.normalize(bundle["knn_description_features"], dim=-1)
    knn_description_targets = bundle["knn_description_targets"]
    cov_image = bundle["cov_image"]

    semantic_proto = F.normalize((1.0 - lambda_t) * text_features + lambda_t * fused_description_proto[:num_cls], dim=-1)
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
    full_acc, full_novel = accuracy_stats(full_probs, query_targets, num_base_cls)
    fused_acc, fused_novel = accuracy_stats(fused_probs, query_targets, num_base_cls)
    knn_acc, knn_novel = accuracy_stats(prob_knn, query_targets, num_base_cls)
    return {
        "full_acc": full_acc,
        "full_novel_acc": full_novel,
        "fused_only_acc": fused_acc,
        "fused_only_novel_acc": fused_novel,
        "knn_only_acc": knn_acc,
        "knn_only_novel_acc": knn_novel,
        "prob_cov": prob_cov,
        "prob_knn": prob_knn,
    }


def run_oracle(bundle, fused_description_proto, query_features, query_targets, num_cls, num_base_cls, beta, lambda_t, ensemble_alpha, base_atom_topk, max_passes):
    image_proto = F.normalize(bundle["image_proto"][:num_cls], dim=-1)
    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, bundle["cov_image"]) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(
            query_features,
            F.normalize(bundle["knn_description_features"], dim=-1),
            bundle["knn_description_targets"],
            num_cls,
        ),
        dim=-1,
    )
    baseline_sem, banks = build_candidate_banks(bundle, fused_description_proto, num_cls, num_base_cls, lambda_t, base_atom_topk)
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
    baseline_acc = float((torch.argmax(baseline_probs, dim=1) == query_targets).float().mean().item() * 100.0)

    current_sem = baseline_sem.clone()
    current_probs = baseline_probs
    current_acc = baseline_acc
    chosen = {cls_id: {"is_baseline": True} for cls_id in range(num_cls)}
    novel_classes = list(range(num_base_cls, num_cls))

    for _ in range(max_passes):
        improved_any = False
        for cls_id in novel_classes:
            best_acc = current_acc
            best_probs = current_probs
            best_vec = current_sem[cls_id]
            best_is_baseline = chosen[cls_id]["is_baseline"]
            for candidate in banks[cls_id]:
                temp_sem = current_sem.clone()
                temp_sem[cls_id] = candidate["vector"]
                probs, _ = compose_probs_from_semantic(
                    query_features,
                    temp_sem,
                    image_proto,
                    num_base_cls,
                    beta,
                    ensemble_alpha,
                    prob_cov,
                    prob_knn,
                )
                acc = float((torch.argmax(probs, dim=1) == query_targets).float().mean().item() * 100.0)
                if acc > best_acc + 1e-6:
                    best_acc = acc
                    best_probs = probs
                    best_vec = candidate["vector"]
                    best_is_baseline = bool(candidate["is_baseline"])
            if best_acc > current_acc + 1e-6:
                current_acc = best_acc
                current_probs = best_probs
                current_sem[cls_id] = best_vec
                chosen[cls_id]["is_baseline"] = best_is_baseline
                improved_any = True
        if not improved_any:
            break

    changed = [cls_id for cls_id in novel_classes if not chosen[cls_id]["is_baseline"]]
    return {
        "oracle_gain": round(float(current_acc - baseline_acc), 3),
        "oracle_changed_rate": round(float(len(changed) / max(len(novel_classes), 1)), 4),
    }


def run_dataset(data_cfg, train_cfg, k, confuser_topk, base_atom_topk, alphas, seed_override=None):
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
        min(int((merged_state["description_targets"] == cls_id).sum().item()) for cls_id in range(num_cls)),
    )
    reference_fused_proto, ref_confuser_sets, ref_confuser_proto = reference_confusers(
        merged_state,
        num_cls,
        num_base_cls,
        beta,
        lambda_t,
        confuser_topk,
    )
    bundle = build_diverse_novel_bundle(
        merged_state,
        num_base_cls,
        k_eff,
        reference_fused_proto,
        ref_confuser_sets,
        ref_confuser_proto,
    )

    results = {}
    for alpha in alphas:
        fused_description_proto = interpolate_description_proto(bundle, num_base_cls, alpha)
        baseline = evaluate_baseline(
            bundle,
            fused_description_proto,
            query_features,
            query_targets,
            num_cls,
            num_base_cls,
            beta,
            lambda_t,
            ensemble_alpha,
        )
        oracle_full = run_oracle(
            bundle,
            fused_description_proto,
            query_features,
            query_targets,
            num_cls,
            num_base_cls,
            beta,
            lambda_t,
            ensemble_alpha,
            base_atom_topk,
            max_passes=2,
        )
        oracle_fused = run_oracle(
            bundle,
            fused_description_proto,
            query_features,
            query_targets,
            num_cls,
            num_base_cls,
            beta,
            lambda_t,
            1.0,
            base_atom_topk,
            max_passes=2,
        )
        results[str(alpha)] = {
            "metrics": {
                "full_acc": baseline["full_acc"],
                "full_novel_acc": baseline["full_novel_acc"],
                "fused_only_acc": baseline["fused_only_acc"],
                "fused_only_novel_acc": baseline["fused_only_novel_acc"],
                "knn_only_acc": baseline["knn_only_acc"],
                "knn_only_novel_acc": baseline["knn_only_novel_acc"],
                "full_oracle_gain": oracle_full["oracle_gain"],
                "full_oracle_changed_rate": oracle_full["oracle_changed_rate"],
                "fused_oracle_gain": oracle_fused["oracle_gain"],
                "fused_oracle_changed_rate": oracle_fused["oracle_changed_rate"],
            }
        }

    runtime_sec = round(time.time() - start_time, 3)
    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": runtime_sec,
        "k_requested": int(k),
        "k_effective": int(k_eff),
        "alphas": [float(alpha) for alpha in alphas],
        "bank_stats": bundle["stats"],
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
        f"novel_diversity: {bundle['stats']['novel_within_class_diversity_mean']}",
        f"novel_confuser_overlap: {bundle['stats']['novel_confuser_overlap_rate_mean']}",
    ]
    for alpha in alphas:
        m = results[str(alpha)]["metrics"]
        lines.extend([
            f"[alpha={alpha}] full_acc={m['full_acc']} full_novel_acc={m['full_novel_acc']} full_oracle_gain={m['full_oracle_gain']}",
            f"[alpha={alpha}] fused_only_acc={m['fused_only_acc']} fused_only_novel_acc={m['fused_only_novel_acc']} fused_oracle_gain={m['fused_oracle_gain']}",
            f"[alpha={alpha}] knn_only_acc={m['knn_only_acc']} knn_only_novel_acc={m['knn_only_novel_acc']}",
        ])
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "runtime_sec": runtime_sec,
        "k_effective": k_eff,
        "results": {
            str(alpha): {
                "full_acc": results[str(alpha)]["metrics"]["full_acc"],
                "full_novel_acc": results[str(alpha)]["metrics"]["full_novel_acc"],
                "full_oracle_gain": results[str(alpha)]["metrics"]["full_oracle_gain"],
            }
            for alpha in alphas
        },
    }, indent=2))
    print(f"Saved semantic bank novel-interp audit to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--confuser_topk", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seed_override", type=int, default=None)
    args = parser.parse_args()
    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        k=args.k,
        confuser_topk=args.confuser_topk,
        base_atom_topk=args.base_atom_topk,
        alphas=args.alphas,
        seed_override=args.seed_override,
    )


if __name__ == "__main__":
    main()
