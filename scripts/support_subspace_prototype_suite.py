import argparse
import contextlib
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import setup_cfg
from prototype_refinement_conservative_dino_fusion import (
    collect_final_queries as collect_tric_queries,
    compose_dino_fusion_probs,
    extract_final_state as extract_tric_state,
    normalize_rows,
)
from semantic_option2_killswitch import (
    collect_final_queries as collect_bimc_queries,
    compose_probs_from_semantic,
    compute_cov_logits as compute_bimc_cov_logits,
    compute_knn_logits as compute_bimc_knn_logits,
    extract_final_state as extract_bimc_state,
)


TRIC_DATA_CFGS = [
    "configs/datasets/cifar100.yaml",
    "configs/datasets/miniimagenet.yaml",
    "configs/datasets/cub200_bimc_dino_fusion.yaml",
]
BIMC_DATA_CFGS = [
    "configs/datasets/cifar100.yaml",
    "configs/datasets/miniimagenet.yaml",
    "configs/datasets/cub200_adaptive.yaml",
]
TRIC_TRAIN_CFG = "configs/trainers/bimc_dino_fusion.yaml"
BIMC_TRAIN_CFG = "configs/trainers/bimc_ensemble.yaml"
BIMC_COV_SCALE = 512.0


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def dataset_output_name(data_cfg):
    stem = Path(data_cfg).stem
    if stem == "cifar100":
        return "CIFAR100"
    if stem in {"cub200", "cub200_bimc_dino_fusion"}:
        return "cub200"
    return stem


def output_root_from_arg(base_output_root, tag):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(base_output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    out_dir = root / f"{tag}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def pct(correct_mask):
    if correct_mask.numel() == 0:
        return 0.0
    return round(float(correct_mask.float().mean().item() * 100.0), 3)


def stats_from_scores(scores, targets, num_base_cls):
    preds = torch.argmax(scores, dim=1)
    correct = preds.eq(targets)
    base_mask = targets < int(num_base_cls)
    novel_mask = ~base_mask
    return {
        "full_acc": pct(correct),
        "base_acc": pct(correct[base_mask]) if base_mask.any() else 0.0,
        "novel_acc": pct(correct[novel_mask]) if novel_mask.any() else 0.0,
        "correct_count": int(correct.sum().item()),
    }


def vec_normalize(vec, eps=1e-12):
    denom = torch.clamp(torch.linalg.vector_norm(vec), min=float(eps))
    return vec / denom


def orthonormalize_columns(matrix, eps=1e-8):
    if matrix.numel() == 0 or matrix.shape[1] == 0:
        return matrix.new_zeros((matrix.shape[0], 0))
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    keep = s > float(eps)
    if not torch.any(keep):
        return matrix.new_zeros((matrix.shape[0], 0))
    return u[:, keep]


def remove_shared_from_vector(vec, shared_basis):
    if shared_basis.shape[1] == 0:
        return vec
    return vec - shared_basis @ (shared_basis.T @ vec)


def remove_shared_from_columns(matrix, shared_basis):
    if matrix.numel() == 0 or matrix.shape[1] == 0 or shared_basis.shape[1] == 0:
        return matrix
    return matrix - shared_basis @ (shared_basis.T @ matrix)


def build_support_geometry(support_features, support_labels, num_cls, shared_rank, subspace_rank, eps):
    dim = int(support_features.shape[1])
    means = support_features.new_zeros((num_cls, dim))
    prototypes = support_features.new_zeros((num_cls, dim))
    class_bases = []
    class_eigvals = []
    class_counts = []
    class_ranks = []

    for cls_id in range(num_cls):
        feats = support_features[support_labels == cls_id]
        count = int(feats.shape[0])
        class_counts.append(count)
        if count <= 0:
            raise RuntimeError(f"class {cls_id} has no support features")

        mu = feats.mean(dim=0)
        means[cls_id] = mu
        prototypes[cls_id] = vec_normalize(mu, eps=eps)

        if count <= 1:
            class_bases.append(support_features.new_zeros((dim, 0)))
            class_eigvals.append(support_features.new_zeros((0,)))
            class_ranks.append(0)
            continue

        centered = (feats - mu).T
        u, s, _ = torch.linalg.svd(centered, full_matrices=False)
        keep = s > float(eps)
        max_rank = int(keep.sum().item())
        k = min(int(subspace_rank), max_rank)
        if k <= 0:
            class_bases.append(support_features.new_zeros((dim, 0)))
            class_eigvals.append(support_features.new_zeros((0,)))
            class_ranks.append(0)
            continue

        basis = u[:, :k]
        eigvals = (s[:k] ** 2) / float(max(count - 1, 1))
        class_bases.append(basis)
        class_eigvals.append(eigvals)
        class_ranks.append(int(k))

    proto_centered = (prototypes - prototypes.mean(dim=0, keepdim=True)).T
    if int(shared_rank) > 0 and proto_centered.shape[1] > 1:
        u_shared, s_shared, _ = torch.linalg.svd(proto_centered, full_matrices=False)
        keep = s_shared > float(eps)
        shared_rank_used = min(int(shared_rank), int(keep.sum().item()))
        if shared_rank_used > 0:
            shared_basis = u_shared[:, :shared_rank_used]
            total_energy = torch.clamp(torch.sum(s_shared ** 2), min=float(eps))
            shared_energy = float(torch.sum(s_shared[:shared_rank_used] ** 2).item() / total_energy.item())
        else:
            shared_basis = support_features.new_zeros((dim, 0))
            shared_energy = 0.0
    else:
        shared_basis = support_features.new_zeros((dim, 0))
        shared_rank_used = 0
        shared_energy = 0.0

    return {
        "means": means,
        "prototypes": prototypes,
        "class_bases": class_bases,
        "class_eigvals": class_eigvals,
        "class_counts": class_counts,
        "class_ranks": class_ranks,
        "shared_basis": shared_basis,
        "shared_rank_used": int(shared_rank_used),
        "shared_energy_ratio": float(shared_energy),
    }


def build_weighted_unique_component(p_minus, class_basis, eigvals, shared_basis, eps):
    if class_basis.shape[1] == 0:
        return p_minus.new_zeros(p_minus.shape)

    cleaned = remove_shared_from_columns(class_basis, shared_basis)
    cleaned_cols = []
    kept_weights = []
    for idx in range(cleaned.shape[1]):
        col = cleaned[:, idx]
        norm = torch.linalg.vector_norm(col)
        if float(norm.item()) <= float(eps):
            continue
        cleaned_cols.append(col / norm)
        kept_weights.append(eigvals[idx])

    if not cleaned_cols:
        return p_minus.new_zeros(p_minus.shape)

    cleaned_basis = torch.stack(cleaned_cols, dim=1)
    weights = torch.stack(kept_weights).to(dtype=p_minus.dtype, device=p_minus.device)
    weights = weights / torch.clamp(weights.sum(), min=float(eps))
    coeff = cleaned_basis.T @ p_minus
    return cleaned_basis @ (weights * coeff)


def build_clean_projector_component(p_minus, class_basis, shared_basis, eps):
    if class_basis.shape[1] == 0:
        return p_minus.new_zeros(p_minus.shape)
    cleaned_basis = orthonormalize_columns(remove_shared_from_columns(class_basis, shared_basis), eps=eps)
    if cleaned_basis.shape[1] == 0:
        return p_minus.new_zeros(p_minus.shape)
    return cleaned_basis @ (cleaned_basis.T @ p_minus)


def build_refined_prototypes(
    orig_prototypes,
    support_features,
    support_labels,
    num_cls,
    num_base_cls,
    shared_rank,
    subspace_rank,
    lambda_unique,
    beta_residual,
    novel_only,
    weighted_unique,
    eps,
):
    geometry = build_support_geometry(
        support_features=support_features,
        support_labels=support_labels,
        num_cls=num_cls,
        shared_rank=shared_rank,
        subspace_rank=subspace_rank,
        eps=eps,
    )
    mean_prototypes_all = geometry["prototypes"]
    shared_basis = geometry["shared_basis"]

    residualized_all = torch.empty_like(mean_prototypes_all)
    for cls_id in range(num_cls):
        p_minus = remove_shared_from_vector(mean_prototypes_all[cls_id], shared_basis)
        if float(torch.linalg.vector_norm(p_minus).item()) <= float(eps):
            p_minus = mean_prototypes_all[cls_id]
        residualized_all[cls_id] = vec_normalize(p_minus, eps=eps)

    mean_variant = orig_prototypes.clone()
    refined_variant = orig_prototypes.clone()
    refine_start = int(num_base_cls) if novel_only else 0
    refine_ids = list(range(refine_start, int(num_cls)))
    if refine_ids:
        index_tensor = torch.tensor(refine_ids, device=orig_prototypes.device, dtype=torch.long)
        mean_variant[index_tensor] = mean_prototypes_all[index_tensor]

    cosine_matrix = residualized_all @ residualized_all.T
    cosine_matrix.fill_diagonal_(-1e9)

    competitor_similarity = {}
    proto_shift_cos = {}
    mean_shift_cos = {}
    class_component_norm = {}
    refined_component_norm = {}

    for cls_id in refine_ids:
        p_minus = residualized_all[cls_id]
        class_basis = geometry["class_bases"][cls_id]
        eigvals = geometry["class_eigvals"][cls_id]
        if weighted_unique:
            unique_component = build_weighted_unique_component(
                p_minus=p_minus,
                class_basis=class_basis,
                eigvals=eigvals,
                shared_basis=shared_basis,
                eps=eps,
            )
        else:
            unique_component = build_clean_projector_component(
                p_minus=p_minus,
                class_basis=class_basis,
                shared_basis=shared_basis,
                eps=eps,
            )

        competitor_id = int(torch.argmax(cosine_matrix[cls_id]).item())
        residual_dir = residualized_all[cls_id] - residualized_all[competitor_id]
        if float(torch.linalg.vector_norm(residual_dir).item()) <= float(eps):
            residual_dir = residualized_all[cls_id]
        residual_dir = vec_normalize(residual_dir, eps=eps)

        refined = residualized_all[cls_id] + float(lambda_unique) * unique_component + float(beta_residual) * residual_dir
        refined = vec_normalize(refined, eps=eps)
        refined_variant[cls_id] = refined

        proto_shift_cos[str(cls_id)] = round(float(torch.dot(orig_prototypes[cls_id], mean_prototypes_all[cls_id]).item()), 6)
        mean_shift_cos[str(cls_id)] = round(float(torch.dot(mean_prototypes_all[cls_id], refined).item()), 6)
        competitor_similarity[str(cls_id)] = round(float(cosine_matrix[cls_id, competitor_id].item()), 6)
        class_component_norm[str(cls_id)] = round(float(torch.linalg.vector_norm(unique_component).item()), 6)
        refined_component_norm[str(cls_id)] = round(float(torch.linalg.vector_norm(refined - residualized_all[cls_id]).item()), 6)

    novel_counts = geometry["class_counts"][refine_start:]
    novel_ranks = geometry["class_ranks"][refine_start:]
    mean_orig_cos = [float(torch.dot(orig_prototypes[idx], mean_variant[idx]).item()) for idx in refine_ids]
    refined_mean_cos = [float(torch.dot(mean_variant[idx], refined_variant[idx]).item()) for idx in refine_ids]
    refined_orig_cos = [float(torch.dot(orig_prototypes[idx], refined_variant[idx]).item()) for idx in refine_ids]

    return {
        "mean_variant": mean_variant,
        "refined_variant": refined_variant,
        "geometry": {
            "shared_rank_used": int(geometry["shared_rank_used"]),
            "shared_energy_ratio": round(float(geometry["shared_energy_ratio"]), 6),
            "novel_class_count": len(refine_ids),
            "novel_support_count_min": int(min(novel_counts)) if novel_counts else 0,
            "novel_support_count_mean": round(float(sum(novel_counts) / max(len(novel_counts), 1)), 3) if novel_counts else 0.0,
            "novel_support_count_max": int(max(novel_counts)) if novel_counts else 0,
            "novel_basis_rank_min": int(min(novel_ranks)) if novel_ranks else 0,
            "novel_basis_rank_mean": round(float(sum(novel_ranks) / max(len(novel_ranks), 1)), 3) if novel_ranks else 0.0,
            "novel_basis_rank_max": int(max(novel_ranks)) if novel_ranks else 0,
            "mean_orig_proto_cos": round(float(sum(mean_orig_cos) / max(len(mean_orig_cos), 1)), 6) if mean_orig_cos else 0.0,
            "mean_refined_vs_mean_cos": round(float(sum(refined_mean_cos) / max(len(refined_mean_cos), 1)), 6) if refined_mean_cos else 0.0,
            "mean_refined_vs_orig_cos": round(float(sum(refined_orig_cos) / max(len(refined_orig_cos), 1)), 6) if refined_orig_cos else 0.0,
            "per_class_orig_to_mean_cos": proto_shift_cos,
            "per_class_mean_to_refined_cos": mean_shift_cos,
            "per_class_competitor_cos": competitor_similarity,
            "per_class_unique_component_norm": class_component_norm,
            "per_class_total_refine_delta_norm": refined_component_norm,
        },
    }


def build_bimc_problem(data_cfg, train_cfg, seed):
    cfg = setup_cfg(data_cfg, train_cfg)
    cfg.defrost()
    cfg.SEED = int(seed)
    cfg.freeze()

    data_manager, model, merged_state, final_task_id = extract_bimc_state(cfg)
    query_features, query_targets = collect_bimc_queries(cfg, data_manager, model, final_task_id)

    device = cfg.DEVICE.DEVICE_NAME
    query_features = normalize_rows(query_features.to(device))
    query_targets = query_targets.to(device)

    num_cls = int(max(data_manager.class_index_in_task[final_task_id]) + 1)
    num_base_cls = int(len(data_manager.class_index_in_task[0]))
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    text_features = normalize_rows(merged_state["text_features"][:num_cls].to(device))
    description_proto = normalize_rows(merged_state["description_proto"][:num_cls].to(device))
    semantic_proto = normalize_rows((1.0 - lambda_t) * text_features + lambda_t * description_proto)
    image_proto = normalize_rows(merged_state["image_proto"][:num_cls].to(device))
    support_features = normalize_rows(merged_state["images_features"].to(device))
    support_labels = merged_state["images_targets"].to(device)
    cov_image = merged_state["cov_image"].to(device).float()
    description_features = normalize_rows(merged_state["description_features"].to(device))
    description_targets = merged_state["description_targets"].to(device)
    prob_knn = F.softmax(
        compute_bimc_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )

    return {
        "framework": "bimc_clip",
        "cfg": cfg,
        "dataset_name": str(cfg.DATASET.NAME),
        "query_features": query_features,
        "query_targets": query_targets,
        "num_cls": num_cls,
        "num_base_cls": num_base_cls,
        "beta": beta,
        "lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "semantic_proto": semantic_proto,
        "orig_prototypes": image_proto,
        "support_features": support_features,
        "support_labels": support_labels,
        "cov_image": cov_image,
        "description_features": description_features,
        "description_targets": description_targets,
        "prob_knn": prob_knn,
    }


def evaluate_bimc_variant(problem, image_proto):
    prob_cov = F.softmax(
        compute_bimc_cov_logits(problem["query_features"], image_proto, problem["cov_image"]) / float(BIMC_COV_SCALE),
        dim=-1,
    )
    full_probs, _ = compose_probs_from_semantic(
        problem["query_features"],
        problem["semantic_proto"],
        image_proto,
        problem["num_base_cls"],
        problem["beta"],
        problem["ensemble_alpha"],
        prob_cov,
        problem["prob_knn"],
    )
    visual_scores = torch.matmul(problem["query_features"], image_proto.T)
    return {
        "visual": stats_from_scores(visual_scores, problem["query_targets"], problem["num_base_cls"]),
        "full": stats_from_scores(full_probs, problem["query_targets"], problem["num_base_cls"]),
    }


def build_tric_problem(data_cfg, train_cfg, seed):
    cfg = setup_cfg(data_cfg, train_cfg)
    cfg.defrost()
    cfg.SEED = int(seed)
    cfg.freeze()

    data_manager, model, merged_state, final_task_id = extract_tric_state(cfg)
    clip_query, dino_query, query_targets = collect_tric_queries(cfg, data_manager, model, final_task_id)

    device = cfg.DEVICE.DEVICE_NAME
    clip_query = normalize_rows(clip_query.to(device))
    dino_query = normalize_rows(dino_query.to(device))
    query_targets = query_targets.to(device)

    num_cls = int(max(data_manager.class_index_in_task[final_task_id]) + 1)
    num_base_cls = int(len(data_manager.class_index_in_task[0]))
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)
    omega = float(getattr(model, "OMEGA", 0.7))
    logit_temp = float(getattr(model, "LOGIT_TEMP", 100.0))
    cov_scale = float(getattr(model, "DINO_COV_SCALE", 1.0))

    text_features = normalize_rows(merged_state["text_features"][:num_cls].to(device))
    description_proto = normalize_rows(merged_state["description_proto"][:num_cls].to(device))
    semantic_proto = normalize_rows((1.0 - lambda_t) * text_features + lambda_t * description_proto)
    clip_image_proto = normalize_rows(merged_state["image_proto"][:num_cls].to(device))
    dino_image_proto = normalize_rows(merged_state["dino_image_proto"][:num_cls].to(device))
    dino_cov = merged_state["dino_cov_image"].to(device).float()
    description_features = normalize_rows(merged_state["description_features"].to(device))
    description_targets = merged_state["description_targets"].to(device)
    support_features = normalize_rows(merged_state["dino_images_features"].to(device))
    support_labels = merged_state["dino_images_targets"].to(device)

    return {
        "framework": "tric_dinov2",
        "cfg": cfg,
        "dataset_name": str(cfg.DATASET.NAME),
        "clip_query": clip_query,
        "dino_query": dino_query,
        "query_targets": query_targets,
        "num_cls": num_cls,
        "num_base_cls": num_base_cls,
        "beta": beta,
        "lambda_t": lambda_t,
        "ensemble_alpha": ensemble_alpha,
        "omega": omega,
        "logit_temp": logit_temp,
        "cov_scale": cov_scale,
        "semantic_proto": semantic_proto,
        "clip_image_proto": clip_image_proto,
        "orig_prototypes": dino_image_proto,
        "support_features": support_features,
        "support_labels": support_labels,
        "dino_cov": dino_cov,
        "description_features": description_features,
        "description_targets": description_targets,
    }


def evaluate_tric_variant(problem, dino_image_proto):
    full_probs = compose_dino_fusion_probs(
        problem["clip_query"],
        problem["dino_query"],
        problem["semantic_proto"],
        problem["clip_image_proto"],
        dino_image_proto,
        problem["dino_cov"],
        problem["description_features"],
        problem["description_targets"],
        problem["num_base_cls"],
        problem["beta"],
        problem["ensemble_alpha"],
        omega=problem["omega"],
        logit_temp=problem["logit_temp"],
        cov_scale=problem["cov_scale"],
    )
    visual_scores = torch.matmul(problem["dino_query"], dino_image_proto.T)
    return {
        "visual": stats_from_scores(visual_scores, problem["query_targets"], problem["num_base_cls"]),
        "full": stats_from_scores(full_probs, problem["query_targets"], problem["num_base_cls"]),
    }


def evaluate_problem(problem, params):
    refined = build_refined_prototypes(
        orig_prototypes=problem["orig_prototypes"],
        support_features=problem["support_features"],
        support_labels=problem["support_labels"],
        num_cls=problem["num_cls"],
        num_base_cls=problem["num_base_cls"],
        shared_rank=params["shared_rank"],
        subspace_rank=params["subspace_rank"],
        lambda_unique=params["lambda_unique"],
        beta_residual=params["beta_residual"],
        novel_only=params["novel_only"],
        weighted_unique=params["weighted_unique"],
        eps=params["eps"],
    )

    if problem["framework"] == "bimc_clip":
        evaluator = evaluate_bimc_variant
    else:
        evaluator = evaluate_tric_variant

    baseline_metrics = evaluator(problem, problem["orig_prototypes"])
    mean_metrics = evaluator(problem, refined["mean_variant"])
    refined_metrics = evaluator(problem, refined["refined_variant"])

    payload = {
        "baseline": baseline_metrics,
        "mean_rebuild": mean_metrics,
        "refined": refined_metrics,
        "gains": {
            "mean_full_gain": round(mean_metrics["full"]["full_acc"] - baseline_metrics["full"]["full_acc"], 3),
            "refined_full_gain": round(refined_metrics["full"]["full_acc"] - baseline_metrics["full"]["full_acc"], 3),
            "mean_visual_gain": round(mean_metrics["visual"]["full_acc"] - baseline_metrics["visual"]["full_acc"], 3),
            "refined_visual_gain": round(refined_metrics["visual"]["full_acc"] - baseline_metrics["visual"]["full_acc"], 3),
        },
        "geometry": refined["geometry"],
    }
    return payload


def summary_lines(result):
    lines = [
        f"framework: {result['framework']}",
        f"dataset: {result['dataset_key']}",
        f"seed: {result['seed']}",
        f"started_at: {result['started_at']}",
        f"completed_at: {result['completed_at']}",
        f"runtime_sec: {result['runtime_sec']}",
        "",
        "params:",
        f"  novel_only: {result['params']['novel_only']}",
        f"  shared_rank: {result['params']['shared_rank']}",
        f"  subspace_rank: {result['params']['subspace_rank']}",
        f"  lambda_unique: {result['params']['lambda_unique']}",
        f"  beta_residual: {result['params']['beta_residual']}",
        f"  weighted_unique: {result['params']['weighted_unique']}",
        "",
        "full accuracy:",
        f"  baseline: {result['baseline']['full']['full_acc']}",
        f"  mean_rebuild: {result['mean_rebuild']['full']['full_acc']}",
        f"  refined: {result['refined']['full']['full_acc']}",
        "",
        "visual accuracy:",
        f"  baseline: {result['baseline']['visual']['full_acc']}",
        f"  mean_rebuild: {result['mean_rebuild']['visual']['full_acc']}",
        f"  refined: {result['refined']['visual']['full_acc']}",
        "",
        "gains:",
        f"  mean_full_gain: {result['gains']['mean_full_gain']}",
        f"  refined_full_gain: {result['gains']['refined_full_gain']}",
        f"  mean_visual_gain: {result['gains']['mean_visual_gain']}",
        f"  refined_visual_gain: {result['gains']['refined_visual_gain']}",
        "",
        "geometry:",
        f"  shared_rank_used: {result['geometry']['shared_rank_used']}",
        f"  shared_energy_ratio: {result['geometry']['shared_energy_ratio']}",
        f"  mean_orig_proto_cos: {result['geometry']['mean_orig_proto_cos']}",
        f"  mean_refined_vs_mean_cos: {result['geometry']['mean_refined_vs_mean_cos']}",
        f"  mean_refined_vs_orig_cos: {result['geometry']['mean_refined_vs_orig_cos']}",
    ]
    return lines


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def run_single_dataset(spec, data_cfg, seed, output_root, params):
    dataset_key = dataset_output_name(data_cfg)
    out_dir = output_root / spec["framework"] / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = now_iso()
    start_time = time.time()
    problem = spec["builder"](data_cfg, spec["train_cfg"], seed)
    metrics = evaluate_problem(problem, params)
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
        "params": {
            "shared_rank": int(params["shared_rank"]),
            "subspace_rank": int(params["subspace_rank"]),
            "lambda_unique": float(params["lambda_unique"]),
            "beta_residual": float(params["beta_residual"]),
            "novel_only": bool(params["novel_only"]),
            "weighted_unique": bool(params["weighted_unique"]),
            "eps": float(params["eps"]),
        },
        "num_seen_classes": int(problem["num_cls"]),
        "num_base_classes": int(problem["num_base_cls"]),
    }
    result.update(metrics)

    write_json(out_dir / "results.json", result)
    (out_dir / "summary.txt").write_text("\n".join(summary_lines(result)) + "\n", encoding="utf-8")
    (out_dir / "completion.txt").write_text(f"{completed_at}\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Support-only shared-subspace prototype refinement for BiMC and TriC.")
    parser.add_argument("--frameworks", nargs="+", default=["bimc_clip", "tric_dinov2"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_root", default="experiments")
    parser.add_argument("--shared_rank", type=int, default=1)
    parser.add_argument("--subspace_rank", type=int, default=3)
    parser.add_argument("--lambda_unique", type=float, default=0.35)
    parser.add_argument("--beta_residual", type=float, default=0.15)
    parser.add_argument("--novel_only", action="store_true", default=True)
    parser.add_argument("--all_seen", action="store_true", help="Refine all seen classes instead of novel-only.")
    parser.add_argument("--no_weighted_unique", action="store_true", help="Use an unweighted clean projector instead of normalized eigenvalue weighting.")
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    params = {
        "shared_rank": int(args.shared_rank),
        "subspace_rank": int(args.subspace_rank),
        "lambda_unique": float(args.lambda_unique),
        "beta_residual": float(args.beta_residual),
        "novel_only": False if args.all_seen else bool(args.novel_only),
        "weighted_unique": not bool(args.no_weighted_unique),
        "eps": float(args.eps),
    }

    suite_root = output_root_from_arg(args.output_root, "support_subspace_prototype_suite")
    suite_started_at = now_iso()
    suite_start = time.time()
    suite_results = []

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
                params=params,
            )
            suite_results.append(result)
            print(
                f"[{now_iso()}] done framework={framework_name} dataset={result['dataset_key']} "
                f"baseline_full={result['baseline']['full']['full_acc']:.3f} "
                f"refined_full={result['refined']['full']['full_acc']:.3f}",
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
        "params": params,
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
                    f"  refined_full: {item['refined']['full']['full_acc']}",
                    f"  refined_gain: {item['gains']['refined_full_gain']}",
                    "",
                ]
            )
        (suite_root / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
