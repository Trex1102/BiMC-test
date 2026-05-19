import argparse
import contextlib
import csv
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
from models.query_branch_router import normalized_entropy, top_prob_margin
from prototype_refinement_conservative import conservative_refine_once
from prototype_refinement_conservative_dino_fusion import (
    collect_final_queries,
    compose_dino_fusion_probs,
    extract_final_state,
    knn_logits,
    mahalanobis_logits,
    normalize_rows,
)
from prototype_refinement_conservative_dino_fusion_sessionwise import (
    choose_alpha_by_pseudo_validation,
    pseudo_validation_split,
    refine_from_index_groups,
    summarize_pseudo_split,
)


DEFAULT_DATA_CFGS = [
    "configs/datasets/cifar100.yaml",
    "configs/datasets/miniimagenet.yaml",
    "configs/datasets/cub200_bimc_dino_fusion.yaml",
]
DEFAULT_TRAIN_CFG = "configs/trainers/bimc_dino_fusion.yaml"
DEFAULT_ALPHA_GRID = [0.5, 0.75, 1.0]
DEFAULT_FALLBACK_ALPHA = 0.75
DEFAULT_SUBSPACE_DIM = 3
DEFAULT_SUBSPACE_TEMP = 40.0
DEFAULT_TOPK_DIAGNOSTICS = 3


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pct(mask):
    return round(float(mask.float().mean().item() * 100.0), 3) if mask.numel() else 0.0


def accuracy_stats_from_probs(probs, targets, num_base_cls):
    preds = torch.argmax(probs, dim=1)
    correct = preds.eq(targets)
    base_mask = targets < int(num_base_cls)
    novel_mask = ~base_mask
    return {
        "full_acc": pct(correct),
        "base_acc": pct(correct[base_mask]) if base_mask.any() else 0.0,
        "novel_acc": pct(correct[novel_mask]) if novel_mask.any() else 0.0,
        "correct_count": int(correct.sum().item()),
    }


def safe_normalize_probs(probs):
    return probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-12)


def blend_probs(lhs, rhs, weight):
    return safe_normalize_probs((1.0 - float(weight)) * lhs + float(weight) * rhs)


def dataset_output_name(data_cfg):
    base = Path(data_cfg).stem
    if base == "cifar100":
        return "CIFAR100"
    if base == "cub200_bimc_dino_fusion":
        return "cub200"
    return base


def make_output_root(base_output_root):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(base_output_root) / f"tric_followup_suite_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_final_problem(data_cfg, train_cfg, seed):
    cfg = setup_cfg(data_cfg, train_cfg)
    cfg.defrost()
    cfg.SEED = int(seed)
    cfg.freeze()

    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    clip_query, dino_query, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)

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
    dino_support_features = normalize_rows(merged_state["dino_images_features"].to(device))
    dino_support_targets = merged_state["dino_images_targets"].to(device)

    text_logits = torch.matmul(clip_query, text_features.T)
    clip_visual_logits = torch.matmul(clip_query, clip_image_proto.T)
    dino_visual_logits = torch.matmul(dino_query, dino_image_proto.T)
    visual_logits = omega * dino_visual_logits + (1.0 - omega) * clip_visual_logits
    fused_logits = beta * text_logits + (1.0 - beta) * visual_logits
    cov_logits = mahalanobis_logits(dino_query, dino_image_proto, dino_cov)
    knn_branch_logits = knn_logits(clip_query, description_features, description_targets, num_cls)

    prob_semantic = F.softmax(text_logits * logit_temp, dim=-1)
    prob_clip_visual = F.softmax(clip_visual_logits * logit_temp, dim=-1)
    prob_dino_visual = F.softmax(dino_visual_logits * logit_temp, dim=-1)
    prob_clip_dino_visual = F.softmax(visual_logits * logit_temp, dim=-1)
    prob_tric_no_mask = F.softmax(fused_logits * logit_temp, dim=-1)
    prob_cov = F.softmax(cov_logits / cov_scale, dim=-1)
    prob_knn = F.softmax(knn_branch_logits, dim=-1)
    prob_tric = compose_dino_fusion_probs(
        clip_query,
        dino_query,
        semantic_proto,
        clip_image_proto,
        dino_image_proto,
        dino_cov,
        description_features,
        description_targets,
        num_base_cls,
        beta,
        ensemble_alpha,
        omega=omega,
        logit_temp=logit_temp,
        cov_scale=cov_scale,
    )

    return {
        "cfg": cfg,
        "model": model,
        "data_manager": data_manager,
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
        "text_features": text_features,
        "description_proto": description_proto,
        "clip_image_proto": clip_image_proto,
        "dino_image_proto": dino_image_proto,
        "dino_cov": dino_cov,
        "description_features": description_features,
        "description_targets": description_targets,
        "dino_support_features": dino_support_features,
        "dino_support_targets": dino_support_targets,
        "text_logits": text_logits,
        "clip_visual_logits": clip_visual_logits,
        "dino_visual_logits": dino_visual_logits,
        "visual_logits": visual_logits,
        "fused_logits": fused_logits,
        "cov_logits": cov_logits,
        "knn_branch_logits": knn_branch_logits,
        "base_probs": {
            "semantic_branch_only": prob_semantic,
            "clip_visual_only": prob_clip_visual,
            "dino_visual_only": prob_dino_visual,
            "clip_dino_visual_only": prob_clip_dino_visual,
            "tric_no_mask": prob_tric_no_mask,
            "tric": prob_tric,
            "covariance_branch": prob_cov,
            "description_knn_branch": prob_knn,
        },
    }


def update_visual_bundle(problem, dino_image_proto):
    visual_logits = problem["omega"] * torch.matmul(problem["dino_query"], dino_image_proto.T) + (
        1.0 - problem["omega"]
    ) * problem["clip_visual_logits"]
    fused_logits = problem["beta"] * problem["text_logits"] + (1.0 - problem["beta"]) * visual_logits
    probs_visual = F.softmax(visual_logits * problem["logit_temp"], dim=-1)
    probs_no_mask = F.softmax(fused_logits * problem["logit_temp"], dim=-1)
    probs_cov = F.softmax(mahalanobis_logits(problem["dino_query"], dino_image_proto, problem["dino_cov"]) / problem["cov_scale"], dim=-1)
    probs_full = compose_dino_fusion_probs(
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
    return {
        "dino_image_proto": dino_image_proto,
        "visual_probs": probs_visual,
        "tric_no_mask_probs": probs_no_mask,
        "cov_probs": probs_cov,
        "full_probs": probs_full,
    }


def compute_global_qpr(problem, alpha_grid, fallback_alpha, mass_thr, min_count, val_fraction, base_val_conf_thr, base_val_max_per_class):
    base_probs = problem["base_probs"]["tric"]
    components = {
        "num_cls": problem["num_cls"],
        "num_base_cls": problem["num_base_cls"],
        "beta": problem["beta"],
        "lambda_t": problem["lambda_t"],
        "ensemble_alpha": problem["ensemble_alpha"],
        "semantic_proto": problem["semantic_proto"],
        "clip_image_proto": problem["clip_image_proto"],
        "dino_image_proto": problem["dino_image_proto"],
        "dino_cov": problem["dino_cov"],
        "description_features": problem["description_features"],
        "description_targets": problem["description_targets"],
    }
    selected_alpha, alpha_selection = choose_alpha_by_pseudo_validation(
        clip_query=problem["clip_query"],
        dino_query=problem["dino_query"],
        components=components,
        baseline_probs=base_probs,
        alpha_grid=alpha_grid,
        mass_thr=mass_thr,
        min_count=min_count,
        val_fraction=val_fraction,
        base_val_conf_thr=base_val_conf_thr,
        base_val_max_per_class=base_val_max_per_class,
        fallback_alpha=fallback_alpha,
        topk_per_class=None,
    )
    refined_dino_proto, stats = conservative_refine_once(
        query_features=problem["dino_query"],
        query_targets=problem["query_targets"],
        baseline_probs=base_probs,
        orig_image_proto=problem["dino_image_proto"],
        num_base_cls=problem["num_base_cls"],
        num_cls=problem["num_cls"],
        alpha=selected_alpha,
        mass_thr=mass_thr,
        min_count=min_count,
        topk_per_class=None,
    )
    bundle = update_visual_bundle(problem, refined_dino_proto)
    bundle["selected_alpha"] = round(float(selected_alpha), 3)
    bundle["alpha_selection"] = alpha_selection
    bundle["qpr_stats"] = {
        "updated_class_count": int(stats["updated_class_count"]),
        "updated_class_rate": round(float(stats["updated_class_rate"]), 4),
        "gated_query_count": int(stats["gated_query_count"]),
        "gated_base_count": int(stats["gated_base_count"]),
        "gated_novel_count": int(stats["gated_novel_count"]),
        "mean_queries_per_updated_class": round(float(stats["mean_queries_per_updated_class"]), 3),
        "mean_selected_per_updated_class": round(float(stats["mean_selected_per_updated_class"]), 3),
    }
    return bundle


def compute_fixed_qpr(problem, alpha, mass_thr, min_count):
    base_probs = problem["base_probs"]["tric"]
    refined_dino_proto, _ = conservative_refine_once(
        query_features=problem["dino_query"],
        query_targets=problem["query_targets"],
        baseline_probs=base_probs,
        orig_image_proto=problem["dino_image_proto"],
        num_base_cls=problem["num_base_cls"],
        num_cls=problem["num_cls"],
        alpha=float(alpha),
        mass_thr=mass_thr,
        min_count=min_count,
        topk_per_class=None,
    )
    bundle = update_visual_bundle(problem, refined_dino_proto)
    bundle["selected_alpha"] = float(alpha)
    return bundle


def compute_classwise_qpr(problem, alpha_grid, fallback_alpha, mass_thr, min_count, val_fraction, base_val_conf_thr, base_val_max_per_class):
    base_probs = problem["base_probs"]["tric"]
    split = pseudo_validation_split(
        baseline_probs=base_probs,
        num_base_cls=problem["num_base_cls"],
        num_cls=problem["num_cls"],
        mass_thr=mass_thr,
        min_count=min_count,
        val_fraction=val_fraction,
        base_val_conf_thr=base_val_conf_thr,
        base_val_max_per_class=base_val_max_per_class,
        topk_per_class=None,
    )
    val_indices = split["validation_indices"]
    val_labels = split["validation_labels"]
    class_alpha = {}
    search_log = {}

    for cls_id in range(problem["num_base_cls"], problem["num_cls"]):
        cls_id = int(cls_id)
        update_indices = split["update_indices_by_class"].get(cls_id)
        if update_indices is None or update_indices.numel() == 0 or val_indices.numel() == 0:
            class_alpha[cls_id] = float(fallback_alpha)
            search_log[str(cls_id)] = {
                "selected_alpha": float(fallback_alpha),
                "fallback_used": True,
                "reason": "no update set or no validation samples",
                "candidates": [],
            }
            continue

        best = None
        records = []
        for alpha in alpha_grid:
            candidate_proto = refine_from_index_groups(
                query_features=problem["dino_query"],
                baseline_probs=base_probs,
                orig_image_proto=problem["dino_image_proto"],
                update_indices_by_class={cls_id: update_indices},
                alpha=float(alpha),
            )
            candidate_probs = update_visual_bundle(problem, candidate_proto)["full_probs"]
            selected = candidate_probs[val_indices, val_labels]
            pseudo_nll = -torch.log(torch.clamp(selected, min=1e-12)).mean()
            score = -float(pseudo_nll.item())
            rec = {
                "alpha": float(alpha),
                "pseudo_val_score": round(score, 6),
                "pseudo_val_nll": round(float(pseudo_nll.item()), 6),
            }
            records.append(rec)
            if best is None or score > best["score"] + 1e-12 or (abs(score - best["score"]) <= 1e-12 and float(alpha) < best["alpha"]):
                best = {"alpha": float(alpha), "score": score}

        class_alpha[cls_id] = float(best["alpha"])
        search_log[str(cls_id)] = {
            "selected_alpha": float(best["alpha"]),
            "fallback_used": False,
            "candidates": records,
        }

    refined_proto = problem["dino_image_proto"].clone()
    for cls_id, alpha in class_alpha.items():
        update_indices = split["update_indices_by_class"].get(int(cls_id))
        if update_indices is None or update_indices.numel() == 0:
            continue
        refined_proto = refine_from_index_groups(
            query_features=problem["dino_query"],
            baseline_probs=base_probs,
            orig_image_proto=refined_proto,
            update_indices_by_class={int(cls_id): update_indices},
            alpha=float(alpha),
        )

    bundle = update_visual_bundle(problem, refined_proto)
    alpha_values = list(class_alpha.values())
    bundle["class_alpha"] = {str(k): round(float(v), 3) for k, v in class_alpha.items()}
    bundle["alpha_summary"] = {
        "min_alpha": round(float(min(alpha_values)), 3) if alpha_values else None,
        "max_alpha": round(float(max(alpha_values)), 3) if alpha_values else None,
        "mean_alpha": round(float(sum(alpha_values) / len(alpha_values)), 3) if alpha_values else None,
    }
    bundle["alpha_selection"] = {
        "mode": "classwise_pseudo_validation",
        "split": summarize_pseudo_split(split),
        "search_log": search_log,
    }
    return bundle


def build_subspace_branch(problem, dino_proto_for_centers, subspace_dim, subspace_temp):
    num_cls = problem["num_cls"]
    features = problem["dino_support_features"]
    labels = problem["dino_support_targets"]
    queries = problem["dino_query"]
    centers = dino_proto_for_centers

    logits = []
    for cls_id in range(num_cls):
        cls_mask = labels == int(cls_id)
        cls_features = features[cls_mask]
        center = centers[int(cls_id)]
        centered = cls_features - center.unsqueeze(0)
        if cls_features.shape[0] <= 1:
            basis = None
        else:
            rank = max(0, min(int(subspace_dim), int(cls_features.shape[0] - 1), int(cls_features.shape[1])))
            if rank <= 0:
                basis = None
            else:
                _, _, vh = torch.linalg.svd(centered.float(), full_matrices=False)
                basis = vh[:rank].to(queries.dtype)
        q_centered = queries - center.unsqueeze(0)
        if basis is not None and basis.numel() > 0:
            coeff = torch.matmul(q_centered, basis.T)
            recon = torch.matmul(coeff, basis)
            residual = q_centered - recon
        else:
            residual = q_centered
        residual_sq = torch.sum(residual * residual, dim=1)
        center_sq = torch.sum(q_centered * q_centered, dim=1)
        score = -(residual_sq + 0.25 * center_sq)
        logits.append(score)
    logits = torch.stack(logits, dim=1)
    probs = F.softmax(logits * float(subspace_temp), dim=-1)
    return logits, probs


def selective_semantic_repair(default_probs, semantic_probs, knn_probs):
    default_top1, default_margin = top_prob_margin(default_probs)
    sem_top1, sem_margin = top_prob_margin(semantic_probs)
    knn_top1, knn_margin = top_prob_margin(knn_probs)
    default_entropy = normalized_entropy(default_probs)
    sem_entropy = normalized_entropy(semantic_probs)
    knn_entropy = normalized_entropy(knn_probs)
    default_pred = torch.argmax(default_probs, dim=1)
    sem_pred = torch.argmax(semantic_probs, dim=1)
    knn_pred = torch.argmax(knn_probs, dim=1)

    semantic_agree = sem_pred.eq(knn_pred)
    low_default_conf = (default_margin < 0.04) | (default_entropy > 0.58)
    strong_semantic = (sem_margin > 0.08) & (knn_margin > 0.08) & (sem_entropy < 0.52) & (knn_entropy < 0.55)
    disagree_with_default = ~sem_pred.eq(default_pred)
    repair_mask = low_default_conf & strong_semantic & semantic_agree & disagree_with_default

    semantic_mix = safe_normalize_probs(0.5 * semantic_probs + 0.5 * knn_probs)
    repaired = default_probs.clone()
    if repair_mask.any():
        weight = 0.35
        repaired[repair_mask] = blend_probs(default_probs[repair_mask], semantic_mix[repair_mask], weight)
    return repaired, {
        "repair_query_count": int(repair_mask.sum().item()),
        "repair_query_rate": round(float(repair_mask.float().mean().item()), 4),
    }


def selective_subspace_repair(default_probs, visual_probs, cov_probs, subspace_probs):
    default_top1, default_margin = top_prob_margin(default_probs)
    visual_top1, visual_margin = top_prob_margin(visual_probs)
    cov_top1, cov_margin = top_prob_margin(cov_probs)
    sub_top1, sub_margin = top_prob_margin(subspace_probs)
    default_entropy = normalized_entropy(default_probs)
    visual_entropy = normalized_entropy(visual_probs)
    sub_entropy = normalized_entropy(subspace_probs)

    default_pred = torch.argmax(default_probs, dim=1)
    visual_pred = torch.argmax(visual_probs, dim=1)
    cov_pred = torch.argmax(cov_probs, dim=1)
    sub_pred = torch.argmax(subspace_probs, dim=1)

    agree_visual_sub = visual_pred.eq(sub_pred)
    agree_sub_cov = sub_pred.eq(cov_pred)
    low_default_conf = (default_margin < 0.05) | (default_entropy > 0.55)
    strong_subspace = (sub_margin > 0.06) & (sub_entropy < 0.5)
    strong_visual = (visual_margin > 0.08) & (visual_entropy < 0.5)
    rescue_mask = low_default_conf & strong_subspace & strong_visual & (agree_visual_sub | agree_sub_cov) & (~sub_pred.eq(default_pred))

    subspace_mix = safe_normalize_probs(0.65 * visual_probs + 0.35 * subspace_probs)
    repaired = default_probs.clone()
    if rescue_mask.any():
        repaired[rescue_mask] = blend_probs(default_probs[rescue_mask], subspace_mix[rescue_mask], 0.4)
    return repaired, {
        "repair_query_count": int(rescue_mask.sum().item()),
        "repair_query_rate": round(float(rescue_mask.float().mean().item()), 4),
    }


def conservative_router(default_probs, cov_probs, knn_probs, visual_probs, text_probs, subspace_probs, num_base_cls):
    default_top1, default_margin = top_prob_margin(default_probs)
    cov_top1, cov_margin = top_prob_margin(cov_probs)
    knn_top1, knn_margin = top_prob_margin(knn_probs)
    visual_top1, visual_margin = top_prob_margin(visual_probs)
    text_top1, text_margin = top_prob_margin(text_probs)
    sub_top1, sub_margin = top_prob_margin(subspace_probs)

    default_entropy = normalized_entropy(default_probs)
    cov_entropy = normalized_entropy(cov_probs)
    knn_entropy = normalized_entropy(knn_probs)
    visual_entropy = normalized_entropy(visual_probs)
    text_entropy = normalized_entropy(text_probs)
    sub_entropy = normalized_entropy(subspace_probs)

    default_pred = torch.argmax(default_probs, dim=1)
    cov_pred = torch.argmax(cov_probs, dim=1)
    knn_pred = torch.argmax(knn_probs, dim=1)
    visual_pred = torch.argmax(visual_probs, dim=1)
    text_pred = torch.argmax(text_probs, dim=1)
    sub_pred = torch.argmax(subspace_probs, dim=1)

    route = torch.full((default_probs.shape[0],), -1, device=default_probs.device, dtype=torch.long)
    out = default_probs.clone()

    base_mask = default_pred < int(num_base_cls)
    novel_mask = ~base_mask

    base_cov_mask = (
        base_mask
        & cov_pred.eq(default_pred)
        & (cov_margin > default_margin + 0.03)
        & (cov_entropy < default_entropy - 0.02)
    )
    out[base_cov_mask] = cov_probs[base_cov_mask]
    route[base_cov_mask] = 1

    visual_sub_mask = (
        novel_mask
        & visual_pred.eq(sub_pred)
        & (visual_margin > default_margin + 0.03)
        & (sub_margin > 0.05)
        & (visual_entropy < default_entropy)
    )
    if visual_sub_mask.any():
        out[visual_sub_mask] = safe_normalize_probs(
            0.7 * visual_probs[visual_sub_mask] + 0.3 * subspace_probs[visual_sub_mask]
        )
        route[visual_sub_mask] = 2

    semantic_mask = (
        (route < 0)
        & (default_margin < 0.04)
        & text_pred.eq(knn_pred)
        & (~text_pred.eq(default_pred))
        & (text_margin > 0.08)
        & (knn_margin > 0.08)
        & (text_entropy < 0.52)
        & (knn_entropy < 0.55)
    )
    if semantic_mask.any():
        out[semantic_mask] = safe_normalize_probs(0.5 * text_probs[semantic_mask] + 0.5 * knn_probs[semantic_mask])
        route[semantic_mask] = 3

    visual_conf_mask = (
        (route < 0)
        & visual_pred.eq(default_pred)
        & (visual_margin > default_margin + 0.04)
        & (visual_entropy < default_entropy - 0.02)
    )
    out[visual_conf_mask] = visual_probs[visual_conf_mask]
    route[visual_conf_mask] = 4

    out[route < 0] = default_probs[route < 0]
    route[route < 0] = 0

    counts = torch.bincount(route.cpu(), minlength=5).tolist()
    return out, {
        "route_counts": {
            "default": int(counts[0]),
            "covariance": int(counts[1]),
            "visual_subspace": int(counts[2]),
            "semantic_repair": int(counts[3]),
            "visual_override": int(counts[4]),
        }
    }


def topk_predictions(probs, k):
    values, indices = torch.topk(probs, k=min(int(k), probs.shape[1]), dim=1)
    return indices.detach().cpu().tolist(), values.detach().cpu().tolist()


def query_diagnostics(problem, variant_probs, extra_metrics, out_path, topk):
    targets = problem["query_targets"]
    num_base_cls = problem["num_base_cls"]
    rows = []

    top_idx, top_val = {}, {}
    preds, margins, entropies = {}, {}, {}
    for name, probs in variant_probs.items():
        pred = torch.argmax(probs, dim=1)
        preds[name] = pred
        _, margin = top_prob_margin(probs)
        margins[name] = margin
        entropies[name] = normalized_entropy(probs)
        idx, val = topk_predictions(probs, topk)
        top_idx[name] = idx
        top_val[name] = val

    target_list = targets.detach().cpu().tolist()
    for i, target in enumerate(target_list):
        row = {
            "query_index": i,
            "target": target,
            "split": "base" if target < num_base_cls else "novel",
        }
        for name in variant_probs:
            row[f"{name}_pred"] = int(preds[name][i].item())
            row[f"{name}_correct"] = int(bool(preds[name][i].eq(targets[i]).item()))
            row[f"{name}_margin"] = round(float(margins[name][i].item()), 6)
            row[f"{name}_entropy"] = round(float(entropies[name][i].item()), 6)
            row[f"{name}_topk_idx"] = json.dumps(top_idx[name][i])
            row[f"{name}_topk_prob"] = json.dumps([round(float(x), 6) for x in top_val[name][i]])
        row["default_base_mass"] = round(float(variant_probs["tric_pv"][i, :num_base_cls].sum().item()), 6)
        row["subspace_margin"] = round(float(margins["dino_subspace_only"][i].item()), 6)
        row["subspace_entropy"] = round(float(entropies["dino_subspace_only"][i].item()), 6)
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["query_index", "target", "split"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_variants(problem, alpha_grid, fallback_alpha, mass_thr, min_count, val_fraction, base_val_conf_thr, base_val_max_per_class, subspace_dim, subspace_temp):
    base = problem["base_probs"]

    fixed_qpr = compute_fixed_qpr(problem, alpha=0.75, mass_thr=mass_thr, min_count=min_count)
    tric_pv = compute_global_qpr(
        problem,
        alpha_grid=alpha_grid,
        fallback_alpha=fallback_alpha,
        mass_thr=mass_thr,
        min_count=min_count,
        val_fraction=val_fraction,
        base_val_conf_thr=base_val_conf_thr,
        base_val_max_per_class=base_val_max_per_class,
    )
    classwise_qpr = compute_classwise_qpr(
        problem,
        alpha_grid=alpha_grid,
        fallback_alpha=fallback_alpha,
        mass_thr=mass_thr,
        min_count=min_count,
        val_fraction=val_fraction,
        base_val_conf_thr=base_val_conf_thr,
        base_val_max_per_class=base_val_max_per_class,
    )

    _, subspace_probs = build_subspace_branch(problem, problem["dino_image_proto"], subspace_dim=subspace_dim, subspace_temp=subspace_temp)
    _, subspace_qpr_probs = build_subspace_branch(problem, tric_pv["dino_image_proto"], subspace_dim=subspace_dim, subspace_temp=subspace_temp)
    _, subspace_classwise_probs = build_subspace_branch(problem, classwise_qpr["dino_image_proto"], subspace_dim=subspace_dim, subspace_temp=subspace_temp)

    semantic_repair_probs, semantic_repair_meta = selective_semantic_repair(
        default_probs=tric_pv["full_probs"],
        semantic_probs=base["semantic_branch_only"],
        knn_probs=base["description_knn_branch"],
    )
    subspace_repair_probs, subspace_repair_meta = selective_subspace_repair(
        default_probs=tric_pv["full_probs"],
        visual_probs=tric_pv["visual_probs"],
        cov_probs=tric_pv["cov_probs"],
        subspace_probs=subspace_qpr_probs,
    )
    safe_router_probs, safe_router_meta = conservative_router(
        default_probs=tric_pv["full_probs"],
        cov_probs=tric_pv["cov_probs"],
        knn_probs=base["description_knn_branch"],
        visual_probs=tric_pv["visual_probs"],
        text_probs=base["semantic_branch_only"],
        subspace_probs=subspace_qpr_probs,
        num_base_cls=problem["num_base_cls"],
    )
    classwise_router_probs, classwise_router_meta = conservative_router(
        default_probs=classwise_qpr["full_probs"],
        cov_probs=classwise_qpr["cov_probs"],
        knn_probs=base["description_knn_branch"],
        visual_probs=classwise_qpr["visual_probs"],
        text_probs=base["semantic_branch_only"],
        subspace_probs=subspace_classwise_probs,
        num_base_cls=problem["num_base_cls"],
    )

    variant_probs = {
        "semantic_branch_only": base["semantic_branch_only"],
        "clip_visual_only": base["clip_visual_only"],
        "dino_visual_only": base["dino_visual_only"],
        "clip_dino_visual_only": base["clip_dino_visual_only"],
        "tric_no_mask": base["tric_no_mask"],
        "tric": base["tric"],
        "tric_fixed_alpha_075": fixed_qpr["full_probs"],
        "tric_pv": tric_pv["full_probs"],
        "tric_classwise_qpr": classwise_qpr["full_probs"],
        "dino_subspace_only": subspace_probs,
        "tric_semantic_repair": semantic_repair_probs,
        "tric_subspace_repair": subspace_repair_probs,
        "tric_safe_router": safe_router_probs,
        "tric_classwise_router": classwise_router_probs,
    }

    metadata = {
        "tric_fixed_alpha_075": {"selected_alpha": fixed_qpr["selected_alpha"]},
        "tric_pv": {
            "selected_alpha": tric_pv["selected_alpha"],
            "alpha_selection": tric_pv["alpha_selection"],
            "qpr_stats": tric_pv["qpr_stats"],
        },
        "tric_classwise_qpr": {
            "alpha_summary": classwise_qpr["alpha_summary"],
            "alpha_selection": classwise_qpr["alpha_selection"],
        },
        "tric_semantic_repair": semantic_repair_meta,
        "tric_subspace_repair": subspace_repair_meta,
        "tric_safe_router": safe_router_meta,
        "tric_classwise_router": classwise_router_meta,
    }
    return variant_probs, metadata


def summarize_variant_set(variant_probs, targets, num_base_cls, reference_name):
    summary = {}
    ref_acc = accuracy_stats_from_probs(variant_probs[reference_name], targets, num_base_cls)["full_acc"]
    for name, probs in variant_probs.items():
        stats = accuracy_stats_from_probs(probs, targets, num_base_cls)
        stats["gain_vs_reference"] = round(float(stats["full_acc"] - ref_acc), 3)
        summary[name] = stats
    return summary


def write_summary(out_dir, dataset_payload):
    lines = [
        f"dataset: {dataset_payload['dataset']}",
        f"data_cfg: {dataset_payload['data_cfg']}",
        f"train_cfg: {dataset_payload['train_cfg']}",
        f"seed: {dataset_payload['seed']}",
        f"started_at: {dataset_payload['started_at']}",
        f"completed_at: {dataset_payload['completed_at']}",
        f"runtime_sec: {dataset_payload['runtime_sec']}",
        f"num_queries: {dataset_payload['num_queries']}",
    ]
    for name, stats in dataset_payload["variant_summary"].items():
        lines.append(
            f"{name}: full={stats['full_acc']} base={stats['base_acc']} novel={stats['novel_acc']} gain_vs_tric={stats['gain_vs_reference']}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(data_cfg, train_cfg, seed, output_root, alpha_grid, fallback_alpha, mass_thr, min_count, val_fraction, base_val_conf_thr, base_val_max_per_class, subspace_dim, subspace_temp, diagnostics_topk):
    dataset_label = dataset_output_name(data_cfg)
    out_dir = output_root / dataset_label / f"seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    driver_log = out_dir / "driver.log"

    started_at = now_iso()
    start_time = time.time()
    with driver_log.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            problem = build_final_problem(data_cfg, train_cfg, seed)
    variant_probs, variant_meta = evaluate_variants(
        problem,
        alpha_grid=alpha_grid,
        fallback_alpha=fallback_alpha,
        mass_thr=mass_thr,
        min_count=min_count,
        val_fraction=val_fraction,
        base_val_conf_thr=base_val_conf_thr,
        base_val_max_per_class=base_val_max_per_class,
        subspace_dim=subspace_dim,
        subspace_temp=subspace_temp,
    )
    variant_summary = summarize_variant_set(
        variant_probs,
        problem["query_targets"],
        problem["num_base_cls"],
        reference_name="tric",
    )
    completed_at = now_iso()
    runtime_sec = round(time.time() - start_time, 3)

    query_diagnostics(
        problem,
        variant_probs,
        variant_meta,
        out_path=out_dir / "query_diagnostics.csv",
        topk=diagnostics_topk,
    )

    payload = {
        "analysis": "tric_followup_variant_suite_final_session",
        "dataset": problem["dataset_name"],
        "data_cfg": data_cfg,
        "train_cfg": train_cfg,
        "seed": int(seed),
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime_sec": runtime_sec,
        "hostname": socket.gethostname(),
        "num_queries": int(problem["query_targets"].numel()),
        "num_seen_classes": int(problem["num_cls"]),
        "num_base_classes": int(problem["num_base_cls"]),
        "beta": problem["beta"],
        "lambda_t": problem["lambda_t"],
        "ensemble_alpha": problem["ensemble_alpha"],
        "omega": problem["omega"],
        "variant_summary": variant_summary,
        "variant_metadata": variant_meta,
        "files": {
            "query_diagnostics_csv": str(out_dir / "query_diagnostics.csv"),
            "results_json": str(out_dir / "results.json"),
            "summary_txt": str(out_dir / "summary.txt"),
            "driver_log": str(driver_log),
        },
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_summary(out_dir, payload)
    (out_dir / "completion.txt").write_text(
        f"started_at={started_at}\ncompleted_at={completed_at}\nruntime_sec={runtime_sec}\nexit_status=0\n",
        encoding="utf-8",
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Final-session TriC follow-up variant suite.")
    parser.add_argument("--data_cfgs", nargs="+", default=DEFAULT_DATA_CFGS)
    parser.add_argument("--train_cfg", default=DEFAULT_TRAIN_CFG)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_root", default="experiments")
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--fallback_alpha", type=float, default=DEFAULT_FALLBACK_ALPHA)
    parser.add_argument("--mass_thr", type=float, default=0.3)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--base_val_conf_thr", type=float, default=0.5)
    parser.add_argument("--base_val_max_per_class", type=int, default=50)
    parser.add_argument("--subspace_dim", type=int, default=DEFAULT_SUBSPACE_DIM)
    parser.add_argument("--subspace_temp", type=float, default=DEFAULT_SUBSPACE_TEMP)
    parser.add_argument("--diagnostics_topk", type=int, default=DEFAULT_TOPK_DIAGNOSTICS)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = make_output_root(args.output_root)
    suite_started_at = now_iso()
    suite_start = time.time()

    runs = []
    for data_cfg in args.data_cfgs:
        payload = run_dataset(
            data_cfg=data_cfg,
            train_cfg=args.train_cfg,
            seed=args.seed,
            output_root=output_root,
            alpha_grid=args.alpha_grid,
            fallback_alpha=args.fallback_alpha,
            mass_thr=args.mass_thr,
            min_count=args.min_count,
            val_fraction=args.val_fraction,
            base_val_conf_thr=args.base_val_conf_thr,
            base_val_max_per_class=args.base_val_max_per_class,
            subspace_dim=args.subspace_dim,
            subspace_temp=args.subspace_temp,
            diagnostics_topk=args.diagnostics_topk,
        )
        runs.append(
            {
                "dataset": payload["dataset"],
                "completed_at": payload["completed_at"],
                "runtime_sec": payload["runtime_sec"],
                "variant_summary": payload["variant_summary"],
                "files": payload["files"],
            }
        )

    suite_payload = {
        "analysis": "tric_followup_variant_suite_final_session",
        "started_at": suite_started_at,
        "completed_at": now_iso(),
        "runtime_sec": round(time.time() - suite_start, 3),
        "seed": int(args.seed),
        "train_cfg": args.train_cfg,
        "data_cfgs": args.data_cfgs,
        "output_root": str(output_root),
        "runs": runs,
    }
    with (output_root / "suite_summary.json").open("w", encoding="utf-8") as f:
        json.dump(suite_payload, f, indent=2)
    print(json.dumps(suite_payload, indent=2))


if __name__ == "__main__":
    main()
