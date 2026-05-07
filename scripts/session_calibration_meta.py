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
    collect_final_queries,
    compute_cov_logits,
    compute_knn_logits,
    extract_final_state,
    normalize_vector,
    to_builtin,
)


PARAM_RANGES = {
    "beta": (0.05, 0.95),
    "alpha_base": (0.0, 1.0),
    "alpha_novel": (0.0, 1.0),
    "delta": (-1.5, 1.5),
    "tau": (0.5, 3.0),
}

MODE_PARAM_KEYS = {
    "bmvc": ["alpha_base", "alpha_novel", "delta", "tau"],
    "astar": ["beta", "alpha_base", "alpha_novel", "delta", "tau"],
}


def create_output_dir(cfg, train_cfg, mode):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"calibration_meta_{mode}_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def accuracy_from_probs(probs, targets):
    return float((probs.argmax(dim=1) == targets).float().mean().item() * 100.0)


def overall_acc_tensor(probs, targets):
    return (probs.argmax(dim=1) == targets).float().mean()


def per_class_mean_nll(probs, targets, class_ids):
    idx = torch.arange(targets.size(0), device=targets.device)
    target_probs = torch.clamp(probs[idx, targets], min=1e-12)
    nll = -torch.log(target_probs)
    losses = []
    for cls_id in class_ids:
        mask = targets == cls_id
        if mask.any():
            losses.append(nll[mask].mean())
    return torch.stack(losses)


def class_balanced_nll(probs, targets, class_ids):
    return per_class_mean_nll(probs, targets, class_ids).mean()


def mean_std(values):
    if values.numel() == 0:
        zero = values.new_tensor(0.0)
        return zero, zero
    if values.numel() == 1:
        return values.mean(), values.new_tensor(0.0)
    return values.mean(), values.std(unbiased=False)


def gini_coefficient(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    values = np.clip(values, a_min=0.0, a_max=None)
    total = values.sum()
    if total <= 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_vals) / (n * total)) - (n + 1.0) / n)


def regularize_covariance(features, eps=1e-4):
    if features.shape[0] <= 1:
        dim = features.shape[1]
        return torch.eye(dim, device=features.device, dtype=features.dtype)
    cov = torch.cov(features.T)
    dim = cov.shape[0]
    eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)
    diag_mean = torch.diagonal(cov).mean()
    return cov + eps * max(float(diag_mean.item()), 1.0) * eye


def mean_pairwise_similarity(features):
    if features.shape[0] <= 1:
        return features.new_tensor(1.0)
    sim = torch.matmul(features, features.T)
    denom = float(features.shape[0] * (features.shape[0] - 1))
    return (sim.sum() - torch.trace(sim)) / max(denom, 1.0)


def split_class_features(class_feats, support_count, query_count, generator, keep_min_proto=1):
    num_items = class_feats.shape[0]
    support_count = min(int(support_count), num_items)
    query_count = min(int(query_count), max(num_items - keep_min_proto, 0))
    if support_count <= 0:
        support_count = min(1, num_items)
    order = torch.randperm(num_items, generator=generator).tolist()

    support_idx = order[:support_count]
    remaining = order[support_count:]
    if not remaining:
        remaining = support_idx[-1:]
        support_idx = support_idx[:-1] or support_idx

    query_idx = remaining[:query_count]
    if not query_idx:
        query_idx = remaining[:1]
        remaining = remaining[1:]
    else:
        remaining = remaining[query_count:]

    proto_idx = support_idx if support_idx else remaining
    if not proto_idx:
        proto_idx = query_idx

    return {
        "support": class_feats[support_idx],
        "query": class_feats[query_idx],
        "proto_source": class_feats[proto_idx],
    }


def compose_semantic_proto(text_features, description_proto, lambda_t):
    return F.normalize((1.0 - lambda_t) * text_features + lambda_t * description_proto, dim=-1)


def build_local_problem(
    merged_state,
    global_base_ids,
    global_novel_ids,
    pseudo_novel_shot,
    base_query_per_class,
    novel_query_per_class,
    lambda_t,
    beta,
    ensemble_alpha,
    seed,
    device,
):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    class_ids = list(global_base_ids) + list(global_novel_ids)
    num_base_cls = len(global_base_ids)
    num_cls = len(class_ids)

    text_features = F.normalize(
        merged_state["text_features"][class_ids].to(device=device, dtype=torch.float32),
        dim=-1,
    )
    description_proto = F.normalize(
        merged_state["description_proto"][class_ids].to(device=device, dtype=torch.float32),
        dim=-1,
    )

    desc_groups = []
    desc_targets = []
    image_proto_list = []
    cov_sources = []
    query_features = []
    query_targets = []
    support_by_class = {}
    support_counts = []

    for local_cls, global_cls in enumerate(class_ids):
        desc_mask = merged_state["description_targets"] == int(global_cls)
        class_desc = F.normalize(
            merged_state["description_features"][desc_mask].to(device=device, dtype=torch.float32),
            dim=-1,
        )
        desc_groups.append(class_desc)
        desc_targets.append(
            torch.full((class_desc.shape[0],), local_cls, dtype=torch.long, device=device)
        )

        class_images = F.normalize(
            merged_state["images_features"][merged_state["images_targets"] == int(global_cls)].to(
                device=device,
                dtype=torch.float32,
            ),
            dim=-1,
        )
        if local_cls < num_base_cls:
            split = split_class_features(
                class_images,
                support_count=max(class_images.shape[0] - base_query_per_class, 1),
                query_count=base_query_per_class,
                generator=generator,
                keep_min_proto=2,
            )
        else:
            split = split_class_features(
                class_images,
                support_count=pseudo_novel_shot,
                query_count=novel_query_per_class,
                generator=generator,
                keep_min_proto=1,
            )

        support = F.normalize(split["support"], dim=-1)
        query = F.normalize(split["query"], dim=-1)
        proto_source = F.normalize(split["proto_source"], dim=-1)
        support_by_class[local_cls] = support
        support_counts.append(float(support.shape[0]))
        image_proto_list.append(normalize_vector(proto_source.mean(dim=0)))
        cov_sources.append(proto_source)
        if query.numel() > 0:
            query_features.append(query)
            query_targets.append(
                torch.full((query.shape[0],), local_cls, dtype=torch.long, device=device)
            )

    description_features = torch.cat(desc_groups, dim=0)
    description_targets = torch.cat(desc_targets, dim=0)
    image_proto = F.normalize(torch.stack(image_proto_list, dim=0), dim=-1)
    cov_image = regularize_covariance(torch.cat(cov_sources, dim=0))
    query_features = torch.cat(query_features, dim=0)
    query_targets = torch.cat(query_targets, dim=0)
    semantic_proto = compose_semantic_proto(text_features, description_proto, lambda_t)
    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )

    return {
        "num_cls": num_cls,
        "num_base_cls": num_base_cls,
        "class_ids": class_ids,
        "text_features": text_features,
        "description_proto": description_proto,
        "description_features": description_features,
        "description_targets": description_targets,
        "image_proto": image_proto,
        "cov_image": cov_image,
        "query_features": query_features,
        "query_targets": query_targets,
        "support_by_class": support_by_class,
        "support_counts": support_counts,
        "semantic_proto": semantic_proto,
        "prob_cov": prob_cov,
        "prob_knn": prob_knn,
        "beta": float(beta),
        "ensemble_alpha": float(ensemble_alpha),
    }


def sample_pseudo_episodes(
    merged_state,
    num_base_cls,
    pseudo_episodes,
    pseudo_base_count,
    pseudo_novel_count,
    pseudo_novel_shot,
    base_query_per_class,
    novel_query_per_class,
    lambda_t,
    beta,
    ensemble_alpha,
    seed,
    device,
):
    episodes = []
    max_available_base = max(num_base_cls - pseudo_novel_count, 1)
    pseudo_base_count = min(max(int(pseudo_base_count), 1), max_available_base)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    for episode_idx in range(int(pseudo_episodes)):
        perm = torch.randperm(num_base_cls, generator=generator).tolist()
        global_base_ids = perm[:pseudo_base_count]
        global_novel_ids = perm[pseudo_base_count:pseudo_base_count + pseudo_novel_count]
        if len(global_novel_ids) < pseudo_novel_count:
            continue
        problem = build_local_problem(
            merged_state=merged_state,
            global_base_ids=global_base_ids,
            global_novel_ids=global_novel_ids,
            pseudo_novel_shot=pseudo_novel_shot,
            base_query_per_class=base_query_per_class,
            novel_query_per_class=novel_query_per_class,
            lambda_t=lambda_t,
            beta=beta,
            ensemble_alpha=ensemble_alpha,
            seed=int(seed) + 997 * (episode_idx + 1),
            device=device,
        )
        episodes.append(problem)
    return episodes


def build_final_problem(
    merged_state,
    query_features,
    query_targets,
    num_cls,
    num_base_cls,
    lambda_t,
    beta,
    ensemble_alpha,
    device,
):
    description_mask = merged_state["description_targets"] < int(num_cls)
    description_features = F.normalize(
        merged_state["description_features"][description_mask].to(device=device, dtype=torch.float32),
        dim=-1,
    )
    description_targets = merged_state["description_targets"][description_mask].to(device)
    image_proto = F.normalize(
        merged_state["image_proto"][:num_cls].to(device=device, dtype=torch.float32),
        dim=-1,
    )
    text_features = F.normalize(
        merged_state["text_features"][:num_cls].to(device=device, dtype=torch.float32),
        dim=-1,
    )
    description_proto = F.normalize(
        merged_state["description_proto"][:num_cls].to(device=device, dtype=torch.float32),
        dim=-1,
    )
    semantic_proto = compose_semantic_proto(text_features, description_proto, lambda_t)
    cov_image = merged_state["cov_image"].to(device=device, dtype=torch.float32)
    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(
        compute_knn_logits(query_features, description_features, description_targets, num_cls),
        dim=-1,
    )

    support_by_class = {}
    support_counts = []
    for cls_id in range(num_cls):
        class_support = F.normalize(
            merged_state["images_features"][merged_state["images_targets"] == int(cls_id)].to(
                device=device,
                dtype=torch.float32,
            ),
            dim=-1,
        )
        support_by_class[cls_id] = class_support
        support_counts.append(float(class_support.shape[0]))

    return {
        "num_cls": num_cls,
        "num_base_cls": num_base_cls,
        "text_features": text_features,
        "description_proto": description_proto,
        "description_features": description_features,
        "description_targets": description_targets,
        "image_proto": image_proto,
        "cov_image": cov_image,
        "query_features": query_features,
        "query_targets": query_targets,
        "support_by_class": support_by_class,
        "support_counts": support_counts,
        "semantic_proto": semantic_proto,
        "prob_cov": prob_cov,
        "prob_knn": prob_knn,
        "beta": float(beta),
        "ensemble_alpha": float(ensemble_alpha),
    }


def compose_probs_from_params(problem, beta, alpha_base, alpha_novel, delta, tau):
    semantic_proto = problem["semantic_proto"]
    image_proto = problem["image_proto"]
    query_features = problem["query_features"]
    num_base_cls = int(problem["num_base_cls"])
    num_cls = int(problem["num_cls"])

    fused_proto = F.normalize(beta * semantic_proto + (1.0 - beta) * image_proto, dim=-1)
    logits_fused = torch.matmul(query_features, fused_proto.T)
    prob_fused = F.softmax(logits_fused, dim=-1)

    base_probs = (
        alpha_base * prob_fused[:, :num_base_cls]
        + (1.0 - alpha_base) * problem["prob_cov"][:, :num_base_cls]
    )
    novel_probs = (
        alpha_novel * prob_fused[:, num_base_cls:]
        + (1.0 - alpha_novel) * problem["prob_knn"][:, num_base_cls:]
    )
    combined_probs = torch.cat([base_probs, novel_probs], dim=1)

    logits = torch.log(torch.clamp(combined_probs, min=1e-12)) / max(float(tau), 1e-12)
    if num_base_cls < num_cls and abs(float(delta)) > 0.0:
        logits = logits.clone()
        logits[:, num_base_cls:] = logits[:, num_base_cls:] + float(delta)
    probs = F.softmax(logits, dim=-1)
    return probs, {
        "prob_fused": prob_fused,
        "combined_probs": combined_probs,
        "fused_proto": fused_proto,
    }


def evaluate_probs(probs, targets, class_ids):
    return {
        "acc": overall_acc_tensor(probs, targets),
        "acc_pct": round(float(overall_acc_tensor(probs, targets).item() * 100.0), 3),
        "nll": class_balanced_nll(probs, targets, class_ids),
    }


def build_episode_features(problem):
    num_cls = int(problem["num_cls"])
    num_base_cls = int(problem["num_base_cls"])
    num_novel_cls = num_cls - num_base_cls

    baseline_probs, aux = compose_probs_from_params(
        problem,
        beta=float(problem["beta"]),
        alpha_base=float(problem["ensemble_alpha"]),
        alpha_novel=float(problem["ensemble_alpha"]),
        delta=0.0,
        tau=1.0,
    )
    prob_fused = aux["prob_fused"]
    ent = -(baseline_probs * torch.log(torch.clamp(baseline_probs, min=1e-12))).sum(dim=1)
    max_prob = baseline_probs.max(dim=1).values
    top2 = torch.topk(baseline_probs, k=min(2, baseline_probs.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        margin = top2[:, 0]
    else:
        margin = top2[:, 0] - top2[:, 1]
    base_mass = baseline_probs[:, :num_base_cls].sum(dim=1)
    novel_mass = baseline_probs[:, num_base_cls:].sum(dim=1)
    pred_labels = baseline_probs.argmax(dim=1)
    pred_counts = torch.bincount(pred_labels, minlength=num_cls).float()
    pred_distribution = pred_counts / max(float(pred_counts.sum().item()), 1.0)
    marginal_entropy = -torch.sum(pred_distribution * torch.log(torch.clamp(pred_distribution, min=1e-12)))

    image_sem_align = torch.sum(problem["image_proto"] * problem["semantic_proto"], dim=1)
    image_text_align = torch.sum(problem["image_proto"] * problem["text_features"], dim=1)
    image_desc_align = torch.sum(problem["image_proto"] * problem["description_proto"], dim=1)

    base_support_counts = torch.tensor(problem["support_counts"][:num_base_cls], dtype=torch.float32)
    novel_support_counts = torch.tensor(problem["support_counts"][num_base_cls:], dtype=torch.float32)
    if base_support_counts.numel() == 0:
        base_support_counts = torch.zeros(1, dtype=torch.float32)
    if novel_support_counts.numel() == 0:
        novel_support_counts = torch.zeros(1, dtype=torch.float32)

    base_count_mean, base_count_std = mean_std(base_support_counts)
    novel_count_mean, novel_count_std = mean_std(novel_support_counts)
    base_align_mean, base_align_std = mean_std(image_sem_align[:num_base_cls])
    novel_align_mean, novel_align_std = mean_std(image_sem_align[num_base_cls:])
    base_text_mean, _ = mean_std(image_text_align[:num_base_cls])
    novel_text_mean, _ = mean_std(image_text_align[num_base_cls:])
    base_desc_mean, _ = mean_std(image_desc_align[:num_base_cls])
    novel_desc_mean, _ = mean_std(image_desc_align[num_base_cls:])

    novel_dispersion = []
    for cls_id in range(num_base_cls, num_cls):
        support = problem["support_by_class"][cls_id]
        novel_dispersion.append(1.0 - mean_pairwise_similarity(support))
    if novel_dispersion:
        novel_dispersion = torch.stack(novel_dispersion)
    else:
        novel_dispersion = torch.zeros(1, device=problem["query_features"].device)
    novel_disp_mean, novel_disp_std = mean_std(novel_dispersion)

    fused_base_mass = prob_fused[:, :num_base_cls].sum(dim=1)
    fused_novel_mass = prob_fused[:, num_base_cls:].sum(dim=1)
    cov_base_mass = problem["prob_cov"][:, :num_base_cls].sum(dim=1)
    knn_novel_mass = problem["prob_knn"][:, num_base_cls:].sum(dim=1)
    base_disagree = torch.abs(fused_base_mass - cov_base_mass)
    novel_disagree = torch.abs(fused_novel_mass - knn_novel_mass)

    if num_base_cls > 0:
        base_arg_fused = prob_fused[:, :num_base_cls].argmax(dim=1)
        base_arg_cov = problem["prob_cov"][:, :num_base_cls].argmax(dim=1)
        base_arg_agree = (base_arg_fused == base_arg_cov).float().mean()
    else:
        base_arg_agree = problem["query_features"].new_tensor(0.0)
    if num_novel_cls > 0:
        novel_arg_fused = prob_fused[:, num_base_cls:].argmax(dim=1)
        novel_arg_knn = problem["prob_knn"][:, num_base_cls:].argmax(dim=1)
        novel_arg_agree = (novel_arg_fused == novel_arg_knn).float().mean()
    else:
        novel_arg_agree = problem["query_features"].new_tensor(0.0)

    if num_base_cls > 0 and num_novel_cls > 0:
        query_to_image = torch.matmul(problem["query_features"], problem["image_proto"].T)
        base_best = query_to_image[:, :num_base_cls].max(dim=1).values
        novel_best = query_to_image[:, num_base_cls:].max(dim=1).values
        image_gap = novel_best - base_best
        query_to_sem = torch.matmul(problem["query_features"], problem["semantic_proto"].T)
        base_best_sem = query_to_sem[:, :num_base_cls].max(dim=1).values
        novel_best_sem = query_to_sem[:, num_base_cls:].max(dim=1).values
        semantic_gap = novel_best_sem - base_best_sem
    else:
        image_gap = problem["query_features"].new_zeros(problem["query_features"].shape[0])
        semantic_gap = problem["query_features"].new_zeros(problem["query_features"].shape[0])

    features = [
        float(num_cls),
        float(num_base_cls),
        float(num_novel_cls),
        float(problem["beta"]),
        float(problem["ensemble_alpha"]),
        float(base_count_mean.item()),
        float(base_count_std.item()),
        float(novel_count_mean.item()),
        float(novel_count_std.item()),
        float(base_align_mean.item()),
        float(base_align_std.item()),
        float(novel_align_mean.item()),
        float(novel_align_std.item()),
        float(base_text_mean.item()),
        float(novel_text_mean.item()),
        float(base_desc_mean.item()),
        float(novel_desc_mean.item()),
        float(novel_disp_mean.item()),
        float(novel_disp_std.item()),
        float(ent.mean().item()),
        float(ent.std(unbiased=False).item() if ent.numel() > 1 else 0.0),
        float(max_prob.mean().item()),
        float(max_prob.std(unbiased=False).item() if max_prob.numel() > 1 else 0.0),
        float(margin.mean().item()),
        float(margin.std(unbiased=False).item() if margin.numel() > 1 else 0.0),
        float(base_mass.mean().item()),
        float(base_mass.std(unbiased=False).item() if base_mass.numel() > 1 else 0.0),
        float(novel_mass.mean().item()),
        float(novel_mass.std(unbiased=False).item() if novel_mass.numel() > 1 else 0.0),
        float((pred_labels >= num_base_cls).float().mean().item()),
        float(marginal_entropy.item()),
        float(gini_coefficient(pred_counts.detach().cpu().numpy())),
        float(base_disagree.mean().item()),
        float(novel_disagree.mean().item()),
        float(base_arg_agree.item()),
        float(novel_arg_agree.item()),
        float(image_gap.mean().item()),
        float(image_gap.std(unbiased=False).item() if image_gap.numel() > 1 else 0.0),
        float(semantic_gap.mean().item()),
        float(semantic_gap.std(unbiased=False).item() if semantic_gap.numel() > 1 else 0.0),
    ]
    return torch.tensor(features, dtype=torch.float32, device=problem["query_features"].device)


def make_default_grids(beta_prior, ensemble_alpha):
    beta_values = sorted({
        round(float(beta_prior), 4),
        0.15,
        0.3,
        0.45,
        0.6,
        0.75,
        0.9,
    })
    alpha_values = sorted({
        round(float(ensemble_alpha), 4),
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    })
    delta_values = [-1.5, -1.125, -0.75, -0.375, 0.0, 0.375, 0.75, 1.125, 1.5]
    tau_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    return {
        "beta": beta_values,
        "alpha": alpha_values,
        "delta": delta_values,
        "tau": tau_values,
    }


def oracle_search(problem, param_keys, grids):
    targets = problem["query_targets"]
    class_ids = list(range(problem["num_cls"]))

    beta_values = grids["beta"] if "beta" in param_keys else [float(problem["beta"])]
    alpha_values = grids["alpha"]
    delta_values = grids["delta"]
    tau_values = grids["tau"]

    best = {
        "beta": float(problem["beta"]),
        "alpha_base": float(problem["ensemble_alpha"]),
        "alpha_novel": float(problem["ensemble_alpha"]),
        "delta": 0.0,
        "tau": 1.0,
        "acc": None,
        "nll": None,
        "probs": None,
    }

    for beta in beta_values:
        fused_proto = F.normalize(
            float(beta) * problem["semantic_proto"] + (1.0 - float(beta)) * problem["image_proto"],
            dim=-1,
        )
        logits_fused = torch.matmul(problem["query_features"], fused_proto.T)
        prob_fused = F.softmax(logits_fused, dim=-1)
        for alpha_base in alpha_values:
            base_probs = (
                float(alpha_base) * prob_fused[:, :problem["num_base_cls"]]
                + (1.0 - float(alpha_base)) * problem["prob_cov"][:, :problem["num_base_cls"]]
            )
            for alpha_novel in alpha_values:
                novel_probs = (
                    float(alpha_novel) * prob_fused[:, problem["num_base_cls"]:]
                    + (1.0 - float(alpha_novel)) * problem["prob_knn"][:, problem["num_base_cls"]:]
                )
                combined = torch.cat([base_probs, novel_probs], dim=1)
                log_probs = torch.log(torch.clamp(combined, min=1e-12))
                for tau in tau_values:
                    scaled_logits = log_probs / max(float(tau), 1e-12)
                    for delta in delta_values:
                        if abs(float(delta)) > 0.0:
                            calibrated_logits = scaled_logits.clone()
                            calibrated_logits[:, problem["num_base_cls"]:] = (
                                calibrated_logits[:, problem["num_base_cls"]:] + float(delta)
                            )
                        else:
                            calibrated_logits = scaled_logits
                        probs = F.softmax(calibrated_logits, dim=-1)
                        metrics = evaluate_probs(probs, targets, class_ids)
                        acc = metrics["acc"]
                        nll = metrics["nll"]
                        if best["acc"] is None:
                            better = True
                        elif acc.item() > best["acc"].item() + 1e-8:
                            better = True
                        elif abs(acc.item() - best["acc"].item()) <= 1e-8 and nll.item() < best["nll"].item() - 1e-8:
                            better = True
                        else:
                            better = False
                        if better:
                            best.update(
                                {
                                    "beta": float(beta),
                                    "alpha_base": float(alpha_base),
                                    "alpha_novel": float(alpha_novel),
                                    "delta": float(delta),
                                    "tau": float(tau),
                                    "acc": acc.detach(),
                                    "nll": nll.detach(),
                                    "probs": probs.detach(),
                                }
                            )
    return best


def attach_episode_features_and_oracles(episodes, mode, grids):
    param_keys = MODE_PARAM_KEYS[mode]
    for problem in episodes:
        problem["feature_vec"] = build_episode_features(problem)
        problem["oracle"] = oracle_search(problem, param_keys=param_keys, grids=grids)
    return episodes


def split_episodes(episodes, val_fraction=0.2):
    if len(episodes) <= 1:
        return episodes, episodes
    split_idx = max(1, int(round(len(episodes) * (1.0 - val_fraction))))
    split_idx = min(split_idx, len(episodes) - 1)
    return episodes[:split_idx], episodes[split_idx:]


def target_tensor(problem, mode, device):
    oracle = problem["oracle"]
    values = [oracle[key] for key in MODE_PARAM_KEYS[mode]]
    return torch.tensor(values, dtype=torch.float32, device=device)


def fit_ridge_regressor(train_x, train_y, reg_lambda):
    x_mean = train_x.mean(dim=0, keepdim=True)
    x_std = train_x.std(dim=0, unbiased=False, keepdim=True)
    x_std = torch.where(x_std < 1e-6, torch.ones_like(x_std), x_std)
    x_norm = (train_x - x_mean) / x_std

    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, unbiased=False, keepdim=True)
    y_std = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)
    y_norm = (train_y - y_mean) / y_std

    ones = torch.ones((x_norm.shape[0], 1), dtype=x_norm.dtype, device=x_norm.device)
    x_aug = torch.cat([x_norm, ones], dim=1)
    gram = x_aug.T @ x_aug
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    eye[-1, -1] = 0.0
    weights = torch.linalg.pinv(gram + float(reg_lambda) * eye) @ (x_aug.T @ y_norm)
    return {
        "weights": weights,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def predict_regressor(model, feature_vec):
    x_norm = (feature_vec.unsqueeze(0) - model["x_mean"]) / model["x_std"]
    x_aug = torch.cat(
        [x_norm, torch.ones((1, 1), dtype=x_norm.dtype, device=x_norm.device)],
        dim=1,
    )
    y_norm = x_aug @ model["weights"]
    return (y_norm * model["y_std"] + model["y_mean"]).squeeze(0)


def pack_params(mode, predicted, beta_prior, ensemble_alpha):
    params = {
        "beta": float(beta_prior),
        "alpha_base": float(ensemble_alpha),
        "alpha_novel": float(ensemble_alpha),
        "delta": 0.0,
        "tau": 1.0,
    }
    for idx, key in enumerate(MODE_PARAM_KEYS[mode]):
        params[key] = float(predicted[idx].item())
    for key, (low, high) in PARAM_RANGES.items():
        params[key] = float(min(max(params[key], low), high))
    return params


def normalized_param_distance(params, ref_params):
    distance = 0.0
    count = 0
    for key, (low, high) in PARAM_RANGES.items():
        scale = max(high - low, 1e-6)
        distance += ((float(params[key]) - float(ref_params[key])) / scale) ** 2
        count += 1
    return distance / max(count, 1)


def info_objective(problem, params, ref_params, diversity_weight, pseudo_weight, reg_weight, confidence):
    probs, _ = compose_probs_from_params(
        problem,
        beta=params["beta"],
        alpha_base=params["alpha_base"],
        alpha_novel=params["alpha_novel"],
        delta=params["delta"],
        tau=params["tau"],
    )
    sample_entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1).mean()
    marginal = probs.mean(dim=0)
    marginal_entropy = -(marginal * torch.log(torch.clamp(marginal, min=1e-12))).sum()
    max_prob, pseudo_labels = probs.max(dim=1)
    mask = max_prob >= float(confidence)
    if mask.any():
        idx = torch.arange(mask.sum(), device=probs.device)
        pseudo = torch.clamp(probs[mask][idx, pseudo_labels[mask]], min=1e-12)
        pseudo_loss = -torch.log(pseudo).mean()
    else:
        pseudo_loss = sample_entropy.new_tensor(0.0)
    reg = sample_entropy.new_tensor(normalized_param_distance(params, ref_params))
    objective = sample_entropy - float(diversity_weight) * marginal_entropy
    objective = objective + float(pseudo_weight) * pseudo_loss
    objective = objective + float(reg_weight) * reg
    return float(objective.item()), probs


def local_candidates(key, center, global_grids):
    low, high = PARAM_RANGES[key]
    if key == "tau":
        local = list(global_grids["tau"])
    elif key == "delta":
        span = [0.75, 0.375, 0.0, -0.375, -0.75]
        local = [center + delta for delta in span]
    else:
        span = [-0.2, -0.1, 0.0, 0.1, 0.2]
        local = [center + delta for delta in span]
    values = []
    for value in local:
        value = float(min(max(value, low), high))
        values.append(round(value, 6))
    return sorted(set(values))


def refine_astar_params(
    problem,
    init_params,
    grids,
    refinement_passes,
    diversity_weight,
    pseudo_weight,
    reg_weight,
    confidence,
):
    current = dict(init_params)
    best_obj, _ = info_objective(
        problem,
        current,
        init_params,
        diversity_weight=diversity_weight,
        pseudo_weight=pseudo_weight,
        reg_weight=reg_weight,
        confidence=confidence,
    )
    for _ in range(int(refinement_passes)):
        improved_any = False
        for key in MODE_PARAM_KEYS["astar"]:
            best_local = dict(current)
            for candidate in local_candidates(key, current[key], grids):
                trial = dict(current)
                trial[key] = candidate
                obj, _ = info_objective(
                    problem,
                    trial,
                    init_params,
                    diversity_weight=diversity_weight,
                    pseudo_weight=pseudo_weight,
                    reg_weight=reg_weight,
                    confidence=confidence,
                )
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best_local = trial
                    improved_any = True
            current = best_local
        if not improved_any:
            break
    return current, best_obj


def evaluate_params(problem, params):
    probs, _ = compose_probs_from_params(
        problem,
        beta=params["beta"],
        alpha_base=params["alpha_base"],
        alpha_novel=params["alpha_novel"],
        delta=params["delta"],
        tau=params["tau"],
    )
    metrics = evaluate_probs(probs, problem["query_targets"], list(range(problem["num_cls"])))
    return probs, metrics


def validate_regressor(model, episodes, mode):
    if not episodes:
        return {
            "val_param_mae": 0.0,
            "val_episode_gain": 0.0,
        }
    maes = []
    gains = []
    for problem in episodes:
        pred = predict_regressor(model, problem["feature_vec"])
        params = pack_params(
            mode=mode,
            predicted=pred,
            beta_prior=problem["beta"],
            ensemble_alpha=problem["ensemble_alpha"],
        )
        target = problem["oracle"]
        diff = []
        for key in MODE_PARAM_KEYS[mode]:
            diff.append(abs(float(params[key]) - float(target[key])))
        maes.append(float(np.mean(diff)))

        _, baseline_metrics = evaluate_params(
            problem,
            {
                "beta": float(problem["beta"]),
                "alpha_base": float(problem["ensemble_alpha"]),
                "alpha_novel": float(problem["ensemble_alpha"]),
                "delta": 0.0,
                "tau": 1.0,
            },
        )
        _, pred_metrics = evaluate_params(problem, params)
        gains.append(float((pred_metrics["acc"] - baseline_metrics["acc"]).item() * 100.0))
    return {
        "val_param_mae": round(float(np.mean(maes)), 6),
        "val_episode_gain": round(float(np.mean(gains)), 4),
    }


def dataset_decision(mode, gain, recovery_rate):
    recovery_rate = float(recovery_rate or 0.0)
    if mode == "bmvc":
        return "continue_direction" if gain >= 0.2 and recovery_rate >= 0.2 else "kill_direction"
    return "continue_direction" if gain >= 0.2 and recovery_rate >= 0.3 else "kill_direction"


def run_dataset(
    data_cfg,
    train_cfg,
    mode,
    pseudo_episodes,
    pseudo_base_count,
    pseudo_novel_count,
    pseudo_novel_shot,
    base_query_per_class,
    novel_query_per_class,
    reg_lambda,
    refinement_passes,
    diversity_weight,
    pseudo_weight,
    refinement_reg,
    confidence,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    out_dir = create_output_dir(cfg, train_cfg, mode)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    data_manager, model, merged_state, final_task_id = extract_final_state(cfg)
    query_features, query_targets = collect_final_queries(cfg, data_manager, model, final_task_id)
    device = torch.device(cfg.DEVICE.DEVICE_NAME)
    query_features = query_features.to(device=device, dtype=torch.float32)
    query_targets = query_targets.to(device)

    num_cls = max(data_manager.class_index_in_task[final_task_id]) + 1
    num_base_cls = len(data_manager.class_index_in_task[0])
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)
    grids = make_default_grids(beta_prior=beta, ensemble_alpha=ensemble_alpha)

    if pseudo_novel_count <= 0:
        pseudo_novel_count = int(cfg.DATASET.NUM_INC_CLS)
    if pseudo_novel_shot <= 0:
        pseudo_novel_shot = int(cfg.DATASET.NUM_INC_SHOT)

    episodes = sample_pseudo_episodes(
        merged_state=merged_state,
        num_base_cls=num_base_cls,
        pseudo_episodes=pseudo_episodes,
        pseudo_base_count=pseudo_base_count,
        pseudo_novel_count=pseudo_novel_count,
        pseudo_novel_shot=pseudo_novel_shot,
        base_query_per_class=base_query_per_class,
        novel_query_per_class=novel_query_per_class,
        lambda_t=lambda_t,
        beta=beta,
        ensemble_alpha=ensemble_alpha,
        seed=int(cfg.SEED),
        device=device,
    )
    episodes = attach_episode_features_and_oracles(episodes, mode=mode, grids=grids)
    train_episodes, val_episodes = split_episodes(episodes, val_fraction=0.2)

    train_x = torch.stack([problem["feature_vec"] for problem in train_episodes], dim=0)
    train_y = torch.stack([target_tensor(problem, mode, device) for problem in train_episodes], dim=0)
    regressor = fit_ridge_regressor(train_x, train_y, reg_lambda=reg_lambda)
    train_info = validate_regressor(regressor, val_episodes, mode=mode)

    final_problem = build_final_problem(
        merged_state=merged_state,
        query_features=query_features,
        query_targets=query_targets,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        lambda_t=lambda_t,
        beta=beta,
        ensemble_alpha=ensemble_alpha,
        device=device,
    )
    final_problem["feature_vec"] = build_episode_features(final_problem)

    baseline_params = {
        "beta": beta,
        "alpha_base": ensemble_alpha,
        "alpha_novel": ensemble_alpha,
        "delta": 0.0,
        "tau": 1.0,
    }
    _, baseline_metrics = evaluate_params(final_problem, baseline_params)

    pred_vector = predict_regressor(regressor, final_problem["feature_vec"])
    predicted_params = pack_params(
        mode=mode,
        predicted=pred_vector,
        beta_prior=beta,
        ensemble_alpha=ensemble_alpha,
    )

    refined_params = None
    transductive_objective = None
    if mode == "astar":
        refined_params, transductive_objective = refine_astar_params(
            problem=final_problem,
            init_params=predicted_params,
            grids=grids,
            refinement_passes=refinement_passes,
            diversity_weight=diversity_weight,
            pseudo_weight=pseudo_weight,
            reg_weight=refinement_reg,
            confidence=confidence,
        )
        method_params = refined_params
    else:
        method_params = predicted_params

    _, method_metrics = evaluate_params(final_problem, method_params)
    oracle = oracle_search(final_problem, param_keys=MODE_PARAM_KEYS[mode], grids=grids)

    method_gain = float((method_metrics["acc"] - baseline_metrics["acc"]).item() * 100.0)
    oracle_gain = float((oracle["acc"] - baseline_metrics["acc"]).item() * 100.0)
    recovery_rate = None if abs(oracle_gain) <= 1e-9 else round(method_gain / oracle_gain, 4)
    recommendation = dataset_decision(mode, method_gain, recovery_rate)
    runtime_sec = round(time.time() - start_time, 3)

    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "mode": mode,
        "runtime_sec": runtime_sec,
        "num_seen_classes": num_cls,
        "num_base_classes": num_base_cls,
        "baseline_beta": beta,
        "baseline_lambda_t": lambda_t,
        "baseline_ensemble_alpha": ensemble_alpha,
        "pseudo_episodes": pseudo_episodes,
        "pseudo_base_count": pseudo_base_count,
        "pseudo_novel_count": pseudo_novel_count,
        "pseudo_novel_shot": pseudo_novel_shot,
        "base_query_per_class": base_query_per_class,
        "novel_query_per_class": novel_query_per_class,
        "reg_lambda": reg_lambda,
        "refinement_passes": refinement_passes,
        "diversity_weight": diversity_weight,
        "pseudo_weight": pseudo_weight,
        "refinement_reg": refinement_reg,
        "confidence": confidence,
        "num_train_episodes": len(train_episodes),
        "num_val_episodes": len(val_episodes),
        "baseline_acc": round(float(baseline_metrics["acc"].item() * 100.0), 3),
        "method_acc": round(float(method_metrics["acc"].item() * 100.0), 3),
        "oracle_acc": round(float(oracle["acc"].item() * 100.0), 3),
        "method_gain": round(method_gain, 3),
        "oracle_gain": round(oracle_gain, 3),
        "recovery_rate": recovery_rate,
        "recommendation": recommendation,
        "predicted_params": predicted_params,
        "method_params": method_params,
        "oracle_params": {key: oracle[key] for key in ["beta", "alpha_base", "alpha_novel", "delta", "tau"]},
        "baseline_nll": round(float(baseline_metrics["nll"].item()), 6),
        "method_nll": round(float(method_metrics["nll"].item()), 6),
        "oracle_nll": round(float(oracle["nll"].item()), 6),
        "transductive_objective": transductive_objective,
        **train_info,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"mode: {mode}",
        f"runtime_sec: {runtime_sec}",
        f"baseline_acc: {payload['baseline_acc']}",
        f"method_acc: {payload['method_acc']}",
        f"oracle_acc: {payload['oracle_acc']}",
        f"method_gain: {payload['method_gain']}",
        f"oracle_gain: {payload['oracle_gain']}",
        f"recovery_rate: {payload['recovery_rate']}",
        f"recommendation: {payload['recommendation']}",
        f"baseline_nll: {payload['baseline_nll']}",
        f"method_nll: {payload['method_nll']}",
        f"oracle_nll: {payload['oracle_nll']}",
        f"val_param_mae: {payload['val_param_mae']}",
        f"val_episode_gain: {payload['val_episode_gain']}",
        f"predicted_params: {payload['predicted_params']}",
        f"method_params: {payload['method_params']}",
        f"oracle_params: {payload['oracle_params']}",
    ]
    if transductive_objective is not None:
        summary_lines.append(f"transductive_objective: {transductive_objective}")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "dataset": cfg.DATASET.NAME,
                "mode": mode,
                "baseline_acc": payload["baseline_acc"],
                "method_acc": payload["method_acc"],
                "oracle_acc": payload["oracle_acc"],
                "method_gain": payload["method_gain"],
                "oracle_gain": payload["oracle_gain"],
                "recovery_rate": payload["recovery_rate"],
                "recommendation": payload["recommendation"],
            },
            indent=2,
        )
    )
    print(f"Saved calibration meta results to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--mode", choices=["bmvc", "astar"], required=True)
    parser.add_argument("--pseudo_episodes", type=int, default=48)
    parser.add_argument("--pseudo_base_count", type=int, default=20)
    parser.add_argument("--pseudo_novel_count", type=int, default=-1)
    parser.add_argument("--pseudo_novel_shot", type=int, default=-1)
    parser.add_argument("--base_query_per_class", type=int, default=2)
    parser.add_argument("--novel_query_per_class", type=int, default=3)
    parser.add_argument("--reg_lambda", type=float, default=0.1)
    parser.add_argument("--refinement_passes", type=int, default=2)
    parser.add_argument("--diversity_weight", type=float, default=0.6)
    parser.add_argument("--pseudo_weight", type=float, default=0.4)
    parser.add_argument("--refinement_reg", type=float, default=0.05)
    parser.add_argument("--confidence", type=float, default=0.8)
    args = parser.parse_args()

    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        mode=args.mode,
        pseudo_episodes=args.pseudo_episodes,
        pseudo_base_count=args.pseudo_base_count,
        pseudo_novel_count=args.pseudo_novel_count,
        pseudo_novel_shot=args.pseudo_novel_shot,
        base_query_per_class=args.base_query_per_class,
        novel_query_per_class=args.novel_query_per_class,
        reg_lambda=args.reg_lambda,
        refinement_passes=args.refinement_passes,
        diversity_weight=args.diversity_weight,
        pseudo_weight=args.pseudo_weight,
        refinement_reg=args.refinement_reg,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
