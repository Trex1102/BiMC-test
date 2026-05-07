import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


def create_output_dir(cfg, train_cfg, mode):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"semantic_meta_{mode}_{Path(train_cfg).stem}"
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
    for latest in reversed(results_files):
        try:
            with latest.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return float(payload["oracle_gain"]), latest.parent
        except Exception:
            continue
    return None, None


def accuracy_from_probs(probs, targets):
    return float((probs.argmax(dim=1) == targets).float().mean().item() * 100.0)


def overall_acc_tensor(probs, targets):
    return (probs.argmax(dim=1) == targets).float().mean()


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
    sample_margin = true_logits - competitor
    for cls_id in class_ids:
        mask = targets == cls_id
        if mask.any():
            margins.append(sample_margin[mask].mean())
    return torch.stack(margins).mean()


def mean_pairwise_similarity(features):
    if features.shape[0] <= 1:
        return features.new_tensor(1.0)
    sim = torch.matmul(features, features.T)
    denom = float(features.shape[0] * (features.shape[0] - 1))
    return (sim.sum() - torch.trace(sim)) / max(denom, 1.0)


def evaluate_probs(probs, logits, targets, class_ids):
    return {
        "acc": overall_acc_tensor(probs, targets),
        "acc_pct": round(float(overall_acc_tensor(probs, targets).item() * 100.0), 3),
        "nll": class_balanced_nll(probs, targets, class_ids),
        "margin": class_balanced_margin(logits, targets, class_ids),
    }


def regularize_covariance(features, eps=1e-4):
    if features.shape[0] <= 1:
        dim = features.shape[1]
        return torch.eye(dim, device=features.device, dtype=features.dtype)
    cov = torch.cov(features.T)
    dim = cov.shape[0]
    eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)
    diag_mean = torch.diagonal(cov).mean()
    return cov + eps * max(float(diag_mean.item()), 1.0) * eye


def select_indices(num_items, count, generator):
    if count <= 0:
        return []
    if count >= num_items:
        return list(range(num_items))
    perm = torch.randperm(num_items, generator=generator).tolist()
    return sorted(perm[:count])


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
    max_desc_atoms,
    base_atom_topk,
    seed,
    device,
):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    class_ids = list(global_base_ids) + list(global_novel_ids)
    num_base_cls = len(global_base_ids)
    num_cls = len(class_ids)

    text_features = F.normalize(merged_state["text_features"][class_ids].to(device=device, dtype=torch.float32), dim=-1)
    description_proto = F.normalize(merged_state["description_proto"][class_ids].to(device=device, dtype=torch.float32), dim=-1)

    desc_groups = []
    desc_targets = []
    image_proto_list = []
    cov_sources = []
    query_features = []
    query_targets = []
    support_by_class = {}

    for local_cls, global_cls in enumerate(class_ids):
        desc_mask = merged_state["description_targets"] == int(global_cls)
        class_desc = F.normalize(
            merged_state["description_features"][desc_mask].to(device=device, dtype=torch.float32),
            dim=-1,
        )
        desc_groups.append(class_desc)
        desc_targets.append(torch.full((class_desc.shape[0],), local_cls, dtype=torch.long, device=device))

        class_images = F.normalize(
            merged_state["images_features"][merged_state["images_targets"] == int(global_cls)].to(device=device, dtype=torch.float32),
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
        image_proto_list.append(normalize_vector(proto_source.mean(dim=0)))
        cov_sources.append(proto_source)
        if query.numel() > 0:
            query_features.append(query)
            query_targets.append(torch.full((query.shape[0],), local_cls, dtype=torch.long, device=device))

    description_features = torch.cat(desc_groups, dim=0)
    description_targets = torch.cat(desc_targets, dim=0)
    image_proto = F.normalize(torch.stack(image_proto_list, dim=0), dim=-1)
    cov_image = regularize_covariance(torch.cat(cov_sources, dim=0))
    query_features = torch.cat(query_features, dim=0)
    query_targets = torch.cat(query_targets, dim=0)

    local_state = {
        "text_features": text_features,
        "description_proto": description_proto,
        "description_features": description_features,
        "description_targets": description_targets,
    }

    base_semantic, banks = build_semantic_banks(
        local_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )
    base_semantic = base_semantic.to(device=device, dtype=torch.float32)
    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(compute_knn_logits(query_features, description_features, description_targets, num_cls), dim=-1)

    bank_vectors = {}
    bank_meta = {}
    for cls_id in range(num_base_cls, num_cls):
        bank = banks[cls_id]
        vectors = torch.stack(
            [normalize_vector(candidate["vector"].to(device=device, dtype=torch.float32)) for candidate in bank],
            dim=0,
        )
        bank_vectors[cls_id] = vectors
        bank_meta[cls_id] = bank

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
        "prob_cov": prob_cov,
        "prob_knn": prob_knn,
        "support_by_class": support_by_class,
        "base_semantic": base_semantic,
        "bank_vectors": bank_vectors,
        "bank_meta": bank_meta,
        "beta": float(beta),
        "ensemble_alpha": float(ensemble_alpha),
        "novel_classes": list(range(num_base_cls, num_cls)),
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
    max_desc_atoms,
    base_atom_topk,
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
            max_desc_atoms=max_desc_atoms,
            base_atom_topk=base_atom_topk,
            seed=int(seed) + 997 * (episode_idx + 1),
            device=device,
        )
        episodes.append(problem)
    return episodes


def build_candidate_feature_matrix(problem, cls_id):
    support_features = F.normalize(problem["support_by_class"][cls_id], dim=-1)
    base_semantic = problem["base_semantic"]
    image_proto = problem["image_proto"]
    bank_vectors = problem["bank_vectors"][cls_id]
    bank_meta = problem["bank_meta"][cls_id]
    beta = float(problem["beta"])
    num_base_cls = int(problem["num_base_cls"])

    prompt_vec = normalize_vector(problem["text_features"][cls_id])
    desc_vec = normalize_vector(problem["description_proto"][cls_id])
    image_vec = normalize_vector(image_proto[cls_id])
    support_proto = normalize_vector(support_features.mean(dim=0))
    support_self = mean_pairwise_similarity(support_features)
    baseline_sem = base_semantic[cls_id]
    baseline_fused = F.normalize(beta * base_semantic + (1.0 - beta) * image_proto, dim=-1)
    if num_base_cls > 0:
        baseline_base_scores = torch.matmul(support_features, baseline_fused[:num_base_cls].T).max(dim=1).values
    else:
        baseline_base_scores = support_features.new_zeros(support_features.shape[0])
    baseline_target_scores = torch.matmul(support_features, baseline_fused[cls_id])
    baseline_margin = (baseline_target_scores - baseline_base_scores).mean()
    base_sem_bank = base_semantic[:num_base_cls] if num_base_cls > 0 else None
    base_img_bank = image_proto[:num_base_cls] if num_base_cls > 0 else None

    features = []
    for cand_idx, cand_vec in enumerate(bank_vectors):
        temp_sem = base_semantic.clone()
        temp_sem[cls_id] = cand_vec
        fused = F.normalize(beta * temp_sem + (1.0 - beta) * image_proto, dim=-1)
        cand_scores = torch.matmul(support_features, cand_vec)
        target_scores = torch.matmul(support_features, fused[cls_id])
        if num_base_cls > 0:
            base_scores = torch.matmul(support_features, fused[:num_base_cls].T).max(dim=1).values
            cand_to_best_base_sem = torch.matmul(cand_vec, base_sem_bank.T).max()
            cand_to_best_base_img = torch.matmul(cand_vec, base_img_bank.T).max()
        else:
            base_scores = support_features.new_zeros(support_features.shape[0])
            cand_to_best_base_sem = cand_vec.new_tensor(0.0)
            cand_to_best_base_img = cand_vec.new_tensor(0.0)
        margins = target_scores - base_scores
        meta = bank_meta[cand_idx]
        features.append(torch.stack([
            torch.matmul(cand_vec, support_proto),
            torch.matmul(cand_vec, baseline_sem),
            torch.matmul(cand_vec, prompt_vec),
            torch.matmul(cand_vec, desc_vec),
            torch.matmul(cand_vec, image_vec),
            cand_to_best_base_sem,
            cand_to_best_base_img,
            cand_scores.mean(),
            cand_scores.min(),
            target_scores.mean(),
            target_scores.min(),
            margins.mean(),
            margins.min(),
            support_self,
            support_features.new_tensor(float(meta["uses_base"])),
            support_features.new_tensor(float(meta["uses_desc_atom"])),
            support_features.new_tensor(float(meta["is_baseline"])),
            support_features.new_tensor(math.log1p(float(support_features.shape[0]))),
            baseline_margin,
            target_scores.mean() - baseline_target_scores.mean(),
        ]))
    return torch.stack(features, dim=0)


def compose_episode_with_choice(problem, cls_id, candidate_vec):
    semantic = problem["base_semantic"].clone()
    semantic[cls_id] = candidate_vec
    probs, logits = compose_probs_from_semantic(
        problem["query_features"],
        semantic,
        problem["image_proto"],
        problem["num_base_cls"],
        problem["beta"],
        problem["ensemble_alpha"],
        problem["prob_cov"],
        problem["prob_knn"],
    )
    return probs, logits


def compute_oracle_index(problem, cls_id):
    class_ids = list(range(problem["num_cls"]))
    baseline_probs, baseline_logits = compose_probs_from_semantic(
        problem["query_features"],
        problem["base_semantic"],
        problem["image_proto"],
        problem["num_base_cls"],
        problem["beta"],
        problem["ensemble_alpha"],
        problem["prob_cov"],
        problem["prob_knn"],
    )
    baseline_metrics = evaluate_probs(baseline_probs, baseline_logits, problem["query_targets"], class_ids)
    best_idx = 0
    best_nll = baseline_metrics["nll"]
    best_acc = baseline_metrics["acc"]

    for cand_idx, candidate_vec in enumerate(problem["bank_vectors"][cls_id]):
        probs, logits = compose_episode_with_choice(problem, cls_id, candidate_vec)
        metrics = evaluate_probs(probs, logits, problem["query_targets"], class_ids)
        if metrics["nll"].item() < best_nll.item() - 1e-8:
            best_idx = cand_idx
            best_nll = metrics["nll"]
            best_acc = metrics["acc"]
        elif abs(metrics["nll"].item() - best_nll.item()) <= 1e-8 and metrics["acc"].item() > best_acc.item() + 1e-8:
            best_idx = cand_idx
            best_nll = metrics["nll"]
            best_acc = metrics["acc"]
    return int(best_idx)


def attach_candidate_features_and_oracles(episodes):
    for problem in episodes:
        candidate_features = {}
        oracle_index = {}
        for cls_id in problem["novel_classes"]:
            candidate_features[cls_id] = build_candidate_feature_matrix(problem, cls_id)
            oracle_index[cls_id] = compute_oracle_index(problem, cls_id)
        problem["candidate_features"] = candidate_features
        problem["oracle_index"] = oracle_index
    return episodes


def split_episodes(episodes, val_fraction=0.2):
    if len(episodes) <= 1:
        return episodes, episodes
    split_idx = max(1, int(round(len(episodes) * (1.0 - val_fraction))))
    split_idx = min(split_idx, len(episodes) - 1)
    return episodes[:split_idx], episodes[split_idx:]


class CandidateScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def train_bmvc_ranker(train_episodes, val_episodes, input_dim, hidden_dim, lr, max_epochs, patience, device):
    model = CandidateScorer(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val_acc = -1.0
    best_val_loss = float("inf")
    stale_epochs = 0

    for _ in range(int(max_epochs)):
        model.train()
        epoch_loss = 0.0
        for problem in train_episodes:
            optimizer.zero_grad()
            losses = []
            for cls_id in problem["novel_classes"]:
                scores = model(problem["candidate_features"][cls_id])
                target = torch.tensor([problem["oracle_index"][cls_id]], dtype=torch.long, device=device)
                losses.append(F.cross_entropy(scores.unsqueeze(0), target))
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for problem in val_episodes:
                for cls_id in problem["novel_classes"]:
                    scores = model(problem["candidate_features"][cls_id])
                    target_idx = int(problem["oracle_index"][cls_id])
                    val_loss += float(F.cross_entropy(scores.unsqueeze(0), torch.tensor([target_idx], dtype=torch.long, device=device)).item())
                    pred_idx = int(torch.argmax(scores).item())
                    correct += int(pred_idx == target_idx)
                    total += 1
            val_acc = float(correct / max(total, 1))
            improved = val_acc > best_val_acc + 1e-8 or (abs(val_acc - best_val_acc) <= 1e-8 and val_loss < best_val_loss - 1e-8)
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(patience):
                    break

    model.load_state_dict(best_state)
    return model, {
        "best_val_oracle_match": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 6),
    }


def build_soft_semantic(problem, scorer, temperature=1.0):
    semantic = problem["base_semantic"].clone()
    weights_by_class = {}
    scores_by_class = {}
    for cls_id in problem["novel_classes"]:
        scores = scorer(problem["candidate_features"][cls_id]) / max(float(temperature), 1e-6)
        weights = F.softmax(scores, dim=0)
        semantic_vec = torch.matmul(weights.unsqueeze(0), problem["bank_vectors"][cls_id]).squeeze(0)
        semantic[cls_id] = normalize_vector(semantic_vec)
        weights_by_class[cls_id] = weights
        scores_by_class[cls_id] = scores
    return semantic, weights_by_class, scores_by_class


def train_astar_selector(
    train_episodes,
    val_episodes,
    input_dim,
    hidden_dim,
    lr,
    max_epochs,
    patience,
    prior_weight,
    margin_weight,
    entropy_weight,
    distill_weight,
    temperature,
    device,
):
    model = CandidateScorer(input_dim=input_dim, hidden_dim=hidden_dim, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val_acc = -1.0
    best_val_nll = float("inf")
    stale_epochs = 0

    for _ in range(int(max_epochs)):
        model.train()
        for problem in train_episodes:
            optimizer.zero_grad()
            semantic, weights_by_class, scores_by_class = build_soft_semantic(problem, model, temperature=temperature)
            probs, logits = compose_probs_from_semantic(
                problem["query_features"],
                semantic,
                problem["image_proto"],
                problem["num_base_cls"],
                problem["beta"],
                problem["ensemble_alpha"],
                problem["prob_cov"],
                problem["prob_knn"],
            )
            class_ids = list(range(problem["num_cls"]))
            nll = class_balanced_nll(probs, problem["query_targets"], class_ids)
            margin = class_balanced_margin(logits, problem["query_targets"], class_ids)

            prior_loss = []
            entropy_loss = []
            distill_loss = []
            for cls_id in problem["novel_classes"]:
                weights = weights_by_class[cls_id]
                prior = torch.zeros_like(weights)
                prior[0] = 1.0
                prior_loss.append(F.kl_div(torch.log(torch.clamp(weights, min=1e-8)), prior, reduction="batchmean"))
                entropy_loss.append(-(weights * torch.log(torch.clamp(weights, min=1e-8))).sum())
                oracle_idx = torch.tensor([problem["oracle_index"][cls_id]], dtype=torch.long, device=device)
                distill_loss.append(F.cross_entropy(scores_by_class[cls_id].unsqueeze(0), oracle_idx))

            loss = nll
            loss = loss - float(margin_weight) * margin
            loss = loss + float(prior_weight) * torch.stack(prior_loss).mean()
            loss = loss + float(entropy_weight) * torch.stack(entropy_loss).mean()
            loss = loss + float(distill_weight) * torch.stack(distill_loss).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = []
            val_nll = []
            for problem in val_episodes:
                semantic, _, _ = build_soft_semantic(problem, model, temperature=temperature)
                probs, logits = compose_probs_from_semantic(
                    problem["query_features"],
                    semantic,
                    problem["image_proto"],
                    problem["num_base_cls"],
                    problem["beta"],
                    problem["ensemble_alpha"],
                    problem["prob_cov"],
                    problem["prob_knn"],
                )
                metrics = evaluate_probs(probs, logits, problem["query_targets"], list(range(problem["num_cls"])))
                val_acc.append(float(metrics["acc"].item()))
                val_nll.append(float(metrics["nll"].item()))
            mean_val_acc = float(np.mean(val_acc)) if val_acc else -1.0
            mean_val_nll = float(np.mean(val_nll)) if val_nll else float("inf")
            improved = mean_val_acc > best_val_acc + 1e-8 or (abs(mean_val_acc - best_val_acc) <= 1e-8 and mean_val_nll < best_val_nll - 1e-8)
            if improved:
                best_val_acc = mean_val_acc
                best_val_nll = mean_val_nll
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(patience):
                    break

    model.load_state_dict(best_state)
    return model, {
        "best_val_episode_acc": round(best_val_acc, 4),
        "best_val_episode_nll": round(best_val_nll, 6),
    }


def build_final_problem(
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
    device,
):
    image_proto = F.normalize(merged_state["image_proto"][:num_cls].to(device=device, dtype=torch.float32), dim=-1)
    description_features = F.normalize(merged_state["description_features"].to(device=device, dtype=torch.float32), dim=-1)
    description_targets = merged_state["description_targets"].to(device)
    text_features = F.normalize(merged_state["text_features"][:num_cls].to(device=device, dtype=torch.float32), dim=-1)
    description_proto = F.normalize(merged_state["description_proto"][:num_cls].to(device=device, dtype=torch.float32), dim=-1)
    cov_image = merged_state["cov_image"].to(device=device, dtype=torch.float32)

    local_state = {
        "text_features": text_features,
        "description_proto": description_proto,
        "description_features": description_features,
        "description_targets": description_targets,
    }
    base_semantic, banks = build_semantic_banks(
        local_state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )
    base_semantic = base_semantic.to(device=device, dtype=torch.float32)
    prob_cov = F.softmax(compute_cov_logits(query_features, image_proto, cov_image) / 512.0, dim=-1)
    prob_knn = F.softmax(compute_knn_logits(query_features, description_features, description_targets, num_cls), dim=-1)
    support_by_class = {}
    for cls_id in range(num_base_cls, num_cls):
        class_support = F.normalize(
            merged_state["images_features"][merged_state["images_targets"] == int(cls_id)].to(device=device, dtype=torch.float32),
            dim=-1,
        )
        support_by_class[cls_id] = class_support

    bank_vectors = {}
    bank_meta = {}
    for cls_id in range(num_base_cls, num_cls):
        bank = banks[cls_id]
        bank_meta[cls_id] = bank
        bank_vectors[cls_id] = torch.stack(
            [normalize_vector(candidate["vector"].to(device=device, dtype=torch.float32)) for candidate in bank],
            dim=0,
        )

    problem = {
        "num_cls": num_cls,
        "num_base_cls": num_base_cls,
        "text_features": text_features,
        "description_proto": description_proto,
        "description_features": description_features,
        "description_targets": description_targets,
        "image_proto": image_proto,
        "query_features": query_features,
        "query_targets": query_targets,
        "prob_cov": prob_cov,
        "prob_knn": prob_knn,
        "support_by_class": support_by_class,
        "base_semantic": base_semantic,
        "bank_vectors": bank_vectors,
        "bank_meta": bank_meta,
        "beta": float(beta),
        "ensemble_alpha": float(ensemble_alpha),
        "novel_classes": list(range(num_base_cls, num_cls)),
    }
    candidate_features = {}
    for cls_id in problem["novel_classes"]:
        candidate_features[cls_id] = build_candidate_feature_matrix(problem, cls_id)
    problem["candidate_features"] = candidate_features
    return problem


def evaluate_selector(problem, scorer, mode, temperature):
    if mode == "bmvc":
        semantic = problem["base_semantic"].clone()
        chosen_candidates = {}
        changed_classes = []
        base_atom_usage = 0
        desc_atom_usage = 0
        for cls_id in problem["novel_classes"]:
            scores = scorer(problem["candidate_features"][cls_id])
            top_idx = int(torch.argmax(scores).item())
            meta = problem["bank_meta"][cls_id][top_idx]
            semantic[cls_id] = problem["bank_vectors"][cls_id][top_idx]
            chosen_candidates[str(cls_id)] = meta["name"]
            if meta["name"] != "baseline_sem":
                changed_classes.append(cls_id)
            if meta["uses_base"]:
                base_atom_usage += 1
            if meta["uses_desc_atom"]:
                desc_atom_usage += 1
    else:
        semantic, weights_by_class, _ = build_soft_semantic(problem, scorer, temperature=temperature)
        chosen_candidates = {}
        changed_classes = []
        base_atom_usage = 0
        desc_atom_usage = 0
        for cls_id in problem["novel_classes"]:
            top_idx = int(torch.argmax(weights_by_class[cls_id]).item())
            meta = problem["bank_meta"][cls_id][top_idx]
            chosen_candidates[str(cls_id)] = meta["name"]
            if meta["name"] != "baseline_sem":
                changed_classes.append(cls_id)
            if meta["uses_base"]:
                base_atom_usage += 1
            if meta["uses_desc_atom"]:
                desc_atom_usage += 1

    probs, logits = compose_probs_from_semantic(
        problem["query_features"],
        semantic,
        problem["image_proto"],
        problem["num_base_cls"],
        problem["beta"],
        problem["ensemble_alpha"],
        problem["prob_cov"],
        problem["prob_knn"],
    )
    metrics = evaluate_probs(probs, logits, problem["query_targets"], list(range(problem["num_cls"])))

    baseline_probs, baseline_logits = compose_probs_from_semantic(
        problem["query_features"],
        problem["base_semantic"],
        problem["image_proto"],
        problem["num_base_cls"],
        problem["beta"],
        problem["ensemble_alpha"],
        problem["prob_cov"],
        problem["prob_knn"],
    )
    baseline_metrics = evaluate_probs(
        baseline_probs,
        baseline_logits,
        problem["query_targets"],
        list(range(problem["num_cls"])),
    )

    novel_slice = slice(problem["num_base_cls"], problem["num_cls"])
    baseline_alignment = 1.0 - torch.sum(problem["image_proto"][novel_slice] * problem["base_semantic"][novel_slice], dim=1)
    new_alignment = 1.0 - torch.sum(problem["image_proto"][novel_slice] * semantic[novel_slice], dim=1)

    return {
        "baseline_acc": round(float(baseline_metrics["acc"].item() * 100.0), 3),
        "method_acc": round(float(metrics["acc"].item() * 100.0), 3),
        "method_gain": round(float((metrics["acc"] - baseline_metrics["acc"]).item() * 100.0), 3),
        "test_class_balanced_nll": round(float(metrics["nll"].item()), 6),
        "test_class_balanced_margin": round(float(metrics["margin"].item()), 6),
        "changed_class_rate": round(float(len(changed_classes) / max(len(problem["novel_classes"]), 1)), 4),
        "base_atom_usage_rate": round(float(base_atom_usage / max(len(problem["novel_classes"]), 1)), 4),
        "desc_atom_usage_rate": round(float(desc_atom_usage / max(len(problem["novel_classes"]), 1)), 4),
        "baseline_alignment_gap": round(float(baseline_alignment.mean().item()), 6) if problem["novel_classes"] else 0.0,
        "method_alignment_gap": round(float(new_alignment.mean().item()), 6) if problem["novel_classes"] else 0.0,
        "chosen_candidates": chosen_candidates,
    }


def dataset_decision(mode, gain, recovery_rate):
    if recovery_rate is None:
        recovery_rate = 0.0
    if mode == "bmvc":
        return "continue_direction" if gain > 0.0 and recovery_rate >= 0.1 else "kill_direction"
    return "continue_direction" if gain > 0.0 and recovery_rate >= 0.25 else "kill_direction"


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
    max_desc_atoms,
    base_atom_topk,
    hidden_dim,
    lr,
    max_epochs,
    patience,
    prior_weight,
    margin_weight,
    entropy_weight,
    distill_weight,
    temperature,
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
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        seed=int(cfg.SEED),
        device=device,
    )
    episodes = attach_candidate_features_and_oracles(episodes)
    train_episodes, val_episodes = split_episodes(episodes, val_fraction=0.2)

    input_dim = int(train_episodes[0]["candidate_features"][train_episodes[0]["novel_classes"][0]].shape[1])
    if mode == "bmvc":
        scorer, train_info = train_bmvc_ranker(
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
        )
    else:
        scorer, train_info = train_astar_selector(
            train_episodes=train_episodes,
            val_episodes=val_episodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            prior_weight=prior_weight,
            margin_weight=margin_weight,
            entropy_weight=entropy_weight,
            distill_weight=distill_weight,
            temperature=temperature,
            device=device,
        )

    final_problem = build_final_problem(
        merged_state=merged_state,
        query_features=query_features,
        query_targets=query_targets,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        beta=beta,
        lambda_t=lambda_t,
        ensemble_alpha=ensemble_alpha,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
        device=device,
    )
    final_results = evaluate_selector(
        problem=final_problem,
        scorer=scorer,
        mode=mode,
        temperature=temperature,
    )

    oracle_gain, oracle_dir = latest_oracle_gain(cfg, train_cfg)
    recovery_rate = None if oracle_gain is None or abs(float(oracle_gain)) <= 1e-9 else round(float(final_results["method_gain"] / oracle_gain), 4)
    recommendation = dataset_decision(mode, final_results["method_gain"], recovery_rate)
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
        "ensemble_alpha": ensemble_alpha,
        "pseudo_episodes": pseudo_episodes,
        "pseudo_base_count": pseudo_base_count,
        "pseudo_novel_count": pseudo_novel_count,
        "pseudo_novel_shot": pseudo_novel_shot,
        "base_query_per_class": base_query_per_class,
        "novel_query_per_class": novel_query_per_class,
        "max_desc_atoms": max_desc_atoms,
        "base_atom_topk": base_atom_topk,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "max_epochs": max_epochs,
        "patience": patience,
        "prior_weight": prior_weight,
        "margin_weight": margin_weight,
        "entropy_weight": entropy_weight,
        "distill_weight": distill_weight,
        "temperature": temperature,
        "num_train_episodes": len(train_episodes),
        "num_val_episodes": len(val_episodes),
        "oracle_gain": oracle_gain,
        "oracle_results_dir": str(oracle_dir) if oracle_dir is not None else None,
        "recovery_rate": recovery_rate,
        "recommendation": recommendation,
        **train_info,
        **final_results,
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"mode: {mode}",
        f"runtime_sec: {runtime_sec}",
        f"baseline_acc: {final_results['baseline_acc']}",
        f"method_acc: {final_results['method_acc']}",
        f"method_gain: {final_results['method_gain']}",
        f"oracle_gain: {oracle_gain}",
        f"recovery_rate: {recovery_rate}",
        f"recommendation: {recommendation}",
    ]
    for key, value in train_info.items():
        summary_lines.append(f"{key}: {value}")
    summary_lines.extend([
        f"test_class_balanced_nll: {final_results['test_class_balanced_nll']}",
        f"test_class_balanced_margin: {final_results['test_class_balanced_margin']}",
        f"changed_class_rate: {final_results['changed_class_rate']}",
        f"base_atom_usage_rate: {final_results['base_atom_usage_rate']}",
        f"desc_atom_usage_rate: {final_results['desc_atom_usage_rate']}",
        f"baseline_alignment_gap: {final_results['baseline_alignment_gap']}",
        f"method_alignment_gap: {final_results['method_alignment_gap']}",
    ])
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "mode": mode,
        "baseline_acc": final_results["baseline_acc"],
        "method_acc": final_results["method_acc"],
        "method_gain": final_results["method_gain"],
        "oracle_gain": oracle_gain,
        "recovery_rate": recovery_rate,
        "recommendation": recommendation,
    }, indent=2))
    print(f"Saved semantic meta-selector results to: {out_dir}")
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
    parser.add_argument("--max_desc_atoms", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--prior_weight", type=float, default=0.1)
    parser.add_argument("--margin_weight", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--distill_weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "astar" and args.hidden_dim == 96:
        args.hidden_dim = 160
        args.lr = 0.0015
        args.max_epochs = max(args.max_epochs, 100)
        args.patience = max(args.patience, 12)
        args.distill_weight = max(args.distill_weight, 0.7)

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
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        prior_weight=args.prior_weight,
        margin_weight=args.margin_weight,
        entropy_weight=args.entropy_weight,
        distill_weight=args.distill_weight,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
