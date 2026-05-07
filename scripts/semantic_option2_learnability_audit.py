import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.data_manager import DatasetManager
from main import setup_cfg
from semantic_option2_killswitch import (
    build_model,
    build_semantic_banks,
    compose_probs_from_semantic,
    compute_cov_logits,
    compute_knn_logits,
    normalize_vector,
    to_builtin,
)
from utils.util import set_gpu, set_seed


def create_output_dir(cfg, train_cfg):
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(cfg.OUTPUT.ROOT)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    trainer_name = f"semantic_learnability_audit_{Path(train_cfg).stem}"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = output_root / cfg.DATASET.NAME / trainer_name / f"seed{cfg.SEED}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def shrink_cov(cov, alpha1=1.0, alpha2=0.0):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag * mask).sum() / torch.clamp(mask.sum(), min=1)
    iden = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))


def collect_loader_features(cfg, data_manager, model, task_id, source):
    loader = data_manager.get_dataloader(task_id, source=source, mode="test", accumulate_past=False)
    model_ref = model.module if hasattr(model, "module") else model
    feats = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(cfg.DEVICE.DEVICE_NAME)
            targets = batch["label"].to(cfg.DEVICE.DEVICE_NAME)
            image_features = model_ref.extract_img_feature(images).float()
            image_features = F.normalize(image_features, dim=-1)
            feats.append(image_features.cpu())
            labels.append(targets.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def group_by_label(features, targets):
    grouped = {}
    for cls_id in torch.unique(targets).tolist():
        mask = targets == int(cls_id)
        grouped[int(cls_id)] = features[mask].clone()
    return grouped


def extract_base_context(cfg):
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)

    data_manager = DatasetManager(cfg)
    model = build_model(cfg, data_manager)
    model.eval()

    base_task_id = 0
    base_classes = data_manager.class_index_in_task[base_task_id]
    class_names = np.array(data_manager.class_names)[base_classes]
    train_loader = data_manager.get_dataloader(base_task_id, source="train", mode="test", accumulate_past=False)
    base_state = model.build_task_statistics(
        class_names,
        train_loader,
        class_index=base_classes,
        calibrate_novel_vision_proto=False,
    )

    train_groups = group_by_label(
        F.normalize(base_state["images_features"].float().cpu(), dim=-1),
        base_state["images_targets"].long().cpu(),
    )
    test_features, test_targets = collect_loader_features(cfg, data_manager, model, base_task_id, source="test")
    test_groups = group_by_label(test_features, test_targets.long())
    desc_groups = group_by_label(
        F.normalize(base_state["description_features"].float().cpu(), dim=-1),
        base_state["description_targets"].long().cpu(),
    )

    return {
        "data_manager": data_manager,
        "model": model,
        "base_state": {
            "text_features": F.normalize(base_state["text_features"].float().cpu(), dim=-1),
            "description_proto": F.normalize(base_state["description_proto"].float().cpu(), dim=-1),
            "description_features": F.normalize(base_state["description_features"].float().cpu(), dim=-1),
            "description_targets": base_state["description_targets"].long().cpu(),
            "cov_image": base_state["cov_image"].float().cpu(),
            "train_groups": train_groups,
            "test_groups": test_groups,
            "desc_groups": desc_groups,
            "class_index": list(map(int, base_classes.tolist())),
        },
    }


def episode_rng(seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def sample_episode(base_state, pseudo_novel_classes, cfg, seed):
    all_base_classes = base_state["class_index"]
    pseudo_novel_set = set(int(x) for x in pseudo_novel_classes)
    pseudo_base_classes = [cls_id for cls_id in all_base_classes if cls_id not in pseudo_novel_set]
    episode_classes = pseudo_base_classes + list(pseudo_novel_classes)
    local_map = {orig_cls: idx for idx, orig_cls in enumerate(episode_classes)}
    num_base_cls = len(pseudo_base_classes)
    support_shot = int(cfg.DATASET.NUM_INC_SHOT)
    gamma_base = float(cfg.TRAINER.BiMC.GAMMA_BASE)
    rng = episode_rng(seed)

    image_proto = []
    cov_source = []
    text_features = []
    description_proto = []
    description_features = []
    description_targets = []
    query_features = []
    query_targets = []
    support_groups = {}
    query_class_sizes = {}

    for orig_cls in episode_classes:
        local_cls = local_map[orig_cls]
        train_feats = base_state["train_groups"][orig_cls]
        test_feats = base_state["test_groups"][orig_cls]
        desc_feats = base_state["desc_groups"][orig_cls]

        if orig_cls in pseudo_novel_set:
            if train_feats.shape[0] <= support_shot:
                raise RuntimeError(f"class {orig_cls} does not have enough train samples for pseudo-novel shot={support_shot}")
            perm = torch.randperm(train_feats.shape[0], generator=rng)
            support_idx = perm[:support_shot]
            proto_feats = train_feats[support_idx]
            support_groups[local_cls] = proto_feats.clone()
        else:
            proto_feats = train_feats

        image_proto.append(normalize_vector(proto_feats.mean(dim=0)))
        cov_source.append(proto_feats)
        text_features.append(base_state["text_features"][orig_cls])
        description_proto.append(base_state["description_proto"][orig_cls])
        description_features.append(desc_feats)
        description_targets.append(torch.full((desc_feats.shape[0],), local_cls, dtype=torch.long))
        query_features.append(test_feats)
        query_targets.append(torch.full((test_feats.shape[0],), local_cls, dtype=torch.long))
        query_class_sizes[local_cls] = int(test_feats.shape[0])

    cov_features = torch.cat(cov_source, dim=0)
    cov_image = torch.cov(cov_features.T)
    cov_image = shrink_cov(cov_image, alpha1=gamma_base)

    episode_state = {
        "image_proto": F.normalize(torch.stack(image_proto, dim=0), dim=-1),
        "text_features": F.normalize(torch.stack(text_features, dim=0), dim=-1),
        "description_proto": F.normalize(torch.stack(description_proto, dim=0), dim=-1),
        "description_features": F.normalize(torch.cat(description_features, dim=0), dim=-1),
        "description_targets": torch.cat(description_targets, dim=0),
        "cov_image": cov_image,
    }
    return {
        "state": episode_state,
        "query_features": torch.cat(query_features, dim=0),
        "query_targets": torch.cat(query_targets, dim=0),
        "support_groups": support_groups,
        "query_class_sizes": query_class_sizes,
        "local_map": local_map,
        "orig_classes": episode_classes,
        "pseudo_novel_local": [local_map[cls_id] for cls_id in pseudo_novel_classes],
        "num_base_cls": num_base_cls,
    }


def build_eval_cache(features, state, baseline_sem, num_base_cls, beta, ensemble_alpha):
    image_proto = state["image_proto"]
    description_features = state["description_features"]
    description_targets = state["description_targets"]
    cov_image = state["cov_image"]

    logits_cov = compute_cov_logits(features, image_proto, cov_image)
    prob_cov = F.softmax(logits_cov / 512.0, dim=-1)
    logits_knn = compute_knn_logits(features, description_features, description_targets, baseline_sem.shape[0])
    prob_knn = F.softmax(logits_knn, dim=-1)

    fused_proto = F.normalize(beta * baseline_sem + (1.0 - beta) * image_proto, dim=-1)
    fused_logits = torch.matmul(features, fused_proto.T)
    fused_logits = fused_logits.float()
    fused_exp = torch.exp(fused_logits)
    fused_z = fused_exp.sum(dim=1)
    fused_probs = fused_exp / fused_z.unsqueeze(1)

    if ensemble_alpha >= 1.0:
        final_probs = fused_probs
    else:
        base_probs = ensemble_alpha * fused_probs[:, :num_base_cls] + (1.0 - ensemble_alpha) * prob_cov[:, :num_base_cls]
        inc_probs = ensemble_alpha * fused_probs[:, num_base_cls:] + (1.0 - ensemble_alpha) * prob_knn[:, num_base_cls:]
        final_probs = torch.cat([base_probs, inc_probs], dim=1)

    return {
        "features": features,
        "fused_logits": fused_logits,
        "fused_probs": fused_probs,
        "fused_z": fused_z,
        "prob_cov": prob_cov,
        "prob_knn": prob_knn,
        "final_probs": final_probs,
    }


def update_final_probs_from_candidate(cache, candidate_sem, class_idx, image_proto, beta, num_base_cls, ensemble_alpha):
    candidate_fused = normalize_vector(beta * candidate_sem + (1.0 - beta) * image_proto[class_idx])
    new_logit = torch.matmul(cache["features"], candidate_fused)
    old_logit = cache["fused_logits"][:, class_idx]
    old_exp = torch.exp(old_logit)
    new_exp = torch.exp(new_logit)
    new_z = torch.clamp(cache["fused_z"] - old_exp + new_exp, min=1e-12)
    scale = cache["fused_z"] / new_z
    fused_probs = cache["fused_probs"] * scale.unsqueeze(1)
    fused_probs[:, class_idx] = new_exp / new_z

    if ensemble_alpha >= 1.0:
        return fused_probs

    base_probs = ensemble_alpha * fused_probs[:, :num_base_cls] + (1.0 - ensemble_alpha) * cache["prob_cov"][:, :num_base_cls]
    inc_probs = ensemble_alpha * fused_probs[:, num_base_cls:] + (1.0 - ensemble_alpha) * cache["prob_knn"][:, num_base_cls:]
    return torch.cat([base_probs, inc_probs], dim=1)


def true_prob_and_margin(probs, cls_id):
    true_prob = probs[:, cls_id]
    wrong = probs.clone()
    wrong[:, cls_id] = 0.0
    conf_prob = wrong.max(dim=1).values
    margin = true_prob - conf_prob
    return float(true_prob.mean().item()), float(margin.mean().item()), int(torch.argmax(probs.mean(dim=0)).item())


def semantic_feature_vector(
    candidate,
    candidate_vec,
    cls_id,
    support_features,
    baseline_sem,
    state,
    baseline_support_probs,
    candidate_support_probs,
    num_base_cls,
):
    support_proto = normalize_vector(support_features.mean(dim=0))
    support_dispersion = float(torch.sum(support_features * support_proto.unsqueeze(0), dim=1).mean().item())

    baseline_true_prob, baseline_margin, _ = true_prob_and_margin(baseline_support_probs, cls_id)
    candidate_true_prob, candidate_margin, _ = true_prob_and_margin(candidate_support_probs, cls_id)

    support_mean_scores = baseline_support_probs.mean(dim=0)
    conf_scores = support_mean_scores.clone()
    conf_scores[cls_id] = float("-inf")
    top_confuser = int(torch.argmax(conf_scores).item())

    candidate_vec = normalize_vector(candidate_vec)
    baseline_vec = baseline_sem[cls_id]
    image_vec = state["image_proto"][cls_id]
    text_vec = state["text_features"][cls_id]
    desc_vec = state["description_proto"][cls_id]

    if num_base_cls > 0:
        base_bank = baseline_sem[:num_base_cls]
        max_base_sim = float(torch.matmul(base_bank, candidate_vec).max().item())
    else:
        max_base_sim = 0.0

    other_mask = torch.ones(baseline_sem.shape[0], dtype=torch.bool, device=baseline_sem.device)
    other_mask[cls_id] = False
    max_other_sim = float(torch.matmul(baseline_sem[other_mask], candidate_vec).max().item()) if other_mask.any() else 0.0

    baseline_confuser_sim = float(torch.dot(baseline_vec, baseline_sem[top_confuser]).item())
    candidate_confuser_sim = float(torch.dot(candidate_vec, baseline_sem[top_confuser]).item())

    return [
        float(torch.dot(candidate_vec, support_proto).item()),
        float(torch.dot(baseline_vec, support_proto).item()),
        float(torch.dot(candidate_vec, image_vec).item()),
        float(torch.dot(baseline_vec, image_vec).item()),
        float(torch.dot(candidate_vec, text_vec).item()),
        float(torch.dot(candidate_vec, desc_vec).item()),
        float(torch.dot(candidate_vec, baseline_vec).item()),
        support_dispersion,
        baseline_true_prob,
        candidate_true_prob,
        candidate_true_prob - baseline_true_prob,
        baseline_margin,
        candidate_margin,
        candidate_margin - baseline_margin,
        max_base_sim,
        max_other_sim,
        baseline_confuser_sim,
        candidate_confuser_sim,
        float(candidate["uses_base"]),
        float(candidate["uses_desc_atom"]),
        float(candidate["name"] == "prompt_only"),
        float(candidate["name"] == "desc_mean_only"),
        float(("plus" in candidate["name"]) or candidate["name"].startswith("prompt_desc")),
    ]


def label_from_deltas(delta_episode_correct, delta_class_correct):
    if delta_episode_correct >= 1 and delta_class_correct >= 1:
        return 1
    if delta_episode_correct <= -1 or delta_class_correct <= -1:
        return 0
    return -1


def compute_candidate_records(
    episode_id,
    split_name,
    episode,
    cfg,
    max_desc_atoms,
    base_atom_topk,
):
    device = torch.device(cfg.DEVICE.DEVICE_NAME)
    state = {k: v.to(device) for k, v in episode["state"].items()}
    query_features = episode["query_features"].to(device)
    query_targets = episode["query_targets"].to(device)
    num_cls = len(episode["orig_classes"])
    num_base_cls = episode["num_base_cls"]
    beta = float(cfg.DATASET.BETA)
    lambda_t = float(cfg.TRAINER.BiMC.LAMBDA_T if cfg.TRAINER.BiMC.TEXT_CALIBRATION else 0.0)
    ensemble_alpha = float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0)

    baseline_sem, banks = build_semantic_banks(
        state,
        num_cls=num_cls,
        num_base_cls=num_base_cls,
        lambda_t=lambda_t,
        max_desc_atoms=max_desc_atoms,
        base_atom_topk=base_atom_topk,
    )
    baseline_sem = baseline_sem.to(device)
    query_cache = build_eval_cache(query_features, state, baseline_sem, num_base_cls, beta, ensemble_alpha)
    baseline_query_probs = query_cache["final_probs"]
    baseline_query_preds = torch.argmax(baseline_query_probs, dim=1)
    baseline_correct = baseline_query_preds.eq(query_targets)
    baseline_correct_count = int(baseline_correct.sum().item())
    baseline_acc = float(baseline_correct.float().mean().item() * 100.0)

    candidate_map = {}
    records = []

    for local_cls in episode["pseudo_novel_local"]:
        support_features = episode["support_groups"][local_cls].to(device)
        support_cache = build_eval_cache(support_features, state, baseline_sem, num_base_cls, beta, ensemble_alpha)
        baseline_support_probs = support_cache["final_probs"]
        candidate_map[local_cls] = {}

        class_mask = query_targets == local_cls
        baseline_class_correct = baseline_correct[class_mask]
        baseline_class_correct_count = int(baseline_class_correct.sum().item())

        for candidate in banks[local_cls]:
            if candidate["is_baseline"]:
                candidate_map[local_cls][candidate["name"]] = candidate["vector"].detach().cpu()
                continue

            candidate_vec = candidate["vector"].to(device)
            candidate_query_probs = update_final_probs_from_candidate(
                query_cache,
                candidate_vec,
                local_cls,
                state["image_proto"],
                beta,
                num_base_cls,
                ensemble_alpha,
            )
            candidate_query_preds = torch.argmax(candidate_query_probs, dim=1)
            candidate_query_correct = candidate_query_preds.eq(query_targets)
            candidate_correct_count = int(candidate_query_correct.sum().item())
            delta_episode_correct = candidate_correct_count - baseline_correct_count

            candidate_class_correct_count = int(candidate_query_correct[class_mask].sum().item())
            delta_class_correct = candidate_class_correct_count - baseline_class_correct_count
            delta_episode_acc = 100.0 * float(delta_episode_correct) / max(int(query_targets.numel()), 1)
            delta_class_acc = 100.0 * float(delta_class_correct) / max(int(class_mask.sum().item()), 1)

            candidate_support_probs = update_final_probs_from_candidate(
                support_cache,
                candidate_vec,
                local_cls,
                state["image_proto"],
                beta,
                num_base_cls,
                ensemble_alpha,
            )
            features = semantic_feature_vector(
                candidate=candidate,
                candidate_vec=candidate_vec,
                cls_id=local_cls,
                support_features=support_features,
                baseline_sem=baseline_sem,
                state=state,
                baseline_support_probs=baseline_support_probs,
                candidate_support_probs=candidate_support_probs,
                num_base_cls=num_base_cls,
            )
            label = label_from_deltas(delta_episode_correct, delta_class_correct)
            sample_weight = float(max(1.0, abs(delta_episode_correct) + abs(delta_class_correct)))

            record = {
                "episode_id": episode_id,
                "split": split_name,
                "orig_class": int(episode["orig_classes"][local_cls]),
                "local_class": int(local_cls),
                "candidate_name": candidate["name"],
                "label": int(label),
                "beneficial": bool(label == 1),
                "delta_episode_correct": int(delta_episode_correct),
                "delta_class_correct": int(delta_class_correct),
                "delta_episode_acc": float(delta_episode_acc),
                "delta_class_acc": float(delta_class_acc),
                "sample_weight": sample_weight,
                "features": features,
            }
            records.append(record)
            candidate_map[local_cls][candidate["name"]] = candidate["vector"].detach().cpu()

    eval_payload = {
        "episode_id": episode_id,
        "split": split_name,
        "baseline_acc": baseline_acc,
        "query_features": episode["query_features"].clone(),
        "query_targets": episode["query_targets"].clone(),
        "num_base_cls": episode["num_base_cls"],
        "image_proto": episode["state"]["image_proto"].clone(),
        "baseline_sem": baseline_sem.detach().cpu(),
        "prob_cov": query_cache["prob_cov"].detach().cpu(),
        "prob_knn": query_cache["prob_knn"].detach().cpu(),
        "candidate_map": candidate_map,
        "num_novel": len(episode["pseudo_novel_local"]),
    }
    return records, eval_payload


class LinearProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def prepare_arrays(records):
    x = np.array([rec["features"] for rec in records], dtype=np.float32)
    y = np.array([rec["label"] for rec in records], dtype=np.int64)
    w = np.array([rec["sample_weight"] for rec in records], dtype=np.float32)
    gain = np.array([rec["delta_episode_acc"] for rec in records], dtype=np.float32)
    return x, y, w, gain


def filter_labeled(records):
    return [rec for rec in records if rec["label"] in (0, 1)]


def standardize(train_x, other_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


def fit_probe(train_records, val_records, max_epochs=300, lr=0.05, weight_decay=1e-4):
    train_x, train_y, train_w, _ = prepare_arrays(train_records)
    val_x, val_y, val_w, _ = prepare_arrays(val_records)
    train_x, val_x, mean, std = standardize(train_x, val_x)

    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    w_train = torch.tensor(train_w, dtype=torch.float32)
    x_val = torch.tensor(val_x, dtype=torch.float32)
    y_val = torch.tensor(val_y, dtype=torch.float32)
    w_val = torch.tensor(val_w, dtype=torch.float32)

    num_pos = float(max(1, int((train_y == 1).sum())))
    num_neg = float(max(1, int((train_y == 0).sum())))
    pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32)

    model = LinearProbe(dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = None
    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        per_sample = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none", pos_weight=pos_weight)
        loss = (per_sample * w_train).mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val, reduction="none", pos_weight=pos_weight)
            val_loss = float((val_loss * w_val).mean().item())
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "val_loss": best_val,
        "pos_rate_train": float((train_y == 1).mean()),
    }


def predict_scores(model_bundle, records):
    x, _, _, gains = prepare_arrays(records)
    x = (x - model_bundle["mean"]) / model_bundle["std"]
    with torch.no_grad():
        logits = model_bundle["model"](torch.tensor(x, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs.astype(np.float32), gains


def pearson_corr(x, y):
    if len(x) < 2:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def precision_at_fraction(scores, labels, frac):
    n = len(scores)
    if n == 0:
        return 0.0, 0
    k = max(1, int(round(n * frac)))
    order = np.argsort(scores)[::-1][:k]
    chosen = labels[order]
    return float((chosen == 1).mean()), int(k)


def group_best_candidates(records, probs):
    grouped = defaultdict(list)
    for rec, prob in zip(records, probs):
        grouped[(rec["episode_id"], rec["local_class"])].append((prob, rec))
    best = {}
    for key, items in grouped.items():
        items.sort(key=lambda item: item[0], reverse=True)
        best[key] = items[0]
    return best


def apply_episode_selection(payload, selected_map, cfg):
    device = torch.device(cfg.DEVICE.DEVICE_NAME)
    query_features = payload["query_features"].to(device)
    query_targets = payload["query_targets"].to(device)
    baseline_sem = payload["baseline_sem"].to(device)
    semantic = baseline_sem.clone()
    for local_cls, candidate_name in selected_map.items():
        semantic[local_cls] = payload["candidate_map"][local_cls][candidate_name].to(device)
    probs, _ = compose_probs_from_semantic(
        query_features,
        semantic,
        payload["image_proto"].to(device),
        payload["num_base_cls"],
        float(cfg.DATASET.BETA),
        float(cfg.DATASET.ENSEMBLE_ALPHA if cfg.TRAINER.BiMC.USING_ENSEMBLE else 1.0),
        payload["prob_cov"].to(device),
        payload["prob_knn"].to(device),
    )
    acc = float((torch.argmax(probs, dim=1) == query_targets).float().mean().item() * 100.0)
    return acc


def oracle_top1_gains(records):
    gains = defaultdict(float)
    for rec in records:
        gains[rec["episode_id"]] = max(gains[rec["episode_id"]], float(rec["delta_episode_acc"]))
    return gains


def simulate_policy(eval_payloads, best_candidates, threshold, mode, cfg, oracle_gains):
    total_classes = 0
    switched_classes = 0
    beneficial_switches = 0
    episode_gains = []
    oracle_top1_values = []

    for episode_id, payload in eval_payloads.items():
        episode_items = []
        for key, (prob, rec) in best_candidates.items():
            rec_episode_id, _ = key
            if rec_episode_id != episode_id:
                continue
            if prob >= threshold:
                episode_items.append((prob, rec))

        total_classes += payload["num_novel"]
        selected = {}
        if mode == "class_local":
            for prob, rec in episode_items:
                selected[rec["local_class"]] = rec["candidate_name"]
        elif mode == "session_top1" and episode_items:
            episode_items.sort(key=lambda item: item[0], reverse=True)
            selected[episode_items[0][1]["local_class"]] = episode_items[0][1]["candidate_name"]
        else:
            selected = {}

        switched_classes += len(selected)
        for _, rec in episode_items:
            pass
        for local_cls, candidate_name in selected.items():
            _, rec = best_candidates[(episode_id, local_cls)]
            beneficial_switches += int(rec["beneficial"])

        new_acc = payload["baseline_acc"] if not selected else apply_episode_selection(payload, selected, cfg)
        episode_gains.append(float(new_acc - payload["baseline_acc"]))

        oracle_top1_values.append(float(oracle_gains.get(episode_id, 0.0)))

    return {
        "threshold": float(threshold),
        "mode": mode,
        "switched_classes": int(switched_classes),
        "switched_precision": round(float(beneficial_switches / max(switched_classes, 1)), 4),
        "changed_class_rate": round(float(switched_classes / max(total_classes, 1)), 4),
        "mean_episode_gain": round(float(np.mean(episode_gains) if episode_gains else 0.0), 4),
        "oracle_top1_mean_gain": round(float(np.mean(oracle_top1_values) if oracle_top1_values else 0.0), 4),
    }


def choose_threshold(eval_payloads, best_candidates, cfg, oracle_gains, target_precision=0.75):
    thresholds = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    picks = {}
    for mode in ("class_local", "session_top1"):
        chosen = None
        best_gain = None
        for threshold in thresholds:
            stats = simulate_policy(eval_payloads, best_candidates, threshold, mode, cfg, oracle_gains)
            if stats["switched_classes"] == 0:
                continue
            if stats["switched_precision"] < target_precision:
                continue
            if best_gain is None or stats["mean_episode_gain"] > best_gain:
                best_gain = stats["mean_episode_gain"]
                chosen = stats
        if chosen is None:
            chosen = {
                "threshold": 1.1,
                "mode": mode,
                "switched_classes": 0,
                "switched_precision": 0.0,
                "changed_class_rate": 0.0,
                "mean_episode_gain": 0.0,
                "oracle_top1_mean_gain": round(float(np.mean(list(oracle_gains.values())) if oracle_gains else 0.0), 4),
            }
        picks[mode] = chosen
    return picks


def summarise_split(records):
    labels = [rec["label"] for rec in records]
    return {
        "num_candidates": len(records),
        "num_positive": int(sum(label == 1 for label in labels)),
        "num_negative": int(sum(label == 0 for label in labels)),
        "num_neutral": int(sum(label == -1 for label in labels)),
    }


def run_dataset(
    data_cfg,
    train_cfg,
    episodes_train,
    episodes_val,
    episodes_test,
    max_desc_atoms,
    base_atom_topk,
):
    cfg = setup_cfg(data_cfg, train_cfg)
    out_dir = create_output_dir(cfg, train_cfg)
    (out_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")

    start_time = time.time()
    context = extract_base_context(cfg)
    base_state = context["base_state"]

    base_classes = list(base_state["class_index"])
    split_rng = np.random.RandomState(int(cfg.SEED))
    shuffled = base_classes.copy()
    split_rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(int(round(n_total * 0.7)), int(cfg.DATASET.NUM_INC_CLS))
    n_val = max(int(round(n_total * 0.15)), int(cfg.DATASET.NUM_INC_CLS))
    train_classes = shuffled[:n_train]
    val_classes = shuffled[n_train:n_train + n_val]
    test_classes = shuffled[n_train + n_val:]
    if len(test_classes) < int(cfg.DATASET.NUM_INC_CLS):
        test_classes = shuffled[-max(int(cfg.DATASET.NUM_INC_CLS), len(test_classes)):]

    split_specs = {
        "train": (train_classes, episodes_train),
        "val": (val_classes, episodes_val),
        "test": (test_classes, episodes_test),
    }

    all_records = {"train": [], "val": [], "test": []}
    eval_payloads = {"val": {}, "test": {}}

    episode_counter = 0
    for split_name, (candidate_classes, num_episodes) in split_specs.items():
        if len(candidate_classes) < int(cfg.DATASET.NUM_INC_CLS):
            raise RuntimeError(f"{split_name} split has too few classes for pseudo-novel episodes")
        for split_episode_id in range(num_episodes):
            novel_choices = split_rng.choice(
                np.array(candidate_classes),
                size=int(cfg.DATASET.NUM_INC_CLS),
                replace=False,
            ).tolist()
            episode = sample_episode(
                base_state=base_state,
                pseudo_novel_classes=novel_choices,
                cfg=cfg,
                seed=int(cfg.SEED) + 10007 * (episode_counter + 1),
            )
            episode_id = f"{split_name}_{split_episode_id}"
            records, payload = compute_candidate_records(
                episode_id=episode_id,
                split_name=split_name,
                episode=episode,
                cfg=cfg,
                max_desc_atoms=max_desc_atoms,
                base_atom_topk=base_atom_topk,
            )
            all_records[split_name].extend(records)
            if split_name in eval_payloads:
                eval_payloads[split_name][episode_id] = payload
            episode_counter += 1

    labeled_train = filter_labeled(all_records["train"])
    labeled_val = filter_labeled(all_records["val"])
    labeled_test = filter_labeled(all_records["test"])
    if not labeled_train or not labeled_val or not labeled_test:
        raise RuntimeError("insufficient labeled candidate instances for learnability audit")

    model_bundle = fit_probe(labeled_train, labeled_val)
    val_probs, _ = predict_scores(model_bundle, all_records["val"])
    test_probs, test_gains = predict_scores(model_bundle, all_records["test"])

    val_best = group_best_candidates(all_records["val"], val_probs)
    test_best = group_best_candidates(all_records["test"], test_probs)
    val_oracle = oracle_top1_gains(all_records["val"])
    test_oracle = oracle_top1_gains(all_records["test"])
    thresholds = choose_threshold(eval_payloads["val"], val_best, cfg, val_oracle)

    test_labels_all = np.array([rec["label"] for rec in all_records["test"]], dtype=np.int64)
    labeled_mask = test_labels_all >= 0
    labeled_scores = test_probs[labeled_mask]
    labeled_flags = test_labels_all[labeled_mask]

    candidate_metrics = {
        "test_pearson_gain": round(pearson_corr(test_probs, test_gains), 4),
        "test_precision_at_10pct": round(precision_at_fraction(labeled_scores, labeled_flags, 0.10)[0], 4),
        "test_precision_at_20pct": round(precision_at_fraction(labeled_scores, labeled_flags, 0.20)[0], 4),
    }

    class_local_test = simulate_policy(
        eval_payloads["test"],
        test_best,
        thresholds["class_local"]["threshold"],
        "class_local",
        cfg,
        test_oracle,
    )
    session_top1_test = simulate_policy(
        eval_payloads["test"],
        test_best,
        thresholds["session_top1"]["threshold"],
        "session_top1",
        cfg,
        test_oracle,
    )

    runtime_sec = round(time.time() - start_time, 3)
    payload = {
        "dataset": cfg.DATASET.NAME,
        "train_cfg": train_cfg,
        "runtime_sec": runtime_sec,
        "episodes": {
            "train": int(episodes_train),
            "val": int(episodes_val),
            "test": int(episodes_test),
        },
        "num_base_classes": len(base_classes),
        "pseudo_novel_per_episode": int(cfg.DATASET.NUM_INC_CLS),
        "split_classes": {
            "train": len(train_classes),
            "val": len(val_classes),
            "test": len(test_classes),
        },
        "split_summary": {
            "train": summarise_split(all_records["train"]),
            "val": summarise_split(all_records["val"]),
            "test": summarise_split(all_records["test"]),
        },
        "probe": {
            "val_loss": round(float(model_bundle["val_loss"]), 6),
            "train_positive_rate": round(float(model_bundle["pos_rate_train"]), 4),
            **candidate_metrics,
        },
        "thresholds": thresholds,
        "test_policy": {
            "class_local": class_local_test,
            "session_top1": session_top1_test,
        },
    }
    payload = to_builtin(payload)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        f"dataset: {cfg.DATASET.NAME}",
        f"train_cfg: {train_cfg}",
        f"runtime_sec: {runtime_sec}",
        f"test_pearson_gain: {payload['probe']['test_pearson_gain']}",
        f"test_precision_at_10pct: {payload['probe']['test_precision_at_10pct']}",
        f"test_precision_at_20pct: {payload['probe']['test_precision_at_20pct']}",
        f"class_local_threshold: {payload['thresholds']['class_local']['threshold']}",
        f"class_local_gain: {payload['test_policy']['class_local']['mean_episode_gain']}",
        f"class_local_precision: {payload['test_policy']['class_local']['switched_precision']}",
        f"class_local_changed_rate: {payload['test_policy']['class_local']['changed_class_rate']}",
        f"session_top1_threshold: {payload['thresholds']['session_top1']['threshold']}",
        f"session_top1_gain: {payload['test_policy']['session_top1']['mean_episode_gain']}",
        f"session_top1_precision: {payload['test_policy']['session_top1']['switched_precision']}",
        f"session_top1_changed_rate: {payload['test_policy']['session_top1']['changed_class_rate']}",
        f"oracle_top1_mean_gain: {payload['test_policy']['session_top1']['oracle_top1_mean_gain']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": cfg.DATASET.NAME,
        "test_pearson_gain": payload["probe"]["test_pearson_gain"],
        "test_precision_at_10pct": payload["probe"]["test_precision_at_10pct"],
        "class_local_gain": payload["test_policy"]["class_local"]["mean_episode_gain"],
        "class_local_precision": payload["test_policy"]["class_local"]["switched_precision"],
        "session_top1_gain": payload["test_policy"]["session_top1"]["mean_episode_gain"],
        "session_top1_precision": payload["test_policy"]["session_top1"]["switched_precision"],
    }, indent=2))
    print(f"Saved semantic learnability audit to: {out_dir}")
    return out_dir, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", required=True)
    parser.add_argument("--train_cfg", required=True)
    parser.add_argument("--episodes_train", type=int, default=60)
    parser.add_argument("--episodes_val", type=int, default=15)
    parser.add_argument("--episodes_test", type=int, default=15)
    parser.add_argument("--max_desc_atoms", type=int, default=3)
    parser.add_argument("--base_atom_topk", type=int, default=2)
    args = parser.parse_args()

    run_dataset(
        data_cfg=args.data_cfg,
        train_cfg=args.train_cfg,
        episodes_train=args.episodes_train,
        episodes_val=args.episodes_val,
        episodes_test=args.episodes_test,
        max_desc_atoms=args.max_desc_atoms,
        base_atom_topk=args.base_atom_topk,
    )


if __name__ == "__main__":
    main()
