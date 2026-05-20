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
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.query_branch_router import normalized_entropy, top_prob_margin
from prototype_refinement_conservative_dino_fusion_sessionwise import pseudo_validation_split
from tric_followup_variant_suite import (
    accuracy_stats_from_probs,
    build_final_problem,
    build_subspace_branch,
    compute_global_qpr,
    conservative_router,
    dataset_output_name,
    selective_semantic_repair,
    selective_subspace_repair,
)


DEFAULT_DATA_CFGS = [
    "configs/datasets/cifar100.yaml",
    "configs/datasets/miniimagenet.yaml",
    "configs/datasets/cub200_bimc_dino_fusion.yaml",
]
DEFAULT_TRAIN_CFG = "configs/trainers/bimc_dino_fusion.yaml"
DEFAULT_ALPHA_GRID = [0.5, 0.75, 1.0]
ALT_BRANCHES = ["covariance", "visual", "subspace"]


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_normalize_probs(probs):
    return probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-12)


def pct(mask):
    return round(float(mask.float().mean().item() * 100.0), 3) if mask.numel() else 0.0


def make_output_root(base_output_root):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(base_output_root) / f"tric_router_reliability_suite_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_class_session_map(problem):
    mapping = {}
    for session_idx, cls_list in enumerate(problem["data_manager"].class_index_in_task):
        for cls_id in cls_list:
            mapping[int(cls_id)] = int(session_idx)
    return mapping


def qpr_gate_stats(problem, baseline_probs, mass_thr, min_count):
    num_base_cls = int(problem["num_base_cls"])
    num_cls = int(problem["num_cls"])
    preds = torch.argmax(baseline_probs, dim=1)
    novel_probs = baseline_probs[:, num_base_cls:]
    novel_mass = novel_probs.sum(dim=1)
    top_novel_local = torch.argmax(novel_probs, dim=1)
    top_novel_global = top_novel_local + num_base_cls
    gate = (preds >= num_base_cls) & (novel_mass >= float(mass_thr))

    stats = {}
    for cls_id in range(num_base_cls, num_cls):
        cls_mask = gate & top_novel_global.eq(int(cls_id))
        count = int(cls_mask.sum().item())
        if count == 0:
            purity = 0.0
        else:
            purity = float(problem["query_targets"][cls_mask].eq(int(cls_id)).float().mean().item())
        stats[int(cls_id)] = {
            "count": count,
            "purity": purity,
            "eligible": bool(count >= int(min_count)),
        }
    total_eligible = sum(1 for payload in stats.values() if payload["eligible"])
    return {
        "by_class": stats,
        "gated_query_count": int(gate.sum().item()),
        "eligible_class_count": int(total_eligible),
    }


def topk_overlap_size(probs_a, probs_b, k):
    idx_a = torch.topk(probs_a, k=min(int(k), probs_a.shape[1]), dim=1).indices
    idx_b = torch.topk(probs_b, k=min(int(k), probs_b.shape[1]), dim=1).indices
    overlap = []
    for row_a, row_b in zip(idx_a.tolist(), idx_b.tolist()):
        overlap.append(len(set(row_a).intersection(row_b)))
    return torch.tensor(overlap, device=probs_a.device, dtype=probs_a.dtype)


def chunked_support_density(query_features, support_features, support_targets, pred_classes, topk_all=5, topk_cls=5, chunk_size=512):
    device = query_features.device
    dtype = query_features.dtype
    support_features = support_features.to(device=device, dtype=dtype)
    support_targets = support_targets.to(device=device)
    pred_classes = pred_classes.to(device=device)
    unique_classes = sorted(set(int(x) for x in support_targets.detach().cpu().tolist()))
    support_by_class = {cls_id: support_features[support_targets.eq(int(cls_id))] for cls_id in unique_classes}

    n = int(query_features.shape[0])
    global_top1 = torch.zeros(n, device=device, dtype=dtype)
    global_topk_mean = torch.zeros(n, device=device, dtype=dtype)
    class_top1 = torch.zeros(n, device=device, dtype=dtype)
    class_topk_mean = torch.zeros(n, device=device, dtype=dtype)

    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        q = query_features[start:end]
        sim_all = torch.matmul(q, support_features.T)
        vals = torch.topk(sim_all, k=min(int(topk_all), sim_all.shape[1]), dim=1).values
        global_top1[start:end] = vals[:, 0]
        global_topk_mean[start:end] = vals.mean(dim=1)

        pred_chunk = pred_classes[start:end]
        for cls_id in torch.unique(pred_chunk).tolist():
            cls_id = int(cls_id)
            cls_mask = pred_chunk.eq(cls_id)
            cls_support = support_by_class.get(cls_id)
            if cls_support is None or cls_support.numel() == 0:
                continue
            cls_q = q[cls_mask]
            cls_sim = torch.matmul(cls_q, cls_support.T)
            cls_vals = torch.topk(cls_sim, k=min(int(topk_cls), cls_sim.shape[1]), dim=1).values
            class_top1[start:end][cls_mask] = cls_vals[:, 0]
            class_topk_mean[start:end][cls_mask] = cls_vals.mean(dim=1)

    return {
        "support_global_top1": global_top1,
        "support_global_topk_mean": global_topk_mean,
        "support_predcls_top1": class_top1,
        "support_predcls_topk_mean": class_topk_mean,
    }


def build_router_features(problem, tric_pv, subspace_probs, gate_stats, support_topk):
    default_probs = tric_pv["full_probs"]
    cov_probs = tric_pv["cov_probs"]
    visual_probs = tric_pv["visual_probs"]
    text_probs = problem["base_probs"]["semantic_branch_only"]
    knn_probs = problem["base_probs"]["description_knn_branch"]

    branch_probs = {
        "default": default_probs,
        "covariance": cov_probs,
        "visual": visual_probs,
        "subspace": subspace_probs,
        "text": text_probs,
        "knn": knn_probs,
    }
    branch_names = list(branch_probs.keys())

    pred = {}
    top1 = {}
    margin = {}
    entropy = {}
    base_mass = {}
    for name, probs in branch_probs.items():
        pred[name] = torch.argmax(probs, dim=1)
        top1[name], margin[name] = top_prob_margin(probs)
        entropy[name] = normalized_entropy(probs)
        base_mass[name] = probs[:, : problem["num_base_cls"]].sum(dim=1)

    class_session = build_class_session_map(problem)
    max_session = max(class_session.values()) if class_session else 1
    session_lookup = torch.zeros(problem["num_cls"], device=default_probs.device, dtype=default_probs.dtype)
    for cls_id, session_idx in class_session.items():
        session_lookup[int(cls_id)] = float(session_idx) / max(float(max_session), 1.0)

    dino_proto = F.normalize(tric_pv["dino_image_proto"].float(), dim=-1)
    dino_sim = torch.matmul(problem["dino_query"].float(), dino_proto.T)
    dino_vals, _ = torch.topk(dino_sim, k=min(2, dino_sim.shape[1]), dim=1)
    dino_top1_gap = dino_vals[:, 0] - dino_vals[:, 1] if dino_vals.shape[1] > 1 else dino_vals[:, 0]

    default_support_density = chunked_support_density(
        query_features=problem["dino_query"].float(),
        support_features=problem["dino_support_features"].float(),
        support_targets=problem["dino_support_targets"],
        pred_classes=pred["default"],
        topk_all=support_topk,
        topk_cls=support_topk,
    )

    qpr_purity = torch.zeros_like(top1["default"])
    qpr_log_count = torch.zeros_like(top1["default"])
    qpr_eligible = torch.zeros_like(top1["default"])
    for cls_id, payload in gate_stats["by_class"].items():
        mask = pred["default"].eq(int(cls_id))
        if not mask.any():
            continue
        qpr_purity[mask] = float(payload["purity"])
        qpr_log_count[mask] = float(torch.log1p(torch.tensor(float(payload["count"]))).item())
        qpr_eligible[mask] = 1.0 if payload["eligible"] else 0.0

    overlap3 = {
        "covariance": topk_overlap_size(default_probs, cov_probs, k=3),
        "visual": topk_overlap_size(default_probs, visual_probs, k=3),
        "subspace": topk_overlap_size(default_probs, subspace_probs, k=3),
        "text": topk_overlap_size(default_probs, text_probs, k=3),
        "knn": topk_overlap_size(default_probs, knn_probs, k=3),
    }
    disagreement_count = (
        torch.stack(
            [
                pred["default"],
                pred["covariance"],
                pred["visual"],
                pred["subspace"],
            ],
            dim=1,
        )
        .detach()
        .cpu()
    )
    unique_count = torch.tensor(
        [len(set(row.tolist())) for row in disagreement_count],
        device=default_probs.device,
        dtype=default_probs.dtype,
    )

    feature_map = {
        "default_top1": top1["default"],
        "default_margin": margin["default"],
        "default_entropy": entropy["default"],
        "default_base_mass": base_mass["default"],
        "cov_top1": top1["covariance"],
        "cov_margin": margin["covariance"],
        "cov_entropy": entropy["covariance"],
        "cov_base_mass": base_mass["covariance"],
        "visual_top1": top1["visual"],
        "visual_margin": margin["visual"],
        "visual_entropy": entropy["visual"],
        "visual_base_mass": base_mass["visual"],
        "sub_top1": top1["subspace"],
        "sub_margin": margin["subspace"],
        "sub_entropy": entropy["subspace"],
        "sub_base_mass": base_mass["subspace"],
        "text_top1": top1["text"],
        "text_margin": margin["text"],
        "text_entropy": entropy["text"],
        "text_base_mass": base_mass["text"],
        "knn_top1": top1["knn"],
        "knn_margin": margin["knn"],
        "knn_entropy": entropy["knn"],
        "knn_base_mass": base_mass["knn"],
        "agree_default_cov": pred["default"].eq(pred["covariance"]).float(),
        "agree_default_visual": pred["default"].eq(pred["visual"]).float(),
        "agree_default_sub": pred["default"].eq(pred["subspace"]).float(),
        "agree_default_text": pred["default"].eq(pred["text"]).float(),
        "agree_default_knn": pred["default"].eq(pred["knn"]).float(),
        "agree_visual_sub": pred["visual"].eq(pred["subspace"]).float(),
        "agree_text_knn": pred["text"].eq(pred["knn"]).float(),
        "overlap3_default_cov": overlap3["covariance"] / 3.0,
        "overlap3_default_visual": overlap3["visual"] / 3.0,
        "overlap3_default_sub": overlap3["subspace"] / 3.0,
        "overlap3_default_text": overlap3["text"] / 3.0,
        "overlap3_default_knn": overlap3["knn"] / 3.0,
        "disagreement_count": unique_count / 4.0,
        "default_pred_is_novel": pred["default"].ge(int(problem["num_base_cls"])).float(),
        "visual_pred_is_novel": pred["visual"].ge(int(problem["num_base_cls"])).float(),
        "sub_pred_is_novel": pred["subspace"].ge(int(problem["num_base_cls"])).float(),
        "default_pred_session": session_lookup[pred["default"]],
        "visual_pred_session": session_lookup[pred["visual"]],
        "sub_pred_session": session_lookup[pred["subspace"]],
        "dino_top1_sim": dino_vals[:, 0],
        "dino_top1_gap": dino_top1_gap,
        "dino_default_pred_sim": dino_sim.gather(1, pred["default"].unsqueeze(1)).squeeze(1),
        "dino_visual_pred_sim": dino_sim.gather(1, pred["visual"].unsqueeze(1)).squeeze(1),
        "dino_sub_pred_sim": dino_sim.gather(1, pred["subspace"].unsqueeze(1)).squeeze(1),
        "support_global_top1": default_support_density["support_global_top1"],
        "support_global_topk_mean": default_support_density["support_global_topk_mean"],
        "support_defaultcls_top1": default_support_density["support_predcls_top1"],
        "support_defaultcls_topk_mean": default_support_density["support_predcls_topk_mean"],
        "qpr_gate_purity_default": qpr_purity,
        "qpr_gate_logcount_default": qpr_log_count,
        "qpr_gate_eligible_default": qpr_eligible,
    }

    feature_names = list(feature_map.keys())
    feature_tensor = torch.stack([feature_map[name].float() for name in feature_names], dim=1)
    return {
        "branch_probs": branch_probs,
        "pred": pred,
        "margin": margin,
        "entropy": entropy,
        "feature_names": feature_names,
        "feature_tensor": feature_tensor,
    }


def choose_alt_teacher(pred_map, correct_map, margin_map):
    default_correct = correct_map["default"]
    n = int(default_correct.numel())
    labels = torch.zeros(n, device=default_correct.device, dtype=torch.long)
    alt_branch_to_idx = {name: idx for idx, name in enumerate(ALT_BRANCHES)}

    for branch_name in ALT_BRANCHES:
        branch_idx = alt_branch_to_idx[branch_name]
        mask = (~default_correct) & correct_map[branch_name]
        better = mask & labels.eq(0)
        labels[better] = branch_idx + 1

    best_margin = torch.full((n,), -1e9, device=default_correct.device, dtype=margin_map["default"].dtype)
    for branch_name in ALT_BRANCHES:
        branch_idx = alt_branch_to_idx[branch_name]
        mask = (~default_correct) & correct_map[branch_name]
        better = mask & margin_map[branch_name].gt(best_margin)
        labels[better] = branch_idx + 1
        best_margin[better] = margin_map[branch_name][better]
    return labels


def build_labels(problem, feature_payload):
    targets = problem["query_targets"]
    pred_map = feature_payload["pred"]
    correct_map = {name: pred.eq(targets) for name, pred in pred_map.items()}
    teacher_label = choose_alt_teacher(pred_map, correct_map, feature_payload["margin"])
    should_override = teacher_label.gt(0)
    status = torch.full_like(teacher_label, 2)
    status[correct_map["default"]] = 0
    status[should_override] = 1
    strata = status + 3 * targets.ge(int(problem["num_base_cls"])).long()
    return {
        "correct_map": correct_map,
        "teacher_label": teacher_label,
        "should_override": should_override,
        "strata": strata,
    }


def split_indices(strata, seed, train_frac=0.6, val_frac=0.2):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_parts = []
    val_parts = []
    test_parts = []
    for group in torch.unique(strata).tolist():
        group_idx = torch.nonzero(strata.eq(int(group)), as_tuple=False).flatten()
        perm = group_idx[torch.randperm(group_idx.numel(), generator=generator)]
        n = int(perm.numel())
        if n <= 2:
            train_parts.append(perm)
            continue
        n_train = max(1, int(round(n * float(train_frac))))
        n_val = max(1, int(round(n * float(val_frac))))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = max(1, n - n_train - n_val)
        if n_train + n_val + n_test > n:
            n_train = max(1, n - n_val - n_test)
        train_parts.append(perm[:n_train])
        val_parts.append(perm[n_train : n_train + n_val])
        test_parts.append(perm[n_train + n_val : n_train + n_val + n_test])

    empty = torch.empty(0, device=strata.device, dtype=torch.long)
    train_idx = torch.cat(train_parts, dim=0) if train_parts else empty
    val_idx = torch.cat(val_parts, dim=0) if val_parts else empty
    test_idx = torch.cat(test_parts, dim=0) if test_parts else empty
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def compute_confusion_priors(pred_map, labels, split):
    train_idx = split["train"]
    device = train_idx.device
    num_cls = int(max(int(pred.max().item()) for pred in pred_map.values()) + 1)

    default_pred = pred_map["default"]
    default_wrong = ~labels["correct_map"]["default"]

    default_error_num = torch.ones(num_cls, device=device)
    default_error_den = torch.full((num_cls,), 2.0, device=device)
    for cls_id in range(num_cls):
        cls_mask = default_pred[train_idx].eq(int(cls_id))
        count = int(cls_mask.sum().item())
        if count == 0:
            continue
        error = default_wrong[train_idx][cls_mask].float().sum()
        default_error_num[int(cls_id)] += error
        default_error_den[int(cls_id)] += float(count)
    default_error_prior = default_error_num / torch.clamp(default_error_den, min=1e-12)

    branch_precision = {}
    pair_win = {}
    for branch_name in ALT_BRANCHES:
        pred_branch = pred_map[branch_name]
        precision_num = torch.ones(num_cls, device=device)
        precision_den = torch.full((num_cls,), 2.0, device=device)
        pair_num = torch.ones(num_cls, num_cls, device=device)
        pair_den = torch.full((num_cls, num_cls), 2.0, device=device)

        train_default = default_pred[train_idx]
        train_branch = pred_branch[train_idx]
        helpful = (~labels["correct_map"]["default"][train_idx]) & labels["correct_map"][branch_name][train_idx]

        for cls_id in range(num_cls):
            cls_mask = train_branch.eq(int(cls_id))
            count = int(cls_mask.sum().item())
            if count > 0:
                precision_num[int(cls_id)] += labels["correct_map"][branch_name][train_idx][cls_mask].float().sum()
                precision_den[int(cls_id)] += float(count)

        pair_index = train_default * num_cls + train_branch
        ones = torch.ones_like(pair_index, dtype=pair_den.dtype)
        pair_den.view(-1).scatter_add_(0, pair_index, ones)
        pair_num.view(-1).scatter_add_(0, pair_index, helpful.float())

        branch_precision[branch_name] = precision_num / torch.clamp(precision_den, min=1e-12)
        pair_win[branch_name] = pair_num / torch.clamp(pair_den, min=1e-12)

    return {
        "default_error_prior": default_error_prior,
        "branch_precision": branch_precision,
        "pair_win": pair_win,
    }


def augment_with_priors(base_features, feature_names, pred_map, priors):
    extras = []
    extra_names = []
    extras.append(priors["default_error_prior"][pred_map["default"]])
    extra_names.append("default_error_prior")
    for branch_name in ALT_BRANCHES:
        extras.append(priors["branch_precision"][branch_name][pred_map[branch_name]])
        extra_names.append(f"{branch_name}_pred_precision_prior")
        pair_table = priors["pair_win"][branch_name]
        extras.append(pair_table[pred_map["default"], pred_map[branch_name]])
        extra_names.append(f"{branch_name}_pair_win_prior")

    feature_tensor = torch.cat([base_features, torch.stack([x.float() for x in extras], dim=1)], dim=1)
    return feature_tensor, feature_names + extra_names


def normalize_from_train(features, train_idx):
    mean = features[train_idx].mean(dim=0, keepdim=True)
    std = features[train_idx].std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (features - mean) / std, mean, std


class BinaryRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, linear=False):
        super().__init__()
        if linear:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MultiRouter(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_binary_model(features, labels, train_idx, val_idx, linear, max_epochs, patience, lr, seed):
    torch.manual_seed(int(seed))
    model = BinaryRouter(input_dim=features.shape[1], linear=linear).to(features.device)
    pos_count = float(labels[train_idx].sum().item())
    neg_count = float(train_idx.numel() - labels[train_idx].sum().item())
    pos_weight = torch.tensor(max(neg_count / max(pos_count, 1.0), 1.0), device=features.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-4)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_loss = float("inf")
    stale = 0

    for _ in range(int(max_epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(features[train_idx])
        loss = criterion(logits, labels[train_idx].float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(features[val_idx])
            val_loss = float(criterion(val_logits, labels[val_idx].float()).item())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(features))


def train_multiclass_model(features, labels, train_idx, val_idx, num_classes, max_epochs, patience, lr, seed):
    torch.manual_seed(int(seed))
    model = MultiRouter(input_dim=features.shape[1], num_classes=num_classes).to(features.device)
    class_counts = torch.bincount(labels[train_idx], minlength=int(num_classes)).float()
    class_weights = torch.ones(int(num_classes), device=features.device)
    present = class_counts > 0
    if present.any():
        denom = torch.clamp(class_counts[present], min=1.0)
        class_weights[present] = float(train_idx.numel()) / denom
        class_weights[present] = class_weights[present] / class_weights[present].mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=1e-4)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_acc = -1.0
    stale = 0

    for _ in range(int(max_epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(features[train_idx])
        loss = criterion(logits, labels[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.argmax(model(features[val_idx]), dim=1)
            val_acc = float(val_pred.eq(labels[val_idx]).float().mean().item())
        if val_acc > best_acc + 1e-6:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(features)


def build_restricted_mask_from_thresholds(problem, feature_payload, margin_thr, base_mass_thr):
    default_pred = feature_payload["pred"]["default"]
    default_margin = feature_payload["margin"]["default"]
    default_base_mass = feature_payload["branch_probs"]["default"][:, : problem["num_base_cls"]].sum(dim=1)
    disagreement = (
        (~default_pred.eq(feature_payload["pred"]["covariance"]))
        | (~default_pred.eq(feature_payload["pred"]["visual"]))
        | (~default_pred.eq(feature_payload["pred"]["subspace"]))
    )
    novel_like = default_pred.ge(int(problem["num_base_cls"])) | default_base_mass.le(float(base_mass_thr))
    low_margin = default_margin.le(float(margin_thr))
    return disagreement & novel_like & low_margin


def route_from_alt_scores(default_probs, alt_branch_probs, default_pred, override_mask, alt_scores):
    device = default_probs.device
    alt_preds = torch.stack([torch.argmax(alt_branch_probs[name], dim=1) for name in ALT_BRANCHES], dim=1)
    order = torch.argsort(alt_scores, dim=1, descending=True)
    chosen_alt = torch.full((default_probs.shape[0],), -1, device=device, dtype=torch.long)

    for rank in range(order.shape[1]):
        branch_choice = order[:, rank]
        row_idx = torch.arange(default_probs.shape[0], device=device)
        branch_pred = alt_preds[row_idx, branch_choice]
        eligible = chosen_alt.lt(0) & override_mask & branch_pred.ne(default_pred)
        chosen_alt[eligible] = branch_choice[eligible]

    out = default_probs.clone()
    for branch_idx, branch_name in enumerate(ALT_BRANCHES):
        mask = chosen_alt.eq(int(branch_idx))
        if mask.any():
            out[mask] = alt_branch_probs[branch_name][mask]
    return out, chosen_alt


def route_full_selector(default_probs, alt_branch_probs, default_pred, selector_logits):
    selector_pred = torch.argmax(selector_logits, dim=1)
    out = default_probs.clone()
    chosen = torch.full_like(selector_pred, -1)
    for branch_idx, branch_name in enumerate(ALT_BRANCHES, start=1):
        mask = selector_pred.eq(int(branch_idx))
        branch_pred = torch.argmax(alt_branch_probs[branch_name], dim=1)
        mask = mask & branch_pred.ne(default_pred)
        if mask.any():
            out[mask] = alt_branch_probs[branch_name][mask]
            chosen[mask] = int(branch_idx - 1)
    return out, selector_pred, chosen


def switch_metrics(default_probs, variant_probs, targets):
    default_pred = torch.argmax(default_probs, dim=1)
    variant_pred = torch.argmax(variant_probs, dim=1)
    default_correct = default_pred.eq(targets)
    variant_correct = variant_pred.eq(targets)
    switched = variant_pred.ne(default_pred)
    helpful = switched & (~default_correct) & variant_correct
    harmful = switched & default_correct & (~variant_correct)
    both_wrong = switched & (~default_correct) & (~variant_correct)
    both_correct = switched & default_correct & variant_correct
    return {
        "switch_count": int(switched.sum().item()),
        "switch_coverage": round(float(switched.float().mean().item()), 4),
        "helpful_switches": int(helpful.sum().item()),
        "harmful_switches": int(harmful.sum().item()),
        "both_wrong_diff": int(both_wrong.sum().item()),
        "both_correct_diff": int(both_correct.sum().item()),
        "helpful_precision": round(float(helpful.sum().item() / max(int(switched.sum().item()), 1)), 4),
        "net_help": int(helpful.sum().item() - harmful.sum().item()),
    }


def summarize_variant(default_probs, probs, targets, num_base_cls):
    metrics = accuracy_stats_from_probs(probs, targets, num_base_cls)
    metrics.update(switch_metrics(default_probs, probs, targets))
    metrics["gain_vs_default"] = round(float(metrics["full_acc"] - accuracy_stats_from_probs(default_probs, targets, num_base_cls)["full_acc"]), 3)
    return metrics


def binary_confusion_metrics(probs, labels, threshold, mask=None):
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    pred = probs.ge(float(threshold)) & mask
    label = labels & mask
    tp = int((pred & label).sum().item())
    fp = int((pred & (~label)).sum().item())
    fn = int(((~pred) & label).sum().item())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    return {
        "binary_precision": round(precision, 4),
        "binary_recall": round(recall, 4),
        "binary_positive_rate": round(float(pred.float().mean().item()), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_binary_router(name, default_probs, alt_branch_probs, default_pred, override_probs, alt_scores, labels, split, num_base_cls, allowed_mask=None):
    if allowed_mask is None:
        allowed_mask = torch.ones_like(labels["should_override"], dtype=torch.bool)

    val_idx = split["val"]
    test_idx = split["test"]
    targets = labels["targets"]

    thresholds = torch.linspace(0.05, 0.95, steps=19, device=default_probs.device)
    best = None
    for threshold in thresholds.tolist():
        val_override_mask = override_probs.ge(float(threshold)) & allowed_mask
        routed, chosen_alt = route_from_alt_scores(
            default_probs=default_probs,
            alt_branch_probs=alt_branch_probs,
            default_pred=default_pred,
            override_mask=val_override_mask,
            alt_scores=alt_scores,
        )
        val_acc = accuracy_stats_from_probs(routed[val_idx], targets[val_idx], num_base_cls)["full_acc"]
        val_switch = switch_metrics(default_probs[val_idx], routed[val_idx], targets[val_idx])["net_help"]
        if best is None or val_acc > best["val_acc"] + 1e-6 or (abs(val_acc - best["val_acc"]) <= 1e-6 and val_switch > best["val_switch"]):
            best = {"threshold": float(threshold), "val_acc": float(val_acc), "val_switch": int(val_switch)}

    override_mask = override_probs.ge(float(best["threshold"])) & allowed_mask
    routed, chosen_alt = route_from_alt_scores(
        default_probs=default_probs,
        alt_branch_probs=alt_branch_probs,
        default_pred=default_pred,
        override_mask=override_mask,
        alt_scores=alt_scores,
    )
    summary = summarize_variant(default_probs[test_idx], routed[test_idx], targets[test_idx], num_base_cls)
    summary["threshold"] = round(float(best["threshold"]), 3)
    summary["allowed_query_rate"] = round(float(allowed_mask.float().mean().item()), 4)
    summary["chosen_alt_counts"] = {
        ALT_BRANCHES[i]: int(chosen_alt[test_idx].eq(i).sum().item()) for i in range(len(ALT_BRANCHES))
    }
    summary.update(binary_confusion_metrics(override_probs[test_idx], labels["should_override"][test_idx], best["threshold"], mask=allowed_mask[test_idx]))
    return routed, summary


def evaluate_full_selector(default_probs, alt_branch_probs, default_pred, selector_logits, labels, split, num_base_cls):
    test_idx = split["test"]
    targets = labels["targets"]
    routed, selector_pred, chosen_alt = route_full_selector(
        default_probs=default_probs,
        alt_branch_probs=alt_branch_probs,
        default_pred=default_pred,
        selector_logits=selector_logits,
    )
    summary = summarize_variant(default_probs[test_idx], routed[test_idx], targets[test_idx], num_base_cls)
    summary["selector_label_counts"] = {
        "default": int(selector_pred[test_idx].eq(0).sum().item()),
        "covariance": int(selector_pred[test_idx].eq(1).sum().item()),
        "visual": int(selector_pred[test_idx].eq(2).sum().item()),
        "subspace": int(selector_pred[test_idx].eq(3).sum().item()),
    }
    summary["effective_alt_counts"] = {
        ALT_BRANCHES[i]: int(chosen_alt[test_idx].eq(i).sum().item()) for i in range(len(ALT_BRANCHES))
    }
    return routed, summary


def query_table(out_path, problem, feature_payload, labels, split, variant_preds):
    rows = []
    targets = labels["targets"].detach().cpu().tolist()
    strata = labels["strata"].detach().cpu().tolist()
    teacher = labels["teacher_label"].detach().cpu().tolist()
    should_override = labels["should_override"].detach().cpu().tolist()
    split_name = {}
    for name, idx in split.items():
        for value in idx.detach().cpu().tolist():
            split_name[int(value)] = name

    feature_tensor = feature_payload["feature_tensor"].detach().cpu()
    pred_map = {name: tensor.detach().cpu().tolist() for name, tensor in feature_payload["pred"].items()}
    for i, target in enumerate(targets):
        row = {
            "query_index": int(i),
            "split_name": split_name.get(int(i), "unused"),
            "target": int(target),
            "strata": int(strata[i]),
            "should_override": int(bool(should_override[i])),
            "teacher_branch": int(teacher[i]),
        }
        for branch_name, values in pred_map.items():
            row[f"{branch_name}_pred"] = int(values[i])
        for feat_idx, feat_name in enumerate(feature_payload["feature_names"]):
            row[feat_name] = round(float(feature_tensor[i, feat_idx].item()), 6)
        for key, tensor in variant_preds.items():
            row[f"{key}_pred"] = int(tensor[i])
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["query_index"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(
    data_cfg,
    train_cfg,
    seed,
    output_root,
    alpha_grid,
    fallback_alpha,
    mass_thr,
    min_count,
    val_fraction,
    base_val_conf_thr,
    base_val_max_per_class,
    subspace_dim,
    subspace_temp,
    support_topk,
    max_epochs,
    patience,
    lr,
):
    dataset_label = dataset_output_name(data_cfg)
    out_dir = output_root / dataset_label / f"seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    driver_log = out_dir / "driver.log"

    started_at = now_iso()
    start_time = time.time()
    with driver_log.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            problem = build_final_problem(data_cfg, train_cfg, seed)

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
    _, subspace_probs = build_subspace_branch(
        problem,
        tric_pv["dino_image_proto"],
        subspace_dim=subspace_dim,
        subspace_temp=subspace_temp,
    )
    safe_router_probs, safe_router_meta = conservative_router(
        default_probs=tric_pv["full_probs"],
        cov_probs=tric_pv["cov_probs"],
        knn_probs=problem["base_probs"]["description_knn_branch"],
        visual_probs=tric_pv["visual_probs"],
        text_probs=problem["base_probs"]["semantic_branch_only"],
        subspace_probs=subspace_probs,
        num_base_cls=problem["num_base_cls"],
    )
    subspace_repair_probs, subspace_repair_meta = selective_subspace_repair(
        default_probs=tric_pv["full_probs"],
        visual_probs=tric_pv["visual_probs"],
        cov_probs=tric_pv["cov_probs"],
        subspace_probs=subspace_probs,
    )
    semantic_repair_probs, semantic_repair_meta = selective_semantic_repair(
        default_probs=tric_pv["full_probs"],
        semantic_probs=problem["base_probs"]["semantic_branch_only"],
        knn_probs=problem["base_probs"]["description_knn_branch"],
    )

    gate_stats = qpr_gate_stats(problem, problem["base_probs"]["tric"], mass_thr=mass_thr, min_count=min_count)
    feature_payload = build_router_features(
        problem=problem,
        tric_pv=tric_pv,
        subspace_probs=subspace_probs,
        gate_stats=gate_stats,
        support_topk=support_topk,
    )
    label_payload = build_labels(problem, feature_payload)
    label_payload["targets"] = problem["query_targets"]
    split = split_indices(label_payload["strata"], seed=seed + 100)

    priors = compute_confusion_priors(feature_payload["pred"], label_payload, split)
    feature_tensor, feature_names = augment_with_priors(
        base_features=feature_payload["feature_tensor"],
        feature_names=feature_payload["feature_names"],
        pred_map=feature_payload["pred"],
        priors=priors,
    )
    feature_tensor, _, _ = normalize_from_train(feature_tensor, split["train"])
    feature_payload["feature_tensor"] = feature_tensor
    feature_payload["feature_names"] = feature_names

    default_probs = tric_pv["full_probs"]
    alt_branch_probs = {
        "covariance": tric_pv["cov_probs"],
        "visual": tric_pv["visual_probs"],
        "subspace": subspace_probs,
    }
    default_pred = feature_payload["pred"]["default"]

    override_probs_linear = train_binary_model(
        features=feature_tensor,
        labels=label_payload["should_override"].long(),
        train_idx=split["train"],
        val_idx=split["val"],
        linear=True,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        seed=seed + 200,
    )
    override_probs_mlp = train_binary_model(
        features=feature_tensor,
        labels=label_payload["should_override"].long(),
        train_idx=split["train"],
        val_idx=split["val"],
        linear=False,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        seed=seed + 300,
    )

    override_train = split["train"][label_payload["teacher_label"][split["train"]].gt(0)]
    override_val = split["val"][label_payload["teacher_label"][split["val"]].gt(0)]
    if override_train.numel() == 0 or override_val.numel() == 0:
        raise RuntimeError(f"No override-positive samples available for selector training on {dataset_label}")

    alt_selector_logits = train_multiclass_model(
        features=feature_tensor,
        labels=label_payload["teacher_label"].clamp(min=1) - 1,
        train_idx=override_train,
        val_idx=override_val,
        num_classes=len(ALT_BRANCHES),
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        seed=seed + 400,
    )
    full_selector_logits = train_multiclass_model(
        features=feature_tensor,
        labels=label_payload["teacher_label"],
        train_idx=split["train"],
        val_idx=split["val"],
        num_classes=1 + len(ALT_BRANCHES),
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        seed=seed + 500,
    )

    default_margin = feature_payload["margin"]["default"]
    default_base_mass = default_probs[:, : problem["num_base_cls"]].sum(dim=1)
    margin_thr = torch.quantile(default_margin[split["train"]], q=0.4)
    base_mass_thr = torch.quantile(default_base_mass[split["train"]], q=0.35)
    restricted_mask = build_restricted_mask_from_thresholds(problem, feature_payload, margin_thr=float(margin_thr.item()), base_mass_thr=float(base_mass_thr.item()))

    test_idx = split["test"]
    teacher_label = label_payload["teacher_label"]
    limited_oracle_probs = default_probs.clone()
    for branch_idx, branch_name in enumerate(ALT_BRANCHES, start=1):
        mask = teacher_label.eq(int(branch_idx))
        if mask.any():
            limited_oracle_probs[mask] = alt_branch_probs[branch_name][mask]

    expanded_teacher = teacher_label.clone()
    extra_branch_order = ["text", "knn"]
    best_margin = torch.full_like(feature_payload["margin"]["default"], -1e9)
    for branch_name in ALT_BRANCHES:
        branch_idx = ALT_BRANCHES.index(branch_name) + 1
        mask = expanded_teacher.eq(int(branch_idx))
        best_margin[mask] = feature_payload["margin"][branch_name][mask]
    for branch_name in extra_branch_order:
        branch_correct = label_payload["correct_map"][branch_name]
        candidate = (~label_payload["correct_map"]["default"]) & branch_correct & feature_payload["margin"][branch_name].gt(best_margin)
        expanded_teacher[candidate] = 100 + extra_branch_order.index(branch_name)
        best_margin[candidate] = feature_payload["margin"][branch_name][candidate]
    expanded_oracle_probs = limited_oracle_probs.clone()
    for branch_name in extra_branch_order:
        branch_code = 100 + extra_branch_order.index(branch_name)
        mask = expanded_teacher.eq(int(branch_code))
        if mask.any():
            expanded_oracle_probs[mask] = feature_payload["branch_probs"][branch_name][mask]

    binary_linear_probs, binary_linear_summary = evaluate_binary_router(
        name="binary_linear_strong",
        default_probs=default_probs,
        alt_branch_probs=alt_branch_probs,
        default_pred=default_pred,
        override_probs=override_probs_linear,
        alt_scores=alt_selector_logits,
        labels=label_payload,
        split=split,
        num_base_cls=problem["num_base_cls"],
    )
    binary_mlp_probs, binary_mlp_summary = evaluate_binary_router(
        name="binary_mlp_strong",
        default_probs=default_probs,
        alt_branch_probs=alt_branch_probs,
        default_pred=default_pred,
        override_probs=override_probs_mlp,
        alt_scores=alt_selector_logits,
        labels=label_payload,
        split=split,
        num_base_cls=problem["num_base_cls"],
    )
    binary_restricted_probs, binary_restricted_summary = evaluate_binary_router(
        name="binary_mlp_restricted",
        default_probs=default_probs,
        alt_branch_probs=alt_branch_probs,
        default_pred=default_pred,
        override_probs=override_probs_mlp,
        alt_scores=alt_selector_logits,
        labels=label_payload,
        split=split,
        num_base_cls=problem["num_base_cls"],
        allowed_mask=restricted_mask,
    )
    full_selector_probs, full_selector_summary = evaluate_full_selector(
        default_probs=default_probs,
        alt_branch_probs=alt_branch_probs,
        default_pred=default_pred,
        selector_logits=full_selector_logits,
        labels=label_payload,
        split=split,
        num_base_cls=problem["num_base_cls"],
    )

    restricted_oracle_probs = default_probs.clone()
    for branch_idx, branch_name in enumerate(ALT_BRANCHES, start=1):
        mask = restricted_mask & teacher_label.eq(int(branch_idx))
        if mask.any():
            restricted_oracle_probs[mask] = alt_branch_probs[branch_name][mask]

    summaries = {
        "tric_pv_default": summarize_variant(default_probs[test_idx], default_probs[test_idx], problem["query_targets"][test_idx], problem["num_base_cls"]),
        "tric_safe_router": summarize_variant(default_probs[test_idx], safe_router_probs[test_idx], problem["query_targets"][test_idx], problem["num_base_cls"]),
        "tric_subspace_repair": summarize_variant(default_probs[test_idx], subspace_repair_probs[test_idx], problem["query_targets"][test_idx], problem["num_base_cls"]),
        "narrow_semantic_repair": summarize_variant(default_probs[test_idx], semantic_repair_probs[test_idx], problem["query_targets"][test_idx], problem["num_base_cls"]),
        "limited_oracle_4way": summarize_variant(default_probs[test_idx], limited_oracle_probs[test_idx], problem["query_targets"][test_idx], problem["num_base_cls"]),
        "expanded_oracle_6way": summarize_variant(default_probs[test_idx], expanded_oracle_probs[test_idx], problem["query_targets"][test_idx], problem["num_base_cls"]),
        "restricted_oracle_4way": summarize_variant(default_probs[test_idx], restricted_oracle_probs[test_idx], problem["query_targets"][test_idx], problem["num_base_cls"]),
        "binary_linear_strong": binary_linear_summary,
        "binary_mlp_strong": binary_mlp_summary,
        "binary_mlp_restricted": binary_restricted_summary,
        "full_selector_mlp": full_selector_summary,
    }

    split_stats = {
        "train_count": int(split["train"].numel()),
        "val_count": int(split["val"].numel()),
        "test_count": int(split["test"].numel()),
        "global_override_rate": round(float(label_payload["should_override"].float().mean().item()), 4),
        "test_override_rate": round(float(label_payload["should_override"][test_idx].float().mean().item()), 4),
        "restricted_query_rate": round(float(restricted_mask.float().mean().item()), 4),
        "restricted_test_query_rate": round(float(restricted_mask[test_idx].float().mean().item()), 4),
        "restricted_test_override_rate": round(
            float(label_payload["should_override"][test_idx][restricted_mask[test_idx]].float().mean().item())
            if restricted_mask[test_idx].any()
            else 0.0,
            4,
        ),
        "margin_threshold": round(float(margin_thr.item()), 6),
        "base_mass_threshold": round(float(base_mass_thr.item()), 6),
    }

    variant_preds = {
        "default": torch.argmax(default_probs, dim=1).detach().cpu(),
        "safe_router": torch.argmax(safe_router_probs, dim=1).detach().cpu(),
        "subspace_repair": torch.argmax(subspace_repair_probs, dim=1).detach().cpu(),
        "binary_linear": torch.argmax(binary_linear_probs, dim=1).detach().cpu(),
        "binary_mlp": torch.argmax(binary_mlp_probs, dim=1).detach().cpu(),
        "binary_restricted": torch.argmax(binary_restricted_probs, dim=1).detach().cpu(),
        "full_selector": torch.argmax(full_selector_probs, dim=1).detach().cpu(),
    }
    query_table(out_dir / "query_diagnostics.csv", problem, feature_payload, label_payload, split, variant_preds)

    completed_at = now_iso()
    runtime_sec = round(time.time() - start_time, 3)
    payload = {
        "analysis": "tric_router_reliability_suite_final_session",
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
        "selected_alpha": tric_pv["selected_alpha"],
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "split_stats": split_stats,
        "qpr_gate_stats": gate_stats,
        "variant_metadata": {
            "tric_safe_router": safe_router_meta,
            "tric_subspace_repair": subspace_repair_meta,
            "narrow_semantic_repair": semantic_repair_meta,
        },
        "variant_summary": summaries,
        "files": {
            "query_diagnostics_csv": str(out_dir / "query_diagnostics.csv"),
            "results_json": str(out_dir / "results.json"),
            "summary_txt": str(out_dir / "summary.txt"),
            "driver_log": str(driver_log),
        },
    }

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"dataset: {problem['dataset_name']}",
        f"started_at: {started_at}",
        f"completed_at: {completed_at}",
        f"runtime_sec: {runtime_sec}",
        f"default_full_acc: {summaries['tric_pv_default']['full_acc']}",
        f"limited_oracle_4way_full_acc: {summaries['limited_oracle_4way']['full_acc']}",
        f"binary_mlp_strong_full_acc: {summaries['binary_mlp_strong']['full_acc']}",
        f"binary_mlp_restricted_full_acc: {summaries['binary_mlp_restricted']['full_acc']}",
        f"full_selector_mlp_full_acc: {summaries['full_selector_mlp']['full_acc']}",
        f"safe_router_full_acc: {summaries['tric_safe_router']['full_acc']}",
        f"restricted_query_rate: {split_stats['restricted_query_rate']}",
        f"global_override_rate: {split_stats['global_override_rate']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "completion.txt").write_text(
        f"started_at={started_at}\ncompleted_at={completed_at}\nruntime_sec={runtime_sec}\nexit_status=0\n",
        encoding="utf-8",
    )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Final-session TriC router reliability diagnostics.")
    parser.add_argument("--data_cfgs", nargs="+", default=DEFAULT_DATA_CFGS)
    parser.add_argument("--train_cfg", default=DEFAULT_TRAIN_CFG)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_root", default="experiments")
    parser.add_argument("--alpha_grid", type=float, nargs="+", default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--fallback_alpha", type=float, default=0.75)
    parser.add_argument("--mass_thr", type=float, default=0.3)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--base_val_conf_thr", type=float, default=0.5)
    parser.add_argument("--base_val_max_per_class", type=int, default=50)
    parser.add_argument("--subspace_dim", type=int, default=3)
    parser.add_argument("--subspace_temp", type=float, default=40.0)
    parser.add_argument("--support_topk", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
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
            support_topk=args.support_topk,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
        )
        runs.append(
            {
                "dataset": payload["dataset"],
                "completed_at": payload["completed_at"],
                "runtime_sec": payload["runtime_sec"],
                "split_stats": payload["split_stats"],
                "variant_summary": payload["variant_summary"],
                "files": payload["files"],
            }
        )

    suite_payload = {
        "analysis": "tric_router_reliability_suite_final_session",
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
