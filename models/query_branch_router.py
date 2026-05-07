import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_entropy(probs):
    probs = probs.float()
    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)
    norm = math.log(max(int(probs.shape[-1]), 2))
    return entropy / norm


def top_prob_margin(probs):
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    top1 = top2[:, 0]
    if top2.shape[1] == 1:
        margin = top1
    else:
        margin = top1 - top2[:, 1]
    return top1, margin


def build_router_features(fused_probs, cov_probs, knn_probs, num_base_cls):
    fused_top1, fused_margin = top_prob_margin(fused_probs)
    cov_top1, cov_margin = top_prob_margin(cov_probs)
    knn_top1, knn_margin = top_prob_margin(knn_probs)

    fused_entropy = normalized_entropy(fused_probs)
    cov_entropy = normalized_entropy(cov_probs)
    knn_entropy = normalized_entropy(knn_probs)

    fused_top_idx = torch.argmax(fused_probs, dim=1)
    cov_top_idx = torch.argmax(cov_probs, dim=1)
    knn_top_idx = torch.argmax(knn_probs, dim=1)

    fused_base_mass = fused_probs[:, :num_base_cls].sum(dim=1)
    cov_base_mass = cov_probs[:, :num_base_cls].sum(dim=1)
    knn_base_mass = knn_probs[:, :num_base_cls].sum(dim=1)

    features = [
        fused_top1,
        fused_margin,
        fused_entropy,
        fused_base_mass,
        cov_top1,
        cov_margin,
        cov_entropy,
        cov_base_mass,
        knn_top1,
        knn_margin,
        knn_entropy,
        knn_base_mass,
        (fused_top_idx == cov_top_idx).float(),
        (fused_top_idx == knn_top_idx).float(),
        (cov_top_idx == knn_top_idx).float(),
        (fused_top_idx < num_base_cls).float(),
        (cov_top_idx < num_base_cls).float(),
        (knn_top_idx < num_base_cls).float(),
    ]
    return torch.stack(features, dim=1)


class QueryBranchRouter(nn.Module):
    """
    Query-dependent branch router for BiMC.

    It predicts one alpha for the base segment (fused vs covariance) and one
    alpha for the novel segment (fused vs description-KNN).
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.base_head = nn.Linear(hidden_dim, 1)
        self.novel_head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        hidden = self.net(features)
        alpha_base = torch.sigmoid(self.base_head(hidden)).squeeze(-1)
        alpha_novel = torch.sigmoid(self.novel_head(hidden)).squeeze(-1)
        return {
            "alpha_base": alpha_base,
            "alpha_novel": alpha_novel,
        }

    @staticmethod
    def route_from_alphas(fused_probs, cov_probs, knn_probs, num_base_cls, alpha_base, alpha_novel):
        while alpha_base.dim() < fused_probs.dim():
            alpha_base = alpha_base.unsqueeze(-1)
        while alpha_novel.dim() < fused_probs.dim():
            alpha_novel = alpha_novel.unsqueeze(-1)
        base_probs = alpha_base * fused_probs[:, :num_base_cls] + (1.0 - alpha_base) * cov_probs[:, :num_base_cls]
        novel_probs = alpha_novel * fused_probs[:, num_base_cls:] + (1.0 - alpha_novel) * knn_probs[:, num_base_cls:]
        scores = torch.cat([base_probs, novel_probs], dim=1)
        return scores / torch.clamp(scores.sum(dim=1, keepdim=True), min=1e-12)
