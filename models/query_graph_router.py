import torch
import torch.nn as nn
import torch.nn.functional as F

from models.query_branch_router import build_router_features


def build_knn_affinity(query_features, k=10, self_loop=True):
    """
    Build a symmetric cosine kNN affinity matrix over normalized query features.
    """
    features = F.normalize(query_features.float(), dim=-1)
    sim = torch.matmul(features, features.T)
    n = sim.shape[0]
    k = max(1, min(int(k), max(n - 1, 1)))

    topk = torch.topk(sim, k=k + 1, dim=1).indices
    affinity = sim.new_zeros(sim.shape)
    rows = torch.arange(n, device=sim.device).unsqueeze(1).expand_as(topk)
    affinity[rows, topk] = torch.clamp(sim[rows, topk], min=0.0)
    affinity = torch.maximum(affinity, affinity.T)

    if self_loop:
        affinity.fill_diagonal_(1.0)
    else:
        affinity.fill_diagonal_(0.0)

    deg = torch.clamp(affinity.sum(dim=1, keepdim=True), min=1e-12)
    affinity = affinity / deg
    return affinity


def graph_smoothness(values, affinity):
    """
    Smoothness penalty for scalar node values over an affinity graph.
    values: [N]
    affinity: [N, N], row-normalized
    """
    affinity = affinity.to(device=values.device, dtype=values.dtype)
    neigh = torch.matmul(affinity, values.unsqueeze(1)).squeeze(1)
    return ((values - neigh) ** 2).mean()


def route_with_query_alphas(prob_fused, prob_cov, prob_knn, num_base_cls, alpha_base, alpha_novel):
    while alpha_base.dim() < prob_fused.dim():
        alpha_base = alpha_base.unsqueeze(-1)
    while alpha_novel.dim() < prob_fused.dim():
        alpha_novel = alpha_novel.unsqueeze(-1)

    base_probs = alpha_base * prob_fused[:, :num_base_cls] + (1.0 - alpha_base) * prob_cov[:, :num_base_cls]
    novel_probs = alpha_novel * prob_fused[:, num_base_cls:] + (1.0 - alpha_novel) * prob_knn[:, num_base_cls:]
    scores = torch.cat([base_probs, novel_probs], dim=1)
    return scores / torch.clamp(scores.sum(dim=1, keepdim=True), min=1e-12)


def smooth_query_alphas(alpha_base, alpha_novel, affinity, mix=0.5, steps=1):
    """
    Smooth per-query router outputs with a fixed graph affinity matrix.
    """
    mix = float(mix)
    steps = max(int(steps), 0)
    if steps == 0 or mix <= 0.0:
        return alpha_base, alpha_novel

    affinity = affinity.to(device=alpha_base.device, dtype=alpha_base.dtype)
    cur_base = alpha_base
    cur_novel = alpha_novel
    for _ in range(steps):
        neigh_base = torch.matmul(affinity, cur_base.unsqueeze(1)).squeeze(1)
        neigh_novel = torch.matmul(affinity, cur_novel.unsqueeze(1)).squeeze(1)
        cur_base = (1.0 - mix) * cur_base + mix * neigh_base
        cur_novel = (1.0 - mix) * cur_novel + mix * neigh_novel
    return cur_base, cur_novel


class QueryGraphBranchRouter(nn.Module):
    """
    Query-dependent branch router that uses both branch confidence summaries and
    normalized query embeddings, then predicts base/novel fusion weights.
    """

    def __init__(self, query_dim, stats_dim, hidden_dim=128, query_proj_dim=64):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, query_proj_dim),
            nn.ReLU(inplace=True),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + query_proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.base_head = nn.Linear(hidden_dim, 1)
        self.novel_head = nn.Linear(hidden_dim, 1)

    def forward(self, query_features, stats_features):
        query_features = F.normalize(query_features.float(), dim=-1)
        stats_features = stats_features.float()
        hidden = torch.cat(
            [self.query_proj(query_features), self.stats_proj(stats_features)],
            dim=1,
        )
        hidden = self.trunk(hidden)
        alpha_base = torch.sigmoid(self.base_head(hidden)).squeeze(-1)
        alpha_novel = torch.sigmoid(self.novel_head(hidden)).squeeze(-1)
        return {
            "alpha_base": alpha_base,
            "alpha_novel": alpha_novel,
        }


def build_graph_router_inputs(query_features, prob_fused, prob_cov, prob_knn, num_base_cls):
    return {
        "query_features": F.normalize(query_features.float(), dim=-1),
        "stats_features": build_router_features(prob_fused, prob_cov, prob_knn, num_base_cls).float(),
    }
