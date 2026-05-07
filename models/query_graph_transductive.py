import torch
import torch.nn.functional as F


def graph_neighbor_average(values, affinity):
    affinity = affinity.to(device=values.device, dtype=values.dtype)
    return torch.matmul(affinity, values)


def distribution_smoothness(probs, affinity):
    neigh = graph_neighbor_average(probs, affinity)
    return ((probs - neigh) ** 2).mean()


def entropy_loss(probs):
    return -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1).mean()


def symmetric_kl(p, q):
    p = torch.clamp(p, min=1e-12)
    q = torch.clamp(q, min=1e-12)
    kl_pq = F.kl_div(torch.log(p), q, reduction="batchmean")
    kl_qp = F.kl_div(torch.log(q), p, reduction="batchmean")
    return 0.5 * (kl_pq + kl_qp)
