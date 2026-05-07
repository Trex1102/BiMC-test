import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_vector(vec):
    return F.normalize(vec.unsqueeze(0), dim=-1).squeeze(0)


class NovelSemanticComposer(nn.Module):
    """
    Lightweight branch-aware composer for novel fused semantics.

    Design:
    - Inputs a baseline semantic anchor plus a small set of candidate components
      such as prompt, description mean, top description atoms, and top base anchors.
    - Predicts mixture weights over the components and a residual strength alpha.
    - Keeps KNN untouched; only the fused semantic anchor is updated.

    Intended training regime:
    - Train offline on pseudo-novel base episodes with final BiMC ensemble loss.
    - Use the class/context features that are available at test time.
    """

    def __init__(self, context_dim, component_feat_dim, hidden_dim=64):
        super().__init__()
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.component_net = nn.Sequential(
            nn.Linear(component_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.component_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)

    def forward(self, baseline_sem, component_vectors, context_features, component_features):
        """
        baseline_sem: [B, D]
        component_vectors: [B, M, D]
        context_features: [B, F]
        component_features: [B, M, C]
        """
        context_hidden = self.context_net(context_features)  # [B, H]
        component_hidden = self.component_net(component_features)  # [B, M, H]
        fused_hidden = component_hidden + context_hidden.unsqueeze(1)
        component_logits = self.component_head(fused_hidden).squeeze(-1)
        alpha_logit = self.alpha_head(context_hidden).squeeze(-1)
        semantic, weights, alpha = self.compose_from_parameters(
            baseline_sem=baseline_sem,
            component_vectors=component_vectors,
            component_logits=component_logits,
            alpha_logit=alpha_logit,
        )
        return {
            "semantic": semantic,
            "weights": weights,
            "alpha": alpha,
            "component_logits": component_logits,
            "alpha_logit": alpha_logit,
        }

    @staticmethod
    def compose_from_parameters(baseline_sem, component_vectors, component_logits, alpha_logit):
        """
        Continuous composition rule used both by the trainable module and the
        module-matched oracle killswitch.

        baseline_sem: [D] or [B, D]
        component_vectors: [M, D] or [B, M, D]
        component_logits: [M] or [B, M]
        alpha_logit: scalar or [B]
        """
        weights = F.softmax(component_logits, dim=-1)
        composer = torch.sum(weights.unsqueeze(-1) * component_vectors, dim=-2)
        composer = F.normalize(composer, dim=-1)
        alpha = torch.sigmoid(alpha_logit)
        while alpha.dim() < baseline_sem.dim():
            alpha = alpha.unsqueeze(-1)
        semantic = (1.0 - alpha) * baseline_sem + alpha * composer
        semantic = F.normalize(semantic, dim=-1)
        return semantic, weights, torch.sigmoid(alpha_logit)


def build_context_features(
    prompt_vec,
    desc_mean_vec,
    image_proto_vec,
    desc_group,
    top_base_anchor=None,
):
    diversity = 0.0
    if desc_group.shape[0] > 1:
        sims = torch.matmul(desc_group, desc_group.T)
        triu = torch.triu_indices(desc_group.shape[0], desc_group.shape[0], offset=1)
        diversity = float((1.0 - sims[triu[0], triu[1]]).mean().item())

    features = [
        float(torch.dot(prompt_vec, desc_mean_vec).item()),
        float(torch.dot(prompt_vec, image_proto_vec).item()),
        float(torch.dot(desc_mean_vec, image_proto_vec).item()),
        diversity,
        float(desc_group.shape[0]),
    ]
    if top_base_anchor is not None:
        features.append(float(torch.dot(prompt_vec, top_base_anchor).item()))
        features.append(float(torch.dot(desc_mean_vec, top_base_anchor).item()))
    else:
        features.extend([0.0, 0.0])
    return torch.tensor(features, dtype=prompt_vec.dtype, device=prompt_vec.device)


def build_component_features(component_vectors, prompt_vec, desc_mean_vec, image_proto_vec, top_base_anchor=None):
    rows = []
    if top_base_anchor is None:
        top_base_anchor = torch.zeros_like(prompt_vec)
    for comp in component_vectors:
        rows.append(torch.tensor([
            float(torch.dot(comp, prompt_vec).item()),
            float(torch.dot(comp, desc_mean_vec).item()),
            float(torch.dot(comp, image_proto_vec).item()),
            float(torch.dot(comp, top_base_anchor).item()),
        ], dtype=comp.dtype, device=comp.device))
    return torch.stack(rows, dim=0)
