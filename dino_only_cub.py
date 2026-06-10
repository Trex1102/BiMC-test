"""
DINOv2-only "degenerate-Gaussian" classifier test on CUB (FSCIL, single seed).

Isolates the DINOv2 visual branch (NO CLIP, NO LLM/description text) and compares:
  B0  Cosine prototype (ProtoNet)                         <- isotropic Gaussian, baseline
  V1  Tied-covariance Mahalanobis  (shared Sigma_bar)     <- kill switch
  V2  + per-class shrinkage toward shared cov (mod 1)
  V3  + common-subspace removal of prototype scatter (mod 2)

Master formula:  d_c(q) = -(z-mu_c)^T (Sigma_hat_c + gamma*I)^-1 (z-mu_c)
ProtoNet is Sigma_hat_c = 0 (pure ridge). Mods #3 (soft spectrum via the inverse) and
#4 (L2-normalised, angular geometry) are baked in. Accuracy is argmax => temperature only
affects NLL/ECE, not accuracy.

Reuses the repo's DatasetManager (exact FSCIL splits + seeded shots) and AccuracyEvaluator
(identical metric), and the existing DINOv2 feature pipeline from models/bimc_dino_fusion.py.
Features are extracted ONCE; baseline and every variation rung are scored on the SAME
features and SAME support sets (fair + fast). Grid over gamma/rho/r; best config per variant
reported, plus a fixed default (gamma=1, rho=0.8, r=5) reference.
"""

import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from main import setup_cfg
from utils.util import set_seed, set_gpu
from datasets.data_manager import DatasetManager
from utils.evaluator import AccuracyEvaluator


# --------------------------------------------------------------------------------------
# DINOv2 feature extraction (copied from models/bimc_dino_fusion.py so the baseline is
# identical to the repo's DINOv2 visual-prototype branch). No CLIP, no descriptions.
# --------------------------------------------------------------------------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def _ensure_scaled_dot_product_attention():
    if hasattr(F, "scaled_dot_product_attention"):
        return

    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        scale_factor = (query.size(-1) ** -0.5) if scale is None else scale
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        if is_causal:
            q_len, k_len = query.size(-2), key.size(-2)
            causal = torch.ones((q_len, k_len), dtype=torch.bool, device=query.device).tril()
            attn = attn.masked_fill(~causal, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            else:
                attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)
        if dropout_p and dropout_p > 0.0:
            attn = torch.dropout(attn, dropout_p, train=True)
        return torch.matmul(attn, value)

    F.scaled_dot_product_attention = scaled_dot_product_attention


def load_dinov2(device):
    _ensure_scaled_dot_product_attention()
    default_hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    default_weights = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
    hub_dir = os.environ.get("DINOV2_HUB_DIR", default_hub)
    weights_path = os.environ.get("DINOV2_WEIGHTS", default_weights)
    if os.path.isdir(hub_dir):
        print(f"Loading DinoV2 ViT-B/14 from local hub: {hub_dir}")
        model = torch.hub.load(hub_dir, "dinov2_vitb14", source="local", pretrained=False)
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False)
    print(f"Loading DinoV2 weights: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(model, loader, device, clip_mean, clip_std, dino_mean, dino_std):
    """Loader yields CLIP-normalised images; convert to DINO normalisation, run DINOv2,
    L2-normalise. Returns (feats[N,768] float32, labels[N]) on GPU."""
    feats, labels = [], []
    for batch in loader:
        images = batch["image"].to(device).float()
        y = batch["label"].to(device)
        imgs = images * clip_std + clip_mean
        imgs = imgs.clamp(0.0, 1.0)
        dino_in = (imgs - dino_mean) / dino_std
        f = model(dino_in)
        f = F.normalize(f, dim=-1).float()
        feats.append(f)
        labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


# --------------------------------------------------------------------------------------
# Classifier scoring (all DINOv2-only, training-free)
# --------------------------------------------------------------------------------------
def softmax_probs(logits, temp):
    logits = logits - logits.max(dim=1, keepdim=True).values
    return torch.softmax(logits / temp, dim=1)


def cosine_logits(Zt, mu):
    return Zt @ mu.t()


def maha_logits_shared(Zt, mu, Minv):
    """V1: shared inverse covariance. logit_c = 2 z^T M mu_c - mu_c^T M mu_c
    (drops the per-row-constant z^T M z; argmax/softmax invariant)."""
    ZM = Zt @ Minv                      # [N,D]
    muM = mu @ Minv                     # [C,D]
    term1 = ZM @ mu.t()                 # [N,C]
    term2 = (muM * mu).sum(dim=1)       # [C]
    return 2.0 * term1 - term2.unsqueeze(0)


def maha_logits_perclass(Zt, mu, Minv_stack):
    """V2/V3: per-class inverse covariance. logit_c = -(z-mu_c)^T M_c (z-mu_c)."""
    C = mu.shape[0]
    out = torch.empty((Zt.shape[0], C), device=Zt.device, dtype=Zt.dtype)
    for c in range(C):
        r = Zt - mu[c]                  # [N,D]
        t = r @ Minv_stack[c]           # [N,D]
        out[:, c] = -(t * r).sum(dim=1)
    return out


def build_inv_perclass(Sigma_bar, S_stack, diag_scale, gamma, rho):
    D = Sigma_bar.shape[0]
    eye = torch.eye(D, device=Sigma_bar.device, dtype=Sigma_bar.dtype)
    ridge = gamma * diag_scale
    Sigma_c = (1.0 - rho) * S_stack + rho * Sigma_bar.unsqueeze(0) + ridge * eye.unsqueeze(0)
    return torch.linalg.inv(Sigma_c)


def build_inv_shared(Sigma_bar, diag_scale, gamma):
    D = Sigma_bar.shape[0]
    eye = torch.eye(D, device=Sigma_bar.device, dtype=Sigma_bar.dtype)
    return torch.linalg.inv(Sigma_bar + gamma * diag_scale * eye)


def main():
    t0 = time.time()
    data_cfg = "configs/datasets/cub200_bimc_dino_fusion.yaml"
    train_cfg = "configs/trainers/bimc_dino_fusion.yaml"
    cfg = setup_cfg(data_cfg, train_cfg)
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    device = "cuda"

    print(f"=== DINOv2-only FSCIL :: dataset={cfg.DATASET.NAME} seed={cfg.SEED} "
          f"inc_shot={cfg.DATASET.NUM_INC_SHOT} ===")

    dm = DatasetManager(cfg)
    evaluator = AccuracyEvaluator(dm.class_index_in_task)
    num_tasks = dm.num_tasks

    model = load_dinov2(device)
    clip_mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    clip_std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    dino_mean = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    dino_std = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)

    # Pre-extract the full test set ONCE (no RNG consumed on the test path).
    print("Extracting full test set features once ...")
    test_loader = dm.get_dataloader(num_tasks - 1, source="test", mode="test")
    test_feats, test_labels = extract_features(
        model, test_loader, device, clip_mean, clip_std, dino_mean, dino_std)
    print(f"  test feats: {tuple(test_feats.shape)}  labels in [{int(test_labels.min())},{int(test_labels.max())}]")

    # Grids
    gammas = [0.1, 1.0, 10.0]
    rhos = [0.5, 0.8, 1.0]
    rs_v3 = [5, 10]
    DEFAULT = {"gamma": 1.0, "rho": 0.8, "r": 5}

    # results[variant][config_key] = {'params':..., 'acc_list':[...], 'last':metrics_dict}
    results = {}

    def record(variant, key, params, acc, metrics, is_last):
        slot = results.setdefault(variant, {}).setdefault(
            key, {"params": params, "acc_list": []})
        slot["acc_list"].append(round(float(acc), 3))
        if is_last:
            slot["last"] = {k: metrics[k] for k in
                            ["mean_acc", "base_avg_acc", "inc_avg_acc", "harmonic_acc",
                             "nll", "ece", "task_acc"]}

    class_feats = {}  # class id -> [K,768] (raw L2-normalised DINOv2 features)

    for i in range(num_tasks):
        seen = int(max(dm.class_index_in_task[i]) + 1)
        # --- support for this session's new classes (mode='test' transform, matches engine) ---
        train_loader = dm.get_dataloader(i, source="train", mode="test", accumulate_past=False)
        sf, sl = extract_features(model, train_loader, device, clip_mean, clip_std, dino_mean, dino_std)
        for c in torch.unique(sl).tolist():
            fc = sf[sl == c]
            class_feats[c] = fc if c not in class_feats else torch.cat([class_feats[c], fc], 0)

        # --- per-class statistics over all SEEN classes (raw space) ---
        D = sf.shape[1]
        mu_raw = torch.zeros((seen, D), device=device)
        S_stack = torch.zeros((seen, D, D), device=device)
        W = torch.zeros((D, D), device=device)
        dof = 0
        for c in range(seen):
            Xc = class_feats[c]
            Kc = Xc.shape[0]
            m = Xc.mean(dim=0)
            mu_raw[c] = m
            d = Xc - m
            scatter = d.t() @ d
            W += scatter
            if Kc > 1:
                S_stack[c] = scatter / (Kc - 1)
                dof += (Kc - 1)
        dof = max(dof, 1)
        Sigma_bar = W / dof
        diag_scale = torch.diagonal(Sigma_bar).mean().clamp_min(1e-8)
        mu = F.normalize(mu_raw, dim=-1)  # shared prototype location for all rungs

        # --- test subset for this session ---
        mask = test_labels < seen
        Zt = test_feats[mask]
        yt = test_labels[mask]
        is_last = (i == num_tasks - 1)

        # B0 cosine baseline
        probs = softmax_probs(cosine_logits(Zt, mu), temp=0.01)  # /0.01 == x100
        m = evaluator.calc_accuracy(probs, yt, i)
        record("B0_cosine", "B0", {}, m["mean_acc"], m, is_last)

        # V1 tied covariance (shared), sweep gamma
        for g in gammas:
            Minv = build_inv_shared(Sigma_bar, diag_scale, g)
            lg = maha_logits_shared(Zt, mu, Minv)
            probs = softmax_probs(lg, temp=lg.std().clamp_min(1e-6))
            m = evaluator.calc_accuracy(probs, yt, i)
            record("V1_tied_cov", f"g={g}", {"gamma": g}, m["mean_acc"], m, is_last)
            del Minv, lg

        # V2 per-class shrinkage, sweep gamma x rho  (r=0)
        for g in gammas:
            for rho in rhos:
                Minv = build_inv_perclass(Sigma_bar, S_stack, diag_scale, g, rho)
                lg = maha_logits_perclass(Zt, mu, Minv)
                probs = softmax_probs(lg, temp=lg.std().clamp_min(1e-6))
                m = evaluator.calc_accuracy(probs, yt, i)
                record("V2_shrinkage", f"g={g}|rho={rho}", {"gamma": g, "rho": rho},
                       m["mean_acc"], m, is_last)
                del Minv, lg

        # V3 + common-subspace removal, sweep r x gamma x rho
        eye = torch.eye(D, device=device)
        centered_mu = mu - mu.mean(dim=0, keepdim=True)
        Cmat = centered_mu.t() @ centered_mu
        evals, evecs = torch.linalg.eigh(Cmat)  # ascending
        for r in rs_v3:
            Ur = evecs[:, -r:]
            P = Ur @ Ur.t()
            IP = eye - P
            mu_n = mu @ IP                       # null normalised prototype (location)
            W_n = IP @ W @ IP
            S_n = torch.matmul(torch.matmul(IP, S_stack), IP)
            Sigma_bar_n = W_n / dof
            diag_scale_n = torch.diagonal(Sigma_bar_n).mean().clamp_min(1e-8)
            Zt_n = Zt @ IP
            for g in gammas:
                for rho in rhos:
                    Minv = build_inv_perclass(Sigma_bar_n, S_n, diag_scale_n, g, rho)
                    lg = maha_logits_perclass(Zt_n, mu_n, Minv)
                    probs = softmax_probs(lg, temp=lg.std().clamp_min(1e-6))
                    m = evaluator.calc_accuracy(probs, yt, i)
                    record("V3_common_subspace", f"g={g}|rho={rho}|r={r}",
                           {"gamma": g, "rho": rho, "r": r}, m["mean_acc"], m, is_last)
                    del Minv, lg
            del W_n, S_n, Zt_n

        print(f"[task {i:2d}] seen={seen:3d} ntest={int(mask.sum()):4d}  "
              f"B0={results['B0_cosine']['B0']['acc_list'][-1]:.2f}  "
              f"V1*={max(results['V1_tied_cov'][k]['acc_list'][-1] for k in results['V1_tied_cov']):.2f}  "
              f"V2*={max(results['V2_shrinkage'][k]['acc_list'][-1] for k in results['V2_shrinkage']):.2f}  "
              f"V3*={max(results['V3_common_subspace'][k]['acc_list'][-1] for k in results['V3_common_subspace']):.2f}")
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------------------------
    # Aggregate: best config per variant (by avg acc across sessions) + fixed default
    # ----------------------------------------------------------------------------------
    def avg(lst):
        return round(sum(lst) / len(lst), 3)

    summary = {"dataset": cfg.DATASET.NAME, "seed": cfg.SEED, "num_tasks": num_tasks,
               "grid": {"gamma": gammas, "rho": rhos, "r_v3": rs_v3}, "variants": {}}

    def best_of(variant):
        best_key, best_avg = None, -1.0
        for key, slot in results[variant].items():
            a = avg(slot["acc_list"])
            if a > best_avg:
                best_avg, best_key = a, key
        slot = results[variant][best_key]
        return {"config": best_key, "params": slot["params"], "avg_acc": best_avg,
                "final_acc": slot["acc_list"][-1], "acc_list": slot["acc_list"],
                "last": slot.get("last", {})}

    for v in ["B0_cosine", "V1_tied_cov", "V2_shrinkage", "V3_common_subspace"]:
        summary["variants"][v] = {"best": best_of(v), "all": {
            k: {"params": s["params"], "avg_acc": avg(s["acc_list"]),
                "final_acc": s["acc_list"][-1], "acc_list": s["acc_list"]}
            for k, s in results[v].items()}}

    # Fixed-default reference rows (in-grid)
    defaults = {
        "B0_cosine": "B0",
        "V1_tied_cov": f"g={DEFAULT['gamma']}",
        "V2_shrinkage": f"g={DEFAULT['gamma']}|rho={DEFAULT['rho']}",
        "V3_common_subspace": f"g={DEFAULT['gamma']}|rho={DEFAULT['rho']}|r={DEFAULT['r']}",
    }
    summary["fixed_default"] = {"params": DEFAULT}
    for v, key in defaults.items():
        s = results[v][key]
        summary["fixed_default"][v] = {"config": key, "avg_acc": avg(s["acc_list"]),
                                       "final_acc": s["acc_list"][-1], "acc_list": s["acc_list"]}

    summary["runtime_sec"] = round(time.time() - t0, 1)

    out_dir = os.path.join(ROOT, "dino_only_runs", "cub_seed1")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "raw": results}, f, indent=2)

    # ---- pretty summary ----
    lines = []
    lines.append(f"DINOv2-only FSCIL on {cfg.DATASET.NAME} (seed {cfg.SEED}) | {num_tasks} sessions")
    lines.append("Reference (context, NOT this experiment): full CLIP+DINOv2+LLM = 84.93 final / ~85.1 avg")
    lines.append("")
    lines.append("BEST CONFIG PER VARIANT (config chosen on these sessions => optimistic upper bound):")
    lines.append(f"  {'variant':22s} {'avg':>7s} {'final':>7s}  {'base':>6s} {'inc':>6s} {'harm':>6s}  config")
    for v in ["B0_cosine", "V1_tied_cov", "V2_shrinkage", "V3_common_subspace"]:
        b = summary["variants"][v]["best"]
        last = b["last"]
        lines.append(f"  {v:22s} {b['avg_acc']:7.2f} {b['final_acc']:7.2f}  "
                     f"{last.get('base_avg_acc','-'):>6} {last.get('inc_avg_acc','-'):>6} "
                     f"{last.get('harmonic_acc','-'):>6}  {b['config']}")
    lines.append("")
    lines.append(f"FIXED DEFAULT (gamma={DEFAULT['gamma']}, rho={DEFAULT['rho']}, r={DEFAULT['r']}) - non-cherry-picked:")
    lines.append(f"  {'variant':22s} {'avg':>7s} {'final':>7s}")
    for v in ["B0_cosine", "V1_tied_cov", "V2_shrinkage", "V3_common_subspace"]:
        d = summary["fixed_default"][v]
        lines.append(f"  {v:22s} {d['avg_acc']:7.2f} {d['final_acc']:7.2f}")
    lines.append("")
    b0 = summary["variants"]["B0_cosine"]["best"]
    v1 = summary["variants"]["V1_tied_cov"]["best"]
    verdict = "PASS (anisotropy helps)" if v1["avg_acc"] > b0["avg_acc"] else "FAIL (no 2nd-order signal)"
    lines.append(f"KILL SWITCH: V1 avg {v1['avg_acc']:.2f} vs B0 avg {b0['avg_acc']:.2f}  =>  {verdict}")
    lines.append("")
    lines.append("Per-session best-config acc_list:")
    for v in ["B0_cosine", "V1_tied_cov", "V2_shrinkage", "V3_common_subspace"]:
        lines.append(f"  {v:22s} {summary['variants'][v]['best']['acc_list']}")
    lines.append(f"\nruntime_sec: {summary['runtime_sec']}")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_dir}/results.json and summary.txt")


if __name__ == "__main__":
    main()
