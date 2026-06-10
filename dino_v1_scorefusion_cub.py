"""
Score-level fusion of two V1 (tied-cov Mahalanobis) heads on DINOv2 -- CUB, seed 1.

Motivation: concatenating CLS with patch features hurt (prev experiment). Here we instead keep
TWO independent V1 heads and fuse them at the SCORE level:
    Head A = V1 on the final CLS token (block 11)
    Head B = V1 on a foreground-pooled patch vector (block 11)
    fused logits = (1-w) * zrow(L_cls) + w * zrow(L_patch),   argmax_c
where zrow standardizes each head's logits per query (so the two heads are comparable).
w=0 == pure CLS (must reproduce 85.38); w=1 == pure patch head.

Stronger foreground masks for Head B (training-free), compared:
    mean   : uniform mean of patch tokens
    fgcos  : softmax( cos(patch, CLS) / tau )          (cosine-to-CLS saliency)
    fgpca  : DINOv2 PCA-segmentation -- top PCA component of the image's patch tokens,
             border-based sign disambiguation (object is off-border), softmax(zscore(proj)/tau)

Sweeps foreground method x patch-ridge gamma_patch in {0.1,1.0} x w in {0..1}.  CLS head gamma=1.
Features extracted ONCE; all heads/fusions scored on identical support + queries.
"""

import os
import sys
import json
import time
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from main import setup_cfg
from utils.util import set_seed, set_gpu
from datasets.data_manager import DatasetManager
from utils.evaluator import AccuracyEvaluator

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)

BLOCK = 11
TAU_COS = 0.1
TAU_PCA = 0.5
DIMU = 768
BANK_ORDER = ["cls11", "fgcos", "fgpca", "mean"]
SL = {n: i for i, n in enumerate(BANK_ORDER)}

GAMMA_CLS = 1.0
GAMMAS_PATCH = [0.1, 1.0]
OMEGAS = [round(0.1 * k, 1) for k in range(0, 11)]  # 0.0 .. 1.0
METHODS = ["mean", "fgcos", "fgpca"]
METHOD_BANK = {"mean": "mean", "fgcos": "fgcos", "fgpca": "fgpca"}


def _ensure_sdpa():
    if hasattr(F, "scaled_dot_product_attention"):
        return

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        sf = (q.size(-1) ** -0.5) if scale is None else scale
        a = torch.matmul(q, k.transpose(-2, -1)) * sf
        if attn_mask is not None:
            a = a + (attn_mask if attn_mask.dtype != torch.bool else (~attn_mask) * -1e9)
        return torch.matmul(torch.softmax(a, dim=-1), v)

    F.scaled_dot_product_attention = sdpa


def load_dinov2(device):
    _ensure_sdpa()
    hub = os.environ.get("DINOV2_HUB_DIR", os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"))
    w = os.environ.get("DINOV2_WEIGHTS", os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"))
    print(f"Loading DinoV2 ViT-B/14 (local hub): {hub}")
    model = torch.hub.load(hub, "dinov2_vitb14", source="local", pretrained=False)
    model.load_state_dict(torch.load(w, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _border_mask(N, device):
    g = int(round(math.sqrt(N)))
    if g * g != N:
        return None
    idx = torch.arange(N, device=device).view(g, g)
    m = torch.zeros(g, g, dtype=torch.bool, device=device)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m.view(N)


@torch.no_grad()
def pca_foreground(patch, pn, border):
    """patch [B,N,D] raw (post-LN) tokens; pn = normalized patch tokens. Returns fg [B,D]."""
    Pc = patch - patch.mean(dim=1, keepdim=True)
    # top right singular vector per image
    _, _, Vh = torch.linalg.svd(Pc, full_matrices=False)   # Vh [B, min(N,D), D]
    pc1 = Vh[:, 0, :]                                       # [B,D]
    proj = torch.einsum("bnd,bd->bn", Pc, pc1)             # [B,N]
    if border is not None:
        bmean = proj[:, border].mean(dim=1, keepdim=True)  # background side
        proj = proj * torch.where(bmean > 0, -1.0, 1.0)    # foreground -> positive
    z = (proj - proj.mean(dim=1, keepdim=True)) / proj.std(dim=1, keepdim=True).clamp_min(1e-6)
    w = torch.softmax(z / TAU_PCA, dim=1)
    return F.normalize(torch.einsum("bn,bnd->bd", w, pn), dim=-1)


@torch.no_grad()
def extract_banks(model, loader, device, cm, cs, dmn, dsn):
    rows, labels, border = [], [], None
    for batch in loader:
        images = batch["image"].to(device).float()
        y = batch["label"].to(device)
        x = ((images * cs + cm).clamp(0.0, 1.0) - dmn) / dsn
        (patch, cls_t), = model.get_intermediate_layers(x, n=[BLOCK], return_class_token=True, norm=True)
        cn = F.normalize(cls_t, dim=-1)
        pn = F.normalize(patch, dim=-1)
        if border is None:
            border = _border_mask(patch.shape[1], device)
        sim = torch.einsum("bnd,bd->bn", pn, cn)
        wcos = torch.softmax(sim / TAU_COS, dim=1)
        fgcos = F.normalize(torch.einsum("bn,bnd->bd", wcos, pn), dim=-1)
        fgpca = pca_foreground(patch, pn, border)
        mean = F.normalize(pn.mean(dim=1), dim=-1)
        rows.append(torch.cat([cn, fgcos, fgpca, mean], dim=1))
        labels.append(y)
    return torch.cat(rows, 0), torch.cat(labels, 0)


def get_rep(bank, name):
    return bank[:, SL[name] * DIMU:(SL[name] + 1) * DIMU]


def softmax_probs(logits, temp=0.1):
    logits = logits - logits.max(dim=1, keepdim=True).values
    return torch.softmax(logits / temp, dim=1)


def zrow(L):
    return (L - L.mean(dim=1, keepdim=True)) / L.std(dim=1, keepdim=True).clamp_min(1e-6)


def v1_logits(Zt, mu, Minv):
    ZM = Zt @ Minv
    muM = mu @ Minv
    return 2.0 * (ZM @ mu.t()) - (muM * mu).sum(dim=1).unsqueeze(0)


def head_logits(test_rep, sup_by_class, seen, gamma, device):
    D = test_rep.shape[1]
    mu_raw = torch.zeros((seen, D), device=device)
    W = torch.zeros((D, D), device=device)
    dof = 0
    for c in range(seen):
        Xc = sup_by_class[c]
        m = Xc.mean(dim=0)
        mu_raw[c] = m
        d = Xc - m
        W += d.t() @ d
        if Xc.shape[0] > 1:
            dof += Xc.shape[0] - 1
    dof = max(dof, 1)
    Sigma_bar = W / dof
    delta = torch.diagonal(Sigma_bar).mean().clamp_min(1e-8)
    mu = F.normalize(mu_raw, dim=-1)
    Minv = torch.linalg.inv(Sigma_bar + gamma * delta * torch.eye(D, device=device))
    return v1_logits(test_rep, mu, Minv)


def main():
    t0 = time.time()
    cfg = setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml",
                    "configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    device = "cuda"
    print(f"=== V1 score-fusion (CLS + fg-patch heads) :: {cfg.DATASET.NAME} seed={cfg.SEED} "
          f"block={BLOCK} tau_cos={TAU_COS} tau_pca={TAU_PCA} ===")

    dm = DatasetManager(cfg)
    ev = AccuracyEvaluator(dm.class_index_in_task)
    T = dm.num_tasks
    model = load_dinov2(device)
    cm = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    cs = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    dmn = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    dsn = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)

    print("Extracting full test-set banks once ...")
    test_bank, test_labels = extract_banks(model, dm.get_dataloader(T - 1, "test", mode="test"),
                                           device, cm, cs, dmn, dsn)
    print(f"  test bank: {tuple(test_bank.shape)}")

    # results: 'CLS' -> acc_list ; (method,gp,w) -> acc_list
    results = {"CLS": {"acc_list": []}}

    def rec(key, acc, metrics=None, is_last=False, store=None):
        slot = results.setdefault(key, {"acc_list": []})
        slot["acc_list"].append(round(float(acc), 3))
        if is_last and metrics is not None:
            slot["last"] = {k: metrics[k] for k in ["base_avg_acc", "inc_avg_acc", "harmonic_acc"]}

    class_bank = {}
    for t in range(T):
        seen = int(max(dm.class_index_in_task[t]) + 1)
        sb, sl = extract_banks(model, dm.get_dataloader(t, "train", mode="test", accumulate_past=False),
                               device, cm, cs, dmn, dsn)
        for c in torch.unique(sl).tolist():
            fc = sb[sl == c]
            class_bank[c] = fc if c not in class_bank else torch.cat([class_bank[c], fc], 0)

        mask = test_labels < seen
        yt = test_labels[mask]
        is_last = (t == T - 1)

        # CLS head (gamma=1)
        Zc = get_rep(test_bank[mask], "cls11")
        sup_c = {c: get_rep(class_bank[c], "cls11") for c in range(seen)}
        Lcls = zrow(head_logits(Zc, sup_c, seen, GAMMA_CLS, device))
        m = ev.calc_accuracy(softmax_probs(Lcls), yt, t)
        rec("CLS", m["mean_acc"], m, is_last)

        for method in METHODS:
            name = METHOD_BANK[method]
            Zp = get_rep(test_bank[mask], name)
            sup_p = {c: get_rep(class_bank[c], name) for c in range(seen)}
            for gp in GAMMAS_PATCH:
                Lp = zrow(head_logits(Zp, sup_p, seen, gp, device))
                for w in OMEGAS:
                    fused = (1.0 - w) * Lcls + w * Lp
                    mm = ev.calc_accuracy(softmax_probs(fused), yt, t)
                    rec(f"{method}|gp={gp}|w={w}", mm["mean_acc"], mm, is_last)
                del Lp
        torch.cuda.empty_cache()
        print(f"[task {t:2d}] seen={seen:3d} CLS={results['CLS']['acc_list'][-1]:.2f}  " + "  ".join(
            f"{mth}*={max(results[f'{mth}|gp={gp}|w={w}']['acc_list'][-1] for gp in GAMMAS_PATCH for w in OMEGAS if w>0):.2f}"
            for mth in METHODS))

    # ---------------- aggregate ----------------
    def avg(lst):
        return round(sum(lst) / len(lst), 3)

    cls_avg = avg(results["CLS"]["acc_list"])
    cls_final = results["CLS"]["acc_list"][-1]
    out = {"dataset": cfg.DATASET.NAME, "seed": cfg.SEED, "block": BLOCK,
           "tau_cos": TAU_COS, "tau_pca": TAU_PCA, "gamma_cls": GAMMA_CLS,
           "gammas_patch": GAMMAS_PATCH, "omegas": OMEGAS,
           "CLS_only": {"avg": cls_avg, "final": cls_final, "acc_list": results["CLS"]["acc_list"],
                        "last": results["CLS"].get("last", {})},
           "methods": {}}

    for method in METHODS:
        # pure patch head (w=1), best gp
        pure = {}
        for gp in GAMMAS_PATCH:
            a = avg(results[f"{method}|gp={gp}|w=1.0"]["acc_list"])
            if not pure or a > pure["avg"]:
                pure = {"gp": gp, "avg": a, "final": results[f"{method}|gp={gp}|w=1.0"]["acc_list"][-1]}
        # best fused over gp x w>0
        best = {"avg": -1}
        curve_gp = None
        for gp in GAMMAS_PATCH:
            for w in OMEGAS:
                if w == 0:
                    continue
                a = avg(results[f"{method}|gp={gp}|w={w}"]["acc_list"])
                if a > best["avg"]:
                    best = {"gp": gp, "w": w, "avg": a,
                            "final": results[f"{method}|gp={gp}|w={w}"]["acc_list"][-1],
                            "last": results[f"{method}|gp={gp}|w={w}"].get("last", {})}
                    curve_gp = gp
        curve = {w: avg(results[f"{method}|gp={curve_gp}|w={w}"]["acc_list"]) for w in OMEGAS}
        out["methods"][method] = {"pure_patch": pure, "best_fused": best,
                                  "curve_gp": curve_gp, "omega_curve": curve}

    out["runtime_sec"] = round(time.time() - t0, 1)
    out_dir = os.path.join(ROOT, "dino_scorefusion_runs", "cub_seed1")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": out, "raw": results}, f, indent=2)

    lines = []
    lines.append(f"V1 score-fusion (CLS + fg-patch heads) | {cfg.DATASET.NAME} seed {cfg.SEED} | block {BLOCK}")
    lines.append(f"CLS-only head (w=0 baseline): avg {cls_avg:.2f}  final {cls_final:.2f}   <- target to beat (prev best 85.38)")
    lines.append("")
    lines.append(f"  {'fg method':8s} | {'pure patch (w=1)':>18s} | {'best fused':>26s} | {'dVS_CLS':>8s}")
    for method in METHODS:
        mm = out["methods"][method]
        p = mm["pure_patch"]; b = mm["best_fused"]
        lines.append(f"  {method:8s} | avg {p['avg']:6.2f} (gp={p['gp']}) | "
                     f"avg {b['avg']:6.2f} final {b['final']:6.2f} (gp={b['gp']},w={b['w']}) | {b['avg']-cls_avg:+7.2f}")
    lines.append("")
    # base/inc/harm for best fused per method
    lines.append(f"  {'fg method':8s} {'base':>6s} {'inc':>6s} {'harm':>6s}  (best fused, final session)")
    lines.append(f"  {'CLS_only':8s} {out['CLS_only']['last'].get('base_avg_acc','-'):>6} "
                 f"{out['CLS_only']['last'].get('inc_avg_acc','-'):>6} {out['CLS_only']['last'].get('harmonic_acc','-'):>6}")
    for method in METHODS:
        last = out["methods"][method]["best_fused"].get("last", {})
        lines.append(f"  {method:8s} {last.get('base_avg_acc','-'):>6} {last.get('inc_avg_acc','-'):>6} {last.get('harmonic_acc','-'):>6}")
    lines.append("")
    for method in METHODS:
        c = out["methods"][method]
        lines.append(f"  omega-curve [{method}, gp={c['curve_gp']}]: " +
                     " ".join(f"{w}:{c['omega_curve'][w]:.2f}" for w in OMEGAS))
    lines.append("")
    best_method = max(METHODS, key=lambda mth: out["methods"][mth]["best_fused"]["avg"])
    bf = out["methods"][best_method]["best_fused"]
    lines.append(f"BEST: {best_method} fused avg={bf['avg']:.2f} (gp={bf['gp']}, w={bf['w']}) "
                 f"vs CLS-only {cls_avg:.2f}  =>  delta {bf['avg']-cls_avg:+.2f}")
    lines.append(f"runtime_sec: {out['runtime_sec']}")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_dir}/results.json and summary.txt")


if __name__ == "__main__":
    main()
