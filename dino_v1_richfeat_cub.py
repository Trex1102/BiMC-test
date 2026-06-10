"""
V1 (tied-covariance Mahalanobis) on ENRICHED DINOv2 representations -- CUB, seed 1.

Keeps the winning classifier (V1: shared pooled covariance, d_c = -(z-mu_c)^T (Sigma_bar+gamma*delta*I)^-1 (z-mu_c))
and only changes the feature z, to test:
  (1) earlier / middle-layer CLS token instead of only the last,
  (2) patch-token aggregation (mean + foreground-weighted pooling on bird regions),
  (3) concatenation of layers (mid + high level),
  (4) covariance computed on part-sensitive features (CLS+patch, multi-layer), not only the
      final global vector.

Foreground weighting (training-free): for each patch token p_i and same-layer CLS c,
  w_i = softmax_i( cos(p_i, c) / tau ),  fg = sum_i w_i * p_hat_i   (focuses on object/bird patches).

Representations (each scored with B0 cosine [reference] and V1 [headline]):
  R0 final CLS (block 11)            -- baseline (== previous best V1)
  R1 mid CLS (block 8)
  R2 early-mid CLS (block 5)
  R3 patch mean-pool (block 11)
  R4 patch fg-pool (block 11)
  R5 CLS (+) fg-patch (block 11)
  R6 multi-layer CLS {5,8,11}
  R7 multi-layer CLS(+)fg {5,8,11}   -- full part-sensitive, multi-level

All sub-vectors L2-normalized before concat, full vector L2-normalized after. Features
extracted ONCE; every representation/classifier scored on the same support + queries.
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

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)

LAYERS = (5, 8, 11)        # early-mid, mid-high, last  (DINOv2 ViT-B/14 has 12 blocks)
TAU = 0.1                  # foreground pooling temperature
BANK_ORDER = ["cls5", "cls8", "cls11", "fg5", "fg8", "fg11", "mean11"]
SL = {name: i for i, name in enumerate(BANK_ORDER)}
DIMU = 768

REPS = {
    "R0_final_cls_b11":       ["cls11"],
    "R1_mid_cls_b8":          ["cls8"],
    "R2_earlymid_cls_b5":     ["cls5"],
    "R3_patch_meanpool_b11":  ["mean11"],
    "R4_patch_fgpool_b11":    ["fg11"],
    "R5_cls+fgpatch_b11":     ["cls11", "fg11"],
    "R6_multicls_5_8_11":     ["cls5", "cls8", "cls11"],
    "R7_multi_cls+fg_5_8_11": ["cls5", "fg5", "cls8", "fg8", "cls11", "fg11"],
}
GAMMAS = [0.1, 1.0, 10.0]
DEFAULT_GAMMA = 1.0


def _ensure_scaled_dot_product_attention():
    if hasattr(F, "scaled_dot_product_attention"):
        return

    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        sf = (query.size(-1) ** -0.5) if scale is None else scale
        attn = torch.matmul(query, key.transpose(-2, -1)) * sf
        if attn_mask is not None:
            attn = attn + (attn_mask if attn_mask.dtype != torch.bool else (~attn_mask) * -1e9)
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, value)

    F.scaled_dot_product_attention = sdpa


def load_dinov2(device):
    _ensure_scaled_dot_product_attention()
    hub_dir = os.environ.get("DINOV2_HUB_DIR", os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"))
    weights = os.environ.get("DINOV2_WEIGHTS", os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"))
    print(f"Loading DinoV2 ViT-B/14 (local hub): {hub_dir}")
    model = torch.hub.load(hub_dir, "dinov2_vitb14", source="local", pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_banks(model, loader, device, cm, cs, dmn, dsn):
    """Return bank [N, 7*768] (BANK_ORDER, each sub-vector L2-normalized) + labels."""
    rows, labels = [], []
    for batch in loader:
        images = batch["image"].to(device).float()
        y = batch["label"].to(device)
        x = ((images * cs + cm).clamp(0.0, 1.0) - dmn) / dsn
        outs = model.get_intermediate_layers(x, n=list(LAYERS), return_class_token=True, norm=True)
        cls, fg, mean11 = {}, {}, None
        for li, l in enumerate(LAYERS):
            patch, cls_t = outs[li]                       # [B,N,D], [B,D]
            cn = F.normalize(cls_t, dim=-1)
            pn = F.normalize(patch, dim=-1)               # [B,N,D]
            sim = torch.einsum("bnd,bd->bn", pn, cn)      # patch-to-CLS cosine
            w = torch.softmax(sim / TAU, dim=1)           # foreground weights
            f = F.normalize(torch.einsum("bn,bnd->bd", w, pn), dim=-1)
            cls[l], fg[l] = cn, f
            if l == 11:
                mean11 = F.normalize(pn.mean(dim=1), dim=-1)
        bank = torch.cat([cls[5], cls[8], cls[11], fg[5], fg[8], fg[11], mean11], dim=1)
        rows.append(bank)
        labels.append(y)
    return torch.cat(rows, 0), torch.cat(labels, 0)


def assemble(bank, names):
    parts = [bank[:, SL[n] * DIMU:(SL[n] + 1) * DIMU] for n in names]
    return F.normalize(torch.cat(parts, dim=1), dim=-1)


def softmax_probs(logits, temp):
    logits = logits - logits.max(dim=1, keepdim=True).values
    return torch.softmax(logits / temp, dim=1)


def v1_logits(Zt, mu, Minv):
    ZM = Zt @ Minv
    muM = mu @ Minv
    return 2.0 * (ZM @ mu.t()) - (muM * mu).sum(dim=1).unsqueeze(0)


def class_stats(Zsup_by_class, seen, device, D):
    mu_raw = torch.zeros((seen, D), device=device)
    W = torch.zeros((D, D), device=device)
    dof = 0
    for c in range(seen):
        Xc = Zsup_by_class[c]
        m = Xc.mean(dim=0)
        mu_raw[c] = m
        d = Xc - m
        W += d.t() @ d
        if Xc.shape[0] > 1:
            dof += (Xc.shape[0] - 1)
    dof = max(dof, 1)
    Sigma_bar = W / dof
    delta = torch.diagonal(Sigma_bar).mean().clamp_min(1e-8)
    mu = F.normalize(mu_raw, dim=-1)
    return mu, Sigma_bar, delta


def main():
    t0 = time.time()
    cfg = setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml",
                    "configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    device = "cuda"
    print(f"=== V1 on enriched DINOv2 reps :: {cfg.DATASET.NAME} seed={cfg.SEED} "
          f"layers={LAYERS} tau={TAU} ===")

    dm = DatasetManager(cfg)
    evaluator = AccuracyEvaluator(dm.class_index_in_task)
    T = dm.num_tasks
    model = load_dinov2(device)
    cm = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    cs = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    dmn = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    dsn = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)

    print("Extracting full test-set banks once ...")
    test_loader = dm.get_dataloader(T - 1, source="test", mode="test")
    test_bank, test_labels = extract_banks(model, test_loader, device, cm, cs, dmn, dsn)
    print(f"  test bank: {tuple(test_bank.shape)}")

    # results[rep][clf_key] = {'acc_list':[...], 'last':{...}}
    results = {}

    def record(rep, key, acc, metrics, is_last):
        slot = results.setdefault(rep, {}).setdefault(key, {"acc_list": []})
        slot["acc_list"].append(round(float(acc), 3))
        if is_last:
            slot["last"] = {k: metrics[k] for k in
                            ["mean_acc", "base_avg_acc", "inc_avg_acc", "harmonic_acc"]}

    class_bank = {}  # class id -> [K, 7*768]

    for t in range(T):
        seen = int(max(dm.class_index_in_task[t]) + 1)
        tr_loader = dm.get_dataloader(t, source="train", mode="test", accumulate_past=False)
        sb, sl = extract_banks(model, tr_loader, device, cm, cs, dmn, dsn)
        for c in torch.unique(sl).tolist():
            fc = sb[sl == c]
            class_bank[c] = fc if c not in class_bank else torch.cat([class_bank[c], fc], 0)

        mask = test_labels < seen
        yt = test_labels[mask]
        is_last = (t == T - 1)

        for rep, names in REPS.items():
            Zt = assemble(test_bank[mask], names)
            Zsup = {c: assemble(class_bank[c], names) for c in range(seen)}
            D = Zt.shape[1]
            mu, Sigma_bar, delta = class_stats(Zsup, seen, device, D)

            # B0 cosine (reference)
            probs = softmax_probs(Zt @ mu.t(), temp=0.01)
            m = evaluator.calc_accuracy(probs, yt, t)
            record(rep, "B0", m["mean_acc"], m, is_last)

            # V1 tied covariance (headline), gamma grid
            eye = torch.eye(D, device=device)
            for g in GAMMAS:
                Minv = torch.linalg.inv(Sigma_bar + g * delta * eye)
                lg = v1_logits(Zt, mu, Minv)
                probs = softmax_probs(lg, temp=lg.std().clamp_min(1e-6))
                m = evaluator.calc_accuracy(probs, yt, t)
                record(rep, f"V1|g={g}", m["mean_acc"], m, is_last)
                del Minv, lg
            del Zt, Zsup, mu, Sigma_bar, eye

        print(f"[task {t:2d}] seen={seen:3d}  " + "  ".join(
            f"{rep.split('_')[0]}:V1*={max(results[rep][k]['acc_list'][-1] for k in results[rep] if k.startswith('V1')):.2f}"
            for rep in REPS))
        torch.cuda.empty_cache()

    # ---------------- aggregate ----------------
    def avg(lst):
        return round(sum(lst) / len(lst), 3)

    out = {"dataset": cfg.DATASET.NAME, "seed": cfg.SEED, "layers": list(LAYERS),
           "tau": TAU, "gammas": GAMMAS, "reps": {}}
    for rep in REPS:
        b0 = results[rep]["B0"]
        v1_keys = [k for k in results[rep] if k.startswith("V1")]
        best_k = max(v1_keys, key=lambda k: avg(results[rep][k]["acc_list"]))
        dflt_k = f"V1|g={DEFAULT_GAMMA}"
        out["reps"][rep] = {
            "B0": {"avg": avg(b0["acc_list"]), "final": b0["acc_list"][-1],
                   "acc_list": b0["acc_list"], "last": b0.get("last", {})},
            "V1_best": {"config": best_k, "avg": avg(results[rep][best_k]["acc_list"]),
                        "final": results[rep][best_k]["acc_list"][-1],
                        "acc_list": results[rep][best_k]["acc_list"],
                        "last": results[rep][best_k].get("last", {})},
            "V1_default_g1": {"avg": avg(results[rep][dflt_k]["acc_list"]),
                              "final": results[rep][dflt_k]["acc_list"][-1],
                              "acc_list": results[rep][dflt_k]["acc_list"]},
        }
    out["runtime_sec"] = round(time.time() - t0, 1)

    out_dir = os.path.join(ROOT, "dino_richfeat_runs", "cub_seed1")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": out, "raw": results}, f, indent=2)

    base = out["reps"]["R0_final_cls_b11"]["V1_best"]["avg"]
    lines = []
    lines.append(f"V1 (tied-cov Mahalanobis) on enriched DINOv2 reps | {cfg.DATASET.NAME} seed {cfg.SEED} | layers {LAYERS} tau {TAU}")
    lines.append(f"Baseline R0 (final CLS) V1 avg = {base:.2f}  (prev experiment best). dim shown per rep.")
    lines.append("")
    lines.append(f"  {'representation':24s} {'dim':>5s} {'B0avg':>6s} | {'V1best':>7s} {'final':>6s} {'gamma':>6s} {'dVS_R0':>7s} | {'V1g=1':>6s}")
    for rep, names in REPS.items():
        r = out["reps"][rep]
        dim = len(names) * DIMU
        d_vs = r["V1_best"]["avg"] - base
        gstr = r["V1_best"]["config"].split("=")[-1]
        lines.append(f"  {rep:24s} {dim:5d} {r['B0']['avg']:6.2f} | "
                     f"{r['V1_best']['avg']:7.2f} {r['V1_best']['final']:6.2f} {gstr:>6s} "
                     f"{d_vs:+7.2f} | {r['V1_default_g1']['avg']:6.2f}")
    lines.append("")
    # base/inc breakdown for the strongest reps
    lines.append(f"  {'representation':24s} {'base':>6s} {'inc':>6s} {'harm':>6s}  (V1 best, final session)")
    for rep in REPS:
        last = out["reps"][rep]["V1_best"]["last"]
        lines.append(f"  {rep:24s} {last.get('base_avg_acc','-'):>6} {last.get('inc_avg_acc','-'):>6} {last.get('harmonic_acc','-'):>6}")
    lines.append("")
    best_rep = max(REPS, key=lambda r: out["reps"][r]["V1_best"]["avg"])
    lines.append(f"BEST REP: {best_rep}  V1 avg={out['reps'][best_rep]['V1_best']['avg']:.2f} "
                 f"(R0 baseline {base:.2f}, delta {out['reps'][best_rep]['V1_best']['avg']-base:+.2f})")
    lines.append(f"runtime_sec: {out['runtime_sec']}")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_dir}/results.json and summary.txt")


if __name__ == "__main__":
    main()
