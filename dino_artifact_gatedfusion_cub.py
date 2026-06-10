"""
Week-1 check B: ARTIFACT-token cleanup + CONDITIONAL/GATED fusion -- CUB seed 1, DINOv2 ViT-B/14 @ 518.

Tests the three critiques of the "patches are redundant" conclusion:
  (i)  Artifact confound: high-norm DINOv2 patch tokens pollute pooling. Detect-and-DROP the
       top-p% by L2 norm before foreground pooling; does the patch head improve?
  (ii) Metric confound: a single GLOBAL omega optimizes total (base-dominated) accuracy and lands
       at 0 even if patches help a useful minority. Refit omega CONDITIONALLY:
         - novel-only queries (true label >= num_base)        [oracle diagnostic]
         - low-CLS-margin queries (bottom 20% top1-top2)      [label-free signal]
       If omega*_conditional > 0 while omega*_global = 0, the redundancy claim is a metric artifact.
  (iii) Deployable gate: inject the patch head only when CLS top-1 margin is low (bottom tau%),
        weight omega_gate; does gated fusion beat pure CLS overall?

Two independent V1 heads (CLS gamma=1, patch gamma=0.1), fused as row-z-scored logits.
Overall accuracy computed directly (== mean_acc). Features extracted once; single pass.
"""
import os, sys, json, time, math
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader
from main import setup_cfg
from utils.util import set_seed, set_gpu
from datasets.data_manager import DatasetManager

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
R = 448
BS = 6
BLOCK = 11
TAU_COS = 0.1
DROPS = [0.0, 0.02, 0.05, 0.10]          # fraction of highest-norm patches dropped
OMEGAS = [round(0.1 * k, 1) for k in range(11)]
TAU_PCTS = [10, 20, 30]                   # gate the bottom-tau% margin queries
WGATES = [0.3, 0.5, 0.7]
GAMMA_CLS = 1.0
GAMMA_PATCH = 0.1
DIMU = 768
NAMES = ["cls"] + [f"fg{int(p*100)}" for p in DROPS]   # cls, fg0, fg2, fg5, fg10
SL = {n: i for i, n in enumerate(NAMES)}


def _ensure_sdpa():
    if hasattr(F, "scaled_dot_product_attention"):
        return
    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        sf = (q.size(-1) ** -0.5) if scale is None else scale
        return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * sf, -1), v)
    F.scaled_dot_product_attention = sdpa


def load_dinov2(device):
    _ensure_sdpa()
    hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    w = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
    model = torch.hub.load(hub, "dinov2_vitb14", source="local", pretrained=False)
    model.load_state_dict(torch.load(w, map_location="cpu"), strict=True)
    return model.to(device).eval()


def transform_R(r):
    return transforms.Compose([
        transforms.Resize(r, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(r), transforms.ToTensor(),
        transforms.Normalize(DINO_MEAN, DINO_STD)])


@torch.no_grad()
def extract(model, loader, device):
    rows, labels, hifrac = [], [], []
    for b in loader:
        x = b["image"].to(device)
        (patch, cls), = model.get_intermediate_layers(x, n=[BLOCK], return_class_token=True, norm=True)
        cn = F.normalize(cls, dim=-1)
        norms = patch.norm(dim=-1)                       # [B,N]
        pn = F.normalize(patch, dim=-1)
        sim = torch.einsum("bnd,bd->bn", pn, cn)
        med = norms.median(dim=1, keepdim=True).values
        hifrac.append((norms > 2 * med).float().mean(dim=1))
        vecs = [cn]
        for p in DROPS:
            if p == 0:
                keep = torch.ones_like(norms, dtype=torch.bool)
            else:
                thr = torch.quantile(norms, 1.0 - p, dim=1, keepdim=True)
                keep = norms <= thr
            sm = sim.masked_fill(~keep, float("-inf"))
            wts = torch.softmax(sm / TAU_COS, dim=1)
            vecs.append(F.normalize(torch.einsum("bn,bnd->bd", wts, pn), dim=-1))
        rows.append(torch.cat(vecs, dim=1))
        labels.append(b["label"].to(device))
    return torch.cat(rows, 0), torch.cat(labels, 0), torch.cat(hifrac, 0)


def get(bank, name):
    return bank[:, SL[name] * DIMU:(SL[name] + 1) * DIMU]


def zrow(L):
    return (L - L.mean(1, keepdim=True)) / L.std(1, keepdim=True).clamp_min(1e-6)


def head_logits(Zt, sup, seen, gamma, device):
    D = Zt.shape[1]
    mu_raw = torch.zeros((seen, D), device=device)
    W = torch.zeros((D, D), device=device)
    dof = 0
    for c in range(seen):
        Xc = sup[c]; m = Xc.mean(0); mu_raw[c] = m; d = Xc - m; W += d.t() @ d
        if Xc.shape[0] > 1:
            dof += Xc.shape[0] - 1
    Sigma = W / max(dof, 1)
    delta = torch.diagonal(Sigma).mean().clamp_min(1e-8)
    mu = F.normalize(mu_raw, dim=-1)
    Minv = torch.linalg.inv(Sigma + gamma * delta * torch.eye(D, device=device))
    ZM = Zt @ Minv
    return 2.0 * (ZM @ mu.t()) - ((mu @ Minv) * mu).sum(1).unsqueeze(0)


def acc(logits, yt):
    return (logits.argmax(1) == yt).float().mean().item() * 100.0


def main():
    t0 = time.time()
    cfg = setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml",
                    "configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    device = "cuda"
    dm = DatasetManager(cfg)
    T = dm.num_tasks
    num_base = len(dm.class_index_in_task[0])
    model = load_dinov2(device)
    print(f"=== artifact + gated fusion :: {cfg.DATASET.NAME} seed {cfg.SEED} R={R} num_base={num_base} ===", flush=True)

    # transforms for support + test at R
    dm.test_transform = transform_R(R)
    def small_loader(ds):
        return DataLoader(ds, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)
    print("Extracting test once ...", flush=True)
    test_bank, test_labels, test_hi = extract(model, small_loader(dm.get_dataset(T - 1, "test", mode="test")), device)
    print(f"  test bank {tuple(test_bank.shape)}  mean high-norm-token fraction = {test_hi.mean().item()*100:.2f}%", flush=True)

    cls_acc, pure = [], {p: [] for p in DROPS}
    omega_curve = {p: {w: [] for w in OMEGAS} for p in DROPS}
    gated = {p: {(tp, wg): [] for tp in TAU_PCTS for wg in WGATES} for p in DROPS}
    final = {}
    class_bank = {}

    for t in range(T):
        seen = int(max(dm.class_index_in_task[t]) + 1)
        sb, sl, _ = extract(model, small_loader(dm.get_dataset(t, "train", mode="test")), device)
        for c in torch.unique(sl).tolist():
            fc = sb[sl == c]
            class_bank[c] = fc if c not in class_bank else torch.cat([class_bank[c], fc], 0)
        mask = test_labels < seen
        yt = test_labels[mask]
        Zc = get(test_bank[mask], "cls")
        Lc = zrow(head_logits(Zc, {c: get(class_bank[c], "cls") for c in range(seen)}, seen, GAMMA_CLS, device))
        cls_acc.append(round(acc(Lc, yt), 3))
        margin = (Lc.topk(2, dim=1).values[:, 0] - Lc.topk(2, dim=1).values[:, 1])
        for p in DROPS:
            nm = f"fg{int(p*100)}"
            Zp = get(test_bank[mask], nm)
            Lp = zrow(head_logits(Zp, {c: get(class_bank[c], nm) for c in range(seen)}, seen, GAMMA_PATCH, device))
            pure[p].append(round(acc(Lp, yt), 3))
            for w in OMEGAS:
                omega_curve[p][w].append(acc((1 - w) * Lc + w * Lp, yt))
            for tp in TAU_PCTS:
                thr = torch.quantile(margin, tp / 100.0)
                gate = margin <= thr
                for wg in WGATES:
                    fused = (1 - wg) * Lc + wg * Lp
                    pred = torch.where(gate.unsqueeze(1), fused, Lc).argmax(1)
                    gated[p][(tp, wg)].append((pred == yt).float().mean().item() * 100.0)
            if t == T - 1:
                final[p] = (Lc.detach(), Lp.detach(), yt.detach(), margin.detach())
        torch.cuda.empty_cache()
        print(f"[task {t:2d}] seen={seen:3d} CLS={cls_acc[-1]:.2f}  "
              + " ".join(f"pure_fg{int(p*100)}={pure[p][-1]:.1f}" for p in DROPS), flush=True)

    def avg(a):
        return round(sum(a) / len(a), 3)

    # pick best drop level by pure patch avg
    p_star = max(DROPS, key=lambda p: avg(pure[p]))
    cls_avg = avg(cls_acc)

    # conditional omega at final session for p_star
    Lc, Lp, yt, margin = final[p_star]
    lowm = margin <= torch.quantile(margin, 0.20)
    novel = yt >= num_base
    cond = {}
    for sub, idx in [("all", torch.ones_like(yt, dtype=torch.bool)), ("novel", novel), ("lowmargin", lowm)]:
        curve = {w: round(((((1 - w) * Lc + w * Lp).argmax(1) == yt)[idx].float().mean().item()) * 100, 2) for w in OMEGAS}
        wstar = max(OMEGAS, key=lambda w: curve[w])
        cond[sub] = {"n": int(idx.sum().item()), "w_star": wstar, "acc_at_wstar": curve[wstar],
                     "acc_at_0": curve[0.0], "curve": curve}

    # best global fused and best gated
    best_global = {"avg": -1}
    for p in DROPS:
        for w in OMEGAS:
            a = avg(omega_curve[p][w])
            if a > best_global["avg"]:
                best_global = {"p": p, "w": w, "avg": a}
    best_gated = {"avg": -1}
    for p in DROPS:
        for key in gated[p]:
            a = avg(gated[p][key])
            if a > best_gated["avg"]:
                best_gated = {"p": p, "tau_pct": key[0], "w_gate": key[1], "avg": a}

    summary = {
        "dataset": cfg.DATASET.NAME, "seed": cfg.SEED, "resolution": R, "num_base": num_base,
        "high_norm_token_fraction_pct": round(test_hi.mean().item() * 100, 3),
        "CLS_only": {"avg": cls_avg, "final": cls_acc[-1], "acc": cls_acc},
        "pure_patch_avg_by_drop": {f"fg{int(p*100)}": avg(pure[p]) for p in DROPS},
        "p_star": p_star,
        "best_global_fused": best_global,
        "conditional_omega_final_session": cond,
        "best_gated_fusion": best_gated,
        "omega_curve_pstar_global": {w: avg(omega_curve[p_star][w]) for w in OMEGAS},
        "runtime_sec": round(time.time() - t0, 1),
    }
    out_dir = os.path.join(ROOT, "dino_artifact_gated_runs", "cub_seed1")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    L = []
    L.append(f"ARTIFACT + GATED FUSION | {cfg.DATASET.NAME} seed {cfg.SEED} | ViT-B/14 @ {R}")
    L.append(f"high-norm artifact-token fraction (norm>2*median): {summary['high_norm_token_fraction_pct']:.2f}%")
    L.append(f"CLS-only V1 @ {R}: avg {cls_avg:.2f}  final {cls_acc[-1]:.2f}")
    L.append("")
    L.append("Artifact drop -> pure patch head (avg acc):")
    for p in DROPS:
        L.append(f"   drop {int(p*100):2d}% : {avg(pure[p]):.2f}")
    L.append(f"   best drop p_star = {int(p_star*100)}%")
    L.append("")
    L.append(f"Global fused best: p={int(best_global['p']*100)}% w={best_global['w']} avg={best_global['avg']:.2f}  (dVS_CLS {best_global['avg']-cls_avg:+.2f})")
    L.append(f"omega-curve (p_star, global avg): " + " ".join(f"{w}:{summary['omega_curve_pstar_global'][w]:.2f}" for w in OMEGAS))
    L.append("")
    L.append("CONDITIONAL omega refit (final session, p_star):")
    for sub in ["all", "novel", "lowmargin"]:
        c = cond[sub]
        L.append(f"   {sub:9s} n={c['n']:4d}  w*={c['w_star']}  acc@w*={c['acc_at_wstar']:.2f}  acc@0={c['acc_at_0']:.2f}  (gain {c['acc_at_wstar']-c['acc_at_0']:+.2f})")
    L.append("")
    L.append(f"Deployable gated fusion best: p={int(best_gated['p']*100)}% tau={best_gated['tau_pct']}% w_gate={best_gated['w_gate']} "
             f"avg={best_gated['avg']:.2f}  (dVS_CLS {best_gated['avg']-cls_avg:+.2f})")
    L.append(f"runtime_sec: {summary['runtime_sec']}")
    text = "\n".join(L)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
