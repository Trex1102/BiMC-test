"""
LLM/oracle CONCEPT-DIRECTION metric for DINOv2-only FSCIL -- CUB seed 1 @ 448, V1 base.

Direction 1 (oracle ceiling + LLM version). DINOv2 has no text encoder, so the LLM/attribute
signal enters as a PARTITION of the label set (which classes have attribute a), grounded into
DINOv2 space via images:
  FastCAV concept direction  d_a = normalize( mean(mu_c : c has a) - mean(mu_c : c lacks a) )
estimated from class PROTOTYPES of every seen class sharing the attribute (pooling shots across
classes -> denoises the K=5 novel prototypes). The bundle is used as an up-weighted metric on
top of V1:
  M(lambda) = (Sigma_bar + gamma*delta*I)^-1  +  lambda * sum_a d_a d_a^T
  d_c(z) = -(z-mu_c)^T M (z-mu_c)
lambda=0 is exactly the V1 baseline. Sweep lambda. Partition source: 'gt' (CUB 312 GT attrs,
the ORACLE CEILING) or a JSON {attr_name: [class_idx,...]} from the LLM.

Reuses Script-1 machinery (extract @R, map seeded support, reseed for identical support).
Memory-robust: batch 8 @448, features kept (tiny) on GPU.
"""
import os, sys, json, time, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from main import setup_cfg
from utils.util import set_seed, set_gpu
from datasets.data_manager import DatasetManager, TaskDataset
from utils.evaluator import AccuracyEvaluator

DINO_MEAN = (0.485, 0.456, 0.406); DINO_STD = (0.229, 0.224, 0.225)
GAMMA = 1.0
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
ATTR_THR = 50.0
MIN_COUNT = 3
ATTR_DIR = "dataset/CUB_200_2011/CUB_200_2011/attributes"


def _ensure_sdpa():
    if hasattr(F, "scaled_dot_product_attention"): return
    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        sf = (q.size(-1) ** -0.5) if scale is None else scale
        return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * sf, -1), v)
    F.scaled_dot_product_attention = sdpa


def load_backbone(device):
    _ensure_sdpa()
    hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    w = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
    m = torch.hub.load(hub, "dinov2_vitb14", source="local", pretrained=False)
    m.load_state_dict(torch.load(w, map_location="cpu"), strict=True)
    return m.to(device).eval()


def transform_R(R):
    return transforms.Compose([transforms.Resize(R, interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.CenterCrop(R), transforms.ToTensor(),
                               transforms.Normalize(DINO_MEAN, DINO_STD)])


@torch.no_grad()
def extract_cls(model, loader, device):
    feats, labels = [], []
    for b in loader:
        feats.append(F.normalize(model(b["image"].to(device)).float(), dim=-1))
        labels.append(b["label"].to(device))
    return torch.cat(feats, 0), torch.cat(labels, 0)


def load_partition(source, num_classes, attr_names):
    """Return binary class-attribute matrix [num_classes, A] and the attribute names used."""
    if source == "gt":
        cont = np.loadtxt(os.path.join(ATTR_DIR, "class_attribute_labels_continuous.txt"))
        return (cont > ATTR_THR), attr_names
    # LLM JSON: {attr_name: [class_idx, ...]}
    part = json.load(open(source))
    names = list(part.keys())
    M = np.zeros((num_classes, len(names)), dtype=bool)
    for j, nm in enumerate(names):
        for ci in part[nm]:
            if 0 <= int(ci) < num_classes:
                M[int(ci), j] = True
    return M, names


def class_stats(sup, seen, device, D):
    mu_raw = torch.zeros((seen, D), device=device); W = torch.zeros((D, D), device=device); dof = 0
    for c in range(seen):
        Xc = sup[c]; m = Xc.mean(0); mu_raw[c] = m; d = Xc - m; W += d.t() @ d
        if Xc.shape[0] > 1: dof += Xc.shape[0] - 1
    Sigma = W / max(dof, 1)
    delta = torch.diagonal(Sigma).mean().clamp_min(1e-8)
    return F.normalize(mu_raw, dim=-1), Sigma, delta


def concept_dirs(mu, seen, attr_bin, device):
    dirs = []
    sub = attr_bin[:seen]
    for a in range(sub.shape[1]):
        h = np.where(sub[:, a])[0]; l = np.where(~sub[:, a])[0]
        if len(h) >= MIN_COUNT and len(l) >= MIN_COUNT:
            d = mu[h].mean(0) - mu[l].mean(0)
            dirs.append(F.normalize(d, dim=0))
    return torch.stack(dirs, 0) if dirs else None


def v1_logits(Zt, mu, M):
    ZM = Zt @ M
    return 2.0 * (ZM @ mu.t()) - ((mu @ M) * mu).sum(1).unsqueeze(0)


def softmax_probs(L, T):
    L = L - L.max(1, keepdim=True).values
    return torch.softmax(L / T, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="gt")  # 'gt' or path to LLM json
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--tag", default="oracle_gt")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    cfg = setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml", "configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED); set_gpu(cfg.DEVICE.GPU_ID); device = "cuda"
    dm = DatasetManager(cfg); ev = AccuracyEvaluator(dm.class_index_in_task)
    pathmap = {p: i for i, p in enumerate(dm.train_data)}
    attr_names = [ln.strip().split(" ", 1)[1] for ln in open("dataset/CUB_200_2011/attributes.txt")]
    attr_bin, used_names = load_partition(args.partition, dm.num_total_classes, attr_names)
    print(f"=== CONCEPT DIRS [{args.tag}] R={args.resolution} | partition={args.partition} "
          f"attrs={attr_bin.shape[1]} ===", flush=True)

    model = load_backbone(device)
    tf = transform_R(args.resolution)
    tr = DataLoader(TaskDataset(np.array(dm.train_data), np.array(dm.train_targets), tf),
                    batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    te = DataLoader(TaskDataset(np.array(dm.test_data), np.array(dm.test_targets), tf),
                    batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    print("extracting train ...", flush=True); train_feats, _ = extract_cls(model, tr, device)
    print("extracting test ...", flush=True); test_feats, test_labels = extract_cls(model, te, device)
    del model; torch.cuda.empty_cache()

    set_seed(cfg.SEED)  # identical seeded support
    T = dm.num_tasks
    res = {f"lam={l}": {"acc": []} for l in LAMBDAS}
    class_feats = {}
    for t in range(T):
        seen = int(max(dm.class_index_in_task[t]) + 1)
        ds = dm.get_dataset(t, "train", mode="test")
        rows = torch.tensor([pathmap[p] for p in ds.images], device=device)
        labs = torch.as_tensor(np.asarray(ds.labels), device=device)
        sf = train_feats[rows]
        for c in torch.unique(labs).tolist():
            fc = sf[labs == c]
            class_feats[c] = fc if c not in class_feats else torch.cat([class_feats[c], fc], 0)
        mask = test_labels < seen; Zt, yt = test_feats[mask], test_labels[mask]
        D = Zt.shape[1]
        mu, Sigma, delta = class_stats({c: class_feats[c] for c in range(seen)}, seen, device, D)
        M0 = torch.linalg.inv(Sigma + GAMMA * delta * torch.eye(D, device=device))
        Dc = concept_dirs(mu, seen, attr_bin, device)
        DD = (Dc.t() @ Dc) if Dc is not None else torch.zeros((D, D), device=device)
        is_last = (t == T - 1)
        for l in LAMBDAS:
            M = M0 + l * DD
            Lg = v1_logits(Zt, mu, M)
            m = ev.calc_accuracy(softmax_probs(Lg, Lg.std().clamp_min(1e-6)), yt, t)
            res[f"lam={l}"]["acc"].append(round(m["mean_acc"], 3))
            if is_last:
                res[f"lam={l}"]["last"] = {k: m[k] for k in ["base_avg_acc", "inc_avg_acc", "harmonic_acc"]}
        if t == T - 1:
            print(f"[task {t}] seen={seen} n_concept_dirs={0 if Dc is None else Dc.shape[0]}", flush=True)
        torch.cuda.empty_cache()

    def avg(a): return round(sum(a) / len(a), 3)
    base = avg(res["lam=0.0"]["acc"]); basef = res["lam=0.0"]["acc"][-1]
    best_l = max(LAMBDAS, key=lambda l: avg(res[f"lam={l}"]["acc"]))
    out = {"tag": args.tag, "partition": args.partition, "resolution": args.resolution,
           "n_attrs": int(attr_bin.shape[1]), "gamma": GAMMA,
           "baseline_lam0": {"avg": base, "final": basef, "last": res["lam=0.0"].get("last", {})},
           "by_lambda": {str(l): {"avg": avg(res[f"lam={l}"]["acc"]), "final": res[f"lam={l}"]["acc"][-1],
                                   "last": res[f"lam={l}"].get("last", {})} for l in LAMBDAS},
           "best": {"lambda": best_l, "avg": avg(res[f"lam={best_l}"]["acc"]),
                    "final": res[f"lam={best_l}"]["acc"][-1], "last": res[f"lam={best_l}"].get("last", {}),
                    "delta_vs_baseline": round(avg(res[f"lam={best_l}"]["acc"]) - base, 3)},
           "runtime_sec": round(time.time() - t0, 1)}
    out_dir = os.path.join(ROOT, "dino_concept_runs", f"cub_seed1_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"summary": out, "raw": res}, open(os.path.join(out_dir, "results.json"), "w"), indent=2)
    L = [f"CONCEPT DIRECTIONS [{args.tag}] | CUB seed1 @ {args.resolution} | partition={args.partition} | {out['n_attrs']} attrs",
         f"V1 baseline (lambda=0): avg {base:.2f} final {basef:.2f}  base {out['baseline_lam0']['last'].get('base_avg_acc')} inc {out['baseline_lam0']['last'].get('inc_avg_acc')} harm {out['baseline_lam0']['last'].get('harmonic_acc')}",
         "", f"  {'lambda':>7s} {'avg':>7s} {'final':>7s} {'base':>6s} {'inc':>6s} {'harm':>6s}"]
    for l in LAMBDAS:
        e = out["by_lambda"][str(l)]; la = e["last"]
        L.append(f"  {l:7.2f} {e['avg']:7.2f} {e['final']:7.2f} {la.get('base_avg_acc','-'):>6} {la.get('inc_avg_acc','-'):>6} {la.get('harmonic_acc','-'):>6}")
    b = out["best"]
    L.append("")
    L.append(f"BEST lambda={b['lambda']}: avg {b['avg']:.2f} (dVS_baseline {b['delta_vs_baseline']:+.2f})  "
             f"base {b['last'].get('base_avg_acc')} inc {b['last'].get('inc_avg_acc')} harm {b['last'].get('harmonic_acc')}")
    L.append(f"runtime_sec: {out['runtime_sec']}")
    txt = "\n".join(L)
    open(os.path.join(out_dir, "summary.txt"), "w").write(txt + "\n")
    print("\n" + txt + f"\nSaved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
