"""
Week-1 check A: CEILING (linear probe) + RESOLUTION sweep -- CUB, seed 1, frozen DINOv2.

Answers two questions before over-investing:
  1. Ceiling: full-data linear probe accuracy of DINOv2 CLS features (ViT-B/14 and ViT-L/14)
     at 224 and 518. Tells us how much headroom the ~85% FSCIL number has.
  2. Resolution: does running CUB at 448/518 (instead of 224) lift the V1-CLS FSCIL number?
     DINOv2 fine-grained quality is resolution-dependent; higher res densifies the patch grid.

Features extracted ONCE per (backbone, resolution). FSCIL support is mapped (no re-extraction)
from the full-train features via image path, and the seed is reset before each config's FSCIL
loop so every resolution sees the IDENTICAL seeded 5-shot support (and reproduces the 224 V1).
Incremental save after each config.
"""
import os, sys, json, time, math
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

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
GAMMAS = [0.1, 1.0, 10.0]
PROBE_C = [1.0, 10.0, 100.0]

# (backbone, resolution, batch_size, do_fscil)
CONFIGS = [
    ("dinov2_vitb14", 224, 48, True),
    ("dinov2_vitb14", 448, 16, True),
    ("dinov2_vitb14", 518, 8,  True),
    ("dinov2_vitl14", 224, 24, False),
    ("dinov2_vitl14", 518, 4,  False),
]
WEIGHTS = {
    "dinov2_vitb14": "~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
    "dinov2_vitl14": "~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth",
}


def _ensure_sdpa():
    if hasattr(F, "scaled_dot_product_attention"):
        return
    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        sf = (q.size(-1) ** -0.5) if scale is None else scale
        a = torch.matmul(q, k.transpose(-2, -1)) * sf
        return torch.matmul(torch.softmax(a, dim=-1), v)
    F.scaled_dot_product_attention = sdpa


def load_backbone(name, device):
    _ensure_sdpa()
    hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    w = os.path.expanduser(WEIGHTS[name])
    print(f"Loading {name} from {hub}")
    model = torch.hub.load(hub, name, source="local", pretrained=False)
    model.load_state_dict(torch.load(w, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def transform_R(R):
    return transforms.Compose([
        transforms.Resize(R, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(R),
        transforms.ToTensor(),
        transforms.Normalize(DINO_MEAN, DINO_STD),
    ])


@torch.no_grad()
def extract_cls(model, loader, device):
    feats, labels = [], []
    for b in loader:
        x = b["image"].to(device)
        f = F.normalize(model(x).float(), dim=-1)
        feats.append(f)
        labels.append(b["label"].to(device))
    return torch.cat(feats, 0), torch.cat(labels, 0)


def linear_probe(Xtr, ytr, Xte, yte, device):
    n_cls = int(max(int(ytr.max()), int(yte.max())) + 1)
    best = -1.0
    for C in PROBE_C:
        W = torch.zeros(Xtr.shape[1], n_cls, device=device, requires_grad=True)
        b = torch.zeros(n_cls, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=120, line_search_fn="strong_wolfe")
        lam = 1.0 / C
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(Xtr @ W + b, ytr) + lam * W.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            acc = (Xte @ W + b).argmax(1).eq(yte).float().mean().item() * 100.0
        best = max(best, acc)
    return round(best, 2)


def class_stats(sup_by_class, seen, device, D):
    mu_raw = torch.zeros((seen, D), device=device)
    W = torch.zeros((D, D), device=device)
    dof = 0
    for c in range(seen):
        Xc = sup_by_class[c]
        m = Xc.mean(0)
        mu_raw[c] = m
        d = Xc - m
        W += d.t() @ d
        if Xc.shape[0] > 1:
            dof += Xc.shape[0] - 1
    dof = max(dof, 1)
    Sigma = W / dof
    delta = torch.diagonal(Sigma).mean().clamp_min(1e-8)
    return F.normalize(mu_raw, dim=-1), Sigma, delta


def v1_logits(Zt, mu, Minv):
    ZM = Zt @ Minv
    return 2.0 * (ZM @ mu.t()) - ((mu @ Minv) * mu).sum(1).unsqueeze(0)


def softmax_probs(logits, temp):
    logits = logits - logits.max(1, keepdim=True).values
    return torch.softmax(logits / temp, dim=1)


def run_fscil(dm, ev, train_feats, train_labels, test_feats, test_labels, pathmap, device, seed):
    """Reseed so support == canonical; reuse cached features. Returns per-clf acc_list + finals."""
    set_seed(seed)  # identical seeded support across resolutions
    T = dm.num_tasks
    res = {"B0": {"acc": []}, **{f"V1|g={g}": {"acc": []} for g in GAMMAS}}
    class_feats = {}
    for t in range(T):
        seen = int(max(dm.class_index_in_task[t]) + 1)
        ds = dm.get_dataset(t, "train", mode="test")  # consumes RNG -> seeded support
        rows = torch.tensor([pathmap[p] for p in ds.images], device=device)
        labs = torch.as_tensor(np.asarray(ds.labels), device=device)
        sf = train_feats[rows]
        for c in torch.unique(labs).tolist():
            fc = sf[labs == c]
            class_feats[c] = fc if c not in class_feats else torch.cat([class_feats[c], fc], 0)
        mask = test_labels < seen
        Zt, yt = test_feats[mask], test_labels[mask]
        D = Zt.shape[1]
        sup = {c: class_feats[c] for c in range(seen)}
        mu, Sigma, delta = class_stats(sup, seen, device, D)
        is_last = (t == T - 1)
        # B0
        m = ev.calc_accuracy(softmax_probs(Zt @ mu.t(), 0.01), yt, t)
        res["B0"]["acc"].append(round(m["mean_acc"], 3))
        if is_last:
            res["B0"]["last"] = {k: m[k] for k in ["base_avg_acc", "inc_avg_acc", "harmonic_acc"]}
        # V1
        eye = torch.eye(D, device=device)
        for g in GAMMAS:
            Minv = torch.linalg.inv(Sigma + g * delta * eye)
            lg = v1_logits(Zt, mu, Minv)
            m = ev.calc_accuracy(softmax_probs(lg, lg.std().clamp_min(1e-6)), yt, t)
            res[f"V1|g={g}"]["acc"].append(round(m["mean_acc"], 3))
            if is_last:
                res[f"V1|g={g}"]["last"] = {k: m[k] for k in ["base_avg_acc", "inc_avg_acc", "harmonic_acc"]}
            del Minv, lg
        torch.cuda.empty_cache()
    return res


def main():
    t0 = time.time()
    cfg = setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml",
                    "configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    device = "cuda"
    dm = DatasetManager(cfg)
    ev = AccuracyEvaluator(dm.class_index_in_task)
    pathmap = {p: i for i, p in enumerate(dm.train_data)}
    print(f"=== ceiling + resolution :: {cfg.DATASET.NAME} seed {cfg.SEED} "
          f"| train {len(dm.train_data)} test {len(dm.test_data)} ===")

    out_dir = os.path.join(ROOT, "dino_ceiling_res_runs", "cub_seed1")
    os.makedirs(out_dir, exist_ok=True)
    summary = {"dataset": cfg.DATASET.NAME, "seed": cfg.SEED, "configs": []}

    model, cur_name = None, None
    for (name, R, bs, do_fscil) in CONFIGS:
        cstart = time.time()
        if name != cur_name:
            del model
            torch.cuda.empty_cache()
            model = load_backbone(name, device)
            cur_name = name
        tf = transform_R(R)
        full_tr = TaskDataset(np.array(dm.train_data), np.array(dm.train_targets), tf)
        full_te = TaskDataset(np.array(dm.test_data), np.array(dm.test_targets), tf)
        tr_loader = DataLoader(full_tr, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        te_loader = DataLoader(full_te, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        print(f"[{name} R={R}] extracting train ...", flush=True)
        train_feats, train_labels = extract_cls(model, tr_loader, device)
        print(f"[{name} R={R}] extracting test ...", flush=True)
        test_feats, test_labels = extract_cls(model, te_loader, device)

        ceiling = linear_probe(train_feats, train_labels, test_feats, test_labels, device)
        entry = {"backbone": name, "resolution": R, "dim": int(train_feats.shape[1]),
                 "linear_probe_acc": ceiling}
        print(f"[{name} R={R}] LINEAR-PROBE CEILING = {ceiling:.2f}", flush=True)

        if do_fscil:
            fr = run_fscil(dm, ev, train_feats, train_labels, test_feats, test_labels, pathmap, device, cfg.SEED)
            def avg(a): return round(sum(a) / len(a), 3)
            v1_keys = [k for k in fr if k.startswith("V1")]
            best_v1 = max(v1_keys, key=lambda k: avg(fr[k]["acc"]))
            entry["fscil"] = {
                "B0": {"avg": avg(fr["B0"]["acc"]), "final": fr["B0"]["acc"][-1], "acc": fr["B0"]["acc"]},
                "V1_best": {"config": best_v1, "avg": avg(fr[best_v1]["acc"]),
                            "final": fr[best_v1]["acc"][-1], "acc": fr[best_v1]["acc"],
                            "last": fr[best_v1].get("last", {})},
                "V1_g1": {"avg": avg(fr["V1|g=1.0"]["acc"]), "final": fr["V1|g=1.0"]["acc"][-1]},
            }
            print(f"[{name} R={R}] FSCIL  B0 avg={entry['fscil']['B0']['avg']:.2f}  "
                  f"V1 avg={entry['fscil']['V1_best']['avg']:.2f} (final {entry['fscil']['V1_best']['final']:.2f}, {best_v1})", flush=True)

        entry["seconds"] = round(time.time() - cstart, 1)
        summary["configs"].append(entry)
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(summary, f, indent=2)
        del train_feats, train_labels, test_feats, test_labels
        torch.cuda.empty_cache()

    summary["runtime_sec"] = round(time.time() - t0, 1)
    # pretty summary
    L = []
    L.append(f"CEILING + RESOLUTION | {cfg.DATASET.NAME} seed {cfg.SEED}")
    L.append(f"  {'backbone':16s} {'res':>4s} {'dim':>5s} {'probe':>6s} | {'B0avg':>6s} {'V1avg':>6s} {'V1fin':>6s} {'base':>6s} {'inc':>6s} {'harm':>6s}")
    for e in summary["configs"]:
        if "fscil" in e:
            f = e["fscil"]; last = f["V1_best"]["last"]
            L.append(f"  {e['backbone']:16s} {e['resolution']:>4d} {e['dim']:>5d} {e['linear_probe_acc']:6.2f} | "
                     f"{f['B0']['avg']:6.2f} {f['V1_best']['avg']:6.2f} {f['V1_best']['final']:6.2f} "
                     f"{last.get('base_avg_acc','-'):>6} {last.get('inc_avg_acc','-'):>6} {last.get('harmonic_acc','-'):>6}")
        else:
            L.append(f"  {e['backbone']:16s} {e['resolution']:>4d} {e['dim']:>5d} {e['linear_probe_acc']:6.2f} | "
                     f"{'(probe-only)':>40s}")
    L.append(f"runtime_sec: {summary['runtime_sec']}")
    text = "\n".join(L)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
