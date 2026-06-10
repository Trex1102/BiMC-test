"""
Corrected CEILING linear probe (the one in Script 1 underfit: L2-normalized unit features +
strong reg + too few iters gave 79.5 < few-shot V1, which is impossible).

Proper protocol: RAW CLS features (no L2 norm) -> per-dim standardization (train stats) ->
multinomial logistic regression (LBFGS, many iters) with a WIDE C sweep -> best test top-1.
ViT-B/14 and ViT-L/14, at 224 and 518. Full CUB train (all shots, 200 classes) -> test.
"""
import os, sys, json, time
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

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
PROBE_C = [1.0, 10.0, 100.0, 1000.0, 10000.0]
CONFIGS = [
    ("dinov2_vitb14", 224, 16),
    ("dinov2_vitb14", 448, 8),
    ("dinov2_vitl14", 224, 12),
    ("dinov2_vitl14", 448, 6),
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
        return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * sf, dim=-1), v)
    F.scaled_dot_product_attention = sdpa


def load_backbone(name, device):
    _ensure_sdpa()
    hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    model = torch.hub.load(hub, name, source="local", pretrained=False)
    model.load_state_dict(torch.load(os.path.expanduser(WEIGHTS[name]), map_location="cpu"), strict=True)
    return model.to(device).eval()


def transform_R(R):
    return transforms.Compose([
        transforms.Resize(R, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(R), transforms.ToTensor(),
        transforms.Normalize(DINO_MEAN, DINO_STD)])


@torch.no_grad()
def extract_raw(model, loader, device):
    feats, labels = [], []
    for b in loader:
        feats.append(model(b["image"].to(device)).float())   # RAW CLS, no normalize
        labels.append(b["label"].to(device))
    return torch.cat(feats, 0), torch.cat(labels, 0)


def probe(Xtr, ytr, Xte, yte, device):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    n_cls = int(max(int(ytr.max()), int(yte.max())) + 1)
    results = {}
    best = -1.0
    for C in PROBE_C:
        W = torch.zeros(Xtr.shape[1], n_cls, device=device, requires_grad=True)
        b = torch.zeros(n_cls, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=500, history_size=20,
                                line_search_fn="strong_wolfe")
        lam = 1.0 / (C * Xtr.shape[0])
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(Xtr @ W + b, ytr) + lam * W.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            tr = (Xtr @ W + b).argmax(1).eq(ytr).float().mean().item() * 100
            te = (Xte @ W + b).argmax(1).eq(yte).float().mean().item() * 100
        results[C] = (round(tr, 2), round(te, 2))
        best = max(best, te)
    return round(best, 2), results


def main():
    cfg = setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml",
                    "configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED)
    set_gpu(cfg.DEVICE.GPU_ID)
    device = "cuda"
    dm = DatasetManager(cfg)
    print(f"=== CEILING (fixed) :: train {len(dm.train_data)} test {len(dm.test_data)} ===", flush=True)
    out_dir = os.path.join(ROOT, "dino_ceiling_res_runs", "cub_seed1")
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    model, cur = None, None
    for (name, R, bs) in CONFIGS:
        if name != cur:
            del model; torch.cuda.empty_cache()
            model = load_backbone(name, device); cur = name
        tf = transform_R(R)
        tr = DataLoader(TaskDataset(np.array(dm.train_data), np.array(dm.train_targets), tf),
                        batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        te = DataLoader(TaskDataset(np.array(dm.test_data), np.array(dm.test_targets), tf),
                        batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        print(f"[{name} R={R}] extracting ...", flush=True)
        Xtr, ytr = extract_raw(model, tr, device)
        Xte, yte = extract_raw(model, te, device)
        best, per_c = probe(Xtr, ytr, Xte, yte, device)
        print(f"[{name} R={R}] CEILING(best test top-1) = {best:.2f}   per-C(tr,te): "
              + " ".join(f"C{int(c)}:{per_c[c][1]:.1f}" for c in PROBE_C), flush=True)
        summary.append({"backbone": name, "resolution": R, "dim": int(Xtr.shape[1]),
                        "ceiling": best, "per_C": {str(c): per_c[c] for c in PROBE_C}})
        with open(os.path.join(out_dir, "ceiling_fixed.json"), "w") as f:
            json.dump(summary, f, indent=2)
        del Xtr, ytr, Xte, yte; torch.cuda.empty_cache()
    L = ["CEILING (fixed, full-data linear probe, best-C test top-1)"]
    for e in summary:
        L.append(f"  {e['backbone']:16s} R={e['resolution']:>4d} dim={e['dim']:>5d}  ceiling={e['ceiling']:.2f}")
    text = "\n".join(L)
    with open(os.path.join(out_dir, "ceiling_fixed.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
