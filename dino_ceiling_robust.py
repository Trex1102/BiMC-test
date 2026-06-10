"""Memory-robust ceiling probe for the remaining configs (the others OOM'd under GPU contention).
Extract to CPU (batch 4), move only the small feature matrices to GPU for the LBFGS probe.
Appends to the existing ceiling_fixed.json (keeps vitb@224=89.7)."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
import sys, json, time
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from main import setup_cfg
from utils.util import set_seed, set_gpu
from datasets.data_manager import DatasetManager, TaskDataset

DINO_MEAN=(0.485,0.456,0.406); DINO_STD=(0.229,0.224,0.225); PROBE_C=[1.,10.,100.,1000.,10000.]
CONFIGS=[("dinov2_vitb14",448,6),("dinov2_vitl14",224,8),("dinov2_vitl14",448,4)]
W={"dinov2_vitb14":"~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
   "dinov2_vitl14":"~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"}
def _sdpa():
    if hasattr(F,"scaled_dot_product_attention"): return
    def s(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,scale=None):
        sf=(q.size(-1)**-0.5) if scale is None else scale
        return torch.matmul(torch.softmax(torch.matmul(q,k.transpose(-2,-1))*sf,-1),v)
    F.scaled_dot_product_attention=s
def load_bb(n,dev):
    _sdpa(); hub=os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    m=torch.hub.load(hub,n,source="local",pretrained=False)
    m.load_state_dict(torch.load(os.path.expanduser(W[n]),map_location="cpu"),strict=True); return m.to(dev).eval()
def tfR(R): return transforms.Compose([transforms.Resize(R,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(R),transforms.ToTensor(),transforms.Normalize(DINO_MEAN,DINO_STD)])
@torch.no_grad()
def extract(m,loader,dev):
    fs,ls=[],[]
    for b in loader:
        fs.append(m(b["image"].to(dev)).float().cpu()); ls.append(b["label"])
        torch.cuda.empty_cache()
    return torch.cat(fs,0),torch.cat(ls,0)
def probe(Xtr,ytr,Xte,yte,dev):
    mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True).clamp_min(1e-6)
    Xtr=((Xtr-mu)/sd).to(dev); Xte=((Xte-mu)/sd).to(dev); ytr=ytr.to(dev); yte=yte.to(dev)
    n=int(max(int(ytr.max()),int(yte.max()))+1); best=-1
    for C in PROBE_C:
        Wt=torch.zeros(Xtr.shape[1],n,device=dev,requires_grad=True); b=torch.zeros(n,device=dev,requires_grad=True)
        opt=torch.optim.LBFGS([Wt,b],lr=1.,max_iter=400,history_size=20,line_search_fn="strong_wolfe")
        lam=1./(C*Xtr.shape[0])
        def cl():
            opt.zero_grad(); loss=F.cross_entropy(Xtr@Wt+b,ytr)+lam*Wt.pow(2).sum(); loss.backward(); return loss
        opt.step(cl)
        with torch.no_grad(): best=max(best,(Xte@Wt+b).argmax(1).eq(yte).float().mean().item()*100)
    return round(best,2)
def main():
    cfg=setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml","configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED); set_gpu(cfg.DEVICE.GPU_ID); dev="cuda"
    dm=DatasetManager(cfg)
    path="dino_ceiling_res_runs/cub_seed1/ceiling_fixed.json"
    summary=json.load(open(path)) if os.path.exists(path) else []
    have={(e["backbone"],e["resolution"]) for e in summary}
    model,cur=None,None
    for (n,R,bs) in CONFIGS:
        if (n,R) in have: continue
        if n!=cur:
            del model; torch.cuda.empty_cache(); model=load_bb(n,dev); cur=n
        tf=tfR(R)
        tr=DataLoader(TaskDataset(np.array(dm.train_data),np.array(dm.train_targets),tf),batch_size=bs,shuffle=False,num_workers=0)
        te=DataLoader(TaskDataset(np.array(dm.test_data),np.array(dm.test_targets),tf),batch_size=bs,shuffle=False,num_workers=0)
        print(f"[{n} R={R}] extracting",flush=True); Xtr,ytr=extract(model,tr,dev); Xte,yte=extract(model,te,dev)
        c=probe(Xtr,ytr,Xte,yte,dev)
        print(f"[{n} R={R}] CEILING={c}",flush=True)
        summary.append({"backbone":n,"resolution":R,"dim":int(Xtr.shape[1]),"ceiling":c})
        json.dump(summary,open(path,"w"),indent=2); del Xtr,ytr,Xte,yte; torch.cuda.empty_cache()
    open("dino_ceiling_res_runs/cub_seed1/ceiling_fixed.txt","w").write(
        "\n".join(f"{e['backbone']} R={e['resolution']} dim={e['dim']} ceiling={e['ceiling']}" for e in summary)+"\n")
    print("DONE ceiling robust",flush=True)
if __name__=="__main__": main()
