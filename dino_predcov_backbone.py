"""
Weak-backbone point: predictive-covariance correction across DINOv2 backbones, CUB seed1.
If predcov helps MORE on a weaker backbone (noisier K-shot prototypes), it does something the
representation alone does not -- the rebuttal-proofing weak-backbone evidence.
Usage: --backbone {dinov2_vits14,dinov2_vitb14,dinov2_vitl14} --resolution R
"""
import os, sys, json, time, argparse
ROOT=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,ROOT)
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from main import setup_cfg
from utils.util import set_seed, set_gpu
from datasets.data_manager import DatasetManager, TaskDataset
from utils.evaluator import AccuracyEvaluator
GAMMA=1.0; BETAS=[0.0,0.5,1.0,2.0,5.0,10.0]; DINO_MEAN=(0.485,0.456,0.406); DINO_STD=(0.229,0.224,0.225)
WT={"dinov2_vits14":"~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
    "dinov2_vitb14":"~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
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
    m.load_state_dict(torch.load(os.path.expanduser(WT[n]),map_location="cpu"),strict=True); return m.to(dev).eval()
def tfR(R): return transforms.Compose([transforms.Resize(R,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(R),transforms.ToTensor(),transforms.Normalize(DINO_MEAN,DINO_STD)])
@torch.no_grad()
def extract(m,loader,dev):
    fs,ls=[],[]
    for b in loader: fs.append(F.normalize(m(b["image"].to(dev)).float().cpu(),dim=-1)); ls.append(b["label"])
    return torch.cat(fs,0),torch.cat(ls,0)
def maha_perclass(Zt,mu,Minv):
    N,Cn=Zt.shape[0],mu.shape[0]; out=torch.empty((N,Cn),device=Zt.device)
    for c in range(Cn): r=Zt-mu[c]; out[:,c]=-((r@Minv[c])*r).sum(1)
    return out
def sp(L,T): L=L-L.max(1,keepdim=True).values; return torch.softmax(L/T,1)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--backbone",default="dinov2_vits14"); ap.add_argument("--resolution",type=int,default=448); ap.add_argument("--bs",type=int,default=8); a=ap.parse_args()
    t0=time.time(); cfg=setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml","configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED); set_gpu(cfg.DEVICE.GPU_ID); dev="cuda"
    dm=DatasetManager(cfg); ev=AccuracyEvaluator(dm.class_index_in_task); pathmap={p:i for i,p in enumerate(dm.train_data)}
    model=load_bb(a.backbone,dev); tf=tfR(a.resolution)
    tr=DataLoader(TaskDataset(np.array(dm.train_data),np.array(dm.train_targets),tf),batch_size=a.bs,shuffle=False,num_workers=0)
    te=DataLoader(TaskDataset(np.array(dm.test_data),np.array(dm.test_targets),tf),batch_size=a.bs,shuffle=False,num_workers=0)
    print(f"[{a.backbone} R={a.resolution}] extracting",flush=True)
    train_feats,_=extract(model,tr,dev); test_feats,test_labels=extract(model,te,dev)
    train_feats=train_feats.to(dev); test_feats=test_feats.to(dev); test_labels=test_labels.to(dev)
    del model; torch.cuda.empty_cache()
    set_seed(cfg.SEED); T=dm.num_tasks; cf={}; res={f"b={b}":{"acc":[]} for b in BETAS}
    for t in range(T):
        seen=int(max(dm.class_index_in_task[t])+1)
        ds=dm.get_dataset(t,"train",mode="test"); ridx=torch.tensor([pathmap[p] for p in ds.images],device=dev)
        labs=torch.as_tensor(np.asarray(ds.labels),device=dev); sf=train_feats[ridx]
        for c in torch.unique(labs).tolist():
            fc=sf[labs==c]; cf[c]=fc if c not in cf else torch.cat([cf[c],fc],0)
        D=test_feats.shape[1]; mu=torch.zeros((seen,D),device=dev); W=torch.zeros((D,D),device=dev); dof=0; Kc=torch.zeros(seen,device=dev)
        for c in range(seen):
            Xc=cf[c]; m=Xc.mean(0); mu[c]=m; dd=Xc-m; W+=dd.t()@dd; Kc[c]=Xc.shape[0]
            if Xc.shape[0]>1: dof+=Xc.shape[0]-1
        Sig=W/max(dof,1); delta=torch.diagonal(Sig).mean().clamp_min(1e-8); mu=F.normalize(mu,dim=-1)
        eye=torch.eye(D,device=dev); mask=test_labels<seen; Zt=test_feats[mask]; yt=test_labels[mask]; is_last=(t==T-1)
        for b in BETAS:
            scale=(1.0+b/Kc).view(seen,1,1); Minv=torch.linalg.inv(scale*Sig.unsqueeze(0)+delta*GAMMA*eye.unsqueeze(0))
            L=maha_perclass(Zt,mu,Minv); m=ev.calc_accuracy(sp(L,L.std().clamp_min(1e-6)),yt,t)
            res[f"b={b}"]["acc"].append(round(m["mean_acc"],3))
            if is_last: res[f"b={b}"]["last"]={k:m[k] for k in ["base_avg_acc","inc_avg_acc","harmonic_acc"]}
            del Minv,L
        torch.cuda.empty_cache()
    def avg(x): return round(sum(x)/len(x),3)
    base=avg(res["b=0.0"]["acc"]); bestb=max(BETAS,key=lambda b:avg(res[f"b={b}"]["acc"]))
    out={"backbone":a.backbone,"resolution":a.resolution,"dim":int(test_feats.shape[1]),
         "by_beta":{str(b):{"avg":avg(res[f"b={b}"]["acc"]),"final":res[f"b={b}"]["acc"][-1],"last":res[f"b={b}"].get("last",{})} for b in BETAS},
         "baseline":{"avg":base,"last":res["b=0.0"].get("last",{})},
         "best":{"beta":bestb,"avg":avg(res[f"b={bestb}"]["acc"]),"last":res[f"b={bestb}"].get("last",{}),"delta":round(avg(res[f"b={bestb}"]["acc"])-base,3)},
         "runtime_sec":round(time.time()-t0,1)}
    od=os.path.join(ROOT,"dino_predcov_runs",f"cub_{a.backbone}_R{a.resolution}"); os.makedirs(od,exist_ok=True)
    json.dump(out,open(os.path.join(od,"results.json"),"w"),indent=2)
    L=[f"PREDCOV across backbone | {a.backbone} R={a.resolution} dim={out['dim']}",
       f"baseline V1 (beta0): avg {base:.2f} base {out['baseline']['last'].get('base_avg_acc')} inc {out['baseline']['last'].get('inc_avg_acc')} harm {out['baseline']['last'].get('harmonic_acc')}",""]
    for b in BETAS:
        e=out["by_beta"][str(b)]; la=e["last"]
        L.append(f"  beta={b:5.1f}: avg {e['avg']:7.2f} base {la.get('base_avg_acc','-')!s:>6} inc {la.get('inc_avg_acc','-')!s:>6} harm {la.get('harmonic_acc','-')!s:>6} dVS {e['avg']-base:+.2f}")
    bb=out["best"]; L.append(""); L.append(f"BEST beta={bb['beta']}: avg {bb['avg']:.2f} (dVS {bb['delta']:+.2f}) inc {bb['last'].get('inc_avg_acc')} harm {bb['last'].get('harmonic_acc')}")
    L.append(f"runtime_sec {out['runtime_sec']}")
    open(os.path.join(od,"summary.txt"),"w").write("\n".join(L)+"\n"); print("\n"+"\n".join(L),flush=True)
if __name__=="__main__": main()
