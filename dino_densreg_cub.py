"""
Weeks-2 item #2: SESSION-STABLE DENSITY REGULARIZER. CUB seed1 @448 V1.

As novel 5-shot classes accumulate, the per-session pooled covariance Sigma_bar_t is increasingly
polluted by noisy low-shot scatter and the density scale drifts (the AUROC gate showed the
absolute density is the weak spot). Stabilize it by shrinking toward the well-estimated
base-session covariance:  Sigma_bar_reg = (1-eta) Sigma_bar_t + eta Sigma_bar_0 .
We sweep eta, alone and on top of the predictive-covariance correction (beta=2):
  M_c = ( Sigma_bar_reg (1 + beta/K_c) + gamma*delta_reg*I )^-1 .
eta=0,beta=0 is the V1 baseline. Reuses cached @448 feats.
"""
import os, sys, json, time
ROOT=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,ROOT)
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from main import setup_cfg
from utils.util import set_seed, set_gpu
from datasets.data_manager import DatasetManager, TaskDataset
from utils.evaluator import AccuracyEvaluator
GAMMA=1.0; CACHE="dino_feats_cls_R448.pt"; ETAS=[0.0,0.2,0.5,0.8,1.0]; BETA_COMBO=2.0
DINO_MEAN=(0.485,0.456,0.406); DINO_STD=(0.229,0.224,0.225)
def _sdpa():
    if hasattr(F,"scaled_dot_product_attention"): return
    def s(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,scale=None):
        sf=(q.size(-1)**-0.5) if scale is None else scale
        return torch.matmul(torch.softmax(torch.matmul(q,k.transpose(-2,-1))*sf,-1),v)
    F.scaled_dot_product_attention=s
def load_bb(dev):
    _sdpa(); hub=os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    m=torch.hub.load(hub,"dinov2_vitb14",source="local",pretrained=False)
    m.load_state_dict(torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"),map_location="cpu"),strict=True); return m.to(dev).eval()
def tfR(R): return transforms.Compose([transforms.Resize(R,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(R),transforms.ToTensor(),transforms.Normalize(DINO_MEAN,DINO_STD)])
@torch.no_grad()
def extract(m,loader,dev):
    fs,ls=[],[]
    for b in loader: fs.append(F.normalize(m(b["image"].to(dev)).float(),dim=-1)); ls.append(b["label"].to(dev))
    return torch.cat(fs,0),torch.cat(ls,0)
def maha_pc(Zt,mu,Minv):
    N,Cn=Zt.shape[0],mu.shape[0]; out=torch.empty((N,Cn),device=Zt.device)
    for c in range(Cn): r=Zt-mu[c]; out[:,c]=-((r@Minv[c])*r).sum(1)
    return out
def sp(L,T): L=L-L.max(1,keepdim=True).values; return torch.softmax(L/T,1)
def main():
    t0=time.time(); cfg=setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml","configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED); set_gpu(cfg.DEVICE.GPU_ID); dev="cuda"
    dm=DatasetManager(cfg); ev=AccuracyEvaluator(dm.class_index_in_task); pathmap={p:i for i,p in enumerate(dm.train_data)}
    if os.path.exists(CACHE):
        d=torch.load(CACHE); train_feats=d["train"].to(dev); test_feats=d["test"].to(dev); test_labels=d["tl"].to(dev)
    else:
        model=load_bb(dev); tf=tfR(448)
        tr=DataLoader(TaskDataset(np.array(dm.train_data),np.array(dm.train_targets),tf),batch_size=8,shuffle=False,num_workers=0)
        te=DataLoader(TaskDataset(np.array(dm.test_data),np.array(dm.test_targets),tf),batch_size=8,shuffle=False,num_workers=0)
        train_feats,_=extract(model,tr,dev); test_feats,test_labels=extract(model,te,dev)
        torch.save({"train":train_feats.cpu(),"test":test_feats.cpu(),"tl":test_labels.cpu()},CACHE); del model; torch.cuda.empty_cache()
    set_seed(cfg.SEED); T=dm.num_tasks; cf={}; Sig0=None
    methods=[("densreg",eta,0.0) for eta in ETAS]+[("densreg+predcov",eta,BETA_COMBO) for eta in ETAS]
    res={m:{"acc":[]} for m in methods}
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
        Sig=W/max(dof,1)
        if t==0: Sig0=Sig.clone()
        mu=F.normalize(mu,dim=-1); eye=torch.eye(D,device=dev); mask=test_labels<seen; Zt=test_feats[mask]; yt=test_labels[mask]; is_last=(t==T-1)
        for (name,eta,beta) in methods:
            Sr=(1-eta)*Sig+eta*Sig0; dl=torch.diagonal(Sr).mean().clamp_min(1e-8)
            if beta==0.0:
                M=torch.linalg.inv(Sr+GAMMA*dl*eye); L=2.0*((Zt@M)@mu.t())-((mu@M)*mu).sum(1).unsqueeze(0)
            else:
                scale=(1.0+beta/Kc).view(seen,1,1); Minv=torch.linalg.inv(scale*Sr.unsqueeze(0)+GAMMA*dl*eye.unsqueeze(0)); L=maha_pc(Zt,mu,Minv)
            mm=ev.calc_accuracy(sp(L,L.std().clamp_min(1e-6)),yt,t)
            res[(name,eta,beta)]["acc"].append(round(mm["mean_acc"],3))
            if is_last: res[(name,eta,beta)]["last"]={k:mm[k] for k in ["base_avg_acc","inc_avg_acc","harmonic_acc"]}
        torch.cuda.empty_cache()
    def avg(x): return round(sum(x)/len(x),3)
    base=avg(res[("densreg",0.0,0.0)]["acc"])
    rows=[]
    for (name,eta,beta) in methods:
        e=res[(name,eta,beta)]; rows.append({"method":name,"eta":eta,"beta":beta,"avg":avg(e["acc"]),"final":e["acc"][-1],"last":e.get("last",{}),"delta":round(avg(e["acc"])-base,3)})
    best=max(rows,key=lambda r:r["avg"])
    out={"baseline_avg":base,"rows":rows,"best":best,"runtime_sec":round(time.time()-t0,1)}
    od=os.path.join(ROOT,"dino_densreg_runs","cub_seed1"); os.makedirs(od,exist_ok=True)
    json.dump(out,open(os.path.join(od,"results.json"),"w"),indent=2)
    L=[f"SESSION-STABLE DENSITY REGULARIZER | CUB seed1 @448 V1 (baseline avg {base:.2f})","",
       f"  {'method':18s} {'eta':>4s} {'beta':>4s} {'avg':>7s} {'base':>6s} {'inc':>6s} {'harm':>6s} {'dVS':>6s}"]
    for r in rows:
        la=r["last"]; L.append(f"  {r['method']:18s} {r['eta']:>4} {r['beta']:>4} {r['avg']:7.2f} {la.get('base_avg_acc','-')!s:>6} {la.get('inc_avg_acc','-')!s:>6} {la.get('harmonic_acc','-')!s:>6} {r['delta']:+6.2f}")
    L.append(""); L.append(f"BEST: {best['method']} eta={best['eta']} beta={best['beta']} avg {best['avg']:.2f} (dVS {best['delta']:+.2f}) inc {best['last'].get('inc_avg_acc')} harm {best['last'].get('harmonic_acc')}")
    L.append(f"runtime_sec {out['runtime_sec']}")
    open(os.path.join(od,"summary.txt"),"w").write("\n".join(L)+"\n"); print("\n"+"\n".join(L),flush=True)
if __name__=="__main__": main()
