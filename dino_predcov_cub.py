"""
Weeks-2-4 item #1: PREDICTIVE-COVARIANCE correction (gate passed). CUB seed1 @448 V1.

A K_c-shot prototype mu_hat_c is itself uncertain: var(z - mu_hat_c) = Sigma + Sigma/K_c. So the
predictive covariance is Sigma_pred,c = Sigma_bar (1 + beta/K_c) (beta=1 is the textbook value;
we sweep it). This inflates the metric for low-shot (novel) classes, countering the over-confident
novel prototypes / the novel->base confusion the AUROC gate exposed.
   M_c = ( Sigma_bar (1 + beta/K_c) + gamma*delta*I )^-1 ,  d_c(z) = -(z-mu_c)^T M_c (z-mu_c)
beta=0 reduces to the shared-covariance V1 baseline. Reuses cached @448 feats; nearly free.
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

GAMMA=1.0; CACHE="dino_feats_cls_R448.pt"; BETAS=[0.0,0.5,1.0,2.0,5.0,10.0,20.0]
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
    m.load_state_dict(torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"),map_location="cpu"),strict=True)
    return m.to(dev).eval()
def tfR(R): return transforms.Compose([transforms.Resize(R,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(R),transforms.ToTensor(),transforms.Normalize(DINO_MEAN,DINO_STD)])
@torch.no_grad()
def extract(m,loader,dev):
    fs,ls=[],[]
    for b in loader: fs.append(F.normalize(m(b["image"].to(dev)).float(),dim=-1)); ls.append(b["label"].to(dev))
    return torch.cat(fs,0),torch.cat(ls,0)
def maha_perclass(Zt,mu,Minv_stack):
    N,Cn=Zt.shape[0],mu.shape[0]; out=torch.empty((N,Cn),device=Zt.device)
    for c in range(Cn):
        r=Zt-mu[c]; out[:,c]=-((r@Minv_stack[c])*r).sum(1)
    return out
def softmax_probs(L,T): L=L-L.max(1,keepdim=True).values; return torch.softmax(L/T,1)
def main():
    t0=time.time()
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=1); a=ap.parse_args()
    cfg=setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml","configs/trainers/bimc_dino_fusion.yaml")
    set_seed(a.seed); set_gpu(cfg.DEVICE.GPU_ID); dev="cuda"
    dm=DatasetManager(cfg); ev=AccuracyEvaluator(dm.class_index_in_task); pathmap={p:i for i,p in enumerate(dm.train_data)}
    if os.path.exists(CACHE):
        d=torch.load(CACHE); train_feats=d["train"].to(dev); test_feats=d["test"].to(dev); test_labels=d["tl"].to(dev)
    else:
        model=load_bb(dev); tf=tfR(448)
        tr=DataLoader(TaskDataset(np.array(dm.train_data),np.array(dm.train_targets),tf),batch_size=8,shuffle=False,num_workers=0)
        te=DataLoader(TaskDataset(np.array(dm.test_data),np.array(dm.test_targets),tf),batch_size=8,shuffle=False,num_workers=0)
        train_feats,_=extract(model,tr,dev); test_feats,test_labels=extract(model,te,dev)
        torch.save({"train":train_feats.cpu(),"test":test_feats.cpu(),"tl":test_labels.cpu()},CACHE); del model; torch.cuda.empty_cache()
    set_seed(a.seed); T=dm.num_tasks; cf={}
    res={f"b={b}":{"acc":[]} for b in BETAS}
    for t in range(T):
        seen=int(max(dm.class_index_in_task[t])+1)
        ds=dm.get_dataset(t,"train",mode="test")
        ridx=torch.tensor([pathmap[p] for p in ds.images],device=dev); labs=torch.as_tensor(np.asarray(ds.labels),device=dev); sf=train_feats[ridx]
        for c in torch.unique(labs).tolist():
            fc=sf[labs==c]; cf[c]=fc if c not in cf else torch.cat([cf[c],fc],0)
        D=test_feats.shape[1]; mu=torch.zeros((seen,D),device=dev); W=torch.zeros((D,D),device=dev); dof=0; Kc=torch.zeros(seen,device=dev)
        for c in range(seen):
            Xc=cf[c]; m=Xc.mean(0); mu[c]=m; dd=Xc-m; W+=dd.t()@dd; Kc[c]=Xc.shape[0]
            if Xc.shape[0]>1: dof+=Xc.shape[0]-1
        Sig=W/max(dof,1); delta=torch.diagonal(Sig).mean().clamp_min(1e-8); mu=F.normalize(mu,dim=-1)
        eye=torch.eye(D,device=dev); mask=test_labels<seen; Zt=test_feats[mask]; yt=test_labels[mask]; is_last=(t==T-1)
        for b in BETAS:
            scale=(1.0+b/Kc).view(seen,1,1)                     # [seen,1,1]
            Sigc=scale*Sig.unsqueeze(0)+delta*GAMMA*eye.unsqueeze(0)
            Minv=torch.linalg.inv(Sigc)
            L=maha_perclass(Zt,mu,Minv)
            m=ev.calc_accuracy(softmax_probs(L,L.std().clamp_min(1e-6)),yt,t)
            res[f"b={b}"]["acc"].append(round(m["mean_acc"],3))
            if is_last: res[f"b={b}"]["last"]={k:m[k] for k in ["base_avg_acc","inc_avg_acc","harmonic_acc"]}
            del Minv,Sigc,L
        torch.cuda.empty_cache()
    def avg(a): return round(sum(a)/len(a),3)
    base=avg(res["b=0.0"]["acc"]); basef=res["b=0.0"]["acc"][-1]; basel=res["b=0.0"].get("last",{})
    bestb=max(BETAS,key=lambda b:avg(res[f"b={b}"]["acc"]))
    out={"resolution":448,"betas":BETAS,"baseline_b0":{"avg":base,"final":basef,"last":basel},
         "by_beta":{str(b):{"avg":avg(res[f"b={b}"]["acc"]),"final":res[f"b={b}"]["acc"][-1],"last":res[f"b={b}"].get("last",{})} for b in BETAS},
         "best":{"beta":bestb,"avg":avg(res[f"b={bestb}"]["acc"]),"final":res[f"b={bestb}"]["acc"][-1],"last":res[f"b={bestb}"].get("last",{}),"delta":round(avg(res[f"b={bestb}"]["acc"])-base,3)},
         "runtime_sec":round(time.time()-t0,1)}
    od=os.path.join(ROOT,"dino_predcov_runs",f"cub_seed{a.seed}"); os.makedirs(od,exist_ok=True)
    json.dump(out,open(os.path.join(od,"results.json"),"w"),indent=2)
    L=[f"PREDICTIVE-COVARIANCE correction | CUB seed1 @448 V1",
       f"baseline (beta=0 = V1): avg {base:.2f} final {basef:.2f} base {basel.get('base_avg_acc')} inc {basel.get('inc_avg_acc')} harm {basel.get('harmonic_acc')}","",
       f"  {'beta':>6s} {'avg':>7s} {'final':>7s} {'base':>6s} {'inc':>6s} {'harm':>6s} {'dVS':>6s}"]
    for b in BETAS:
        e=out["by_beta"][str(b)]; la=e["last"]
        L.append(f"  {b:6.1f} {e['avg']:7.2f} {e['final']:7.2f} {la.get('base_avg_acc','-')!s:>6} {la.get('inc_avg_acc','-')!s:>6} {la.get('harmonic_acc','-')!s:>6} {e['avg']-base:+6.2f}")
    bb=out["best"]; L.append("")
    L.append(f"BEST beta={bb['beta']}: avg {bb['avg']:.2f} (dVS {bb['delta']:+.2f}) final {bb['final']:.2f} inc {bb['last'].get('inc_avg_acc')} harm {bb['last'].get('harmonic_acc')}")
    L.append(f"runtime_sec: {out['runtime_sec']}")
    open(os.path.join(od,"summary.txt"),"w").write("\n".join(L)+"\n"); print("\n"+"\n".join(L)+f"\nSaved {od}",flush=True)
if __name__=="__main__": main()
