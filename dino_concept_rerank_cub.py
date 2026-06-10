"""
Oracle/LLM concept directions as a PER-GROUP LOCAL metric (the faithful Direction-1 form).
The global up-weighted metric washed out (+0.01); here we apply attribute axes only where they
matter: re-rank each query's top-K confusable candidates using ONLY the attributes that DIFFER
within that K-group.

For query z with baseline-V1 top-K candidate classes G(z):
   varying attrs A(z) = {a : candidates in G(z) disagree on a}
   new_score(c) = L_base(c) - lambda * sum_{a in A(z)} ( d_a^T (z - mu_c) )^2      for c in G(z)
   prediction = argmax_{c in G(z)} new_score(c)
d_a = FastCAV concept direction (has-mean minus lacks-mean over seen-class prototypes).
Partition: 'gt' (CUB 312 attrs) or LLM json. Sweep K and lambda. Feature-cached @448.
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

DINO_MEAN=(0.485,0.456,0.406); DINO_STD=(0.229,0.224,0.225)
GAMMA=1.0; ATTR_THR=50.0; MIN_COUNT=3
KS=[2,5,10]; LAMBDAS=[0.0,0.5,1.0,2.0,4.0,8.0,16.0,32.0]
ATTR_DIR="dataset/CUB_200_2011/CUB_200_2011/attributes"
FEAT_CACHE="dino_feats_cls_R{R}.pt"

def _sdpa():
    if hasattr(F,"scaled_dot_product_attention"): return
    def s(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None):
        sf=(q.size(-1)**-0.5) if scale is None else scale
        return torch.matmul(torch.softmax(torch.matmul(q,k.transpose(-2,-1))*sf,-1),v)
    F.scaled_dot_product_attention=s

def load_bb(device):
    _sdpa(); hub=os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    w=os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
    m=torch.hub.load(hub,"dinov2_vitb14",source="local",pretrained=False)
    m.load_state_dict(torch.load(w,map_location="cpu"),strict=True); return m.to(device).eval()

def tfR(R): return transforms.Compose([transforms.Resize(R,interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(R),transforms.ToTensor(),transforms.Normalize(DINO_MEAN,DINO_STD)])

@torch.no_grad()
def extract(model,loader,device):
    fs,ls=[],[]
    for b in loader:
        fs.append(F.normalize(model(b["image"].to(device)).float(),dim=-1)); ls.append(b["label"].to(device))
    return torch.cat(fs,0),torch.cat(ls,0)

def load_partition(source,n,attr_names):
    if source=="gt":
        return (np.loadtxt(os.path.join(ATTR_DIR,"class_attribute_labels_continuous.txt"))>ATTR_THR)
    part=json.load(open(source)); M=np.zeros((n,len(part)),dtype=bool)
    for j,nm in enumerate(part):
        for ci in part[nm]:
            if 0<=int(ci)<n: M[int(ci),j]=True
    return M

def class_stats(sup,seen,device,D):
    mu=torch.zeros((seen,D),device=device); W=torch.zeros((D,D),device=device); dof=0
    for c in range(seen):
        Xc=sup[c]; m=Xc.mean(0); mu[c]=m; d=Xc-m; W+=d.t()@d
        if Xc.shape[0]>1: dof+=Xc.shape[0]-1
    Sig=W/max(dof,1); delta=torch.diagonal(Sig).mean().clamp_min(1e-8)
    return F.normalize(mu,dim=-1),Sig,delta

def concept_dirs(mu,seen,attr_bin,device):
    sub=attr_bin[:seen]; dirs=[]; cols=[]
    for a in range(sub.shape[1]):
        h=np.where(sub[:,a])[0]; l=np.where(~sub[:,a])[0]
        if len(h)>=MIN_COUNT and len(l)>=MIN_COUNT:
            dirs.append(F.normalize(mu[h].mean(0)-mu[l].mean(0),dim=0)); cols.append(a)
    if not dirs: return None,None
    return torch.stack(dirs,0), np.array(cols)   # D[Ad,768], used attr cols

def v1_logits(Zt,mu,M):
    ZM=Zt@M; return 2.0*(ZM@mu.t())-((mu@M)*mu).sum(1).unsqueeze(0)

def softmax_probs(L,T):
    L=L-L.max(1,keepdim=True).values; return torch.softmax(L/T,dim=1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--partition",default="gt"); ap.add_argument("--tag",default="oracle_gt_rerank")
    ap.add_argument("--resolution",type=int,default=448); ap.add_argument("--bs",type=int,default=8); a=ap.parse_args()
    t0=time.time()
    cfg=setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml","configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED); set_gpu(cfg.DEVICE.GPU_ID); device="cuda"
    dm=DatasetManager(cfg); ev=AccuracyEvaluator(dm.class_index_in_task)
    pathmap={p:i for i,p in enumerate(dm.train_data)}
    attr_names=[ln.strip().split(" ",1)[1] for ln in open("dataset/CUB_200_2011/attributes.txt")]
    attr_bin=load_partition(a.partition,dm.num_total_classes,attr_names)
    print(f"=== RERANK [{a.tag}] R={a.resolution} partition={a.partition} attrs={attr_bin.shape[1]} ===",flush=True)
    cache=FEAT_CACHE.format(R=a.resolution)
    if os.path.exists(cache):
        print("loading cached feats",flush=True); d=torch.load(cache)
        train_feats=d["train"].to(device); test_feats=d["test"].to(device); test_labels=d["tl"].to(device)
    else:
        model=load_bb(device); tf=tfR(a.resolution)
        tr=DataLoader(TaskDataset(np.array(dm.train_data),np.array(dm.train_targets),tf),batch_size=a.bs,shuffle=False,num_workers=4)
        te=DataLoader(TaskDataset(np.array(dm.test_data),np.array(dm.test_targets),tf),batch_size=a.bs,shuffle=False,num_workers=4)
        print("extracting train",flush=True); train_feats,_=extract(model,tr,device)
        print("extracting test",flush=True); test_feats,test_labels=extract(model,te,device)
        torch.save({"train":train_feats.cpu(),"test":test_feats.cpu(),"tl":test_labels.cpu()},cache)
        del model; torch.cuda.empty_cache()

    set_seed(cfg.SEED); T=dm.num_tasks
    # results: ('base') and (K,lam)
    res={"base":{"acc":[]}}
    for K in KS:
        for lam in LAMBDAS: res[(K,lam)]={"acc":[]}
    cf={}
    for t in range(T):
        seen=int(max(dm.class_index_in_task[t])+1)
        ds=dm.get_dataset(t,"train",mode="test")
        rows=torch.tensor([pathmap[p] for p in ds.images],device=device)
        labs=torch.as_tensor(np.asarray(ds.labels),device=device); sf=train_feats[rows]
        for c in torch.unique(labs).tolist():
            fc=sf[labs==c]; cf[c]=fc if c not in cf else torch.cat([cf[c],fc],0)
        mask=test_labels<seen; Zt=test_feats[mask]; yt=test_labels[mask]; D=Zt.shape[1]; N=Zt.shape[0]
        mu,Sig,delta=class_stats({c:cf[c] for c in range(seen)},seen,device,D)
        M0=torch.linalg.inv(Sig+GAMMA*delta*torch.eye(D,device=device))
        Lbase=v1_logits(Zt,mu,M0)
        is_last=(t==T-1)
        m=ev.calc_accuracy(softmax_probs(Lbase,Lbase.std().clamp_min(1e-6)),yt,t)
        res["base"]["acc"].append(round(m["mean_acc"],3))
        if is_last: res["base"]["last"]={k:m[k] for k in ["base_avg_acc","inc_avg_acc","harmonic_acc"]}
        Dc,cols=concept_dirs(mu,seen,attr_bin,device)
        attr_seen=torch.as_tensor(attr_bin[:seen][:,cols].astype(np.float32),device=device) if Dc is not None else None  # [seen,Ad]
        for K in KS:
            topk=Lbase.topk(min(K,seen),dim=1).indices  # [N,Kk]
            Kk=topk.shape[1]
            if Dc is None:
                for lam in LAMBDAS:
                    res[(K,lam)]["acc"].append(res["base"]["acc"][-1])
                continue
            # differing-attr mask per query: attr varies among the K candidates
            am=attr_seen[topk]                      # [N,Kk,Ad]
            s=am.sum(1)                              # [N,Ad]
            vary=((s>0)&(s<Kk)).float()             # [N,Ad]
            # projections d_a^T (z - mu_c) for c in topk
            mu_k=mu[topk]                           # [N,Kk,768]
            r=Zt.unsqueeze(1)-mu_k                  # [N,Kk,768]
            P=torch.einsum("nkd,ad->nka",r,Dc)      # [N,Kk,Ad]
            base_scores=torch.gather(Lbase,1,topk)  # [N,Kk]
            for lam in LAMBDAS:
                pen=lam*(vary.unsqueeze(1)*P.pow(2)).sum(-1)   # [N,Kk]
                new=base_scores-pen
                pred=topk[torch.arange(N,device=device),new.argmax(1)]
                acc=(pred==yt).float().mean().item()*100.0
                res[(K,lam)]["acc"].append(round(acc,3))
                if is_last:
                    # base/inc/harm via evaluator using a one-hot prob from pred
                    probs=torch.zeros(N,seen,device=device); probs[torch.arange(N,device=device),pred]=1.0
                    mm=ev.calc_accuracy(probs,yt,t)
                    res[(K,lam)]["last"]={k:mm[k] for k in ["base_avg_acc","inc_avg_acc","harmonic_acc"]}
        if is_last: print(f"[task {t}] seen={seen} n_dirs={0 if Dc is None else Dc.shape[0]}",flush=True)
        torch.cuda.empty_cache()

    def avg(x): return round(sum(x)/len(x),3)
    base=avg(res["base"]["acc"]); basef=res["base"]["acc"][-1]
    best=("base",0,0,base,basef,res["base"].get("last",{}))
    grid={}
    for K in KS:
        for lam in LAMBDAS:
            av=avg(res[(K,lam)]["acc"]); grid[f"K{K}_lam{lam}"]={"avg":av,"final":res[(K,lam)]["acc"][-1],"last":res[(K,lam)].get("last",{})}
            if av>best[3]: best=(f"K{K}_lam{lam}",K,lam,av,res[(K,lam)]["acc"][-1],res[(K,lam)].get("last",{}))
    out={"tag":a.tag,"partition":a.partition,"resolution":a.resolution,"n_attrs":int(attr_bin.shape[1]),
         "baseline":{"avg":base,"final":basef,"last":res["base"].get("last",{})},
         "best":{"config":best[0],"K":best[1],"lambda":best[2],"avg":best[3],"final":best[4],"last":best[5],"delta":round(best[3]-base,3)},
         "grid":grid,"runtime_sec":round(time.time()-t0,1)}
    od=os.path.join(ROOT,"dino_concept_runs",f"cub_seed1_{a.tag}"); os.makedirs(od,exist_ok=True)
    json.dump(out,open(os.path.join(od,"results.json"),"w"),indent=2)
    L=[f"CONCEPT RERANK [{a.tag}] @ {a.resolution} | partition={a.partition} | {out['n_attrs']} attrs",
       f"V1 baseline: avg {base:.2f} final {basef:.2f} base {out['baseline']['last'].get('base_avg_acc')} inc {out['baseline']['last'].get('inc_avg_acc')} harm {out['baseline']['last'].get('harmonic_acc')}",""]
    for K in KS:
        row=" ".join(f"l{lam}:{grid[f'K{K}_lam{lam}']['avg']:.2f}" for lam in LAMBDAS)
        L.append(f"  K={K:2d}: {row}")
    b=out["best"]; L.append("")
    L.append(f"BEST: {b['config']} avg {b['avg']:.2f} (delta {b['delta']:+.2f}) final {b['final']:.2f} base {b['last'].get('base_avg_acc')} inc {b['last'].get('inc_avg_acc')} harm {b['last'].get('harmonic_acc')}")
    L.append(f"runtime_sec: {out['runtime_sec']}")
    open(os.path.join(od,"summary.txt"),"w").write("\n".join(L)+"\n"); print("\n"+"\n".join(L)+f"\nSaved {od}",flush=True)

if __name__=="__main__": main()
