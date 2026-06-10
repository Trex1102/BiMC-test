"""
Direction 2 -- anti-circular prototype calibration (oracle ceiling first), CUB seed1 @448 V1.

A K=5 novel prototype is a noisy estimator. TEEN shifts it toward base prototypes by VISUAL
similarity (circular: uses the noisy 5-shot feature similarity). Test whether shifting toward
related base prototypes helps, with relatedness from independent sources:
   mu'_novel = normalize( (1-alpha) mu_novel + alpha * sum_b w_b mu_b^base ),  w = softmax(S[novel,base]/tau)
Relatedness S:
   teen_visual : cosine(mu_novel, mu_base)            (the circular baseline)
   oracle_attr : cosine(GT 312-attr vec)              (oracle ceiling)
   oracle_fam  : 1 if same name-family (last word)    (oracle ceiling)
   llm_tax     : LLM genus(1.0)/family(0.5)           (if llm_partition/taxonomy.json present)
Base prototypes are the clean full-data session-0 prototypes. Classify with V1 (gamma=1) on the
calibrated prototype set. Sweep alpha. Report novel/harmonic (where calibration matters).
Reuses cached @448 feats (dino_feats_cls_R448.pt).
"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from main import setup_cfg
from utils.util import set_seed, set_gpu
from datasets.data_manager import DatasetManager, TaskDataset
from utils.evaluator import AccuracyEvaluator
from datasets.cub200 import CLASSES

DINO_MEAN=(0.485,0.456,0.406); DINO_STD=(0.229,0.224,0.225); GAMMA=1.0; TAU=0.1
ALPHAS=[0.0,0.1,0.2,0.3,0.5,0.7]
ATTR_CONT="dataset/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"
CACHE="dino_feats_cls_R448.pt"; TAX="llm_partition/taxonomy.json"

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
def class_stats(sup,seen,dev,D):
    mu=torch.zeros((seen,D),device=dev); W=torch.zeros((D,D),device=dev); dof=0
    for c in range(seen):
        Xc=sup[c]; m=Xc.mean(0); mu[c]=m; d=Xc-m; W+=d.t()@d
        if Xc.shape[0]>1: dof+=Xc.shape[0]-1
    Sig=W/max(dof,1); delta=torch.diagonal(Sig).mean().clamp_min(1e-8)
    return mu,Sig,delta  # mu RAW (unnormalized) so we can calibrate then normalize
def v1_logits(Zt,mu,M): ZM=Zt@M; return 2.0*(ZM@mu.t())-((mu@M)*mu).sum(1).unsqueeze(0)
def softmax_probs(L,T): L=L-L.max(1,keepdim=True).values; return torch.softmax(L/T,dim=1)

def main():
    t0=time.time()
    cfg=setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml","configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED); set_gpu(cfg.DEVICE.GPU_ID); dev="cuda"
    dm=DatasetManager(cfg); ev=AccuracyEvaluator(dm.class_index_in_task)
    pathmap={p:i for i,p in enumerate(dm.train_data)}; nb=len(dm.class_index_in_task[0])
    # relatedness sources (200x200)
    attr=np.loadtxt(ATTR_CONT); attr=attr/ (np.linalg.norm(attr,axis=1,keepdims=True)+1e-8)
    S_attr=torch.tensor(attr@attr.T,dtype=torch.float32,device=dev)
    fam=[c.split()[-1] for c in CLASSES]
    S_fam=torch.tensor([[1.0 if fam[i]==fam[j] else 0.0 for j in range(len(CLASSES))] for i in range(len(CLASSES))],device=dev)
    S_tax=None
    if os.path.exists(TAX):
        tx=json.load(open(TAX)); gen=[tx.get(c,{}).get("genus","") for c in CLASSES]; famn=[tx.get(c,{}).get("family","") for c in CLASSES]
        S_tax=torch.tensor([[ (1.0 if gen[i] and gen[i]==gen[j] else (0.5 if famn[i] and famn[i]==famn[j] else 0.0)) for j in range(len(CLASSES))] for i in range(len(CLASSES))],device=dev)
    methods=["baseline","teen_visual","oracle_attr","oracle_fam"]+(["llm_tax"] if S_tax is not None else [])

    if os.path.exists(CACHE):
        d=torch.load(CACHE); train_feats=d["train"].to(dev); test_feats=d["test"].to(dev); test_labels=d["tl"].to(dev)
    else:
        model=load_bb(dev); tf=tfR(448)
        tr=DataLoader(TaskDataset(np.array(dm.train_data),np.array(dm.train_targets),tf),batch_size=8,shuffle=False,num_workers=4)
        te=DataLoader(TaskDataset(np.array(dm.test_data),np.array(dm.test_targets),tf),batch_size=8,shuffle=False,num_workers=4)
        train_feats,_=extract(model,tr,dev); test_feats,test_labels=extract(model,te,dev)
        torch.save({"train":train_feats.cpu(),"test":test_feats.cpu(),"tl":test_labels.cpu()},CACHE); del model; torch.cuda.empty_cache()

    set_seed(cfg.SEED); T=dm.num_tasks
    res={(meth,al):{"acc":[]} for meth in methods for al in ALPHAS}
    cf={}; base_protos=None
    for t in range(T):
        seen=int(max(dm.class_index_in_task[t])+1)
        ds=dm.get_dataset(t,"train",mode="test")
        rows=torch.tensor([pathmap[p] for p in ds.images],device=dev); labs=torch.as_tensor(np.asarray(ds.labels),device=dev); sf=train_feats[rows]
        for c in torch.unique(labs).tolist():
            fc=sf[labs==c]; cf[c]=fc if c not in cf else torch.cat([cf[c],fc],0)
        mask=test_labels<seen; Zt=test_feats[mask]; yt=test_labels[mask]; D=Zt.shape[1]
        mu_raw,Sig,delta=class_stats({c:cf[c] for c in range(seen)},seen,dev,D)
        if t==0: base_protos=mu_raw[:nb].clone()   # clean full-data base prototypes
        M0=torch.linalg.inv(Sig+GAMMA*delta*torch.eye(D,device=dev))
        is_last=(t==T-1)
        S_vis=F.normalize(mu_raw,dim=-1)@F.normalize(base_protos,dim=-1).t()  # [seen,nb]
        for meth in methods:
            if meth=="teen_visual": S=S_vis
            elif meth=="oracle_attr": S=S_attr[:seen,:nb]
            elif meth=="oracle_fam": S=S_fam[:seen,:nb]
            elif meth=="llm_tax": S=S_tax[:seen,:nb]
            else: S=None
            for al in ALPHAS:
                mu=mu_raw.clone()
                if al>0 and S is not None and seen>nb:
                    nov=torch.arange(nb,seen,device=dev)
                    Snov=S[nov]                              # [n_novel, nb]
                    w=torch.softmax(Snov/TAU,dim=1) if meth=="teen_visual" or "attr" in meth else (Snov/ (Snov.sum(1,keepdim=True)+1e-8))
                    shift=w@base_protos                      # [n_novel, D]
                    mu[nov]=(1-al)*mu_raw[nov]+al*shift
                mu=F.normalize(mu,dim=-1)
                L=v1_logits(Zt,mu,M0)
                m=ev.calc_accuracy(softmax_probs(L,L.std().clamp_min(1e-6)),yt,t)
                res[(meth,al)]["acc"].append(round(m["mean_acc"],3))
                if is_last: res[(meth,al)]["last"]={k:m[k] for k in ["base_avg_acc","inc_avg_acc","harmonic_acc"]}
        torch.cuda.empty_cache()
    def avg(a): return round(sum(a)/len(a),3)
    base=avg(res[("baseline",0.0)]["acc"]); basef=res[("baseline",0.0)]["acc"][-1]; basel=res[("baseline",0.0)].get("last",{})
    out={"resolution":448,"methods":methods,"alphas":ALPHAS,"has_llm_tax":S_tax is not None,
         "baseline":{"avg":base,"final":basef,"last":basel},"by_method":{}}
    for meth in methods:
        best=max(ALPHAS,key=lambda al:avg(res[(meth,al)]["acc"]))
        out["by_method"][meth]={"best_alpha":best,"avg":avg(res[(meth,best)]["acc"]),"final":res[(meth,best)]["acc"][-1],
                                "last":res[(meth,best)].get("last",{}),"delta":round(avg(res[(meth,best)]["acc"])-base,3),
                                "by_alpha":{str(al):avg(res[(meth,al)]["acc"]) for al in ALPHAS}}
    out["runtime_sec"]=round(time.time()-t0,1)
    od=os.path.join(ROOT,"dino_calibration_runs","cub_seed1"); os.makedirs(od,exist_ok=True)
    json.dump(out,open(os.path.join(od,"results.json"),"w"),indent=2)
    L=[f"PROTOTYPE CALIBRATION (Direction 2) | CUB seed1 @448 | llm_tax={'yes' if S_tax is not None else 'no'}",
       f"V1 baseline: avg {base:.2f} final {basef:.2f} base {basel.get('base_avg_acc')} inc {basel.get('inc_avg_acc')} harm {basel.get('harmonic_acc')}",""]
    L.append(f"  {'method':14s} {'bestA':>5s} {'avg':>7s} {'final':>7s} {'inc':>6s} {'harm':>6s} {'dVS':>6s}")
    for meth in methods:
        e=out["by_method"][meth]; la=e["last"]
        L.append(f"  {meth:14s} {e['best_alpha']:>5} {e['avg']:7.2f} {e['final']:7.2f} {la.get('inc_avg_acc','-'):>6} {la.get('harmonic_acc','-'):>6} {e['delta']:+6.2f}")
    open(os.path.join(od,"summary.txt"),"w").write("\n".join(L)+"\n"); print("\n".join(L)+f"\nSaved {od}",flush=True)

if __name__=="__main__": main()
