"""
WEEK-1 GATE: AUROC-versus-session for the V1 Mahalanobis density. CUB seed1 @448, ViT-B/14.

Question: is the class-conditional Gaussian density (the V1 score) an INFORMATIVE, SESSION-STABLE
confidence signal? If yes -> predictive-covariance correction + session-stable density regularizer
(weeks 2-4) have something to exploit. If the density is uninformative (AUROC ~0.5) or collapses
across sessions -> stop.

We compute, per session t, AUROC of several density-derived scores for two tasks:
  (A) Misclassification detection: label = (prediction correct), score = confidence.
      scores: msp (max softmax prob), margin (d_top1 - d_top2), maxdens (d_top1 = -min Maha dist),
              rmd (relative-Maha confidence d_top1 - d_0, d_0 = global background Gaussian).
  (B) Base-vs-novel separability: label = (true class is novel, y>=num_base),
      score = max_{novel} d_c - max_{base} d_c  (how 'novel-like' the density says the query is).
Also report the novel->base confusion rate (fraction of true-novel queries argmaxed to a base class).
Reuses cached features dino_feats_cls_R448.pt; nearly free.
"""
import os, sys, json, time
ROOT=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,ROOT)
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from main import setup_cfg
from utils.util import set_seed, set_gpu
from datasets.data_manager import DatasetManager, TaskDataset

DINO_MEAN=(0.485,0.456,0.406); DINO_STD=(0.229,0.224,0.225); GAMMA=1.0; CACHE="dino_feats_cls_R448.pt"
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
def maha_quad(Zt,mu,M):
    # returns d_c = -(z-mu_c)^T M (z-mu_c) for all c  [N,seen], keeping the z^T M z term (needed for absolute density)
    zMz=(Zt@M*Zt).sum(1,keepdim=True)               # [N,1]
    cross=2.0*(Zt@M)@mu.t()                          # [N,seen]
    muMmu=((mu@M)*mu).sum(1).unsqueeze(0)            # [1,seen]
    return -(zMz - cross + muMmu)

def main():
    t0=time.time()
    cfg=setup_cfg("configs/datasets/cub200_bimc_dino_fusion.yaml","configs/trainers/bimc_dino_fusion.yaml")
    set_seed(cfg.SEED); set_gpu(cfg.DEVICE.GPU_ID); dev="cuda"
    dm=DatasetManager(cfg); pathmap={p:i for i,p in enumerate(dm.train_data)}; nb=len(dm.class_index_in_task[0])
    if os.path.exists(CACHE):
        d=torch.load(CACHE); train_feats=d["train"].to(dev); test_feats=d["test"].to(dev); test_labels=d["tl"].to(dev)
        print("loaded cached @448 feats",flush=True)
    else:
        model=load_bb(dev); tf=tfR(448)
        tr=DataLoader(TaskDataset(np.array(dm.train_data),np.array(dm.train_targets),tf),batch_size=8,shuffle=False,num_workers=0)
        te=DataLoader(TaskDataset(np.array(dm.test_data),np.array(dm.test_targets),tf),batch_size=8,shuffle=False,num_workers=0)
        train_feats,_=extract(model,tr,dev); test_feats,test_labels=extract(model,te,dev)
        torch.save({"train":train_feats.cpu(),"test":test_feats.cpu(),"tl":test_labels.cpu()},CACHE); del model; torch.cuda.empty_cache()

    set_seed(cfg.SEED); T=dm.num_tasks; cf={}
    rows=[]
    SCORES=["msp","margin","maxdens","rmd"]
    for t in range(T):
        seen=int(max(dm.class_index_in_task[t])+1)
        ds=dm.get_dataset(t,"train",mode="test")
        ridx=torch.tensor([pathmap[p] for p in ds.images],device=dev); labs=torch.as_tensor(np.asarray(ds.labels),device=dev); sf=train_feats[ridx]
        for c in torch.unique(labs).tolist():
            fc=sf[labs==c]; cf[c]=fc if c not in cf else torch.cat([cf[c],fc],0)
        # V1 stats
        mu=torch.zeros((seen,test_feats.shape[1]),device=dev); W=torch.zeros((test_feats.shape[1],)*2,device=dev); dof=0
        all_sup=[]
        for c in range(seen):
            Xc=cf[c]; m=Xc.mean(0); mu[c]=m; dd=Xc-m; W+=dd.t()@dd; all_sup.append(Xc)
            if Xc.shape[0]>1: dof+=Xc.shape[0]-1
        D=mu.shape[1]; Sig=W/max(dof,1); delta=torch.diagonal(Sig).mean().clamp_min(1e-8)
        mu=F.normalize(mu,dim=-1)
        M=torch.linalg.inv(Sig+GAMMA*delta*torch.eye(D,device=dev))
        mask=test_labels<seen; Zt=test_feats[mask]; yt=test_labels[mask]
        dC=maha_quad(Zt,mu,M)                                  # [N,seen] absolute log-density (up to const)
        pred=dC.argmax(1); correct=(pred==yt)
        # global background Gaussian (RMD)
        Xall=torch.cat(all_sup,0); mu0=Xall.mean(0); d0c=Xall-mu0; Sig0=d0c.t()@d0c/max(Xall.shape[0]-1,1)
        delta0=torch.diagonal(Sig0).mean().clamp_min(1e-8); M0=torch.linalg.inv(Sig0+GAMMA*delta0*torch.eye(D,device=dev))
        d0=-((Zt-mu0)@M0*(Zt-mu0)).sum(1)                     # [N] background log-density
        # scores (higher = more confident)
        T_=dC.std().clamp_min(1e-6); P=torch.softmax((dC-dC.max(1,keepdim=True).values)/T_,1)
        top2=dC.topk(2,dim=1).values
        sc={"msp":P.max(1).values, "margin":top2[:,0]-top2[:,1], "maxdens":dC.max(1).values, "rmd":dC.max(1).values-d0}
        y=correct.detach().cpu().numpy().astype(int)
        row={"task":t,"seen":seen,"ntest":int(mask.sum()),"acc":round(100*y.mean(),2)}
        for s in SCORES:
            sv=sc[s].detach().cpu().numpy()
            row[f"auroc_err_{s}"]=round(float(roc_auc_score(y,sv)),4) if (y.min()!=y.max()) else None
        # base vs novel
        if seen>nb:
            isnov=(yt>=nb).detach().cpu().numpy().astype(int)
            base_max=dC[:,:nb].max(1).values; nov_max=dC[:,nb:].max(1).values
            s_nov=(nov_max-base_max).detach().cpu().numpy()
            row["auroc_basenovel"]=round(float(roc_auc_score(isnov,s_nov)),4) if isnov.min()!=isnov.max() else None
            nov_mask=(yt>=nb); novpred=pred[nov_mask]
            row["novel_to_base_rate"]=round(100*float((novpred<nb).float().mean().item()),2)
        rows.append(row); print(row,flush=True); torch.cuda.empty_cache()

    # aggregate
    def col(k): return [r[k] for r in rows if r.get(k) is not None]
    summary={"resolution":448,"per_session":rows,
             "avg":{f"auroc_err_{s}":round(float(np.mean(col(f'auroc_err_{s}'))),4) for s in SCORES}}
    summary["avg"]["auroc_basenovel"]=round(float(np.mean(col("auroc_basenovel"))),4)
    summary["avg"]["novel_to_base_rate"]=round(float(np.mean(col("novel_to_base_rate"))),2)
    # stability: best error-AUROC score, first-vs-last incremental session
    best_s=max(SCORES,key=lambda s:summary["avg"][f"auroc_err_{s}"])
    seq=[r[f"auroc_err_{best_s}"] for r in rows]
    summary["best_err_score"]=best_s; summary["err_auroc_seq"]=seq
    summary["err_auroc_first"]=seq[0]; summary["err_auroc_last"]=seq[-1]; summary["err_auroc_drop"]=round(seq[0]-seq[-1],4)
    summary["runtime_sec"]=round(time.time()-t0,1)
    # verdict
    avg_err=summary["avg"][f"auroc_err_{best_s}"]; drop=summary["err_auroc_drop"]
    verdict="PASS" if (avg_err>=0.80 and drop<=0.07) else ("WEAK" if avg_err>=0.70 else "FAIL")
    summary["verdict"]=verdict
    od=os.path.join(ROOT,"dino_auroc_gate_runs","cub_seed1"); os.makedirs(od,exist_ok=True)
    json.dump(summary,open(os.path.join(od,"results.json"),"w"),indent=2)
    L=[f"AUROC-VS-SESSION GATE | CUB seed1 @448 V1",""]
    L.append(f"  {'task':>4s} {'seen':>4s} {'acc':>6s} | {'err_msp':>7s} {'err_marg':>8s} {'err_dens':>8s} {'err_rmd':>7s} | {'base/nov':>8s} {'nov2base%':>9s}")
    for r in rows:
        L.append(f"  {r['task']:>4d} {r['seen']:>4d} {r['acc']:>6.2f} | "
                 f"{r['auroc_err_msp']!s:>7} {r['auroc_err_margin']!s:>8} {r['auroc_err_maxdens']!s:>8} {r['auroc_err_rmd']!s:>7} | "
                 f"{r.get('auroc_basenovel','-')!s:>8} {r.get('novel_to_base_rate','-')!s:>9}")
    L.append("")
    L.append(f"AVG error-detection AUROC: "+", ".join(f"{s}={summary['avg'][f'auroc_err_{s}']:.3f}" for s in SCORES))
    L.append(f"AVG base-vs-novel AUROC: {summary['avg']['auroc_basenovel']:.3f}   AVG novel->base confusion: {summary['avg']['novel_to_base_rate']:.1f}%")
    L.append(f"Best error score: {best_s}; AUROC seq first->last = {summary['err_auroc_first']:.3f} -> {summary['err_auroc_last']:.3f} (drop {summary['err_auroc_drop']:+.3f})")
    L.append(f"VERDICT: {verdict}  (pass if avg err-AUROC>=0.80 and session drop<=0.07)")
    L.append(f"runtime_sec: {summary['runtime_sec']}")
    open(os.path.join(od,"summary.txt"),"w").write("\n".join(L)+"\n"); print("\n"+"\n".join(L)+f"\nSaved {od}",flush=True)

if __name__=="__main__": main()
