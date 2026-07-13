"""Mechanism, ensemble and class-level analyses for the Q1 revision."""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

ROOT=Path(__file__).resolve().parent;SRC=ROOT/"source_data";RNG=np.random.default_rng(20260712)
METRICS=["accuracy","macro_f1","macro_auprc","macro_auroc","nll","brier","ece15"]
FINAL="PDRF_RCIA_R30"

def paired_tests(metrics):
 rows=[]
 for subset in ["all","affected","unaffected"]:
  for metric in METRICS:
   for right in ["PDRF","CAGF","PDRF_REC","PDRF_RC"]:
    a=metrics[(metrics.method==FINAL)&(metrics.subset==subset)&(metrics.metric==metric)].sort_values("seed")
    b=metrics[(metrics.method==right)&(metrics.subset==subset)&(metrics.metric==metric)].sort_values("seed")
    d=a.value.to_numpy()-b.value.to_numpy()
    if metric in ["nll","brier","ece15"]:d=-d
    t=wilcoxon(d,method="exact",alternative="two-sided")
    rows.append({"contrast":f"{FINAL}-{right}","subset":subset,"metric":metric,"n":len(d),
                 "aligned_mean_difference":d.mean(),"wins":int((d>0).sum()),"exact_p":t.pvalue})
 return pd.DataFrame(rows)

def ensemble_arrays(pred,method,pool="probability_mean"):
 z=pred[(pred.method==method)&(pred.pool==pool)].sort_values("sample")
 pc=[f"p{i}" for i in range(1,7)]
 return z.y.to_numpy(),z.affected.to_numpy(bool),z.batch.to_numpy(),z[pc].to_numpy(),z

def bootstrap_auc(y,affected,pa,pb,reps=5000):
 rows=[]
 for subset,take in [("all",np.ones(len(y),bool)),("affected",affected)]:
  yy=y[take];aa=pa[take];bb=pb[take];yb=label_binarize(yy,classes=np.arange(aa.shape[1]))
  est=roc_auc_score(yb,aa,average="macro",multi_class="ovr")-roc_auc_score(yb,bb,average="macro",multi_class="ovr")
  vals=[]
  for _ in range(reps):
   ix=RNG.integers(0,len(yy),len(yy));yb2=label_binarize(yy[ix],classes=np.arange(aa.shape[1]))
   vals.append(roc_auc_score(yb2,aa[ix],average="macro",multi_class="ovr")-roc_auc_score(yb2,bb[ix],average="macro",multi_class="ovr"))
  lo,hi=np.quantile(vals,[.025,.975]);rows.append({"subset":subset,"metric":"macro_auroc","difference":est,"ci_low":lo,"ci_high":hi,"reps":reps})
 return rows

def main():
 metrics=pd.read_csv(SRC/"q1_risk_metrics.csv");tests=paired_tests(metrics);tests.to_csv(SRC/"q1_seed_tests.csv",index=False)
 pred_seed=pd.read_csv(SRC/"q1_risk_predictions.csv");pred_ens=pd.read_csv(SRC/"q1_ensemble_predictions.csv")
 mech=pd.read_csv(SRC/"q1_mechanism_samples.csv");m=mech[mech.method==FINAL].copy()
 corr=[]
 for seed,g in m.groupby("seed"):
  for x in ["score_change","weight_reduction"]:
   for y in ["true_probability_change","fault_correct"]:
    r,p=spearmanr(g[x],g[y].astype(float));corr.append({"seed":seed,"x":x,"y":y,"spearman_r":r,"p":p})
 pd.DataFrame(corr).to_csv(SRC/"q1_mechanism_correlations.csv",index=False)

 # Downweighting succeeds when the affected group's normalized weight falls.
 audit=[]
 for method,g in mech.groupby("method"):
  for seed,s in g.groupby("seed"):
   down=s.weight_reduction>0
   audit.append({"method":method,"seed":seed,"n":len(s),"downweight_success":down.mean(),
    "failure_despite_downweight":((down)&(~s.fault_correct)).mean(),
    "clean_correct_retained":((s.clean_correct)&(s.fault_correct)).sum()/max(1,s.clean_correct.sum()),
    "fault_accuracy":s.fault_correct.mean(),"saturation":(s.fault_score.abs()>=2.85).mean()})
 pd.DataFrame(audit).to_csv(SRC/"q1_mechanism_audit.csv",index=False)

 # Matched correctness decomposition against CAGF and per-class ensemble results.
 decomposition=[]
 for seed in sorted(pred_seed.seed.unique()):
  a=pred_seed[(pred_seed.method==FINAL)&(pred_seed.seed==seed)].sort_values("sample")
  b=pred_seed[(pred_seed.method=="CAGF")&(pred_seed.seed==seed)].sort_values("sample")
  pc=[f"p{i}" for i in range(1,7)];ca=a[pc].to_numpy().argmax(1)==a.y.to_numpy();cb=b[pc].to_numpy().argmax(1)==b.y.to_numpy()
  for subset,take in [("affected",a.affected.to_numpy(bool)),("unaffected",~a.affected.to_numpy(bool))]:
   decomposition.append({"seed":seed,"subset":subset,"rci_only_correct":int((ca&~cb&take).sum()),
    "cagf_only_correct":int((~ca&cb&take).sum()),"both_correct":int((ca&cb&take).sum()),"n":int(take.sum())})
 pd.DataFrame(decomposition).to_csv(SRC/"q1_correctness_decomposition.csv",index=False)

 y,aff,batch,pr,zr=ensemble_arrays(pred_ens,FINAL);_,_,_,pc,zc=ensemble_arrays(pred_ens,"CAGF")
 boot=bootstrap_auc(y,aff,pr,pc);pd.DataFrame(boot).to_csv(SRC/"q1_ensemble_bootstrap.csv",index=False)
 class_rows=[]
 for method,p in [(FINAL,pr),("CAGF",pc)]:
  pred=p.argmax(1);cm=confusion_matrix(y[aff],pred[aff],labels=np.arange(6))
  for i in range(6):
   class_rows.append({"method":method,"class":i+1,"affected_recall":recall_score(y[aff]==i,pred[aff]==i,zero_division=0),
    **{f"pred_{j+1}":int(cm[i,j]) for j in range(6)}})
 pd.DataFrame(class_rows).to_csv(SRC/"q1_affected_class_confusion.csv",index=False)

 # Seed diversity and sharpness from individual models.
 div=[];pcols=[f"p{i}" for i in range(1,7)]
 for method,g in pred_seed.groupby("method"):
  seeds=sorted(g.seed.unique());arr=np.stack([g[g.seed==s].sort_values("sample")[pcols].to_numpy() for s in seeds])
  disagreements=[];js=[]
  for i,j in combinations(range(len(seeds)),2):
   disagreements.append((arr[i].argmax(1)!=arr[j].argmax(1)).mean())
   js.append(np.mean([jensenshannon(arr[i,k],arr[j,k])**2 for k in range(arr.shape[1])]))
  meanp=arr.mean(0);entropy=-(meanp*np.log(np.clip(meanp,1e-12,1))).sum(1)
  div.append({"method":method,"pairwise_disagreement":np.mean(disagreements),"pairwise_js":np.mean(js),
              "ensemble_sharpness":meanp.max(1).mean(),"ensemble_entropy":entropy.mean()})
 pd.DataFrame(div).to_csv(SRC/"q1_ensemble_diversity.csv",index=False)

 # Confidence-error curves and concrete high-downweight failures.
 curve=[]
 for method,p in [(FINAL,pr),("CAGF",pc)]:
  conf=p.max(1);correct=p.argmax(1)==y
  for lo in np.linspace(0,1,10,endpoint=False):
   take=(conf>=lo)&(conf<lo+.1)
   if take.any():curve.append({"method":method,"bin_low":lo,"n":int(take.sum()),"mean_confidence":conf[take].mean(),"error_rate":1-correct[take].mean()})
 pd.DataFrame(curve).to_csv(SRC/"q1_confidence_error.csv",index=False)
 failures=m[(m.weight_reduction>0)&(~m.fault_correct)].sort_values("weight_reduction",ascending=False).head(40)
 failures.to_csv(SRC/"q1_high_downweight_failures.csv",index=False)

if __name__=="__main__":main()
