"""Confirmatory statistics and revision tables for the major-revision runs."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             log_loss, roc_auc_score)
from sklearn.preprocessing import label_binarize

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"source_data"; TABLES=ROOT/"tables"
RNG=np.random.default_rng(20260712); REPS=5000


def metric(y,p,name):
    pred=p.argmax(1); c=p.shape[1]; yb=label_binarize(y,classes=np.arange(c))
    if name=="accuracy": return accuracy_score(y,pred)
    if name=="macro_f1": return f1_score(y,pred,average="macro")
    if name=="macro_auroc": return roc_auc_score(yb,p,average="macro",multi_class="ovr")
    if name=="macro_auprc": return average_precision_score(yb,p,average="macro")
    if name=="nll": return log_loss(y,p,labels=np.arange(c))
    raise ValueError(name)


def ensemble(frame,method):
    x=frame[frame.method==method]
    pcols=[c for c in x if c.startswith("p") and c[1:].isdigit()]
    g=x.groupby(["sample","batch","y"],as_index=False)[pcols].mean()
    p=g[pcols].to_numpy(); p=np.clip(p,1e-12,None); p/=p.sum(1,keepdims=True)
    return g.y.to_numpy(),g.batch.to_numpy(),p,g


def bootstrap(y,batch,pa,pb,name,stratified=False):
    observed=metric(y,pa,name)-metric(y,pb,name)
    if name=="nll": observed=-observed
    vals=[]; unique=np.unique(batch)
    for _ in range(REPS):
        if stratified:
            idx=np.concatenate([RNG.choice(np.flatnonzero(batch==b),size=(batch==b).sum(),replace=True) for b in unique])
        else: idx=RNG.integers(0,len(y),len(y))
        v=metric(y[idx],pa[idx],name)-metric(y[idx],pb[idx],name)
        vals.append(-v if name=="nll" else v)
    lo,hi=np.quantile(vals,[.025,.975])
    return observed,float(lo),float(hi)


def main():
    pred=pd.read_csv(OUT/"major_seed_predictions.csv")
    pred=pred[pred.fault=="silent"]
    y,batch,pp,gp=ensemble(pred,"PDRF"); y2,b2,pc,gc=ensemble(pred,"CAGF")
    assert np.array_equal(y,y2) and np.array_equal(batch,b2)
    boot=[]
    for name in ("macro_auroc","accuracy","macro_f1","macro_auprc","nll"):
        for design in ("sample","batch_stratified"):
            est,lo,hi=bootstrap(y,batch,pp,pc,name,design=="batch_stratified")
            boot.append({"contrast":"PDRF-CAGF","metric":name,"design":design,"reps":REPS,
                         "aligned_difference":est,"ci_low":lo,"ci_high":hi})
    pd.DataFrame(boot).to_csv(OUT/"major_bootstrap_5000.csv",index=False)

    # Paired exact McNemar test on ensemble hard decisions.
    cp=pp.argmax(1)==y; cc=pc.argmax(1)==y
    n10=int((cp&~cc).sum()); n01=int((~cp&cc).sum())
    mcn={"pdrf_only_correct":n10,"cagf_only_correct":n01,"discordant":n10+n01,
         "exact_two_sided_p":float(binomtest(min(n10,n01),n10+n01,.5).pvalue)}
    (OUT/"major_mcnemar.json").write_text(json.dumps(mcn,indent=2),encoding="utf-8")
    dis=pd.crosstab(pd.Series(pc.argmax(1)+1,name="CAGF prediction"),
                    pd.Series(pp.argmax(1)+1,name="PDRF prediction"),dropna=False)
    dis.to_csv(OUT/"major_prediction_disagreement.csv")

    # Per-seed paired effects and distinction between mean seed accuracy and ensemble accuracy.
    pcols=[c for c in pred if c.startswith("p") and c[1:].isdigit()]
    seedrows=[]
    for seed in sorted(pred.seed.unique()):
        a=pred[(pred.seed==seed)&(pred.method=="PDRF")]; b=pred[(pred.seed==seed)&(pred.method=="CAGF")]
        yy=a.y.to_numpy(); pa=a[pcols].to_numpy(); pb=b[pcols].to_numpy()
        for name in ("macro_auroc","accuracy","macro_f1","macro_auprc","nll"):
            d=metric(yy,pa,name)-metric(yy,pb,name)
            seedrows.append({"seed":seed,"metric":name,"difference":-d if name=="nll" else d})
    pd.DataFrame(seedrows).to_csv(OUT/"major_seed_paired_effects.csv",index=False)
    explanation={"mean_seed_accuracy_PDRF":float(pred[pred.method=="PDRF"].groupby("seed").apply(lambda z:accuracy_score(z.y,z.pred),include_groups=False).mean()),
                 "mean_seed_accuracy_CAGF":float(pred[pred.method=="CAGF"].groupby("seed").apply(lambda z:accuracy_score(z.y,z.pred),include_groups=False).mean()),
                 "ensemble_accuracy_PDRF":float(accuracy_score(y,pp.argmax(1))),
                 "ensemble_accuracy_CAGF":float(accuracy_score(y,pc.argmax(1))),
                 "reason":"Averaging probabilities can change the winning class; mean per-model hard-decision accuracy and hard decisions from the probability ensemble are different estimands."}
    (OUT/"major_seed_ensemble_explanation.json").write_text(json.dumps(explanation,indent=2),encoding="utf-8")

    # Compact summaries for manuscript tables.
    met=pd.read_csv(OUT/"major_ablation_metrics.csv")
    head=(met[(met.fault=="silent")&(met.subset=="all")&met.metric.isin(["accuracy","macro_f1","macro_auprc","macro_auroc","nll"])]
          .groupby(["method","metric"],as_index=False).value.agg(["mean","std"]).reset_index())
    head.to_csv(OUT/"major_ablation_summary.csv",index=False)
    hyd=pd.read_csv(OUT/"hydraulic_validation_metrics.csv") if (OUT/"hydraulic_validation_metrics.csv").exists() else pd.DataFrame()
    if len(hyd):
        hs=(hyd[(hyd.subset=="all")&hyd.metric.isin(["accuracy","macro_f1","macro_auroc"])]
            .groupby(["task","method","fault","metric"],as_index=False).value.agg(["mean","std"]).reset_index())
        hs.to_csv(OUT/"hydraulic_validation_summary.csv",index=False)

if __name__=="__main__": main()
