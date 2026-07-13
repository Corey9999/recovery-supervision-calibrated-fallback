"""Paired inference for the Section 2.3 exploratory remedies."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import analyse_major_revision as major_stats

ROOT=Path(__file__).resolve().parent; SRC=ROOT/"source_data"
RNG=np.random.default_rng(20260712); REPS=5000

metrics=pd.read_csv(SRC/"targeted_improvement_metrics.csv")
rows=[]
for subset in ["all","affected","unaffected"]:
  for metric in ["accuracy","macro_f1","macro_auprc","macro_auroc","nll"]:
    for left,right in [("PDRF_AUC","PDRF"),("PDRF_AUC","CAGF")]:
      a=metrics[(metrics.level=="single_model")&(metrics.method==left)&(metrics.subset==subset)&(metrics.metric==metric)].sort_values("seed")
      b=metrics[(metrics.level=="single_model")&(metrics.method==right)&(metrics.subset==subset)&(metrics.metric==metric)].sort_values("seed")
      d=a.value.to_numpy()-b.value.to_numpy()
      if metric=="nll": d=-d
      test=wilcoxon(d,alternative="two-sided",method="exact")
      rows.append({"contrast":f"{left}-{right}","subset":subset,"metric":metric,"n":len(d),
                   "aligned_mean_difference":d.mean(),"wins":int((d>0).sum()),
                   "exact_wilcoxon_p":test.pvalue})
pd.DataFrame(rows).to_csv(SRC/"targeted_seed_tests.csv",index=False)

pred=pd.read_csv(SRC/"targeted_ensemble_predictions.csv")
pcols=[f"p{i}" for i in range(1,7)]
boot=[]
for pool in ["equal_probability","calibration_stacking"]:
  for left,right in [("PDRF","CAGF"),("PDRF_AUC","CAGF")]:
    a=pred[(pred.pool==pool)&(pred.method==left)].sort_values("sample")
    b=pred[(pred.pool==pool)&(pred.method==right)].sort_values("sample")
    y=a.y.to_numpy();batch=a.batch.to_numpy();pa=a[pcols].to_numpy();pb=b[pcols].to_numpy()
    for subset,take in [("all",np.ones(len(y),bool)),("affected",a.affected.to_numpy(bool))]:
      yy=y[take];bb=batch[take];aa=pa[take];cc=pb[take]
      for metric in ["macro_auroc"]:
        est=major_stats.metric(yy,aa,metric)-major_stats.metric(yy,cc,metric)
        if metric=="nll": est=-est
        vals=[]
        for _ in range(REPS):
          idx=RNG.integers(0,len(yy),len(yy));v=major_stats.metric(yy[idx],aa[idx],metric)-major_stats.metric(yy[idx],cc[idx],metric)
          vals.append(-v if metric=="nll" else v)
        lo,hi=np.quantile(vals,[.025,.975])
        boot.append({"pool":pool,"contrast":f"{left}-{right}","subset":subset,"metric":metric,
                     "aligned_difference":est,"ci_low":lo,"ci_high":hi,"reps":REPS})
pd.DataFrame(boot).to_csv(SRC/"targeted_ensemble_bootstrap.csv",index=False)
