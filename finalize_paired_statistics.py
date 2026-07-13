from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT=Path(__file__).resolve().parent/"source_data"
seed=pd.read_csv(OUT/"major_seed_paired_effects.csv")
rows=[]
for metric,g in seed.groupby("metric"):
    x=g.difference.to_numpy()
    test=wilcoxon(x,alternative="two-sided",method="exact")
    rows.append({"dataset":"gas_array","contrast":"PDRF-CAGF","metric":metric,
                 "n_seeds":len(x),"mean_difference":x.mean(),"median_difference":np.median(x),
                 "wins":int((x>0).sum()),"wilcoxon_statistic":test.statistic,"exact_p":test.pvalue})

hyd=pd.read_csv(OUT/"hydraulic_validation_metrics.csv")
for task in sorted(hyd.task.unique()):
    for fault in ("natural","silent_pressure","silent_vibration"):
        x=hyd[(hyd.task==task)&(hyd.fault==fault)&(hyd.subset=="all")&
              (hyd.metric=="macro_auroc")]
        a=x[x.method=="PDRF_NOQ"].sort_values("seed").value.to_numpy()
        b=x[x.method=="CAGF"].sort_values("seed").value.to_numpy()
        d=a-b; test=wilcoxon(d,alternative="two-sided",method="exact")
        rows.append({"dataset":"hydraulic","task":task,"fault":fault,
                     "contrast":"PDRF_NOQ-CAGF","metric":"macro_auroc","n_seeds":len(d),
                     "mean_difference":d.mean(),"median_difference":np.median(d),
                     "wins":int((d>0).sum()),"wilcoxon_statistic":test.statistic,"exact_p":test.pvalue})
pd.DataFrame(rows).to_csv(OUT/"major_paired_tests.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
