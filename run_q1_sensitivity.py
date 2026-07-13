"""Three-seed sensitivity of PDRF-RCI score geometry and ranking strength."""
from dataclasses import replace
from pathlib import Path
import pandas as pd
import numpy as np

import run_q1_risk_sensitive as q1
import run_major_revision_experiments as major
import run_q2_revision as q2
import run_uci_validation as base

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"source_data";SEEDS=(101,102,103)
BASE=replace(major.SPECS["PDRF"],recovery_distillation=True,w_recovery=.30,
             brier_regularization=True,interior_barrier=True,w_interior=.005,
             degraded_auc=True)
SETTINGS=[]
for b in [2.,3.,4.,5.]:SETTINGS.append((f"B={b:g}",replace(BASE,bound=b)))
for beta in [.10,.25,.50,.75]:SETTINGS.append((f"beta={beta:g}",replace(BASE,beta=beta)))
for wr in [.05,.15,.30]:SETTINGS.append((f"rank={wr:g}",replace(BASE,w_rank=wr)))

zip_path=base.ensure_zip(None);groups=[list(range(i,i+4)) for i in range(0,16,4)]
split=q2.prepare_grouped(zip_path,groups);tx,ty,_=split["test"]
clean,fault,mask,q,affected=q1.paired_fault(tx,70001)
rows=[]
for label,spec in SETTINGS:
 for seed in SEEDS:
  train,select,calibration,cw=q2.train_views(split,seed)
  model,_=major.fit(spec,train,select,cw,seed,prior=(.45,.25,.18,.12),epochs=60)
  temp=major.fit_temperature(model,calibration)
  p,s,_,_,_=major.predict(model,fault,mask,q,temp)
  vals=q1.all_metrics(ty,p);av=q1.all_metrics(ty[affected],p[affected])
  rows.append({"setting":label,"seed":seed,"bound":spec.bound,"beta":spec.beta,"rank_weight":spec.w_rank,
   "saturation":float((np.abs(s[affected,0])>=.95*spec.bound).mean()),
   "macro_auroc":vals["macro_auroc"],"affected_auroc":av["macro_auroc"],
   "nll":vals["nll"],"ece15":vals["ece15"],"brier":vals["brier"]})
pd.DataFrame(rows).to_csv(OUT/"q1_sensitivity.csv",index=False)
