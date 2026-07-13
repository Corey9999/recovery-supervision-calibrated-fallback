"""Leave-one-recovery-loss-out ablation on the fixed chemical fault."""
from dataclasses import replace
from pathlib import Path
import os
import pandas as pd

import run_major_revision_experiments as major
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"source_data"
QUICK=os.getenv("M3_LOSS_QUICK")=="1"
SEEDS=(101,) if QUICK else tuple(range(101,111))
EPOCHS=3 if QUICK else 60
FULL=replace(major.SPECS["PDRF"],recovery_distillation=True,recovery_teacher="clean",w_recovery=.30,
             brier_regularization=True,interior_barrier=True,w_interior=.005,
             degraded_auc=True)
METHODS={
    "full":FULL,
    "minus_recovery":replace(FULL,recovery_distillation=False),
    "minus_brier":replace(FULL,brier_regularization=False),
    "minus_interior":replace(FULL,interior_barrier=False),
    "minus_auc":replace(FULL,degraded_auc=False),
}

def main():
    split=q2.prepare_grouped(base.ensure_zip(None),[list(range(i,i+4)) for i in range(0,16,4)])
    testx,testy,_=split["test"]
    xx,mm,qq,aff=major.fixed_fault(testx,"silent",0,3.,.40,70001)
    rows=[]
    for seed in SEEDS:
        train,select,calibration,cw=q2.train_views(split,seed)
        for method,spec in METHODS.items():
            model,cost=major.fit(spec,train,select,cw,seed,prior=(.45,.25,.18,.12),epochs=EPOCHS)
            temp=major.fit_temperature(model,calibration)
            p,_,_,_,infer=major.predict(model,xx,mm,qq,temp)
            for subset,take in (("all",slice(None)),("affected",aff),("unaffected",~aff)):
                for metric,value in risk.all_metrics(testy[take],p[take]).items():
                    rows.append({"seed":seed,"variant":method,"subset":subset,
                                 "metric":metric,"value":value,
                                 "parameters":cost["parameters"],
                                 "train_seconds":cost["train_seconds"],
                                 "inference_ms_per_observation":1000*infer/len(testy)})
    pd.DataFrame(rows).to_csv(OUT/"major3_loss_ablation.csv",index=False)

if __name__=="__main__":main()
