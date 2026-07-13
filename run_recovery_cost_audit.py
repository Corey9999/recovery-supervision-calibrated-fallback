"""Three-seed matched computational audit for the recovery-oriented method."""
from dataclasses import replace
from pathlib import Path
import time
import pandas as pd

import run_major_revision_experiments as major
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"source_data"
FINAL=replace(major.SPECS["PDRF"],recovery_distillation=True,w_recovery=.30,
              brier_regularization=True,interior_barrier=True,w_interior=.005,degraded_auc=True)
METHODS={"CAGF":major.SPECS["CAGF"],"ENT-PD":major.SPECS["ENTROPY_PD"],
         "PDRF":major.SPECS["PDRF"],"RO-PDRF":FINAL}

def main():
    split=q2.prepare_grouped(base.ensure_zip(None),[list(range(i,i+4)) for i in range(0,16,4)])
    tx,_,_=split["test"];_,testx,mask,q,_=risk.paired_fault(tx,70001)
    rows=[]
    for seed in (101,102,103):
        train,select,calibration,cw=q2.train_views(split,seed)
        for method,spec in METHODS.items():
            model,cost=major.fit(spec,train,select,cw,seed,prior=(.45,.25,.18,.12),epochs=60)
            temp=major.fit_temperature(model,calibration,neutral_q=(method=="ENT-PD"))
            t0=time.perf_counter();major.predict(model,testx,mask,q,temp);infer=time.perf_counter()-t0
            rows.append({"method":method,"seed":seed,"parameters":cost["parameters"],
                         "train_seconds":cost["train_seconds"],"epochs":cost["epochs"],
                         "inference_ms_per_observation":1000*infer/len(testx)})
    pd.DataFrame(rows).to_csv(OUT/"recovery_cost_audit.csv",index=False)

if __name__=="__main__":main()
