"""Corrected clean-distillation controls for early, uniform and ModDrop fusion."""
from pathlib import Path
import pandas as pd

import run_major5_consistency_controls as m5
import run_major_revision_experiments as major
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"source_data"
METHODS={k:m5.CONTROL_METHODS[k] for k in ("EF-PD+CD","UF-PD+CD","ModDrop-SD")}


def main():
    split=q2.prepare_grouped(base.ensure_zip(None),[list(range(i,i+4)) for i in range(0,16,4)])
    testx,testy,batches=split["test"]
    xx,mm,qq,affected=major.fixed_fault(testx,"silent",0,3.,.40,70001)
    rows=[];costs=[];traces=[]
    for seed in range(101,111):
        train,select,calibration,cw=q2.train_views(split,seed)
        for method,spec in METHODS.items():
            trace=[]
            model,cost=major.fit(spec,train,select,cw,seed,prior=(.45,.25,.18,.12),epochs=60,trace=trace)
            temp=major.fit_temperature(model,calibration)
            p,_,_,_,infer=major.predict(model,xx,mm,qq,temp)
            rows.extend(m5.metric_rows(seed,method,70001,batches,testy,p,affected))
            costs.append({"seed":seed,"method":method,**cost,
                          "inference_ms_per_observation":1000*infer/len(testy)})
            traces.extend({"method":method,**x} for x in trace)
    pd.DataFrame(rows).to_csv(OUT/"major5_basic_distillation_metrics.csv",index=False)
    pd.DataFrame(costs).to_csv(OUT/"major5_basic_distillation_costs.csv",index=False)
    pd.DataFrame(traces).to_csv(OUT/"major5_basic_distillation_stability.csv",index=False)


if __name__=="__main__":main()
