"""Q1-oriented recovery/calibration extensions and mechanism outputs.

The final extension was fixed before its formal ten-seed run. It combines
best-view recovery distillation, clean/degraded Brier proper-scoring terms,
a weak paired interior barrier, and degraded-view pairwise-AUC supervision.
Hard-group/CVaR variants are retained as development ablations, not as the
reported final extension.
"""

from __future__ import annotations
import json, os
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

import run_major_revision_experiments as major
import run_q2_revision as q2
import run_uci_validation as base

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"source_data"
SEEDS=tuple(range(101,104)) if os.getenv("Q1_QUICK")=="1" else tuple(range(101,111))
BASE=major.SPECS["PDRF"]
SPECS={
 "CAGF":major.SPECS["CAGF"],
 "PDRF":BASE,
 "PDRF_INT":replace(BASE,interior_barrier=True,boundary_penalty=0.0),
 "PDRF_REC":replace(BASE,recovery_distillation=True),
 "PDRF_RISK":replace(BASE,hard_group_sampling=True,degraded_cvar=True),
 "PDRF_CAL":replace(BASE,brier_regularization=True),
 "PDRF_RC":replace(BASE,recovery_distillation=True,brier_regularization=True),
 "PDRF_RCI":replace(BASE,recovery_distillation=True,brier_regularization=True,
                    interior_barrier=True,w_interior=0.005),
 "PDRF_RCIA":replace(BASE,recovery_distillation=True,brier_regularization=True,
                     interior_barrier=True,w_interior=0.005,degraded_auc=True),
 "PDRF_RCI_R30":replace(BASE,recovery_distillation=True,w_recovery=0.30,
                        brier_regularization=True,interior_barrier=True,w_interior=0.005),
 "PDRF_RCIA_R30":replace(BASE,recovery_distillation=True,w_recovery=0.30,
                         brier_regularization=True,interior_barrier=True,w_interior=0.005,
                         degraded_auc=True),
 "PDRF_RS":replace(BASE,interior_barrier=True,boundary_penalty=0.0,
                    recovery_distillation=True,hard_group_sampling=True,
                    degraded_cvar=True,brier_regularization=True),
}
if os.getenv("Q1_METHODS"):
 names=[x.strip() for x in os.getenv("Q1_METHODS","").split(",") if x.strip()]
 SPECS={k:SPECS[k] for k in names}

def paired_fault(x,seed,modality=0,scale=3.0,prevalence=.40):
 rng=np.random.default_rng(seed);n,m,d=x.shape
 mask=base.random_mask(n,rng,.20);affected=rng.random(n)<prevalence
 clean=x.copy()*mask[:,:,None];fault=x.copy()
 fault[affected,modality]+=rng.normal(0,scale,(affected.sum(),d)).astype(np.float32)
 fault*=mask[:,:,None];q=np.ones((n,m),np.float32)
 return clean.astype(np.float32),fault.astype(np.float32),mask,q,affected

def ece(y,p,bins=15):
 conf=p.max(1);pred=p.argmax(1);edges=np.linspace(0,1,bins+1);v=0.0
 for lo,hi in zip(edges[:-1],edges[1:]):
  take=(conf>lo)&(conf<=hi) if hi<1 else (conf>lo)&(conf<=hi)
  if take.any():v+=take.mean()*abs((pred[take]==y[take]).mean()-conf[take].mean())
 return float(v)

def all_metrics(y,p):
 p=np.clip(p,1e-12,None);p/=p.sum(1,keepdims=True);yb=label_binarize(y,classes=np.arange(p.shape[1]))
 if p.shape[1]==2 and yb.shape[1]==1: yb=np.column_stack([1-yb[:,0],yb[:,0]])
 return {"accuracy":accuracy_score(y,p.argmax(1)),"macro_f1":f1_score(y,p.argmax(1),average="macro"),
  "macro_auroc":roc_auc_score(yb,p,average="macro",multi_class="ovr"),
  "macro_auprc":average_precision_score(yb,p,average="macro"),"nll":log_loss(y,p),
  "brier":np.square(p-yb).sum(1).mean(),"ece15":ece(y,p)}

def main():
 zip_path=base.ensure_zip(None);groups=[list(range(i,i+4)) for i in range(0,16,4)]
 split=q2.prepare_grouped(zip_path,groups);cx,cy,_=split["calibration"];tx,ty,tbatch=split["test"]
 cclean,cfault,cm,cq,caff=paired_fault(cx,76001);tclean,tfault,tm,tq,aff=paired_fault(tx,70001)
 metric_rows=[];pred_rows=[];mech_rows=[];cal_store={};test_store={};cal_nll={}
 for method,spec in SPECS.items():
  cal_members=[];test_members=[]
  for seed in SEEDS:
   train,select,calibration,cw=q2.train_views(split,seed)
   model,_=major.fit(spec,train,select,cw,seed,prior=(.45,.25,.18,.12),epochs=60)
   temp=major.fit_temperature(model,calibration)
   pc,_,_,_,_=major.predict(model,cfault,cm,cq,temp)
   p0,s0,w0,_,_=major.predict(model,tclean,tm,tq,temp)
   p1,s1,w1,_,_=major.predict(model,tfault,tm,tq,temp)
   cal_members.append(pc);test_members.append(p1);cal_nll[(method,seed)]=log_loss(cy,pc)
   for subset,take in [("all",np.ones(len(ty),bool)),("affected",aff),("unaffected",~aff)]:
    for metric,value in all_metrics(ty[take],p1[take]).items():
     metric_rows.append({"method":method,"seed":seed,"subset":subset,"metric":metric,"value":value})
   if spec.architecture=="bounded":
    true0=p0[np.arange(len(ty)),ty];true1=p1[np.arange(len(ty)),ty]
    for i in np.flatnonzero(aff):
     mech_rows.append({"method":method,"seed":seed,"sample":int(i),"y":int(ty[i]),
      "clean_correct":bool(p0[i].argmax()==ty[i]),"fault_correct":bool(p1[i].argmax()==ty[i]),
      "true_probability_change":float(true1[i]-true0[i]),
      "score_change":float(s1[i,0]-s0[i,0]),"weight_reduction":float(w0[i,0]-w1[i,0]),
      "fault_score":float(s1[i,0]),"fault_weight":float(w1[i,0])})
   for i in range(len(ty)):
    row={"method":method,"seed":seed,"sample":i,"batch":int(tbatch[i]),"y":int(ty[i]),"affected":bool(aff[i])}
    row.update({f"p{j+1}":float(p1[i,j]) for j in range(p1.shape[1])});pred_rows.append(row)
  cal_store[method]=np.stack(cal_members);test_store[method]=np.stack(test_members)
 pd.DataFrame(metric_rows).to_csv(OUT/"q1_risk_metrics.csv",index=False)
 pd.DataFrame(pred_rows).to_csv(OUT/"q1_risk_predictions.csv",index=False)
 pd.DataFrame(mech_rows).to_csv(OUT/"q1_mechanism_samples.csv",index=False)

 ens_rows=[];ens_pred=[]
 for method,arr in test_store.items():
  cal=cal_store[method];best_i=int(np.argmin([cal_nll[(method,s)] for s in SEEDS]))
  pools={"probability_mean":arr.mean(0),
         "log_probability_mean":np.exp(np.log(np.clip(arr,1e-12,1)).mean(0)),
         "majority_vote":np.eye(arr.shape[2])[arr.argmax(2)].mean(0),
         "validation_selected":arr[best_i]}
  for pool,p in pools.items():
   p=np.clip(p,1e-12,None);p/=p.sum(1,keepdims=True)
   for subset,take in [("all",np.ones(len(ty),bool)),("affected",aff),("unaffected",~aff)]:
    for metric,value in all_metrics(ty[take],p[take]).items():ens_rows.append({"method":method,"pool":pool,"subset":subset,"metric":metric,"value":value})
   for i in range(len(ty)):
    row={"method":method,"pool":pool,"sample":i,"batch":int(tbatch[i]),"y":int(ty[i]),"affected":bool(aff[i])}
    row.update({f"p{j+1}":float(p[i,j]) for j in range(p.shape[1])});ens_pred.append(row)
 pd.DataFrame(ens_rows).to_csv(OUT/"q1_ensemble_metrics.csv",index=False)
 pd.DataFrame(ens_pred).to_csv(OUT/"q1_ensemble_predictions.csv",index=False)
 (OUT/"q1_risk_design.json").write_text(json.dumps({"seeds":SEEDS,"methods":list(SPECS),
  "fixed_test_fault_seed":70001,"fixed_calibration_fault_seed":76001,
  "full_method":"PDRF_RCIA_R30","selection":"PDRF-RCIA-R30 was fixed after staged three-seed development audits and before its formal ten-seed run"},indent=2),encoding="utf-8")

if __name__=="__main__":main()
