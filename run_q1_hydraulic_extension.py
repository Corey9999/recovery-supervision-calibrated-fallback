"""Hydraulic transfer of PDRF-RCI and a no-interpolation feature baseline."""
from dataclasses import replace
from pathlib import Path
import io, zipfile
import numpy as np
import pandas as pd

import run_hydraulic_validation as hyd
import run_major_revision_experiments as major

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"source_data";SEEDS=tuple(range(101,106))
RCIA_R30_NOQ=replace(major.SPECS["PDRF"],use_quality=False,train_quality=False,
 recovery_distillation=True,w_recovery=.30,brier_regularization=True,
 interior_barrier=True,w_interior=.005,degraded_auc=True)
METHODS={"CAGF":major.SPECS["CAGF"],"PDRF_NOQ":major.SPECS["PDRF_NOQ"],
         "PDRF_RCIA_R30_NOQ":RCIA_R30_NOQ}

def padded_features():
 feats={}
 with zipfile.ZipFile(hyd.ZIP) as z:
  profile=np.loadtxt(io.BytesIO(z.read("profile.txt"))).astype(np.int64)
  for sensor in hyd.SENSORS:
   arr=np.loadtxt(io.BytesIO(z.read(sensor+".txt")),dtype=np.float64);feats[sensor]=hyd.summarize_signal(arr)
 x=np.zeros((len(profile),len(hyd.GROUPS),48),np.float32)
 for j,names in enumerate(hyd.GROUPS.values()):
  g=np.concatenate([feats[n] for n in names],axis=1);x[:,j,:g.shape[1]]=g
 return x,profile

rows=[]
for representation,loader in [("linear_resampling",hyd.load_features),("zero_padding",padded_features)]:
 x,profile=loader()
 for task,col in hyd.TASKS.items():
  train,select,calibration,test,values=hyd.split_task(x,profile,col);testx,_,_,testy=test;ty=train[3]
  cw=len(ty)/(len(values)*np.bincount(ty,minlength=len(values)))
  for seed in SEEDS:
   for method,spec in METHODS.items():
    model,_=major.fit(spec,train,select,cw,seed,prior=tuple([.2]*5),epochs=80)
    temp=major.fit_temperature(model,calibration,neutral_q=(method!="CAGF"))
    for fault,modality in [("pressure",0),("vibration",4)]:
     xx,mm,qq,aff=hyd.hydraulic_fault(testx,modality,81000+modality)
     p,_,_,_,_=major.predict(model,xx,mm,qq,temp)
     for subset,take in [("all",np.ones(len(testy),bool)),("affected",aff),("unaffected",~aff)]:
      for metric,value in major.metrics(testy[take],p[take]).items():
       if metric in ["accuracy","macro_f1","macro_auroc","macro_auprc","nll"]:
        rows.append({"representation":representation,"task":task,"seed":seed,"method":method,
                     "fault":fault,"subset":subset,"metric":metric,"value":value})
pd.DataFrame(rows).to_csv(OUT/"q1_hydraulic_extension.csv",index=False)
