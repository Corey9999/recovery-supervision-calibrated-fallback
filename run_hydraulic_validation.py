"""Independent validation on the UCI hydraulic condition-monitoring test rig.

The dataset supplies native cycle-level component-condition labels and raw time
series from 14 physical sensors. Sensor corruptions remain controlled test
interventions; component health states are not synthetically assigned.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from run_major_revision_experiments import SPECS, fit, fit_temperature, metrics, predict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"source_data"
ZIP=ROOT/"external_data"/"hydraulic_condition_monitoring.zip"
SEEDS=(101,) if os.getenv("HYDRAULIC_QUICK")=="1" else tuple(range(101,106))
EPOCHS=3 if os.getenv("HYDRAULIC_QUICK")=="1" else 80
SENSORS=["PS1","PS2","PS3","PS4","PS5","PS6","FS1","FS2",
         "TS1","TS2","TS3","TS4","EPS1","VS1"]
GROUPS={"pressure":["PS1","PS2","PS3","PS4","PS5","PS6"],
        "flow":["FS1","FS2"],"temperature":["TS1","TS2","TS3","TS4"],
        "motor_power":["EPS1"],"vibration":["VS1"]}
TASKS={"cooler":0,"valve":1,"pump":2,"accumulator":3}
METHODS=("EF_PD","UF_PD","CAGF","DWR","PDRF_NOQ")


def summarize_signal(a):
    n=a.shape[1]; t=np.linspace(-1,1,n,dtype=np.float64)
    slope=(a@t)/(t@t)
    return np.column_stack([a.mean(1),a.std(1),a.min(1),a.max(1),
                            np.quantile(a,.25,axis=1),np.quantile(a,.75,axis=1),
                            np.sqrt(np.mean(a*a,axis=1)),slope]).astype(np.float32)


def load_features():
    cache=OUT/"hydraulic_features_v2.npz"
    if cache.exists():
        z=np.load(cache); return z["features"],z["profile"]
    feats={}
    with zipfile.ZipFile(ZIP) as z:
        profile=np.loadtxt(io.BytesIO(z.read("profile.txt"))).astype(np.int64)
        for sensor in SENSORS:
            arr=np.loadtxt(io.BytesIO(z.read(sensor+".txt")),dtype=np.float64)
            feats[sensor]=summarize_signal(arr)
    # Five physically predefined sensor groups. Each flattened summary vector is
    # linearly resampled to 48 positions so paired noise never activates padding.
    x=np.zeros((len(profile),len(GROUPS),48),np.float32)
    for j,names in enumerate(GROUPS.values()):
        g=np.concatenate([feats[n] for n in names],axis=1)
        old=np.linspace(0,1,g.shape[1]); new=np.linspace(0,1,48)
        x[:,j]=np.stack([np.interp(new,old,row) for row in g]).astype(np.float32)
    np.savez_compressed(cache,features=x,profile=profile)
    return x,profile


def split_task(x,profile,column):
    stable=profile[:,4]==0; x=x[stable]; labels=profile[stable,column]
    values=np.sort(np.unique(labels)); mapping={v:i for i,v in enumerate(values)}
    y=np.array([mapping[v] for v in labels],np.int64); idx=np.arange(len(y))
    train,rest=train_test_split(idx,test_size=.40,random_state=447,stratify=y)
    select,rest=train_test_split(rest,test_size=.625,random_state=448,stratify=y[rest])
    calibration,test=train_test_split(rest,test_size=.60,random_state=449,stratify=y[rest])
    scaler=StandardScaler().fit(x[train].reshape(len(train),-1))
    z=np.clip(scaler.transform(x.reshape(len(x),-1)),-8,8).reshape(x.shape).astype(np.float32)
    def view(ids):
        q=np.ones((len(ids),5),np.float32); mask=np.ones_like(q)
        return z[ids],mask,q,y[ids]
    return view(train),view(select),view(calibration),view(test),values


def hydraulic_fault(x,modality,seed):
    rng=np.random.default_rng(seed); xx=x.copy(); n,m,d=x.shape
    mask=(rng.random((n,m))>.20).astype(np.float32)
    empty=mask.sum(1)==0; mask[empty,rng.integers(0,m,size=empty.sum())]=1
    affected=rng.random(n)<.40
    xx[affected,modality]+=rng.normal(0,3,(affected.sum(),d)).astype(np.float32)
    xx*=mask[:,:,None]
    return xx.astype(np.float32),mask,np.ones((n,m),np.float32),affected


def main():
    x,profile=load_features(); rows=[]; preds=[]; costs=[]; counts=[]
    for task,col in TASKS.items():
        train,select,calibration,test,values=split_task(x,profile,col)
        tx,tm,tq,ty=train; testx,testm,testq,testy=test
        cw=len(ty)/(len(values)*np.bincount(ty,minlength=len(values)))
        for split_name,data in (("train",train),("selection",select),("calibration",calibration),("test",test)):
            for j,v in enumerate(values): counts.append({"task":task,"split":split_name,"class_value":int(v),"n":int((data[3]==j).sum())})
        for seed in SEEDS:
            for name in METHODS:
                model,cost=fit(SPECS[name],train,select,cw,seed,prior=tuple([.2]*5),epochs=EPOCHS)
                temp=fit_temperature(model,calibration,neutral_q=(name=="PDRF_NOQ"))
                costs.append({"dataset":"hydraulic","task":task,"seed":seed,"method":name,**cost})
                for fault in ("natural","silent_pressure","silent_vibration"):
                    if fault=="natural": xx,mm,qq,aff=testx,testm,testq,np.zeros(len(testy),bool)
                    else:
                        modality=0 if fault=="silent_pressure" else 4
                        xx,mm,qq,aff=hydraulic_fault(testx,modality,81000+modality)
                    p,_,_,_,_=predict(model,xx,mm,qq,temp)
                    for subset,sel in (("all",np.ones(len(testy),bool)),("affected",aff),("unaffected",~aff)):
                        if not sel.any(): continue
                        for k,v in metrics(testy[sel],p[sel]).items():
                            rows.append({"task":task,"seed":seed,"method":name,"fault":fault,"subset":subset,"metric":k,"value":v})
                    for i in range(len(testy)):
                        row={"task":task,"seed":seed,"method":name,"fault":fault,"sample":i,
                             "y":int(testy[i]),"class_value":int(values[testy[i]]),"affected":bool(aff[i]),"pred":int(p[i].argmax())}
                        row.update({f"p{j+1}":float(p[i,j]) for j in range(len(values))}); preds.append(row)
    pd.DataFrame(rows).to_csv(OUT/"hydraulic_validation_metrics.csv",index=False)
    pd.DataFrame(preds).to_csv(OUT/"hydraulic_seed_predictions.csv",index=False)
    pd.DataFrame(costs).to_csv(OUT/"hydraulic_compute_costs.csv",index=False)
    pd.DataFrame(counts).to_csv(OUT/"hydraulic_split_counts.csv",index=False)
    (OUT/"hydraulic_design.json").write_text(json.dumps({"doi":"10.24432/C5CW21",
      "stable_cycles_only":True,"n_stable":int((profile[:,4]==0).sum()),"sensor_groups":GROUPS,
      "features_per_sensor":["mean","sd","min","max","q25","q75","rms","linear_slope"],
      "split":"stratified 60% train, 15% selection, 10% calibration, 15% test",
      "seeds":SEEDS,"methods":METHODS},indent=2),encoding="utf-8")


if __name__=="__main__": main()
