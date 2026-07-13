"""Reproduce the paired score distribution for one fixed representative seed."""

from pathlib import Path
import numpy as np
import pandas as pd

import run_major_revision_experiments as major
import run_q2_revision as q2
import run_uci_validation as base

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"source_data"

zip_path=base.ensure_zip(None)
groups=[list(range(i,i+4)) for i in range(0,16,4)]
split=q2.prepare_grouped(zip_path,groups)
train,select,calibration,cw=q2.train_views(split,101)
model,_=major.fit(major.SPECS["PDRF"],train,select,cw,101,prior=(.45,.25,.18,.12),epochs=60)
testx,_,_=split["test"]
rng=np.random.default_rng(71001); n,m,d=testx.shape
mask=base.random_mask(n,rng,.20); q=np.ones((n,m),np.float32)
affected=(rng.random(n)<.40)&(mask[:,0]>0)
clean=testx.copy()*mask[:,:,None]; degraded=testx.copy()
degraded[affected,0]+=rng.normal(0,3,(affected.sum(),d)).astype(np.float32)
degraded*=mask[:,:,None]
_,s0,w0,_,_=major.predict(model,clean.astype(np.float32),mask,q)
_,s1,w1,_,_=major.predict(model,degraded.astype(np.float32),mask,q)
pd.DataFrame({"score_clean":s0[affected,0],"score_degraded":s1[affected,0],
              "weight_clean":w0[affected,0],"weight_degraded":w1[affected,0]}).to_csv(
                  OUT/"representative_score_distribution.csv",index=False)
