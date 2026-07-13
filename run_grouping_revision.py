"""Grouping-distribution and 16-independent-sensor sensitivity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_major_revision_experiments import SPECS,fit,fit_temperature,metrics,predict
import run_q2_revision as q2
import run_uci_validation as base

OUT=Path(__file__).resolve().parent/'source_data';SEEDS=range(101,106)

def mask(n,m,p,seed):
 r=np.random.default_rng(seed);a=(r.random((n,m))>p).astype(np.float32);e=a.sum(1)==0;a[e,r.integers(0,m,e.sum())]=1;return a
def views(split,seed):
 out=[]
 for k,s in enumerate(['train','select','calibration']):
  x,y,_=split[s];m=mask(len(x),x.shape[1],.10,seed+1000*k);q=np.ones_like(m);out.append((x*m[:,:,None],m,q,y))
 y=split['train'][1];cw=len(y)/(6*np.bincount(y,minlength=6));return *out,cw
def fault(x,seed):
 r=np.random.default_rng(seed);n,m,d=x.shape;mm=mask(n,m,.20,seed);a=r.random(n)<.40;xx=x.copy();xx[a,0]+=r.normal(0,3,(a.sum(),d));xx*=mm[:,:,None];return xx.astype(np.float32),mm,np.ones((n,m),np.float32),a

def main():
 path=base.ensure_zip(None);groupings={'consecutive':[list(range(i,i+4)) for i in range(0,16,4)],
 'interleaved':[[0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15]],
 'independent_16':[[i] for i in range(16)]}
 for k in range(10):
  p=np.random.default_rng(9200+k).permutation(16);groupings[f'random_{k+1:02d}']=[p[i:i+4].tolist() for i in range(0,16,4)]
 rows=[]
 for name,groups in groupings.items():
  split=q2.prepare_grouped(path,groups);x,y,_=split['test']
  xx,mm,qq,aff=fault(x,93000+len(groups))
  for seed in SEEDS:
   train,select,cal,cw=views(split,seed)
   for method in ['CAGF','PDRF']:
    model,_=fit(SPECS[method],train,select,cw,seed,prior=tuple([1/len(groups)]*len(groups)),epochs=60)
    temp=fit_temperature(model,cal);p,_,_,_,_=predict(model,xx,mm,qq,temp)
    z=metrics(y,p)
    rows.append({'grouping':name,'groups':json.dumps(groups),'n_groups':len(groups),'seed':seed,'method':method,'macro_auroc':z['macro_auroc'],'accuracy':z['accuracy']})
 pd.DataFrame(rows).to_csv(OUT/'major_grouping_distribution.csv',index=False)
if __name__=='__main__':main()
