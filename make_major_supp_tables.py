from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,average_precision_score
from sklearn.preprocessing import label_binarize

ROOT=Path(__file__).resolve().parent;O=ROOT/"source_data";T=ROOT/"tables"
def esc(x):return str(x).replace('_','\\_')
def write(name,header,align,rows):
 with open(T/name,'w',encoding='utf-8') as f:
  f.write(f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"+" & ".join(header)+"\\\\\n\\midrule\n")
  for r in rows:f.write(" & ".join(map(str,r))+"\\\\\n")
  f.write("\\bottomrule\n\\end{tabular}\n")

# Bootstrap
b=pd.read_csv(O/'major_bootstrap_5000.csv');rows=[]
for _,r in b.iterrows():rows.append([esc(r.metric),esc(r.design),f"{r.aligned_difference:.4f}",f"[{r.ci_low:.4f}, {r.ci_high:.4f}]"])
write('major_bootstrap_table.tex',['Metric','Resampling','Aligned difference','95\\% interval'],'llrr',rows)

# Seed effects
s=pd.read_csv(O/'major_paired_tests.csv');g=s[s.dataset=='gas_array'];rows=[]
for _,r in g.iterrows():rows.append([esc(r.metric),f"{r.mean_difference:.4f}",f"{int(r.wins)}/10",f"{r.exact_p:.4f}"])
write('major_seed_effects_table.tex',['Metric','Mean paired difference','PDRF wins','Exact Wilcoxon $P$'],'lrrr',rows)

# Compute
c=pd.read_csv(O/'major_compute_costs.csv');z=c.groupby('method',as_index=False).agg(parameters=('parameters','first'),train=('train_seconds','mean'),infer=('inference_ms_silent','mean'))
order=['EF','EF_PD','UF','UF_PD','Q_ONLY','CAGF','DWR','PDRF_NOQ','PDRF'];z=z.set_index('method').loc[order].reset_index();rows=[]
for _,r in z.iterrows():rows.append([esc(r.method),f"{int(r.parameters):,}",f"{r.train:.2f}",f"{r.infer:.6f}"])
write('major_compute_table.tex',['Method','Parameters','Training (s)','Inference (ms/observation)'],'lrrr',rows)

# Quality controls
q=pd.read_csv(O/'major_quality_controls.csv');q=q[q.metric=='macro_auroc'].groupby(['training','test_q'],as_index=False).value.agg(['mean','std']).reset_index();rows=[]
for _,r in q.iterrows():rows.append([esc(r.training),esc(r.test_q),f"{r['mean']:.3f}",f"{r['std']:.3f}"])
write('major_quality_table.tex',['Training','Test metadata','Macro-AUROC','Seed SD'],'llrr',rows)

# Leave group
l=pd.read_csv(O/'major_leave_group_out.csv');l=l.groupby(['held_out_group','method'],as_index=False).agg(auroc=('macro_auroc','mean'),acc=('accuracy','mean'));rows=[]
for _,r in l.iterrows():rows.append([int(r.held_out_group),esc(r.method),f"{r.auroc:.3f}",f"{r.acc:.3f}"])
write('major_logo_table.tex',['Excluded/test group','Method','Macro-AUROC','Accuracy'],'rlrr',rows)

# Hyperparameters
h=pd.read_csv(O/'major_hyperparameter_sensitivity.csv');h=h.groupby('setting',as_index=False).agg(auroc=('macro_auroc','mean'),acc=('accuracy','mean'),sat=('score_saturation','mean'));rows=[]
for _,r in h.iterrows():rows.append([esc(r.setting),f"{r.auroc:.3f}",f"{r.acc:.3f}",f"{r.sat:.3f}"])
write('major_hyperparameter_table.tex',['Setting','Macro-AUROC','Accuracy','Saturation'],'lrrr',rows)

# Fault subsets
m=pd.read_csv(O/'major_ablation_metrics.csv');m=m[(m.fault=='silent')&m.method.isin(['CAGF','PDRF'])&m.metric.isin(['accuracy','macro_f1','macro_auprc','macro_auroc'])]
m=m.groupby(['method','subset','metric'],as_index=False).value.mean();rows=[]
for method in ['CAGF','PDRF']:
 for subset in ['affected','unaffected','all']:
  x=m[(m.method==method)&(m.subset==subset)].set_index('metric').value
  rows.append([method,subset]+[f"{x[k]:.3f}" for k in ['accuracy','macro_f1','macro_auprc','macro_auroc']])
write('major_subset_table.tex',['Method','Subset','Accuracy','Macro-F1','Macro-AUPRC','Macro-AUROC'],'llrrrr',rows)

# Per-batch and class results from seed-level probabilities.
p=pd.read_csv(O/'major_seed_predictions.csv');p=p[(p.fault=='silent')&p.method.isin(['CAGF','PDRF'])];pc=[x for x in p if x.startswith('p') and x[1:].isdigit()]
batchrows=[];classrows=[]
for method in ['CAGF','PDRF']:
 for seed in sorted(p.seed.unique()):
  z=p[(p.method==method)&(p.seed==seed)]
  for batch in sorted(z.batch.unique()):
   zz=z[z.batch==batch];y=zz.y.to_numpy();pr=zz[pc].to_numpy();pred=pr.argmax(1)
   yb=label_binarize(y,classes=np.arange(6))
   batchrows.append([method,batch,accuracy_score(y,pred),f1_score(y,pred,average='macro'),roc_auc_score(yb,pr,average='macro',multi_class='ovr')])
  y=z.y.to_numpy();pr=z[pc].to_numpy();pred=pr.argmax(1);yb=label_binarize(y,classes=np.arange(6))
  for cls in range(6):
   classrows.append([method,cls+1,precision_score(y==cls,pred==cls,zero_division=0),recall_score(y==cls,pred==cls,zero_division=0),f1_score(y==cls,pred==cls,zero_division=0),roc_auc_score(yb[:,cls],pr[:,cls]),average_precision_score(yb[:,cls],pr[:,cls])])
br=pd.DataFrame(batchrows,columns=['method','batch','accuracy','f1','auroc']).groupby(['method','batch'],as_index=False).mean();rows=[]
for _,r in br.iterrows():rows.append([r.method,int(r.batch),f"{r.accuracy:.3f}",f"{r.f1:.3f}",f"{r.auroc:.3f}"])
write('major_batch_table.tex',['Method','Batch','Accuracy','Macro-F1','Macro-AUROC'],'lrrrr',rows)
cr=pd.DataFrame(classrows,columns=['method','class','precision','recall','f1','auroc','auprc']).groupby(['method','class'],as_index=False).mean();rows=[]
for _,r in cr.iterrows():rows.append([r.method,int(r['class'])]+[f"{r[k]:.3f}" for k in ['precision','recall','f1','auroc','auprc']])
write('major_class_table.tex',['Method','Class','Precision','Recall','F1','AUROC','AUPRC'],'lrrrrrr',rows)

# Hydraulic split counts and paired task effects
hc=pd.read_csv(O/'hydraulic_split_counts.csv');rows=[]
for _,r in hc.iterrows():rows.append([r.task,r.split,int(r.class_value),int(r.n)])
write('hydraulic_counts_table.tex',['Task','Split','Condition value','$n$'],'llrr',rows)
for task in sorted(hc.task.unique()):
 rows=[]
 for _,r in hc[hc.task==task].iterrows():rows.append([r.split,int(r.class_value),int(r.n)])
 write(f'hydraulic_counts_{task}.tex',['Split','Condition value','$n$'],'lrr',rows)
hp=s[(s.dataset=='hydraulic')&(s.fault.isin(['silent_pressure','silent_vibration']))];rows=[]
for _,r in hp.iterrows():rows.append([r.task,esc(r.fault),f"{r.mean_difference:.3f}",f"{int(r.wins)}/5",f"{r.exact_p:.4f}"])
write('hydraulic_effects_table.tex',['Task','Fault','Mean AUROC difference','PDRF wins','Exact $P$'],'llrrr',rows)

# Grouping distribution and paired ranking
gd=pd.read_csv(O/'major_grouping_distribution.csv')
sm=gd.groupby(['grouping','method'],as_index=False).agg(mean=('macro_auroc','mean'),sd=('macro_auroc','std'))
pair=gd.pivot(index=['grouping','seed'],columns='method',values='macro_auroc').reset_index();pair['diff']=pair.PDRF-pair.CAGF
ps=pair.groupby('grouping',as_index=False).agg(diff=('diff','mean'),wins=('diff',lambda x:int((x>0).sum())))
rows=[]
for grouping in gd.grouping.drop_duplicates():
 for method in ['CAGF','PDRF']:
  r=sm[(sm.grouping==grouping)&(sm.method==method)].iloc[0];p=ps[ps.grouping==grouping].iloc[0]
  rows.append([esc(grouping),method,f"{r['mean']:.3f}",f"{r['sd']:.3f}",f"{p['diff']:.3f}" if method=='PDRF' else '',f"{int(p['wins'])}/5" if method=='PDRF' else ''])
write('major_grouping_distribution_table.tex',['Grouping','Method','Macro-AUROC','Seed SD','PDRF--CAGF','PDRF wins'],'llrrrr',rows)
print('Supplementary tables written')
