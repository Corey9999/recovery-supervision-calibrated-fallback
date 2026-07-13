"""Generate compact LaTeX tables for the Q2-review revision."""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent; S=ROOT/'source_data'; T=ROOT/'tables'; T.mkdir(exist_ok=True)

def write(name, cols, rows, align=None):
    align=align or ('l'+'c'*(len(cols)-1))
    lines=[f'\\begin{{tabular}}{{{align}}}','\\toprule',' & '.join(cols)+' \\\\','\\midrule']
    lines += [' & '.join(map(str,r))+' \\\\' for r in rows]
    lines += ['\\bottomrule','\\end{tabular}']
    (T/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')

b=pd.read_csv(S/'q2_bootstrap_effects.csv')
d=b[(b.fault=='silent_gaussian') & b.metric.isin(['macro_auroc','accuracy','nll'])]
rows=[]
for _,r in d.iterrows():
    rows.append([r.contrast.replace('PDRF-','vs '),r.metric.replace('_',' '),f'{r.aligned_difference:.3f}',
                 f'[{r.sample_ci_low:.3f}, {r.sample_ci_high:.3f}]',
                 f'[{r.batch_stratified_ci_low:.3f}, {r.batch_stratified_ci_high:.3f}]'])
write('q2_bootstrap_table.tex',['Contrast','Metric','Difference','Sample bootstrap','Batch-stratified'],rows,'llccc')

g=pd.read_csv(S/'q2_grouping_sensitivity.csv'); rows=[]
for (group,fault),x in g.groupby(['grouping','fault']): rows.append([group.replace('_',' '),fault.replace('_',' '),f'{x.macro_auroc.mean():.3f}',f'{x.macro_auroc.std():.3f}'])
write('q2_grouping_table.tex',['Grouping','View','Mean AUROC','SD'],rows,'llcc')

s=pd.read_csv(S/'q2_sampling_unseen_sensor.csv'); rows=[]
for (sampling,m),x in s.groupby(['sampling','test_fault_modality']): rows.append([sampling.replace('_',' '),m,f'{x.macro_auroc.mean():.3f}',f'{x.macro_auroc.std():.3f}'])
write('q2_sampling_table.tex',['Training sampling','Faulted group','Mean AUROC','SD'],rows,'llcc')

a=pd.read_csv(S/'q2_target_margin_ablation.csv'); rows=[]
for (target,scale),x in a.groupby(['target','fault_scale']): rows.append([target.replace('_',' '),f'{scale:.1f}',f'{x.macro_auroc.mean():.3f}',f'{x.macro_auroc.std():.3f}'])
write('q2_target_table.tex',['Consistency/margin','Fault scale','Mean AUROC','SD'],rows,'llcc')

c=pd.read_csv(S/'score_comparability_summary.csv'); rows=[]
for _,r in c.iterrows(): rows.append([r.metric.replace('_',' '),f'{r["mean"]:.3f}',f'{r.ci95:.3f}',int(r.n)])
write('q2_score_table.tex',['Audit metric','Mean','95\\% half-width','Seeds'],rows,'lccc')

cal=pd.read_csv(S/'q2_calibration_summary.csv'); rows=[]
for method in ['EF','UF','CAGF','DWR','BCRF']:
 for fault in ['natural','silent_gaussian']:
  for calibrated in [False,True]:
   z=cal[(cal.method==method)&(cal.fault==fault)&(cal.calibrated==calibrated)].set_index('metric').value
   rows.append([method,fault.replace('_',' '),'scaled' if calibrated else 'raw',f'{z.nll:.3f}',f'{z.brier:.3f}',f'{z.ece_10:.3f}',f'{z.ece_15:.3f}',f'{z.ece_20:.3f}',f'{z.adaptive_ece_15:.3f}'])
write('q2_calibration_table.tex',['Method','View','Prob.','NLL','Brier','ECE10','ECE15','ECE20','Adaptive ECE'],rows,'lllcccccc')

th=pd.read_csv(S/'q2_boundary_thresholds.csv'); rows=[]
for threshold,x in th.groupby('threshold'): rows.append([f'{threshold:.2f}',f'{100*x.fraction.mean():.1f}\\%',f'{100*x.fraction.std():.1f}\\%'])
write('q2_threshold_table.tex',['Threshold','Mean fraction','SD'],rows,'lcc')

batch=pd.read_csv(S/'q2_batch_performance.csv'); rows=[]
for method in ['EF','CAGF','BCRF']:
 for fault in ['natural','silent_gaussian']:
  for bn in [8,9,10]:
   z=batch[(batch.method==method)&(batch.fault==fault)&(batch.batch==bn)].set_index('metric')
   rows.append([method,fault.replace('_',' '),bn,int(z.n_samples.iloc[0]),f'{z.loc["accuracy","value"]:.3f}',f'{z.loc["macro_f1","value"]:.3f}',f'{z.loc["macro_auroc","value"]:.3f}',f'{z.loc["macro_auprc","value"]:.3f}'])
write('q2_batch_table.tex',['Method','View','Batch','n','Accuracy','Macro-F1','AUROC','AUPRC'],rows,'lllccccc')

m=pd.read_csv(S/'q2_metrics_long.csv'); m=m[(m.calibrated==True)]
rows=[]
for method in ['EF','CAGF','BCRF']:
 for fault in ['natural','silent_gaussian']:
  z=m[(m.method==method)&(m.fault==fault)].groupby('metric').value.mean()
  for cls in range(1,7): rows.append([method,fault.replace('_',' '),cls,f'{z[f"class_{cls}_auroc"]:.3f}',f'{z[f"class_{cls}_auprc"]:.3f}',f'{z[f"class_{cls}_recall"]:.3f}'])
write('q2_classwise_table.tex',['Method','View','Class','AUROC','AUPRC','Recall'],rows,'lllccc')

br=pd.read_csv(S/'q2_break_even.csv'); br=br[(br.cost_scheme=='uniform')&(br.coverage==1.0)&(br.reject_cost==0.25)]
rows=[[r.comparator,f'{r.break_even_fault_prevalence:.2f}',f'{r.pdrf_cost_at_zero:.3f}',f'{r.comparator_cost_at_zero:.3f}',f'{r.pdrf_cost_at_one:.3f}',f'{r.comparator_cost_at_one:.3f}'] for _,r in br.iterrows()]
write('q2_utility_table.tex',['Comparator','Break-even','PDRF clean','Control clean','PDRF fault','Control fault'],rows,'lccccc')
print('Q2 tables written')
