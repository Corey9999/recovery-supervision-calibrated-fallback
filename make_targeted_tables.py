"""LaTeX tables for the targeted aggregation and affected-subset remedies."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent;SRC=ROOT/"source_data";TAB=ROOT/"tables"
m=pd.read_csv(SRC/"targeted_improvement_metrics.csv")

g=(m[(m.level=="single_model")&(m.subset=="affected")&m.metric.isin(["accuracy","macro_f1","macro_auprc","macro_auroc","nll"])]
   .groupby(["method","metric"]).value.agg(["mean","std"]))
with open(TAB/"targeted_affected_table.tex","w",encoding="utf-8") as f:
  f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nMethod & Accuracy & Macro-F1 & Macro-AUPRC & Macro-AUROC & NLL\\\\\n\\midrule\n")
  for method in ["CAGF","PDRF","PDRF_AUC"]:
    z=g.loc[method];vals=[f"{z.loc[k,'mean']:.3f}$\\pm${z.loc[k,'std']:.3f}" for k in ["accuracy","macro_f1","macro_auprc","macro_auroc","nll"]]
    f.write(method.replace("_","\\_")+" & "+" & ".join(vals)+"\\\\\n")
  f.write("\\bottomrule\n\\end{tabular}\n")

e=m[(m.level=="ensemble")&(m.metric=="macro_auroc")]
with open(TAB/"targeted_ensemble_table.tex","w",encoding="utf-8") as f:
  f.write("\\begin{tabular}{llrr}\n\\toprule\nMethod & Aggregation & All & Affected\\\\\n\\midrule\n")
  labels={"equal_probability":"Equal probability","calibration_stacking":"Calibration stacking","stacking_dirichlet":"Stacking + Dirichlet"}
  for method in ["CAGF","PDRF","PDRF_AUC"]:
    for pool in labels:
      z=e[(e.method==method)&(e.pool==pool)].set_index("subset").value
      f.write(method.replace("_","\\_")+" & "+labels[pool]+f" & {z['all']:.3f} & {z['affected']:.3f}\\\\\n")
    f.write("\\addlinespace\n")
  f.write("\\bottomrule\n\\end{tabular}\n")
