"""Compare entropy-weighted fusion with matched gates and bounded weighting."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parent;SRC=ROOT/"source_data";TAB=ROOT/"tables"
base=pd.read_csv(SRC/"q1_risk_metrics.csv")
ent=pd.read_csv(SRC/"uncertainty_baseline_metrics.csv")
final="PDRF_RCIA_R30"
frame=pd.concat([base[base.method.isin(["CAGF","PDRF",final])],ent],ignore_index=True)
labels={"CAGF":"CAGF","PDRF":"PDRF",final:"RO-PDRF","ENT-PD":"ENT-PD"}
metrics=["macro_auroc","macro_auprc","nll","ece15"]
summary=frame[frame.metric.isin(metrics)].groupby(["method","subset","metric"]).value.agg(["mean","std"]).reset_index()
summary.to_csv(SRC/"uncertainty_baseline_summary.csv",index=False)
lines=[r"\begin{tabular}{lrrrrrrrr}",r"\toprule",
       "Method & All AUROC & All AUPRC & All NLL & All ECE & Aff. AUROC & Aff. AUPRC & Aff. NLL & Aff. ECE \\\\",r"\midrule"]
for method in ["CAGF","ENT-PD","PDRF",final]:
    vals=[]
    for subset in ["all","affected"]:
        for metric in metrics:
            z=summary[(summary.method==method)&(summary.subset==subset)&(summary.metric==metric)]
            vals.append(z.iloc[0]["mean"] if len(z) else np.nan)
    lines.append(labels[method]+" & "+" & ".join(f"{v:.3f}" for v in vals)+" \\\\")
lines += [r"\bottomrule",r"\end{tabular}"]
(TAB/"modern_uncertainty_baseline.tex").write_text("\n".join(lines),encoding="utf-8")

tests=[]
sel=frame[(frame.subset=="affected")&(frame.metric=="macro_auroc")]
for left,right in [(final,"ENT-PD"),(final,"CAGF"),("PDRF","ENT-PD")]:
    a=sel[sel.method==left].sort_values("seed").value.to_numpy()
    b=sel[sel.method==right].sort_values("seed").value.to_numpy()
    n=min(len(a),len(b));a,b=a[:n],b[:n]
    stat,p=wilcoxon(a,b,method="exact")
    tests.append({"contrast":f"{labels[left]} - {labels[right]}","n":n,
                  "mean_difference":float(np.mean(a-b)),"wins":int((a>b).sum()),
                  "losses":int((a<b).sum()),"p_two_sided_exact":float(p)})
pd.DataFrame(tests).to_csv(SRC/"uncertainty_baseline_paired_tests.csv",index=False)
