"""Python-only tables and submission figures for the second major revision.

Figure conclusion: PDRF improves the typical single-model decision metrics and
transfers to a physically controlled hydraulic test rig, but probability-ensemble
and calibration results expose a distinct aggregation limitation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"source_data"; TAB=ROOT/"tables"; FIG=ROOT/"figures"
TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.spines.top":False,
 "axes.spines.right":False,"axes.linewidth":.8,"legend.frameon":False})
COL={"EF_PD":"#9b9b9b","UF_PD":"#b8b8b8","CAGF":"#6597c5","DWR":"#9273b5",
     "PDRF":"#459b78","PDRF_NOQ":"#2f7f62"}

def mean(frame,keys): return frame.groupby(keys,as_index=False).value.agg(["mean","std"]).reset_index()
def tex_escape(s): return str(s).replace("_","\\_")

met=pd.read_csv(OUT/"major_ablation_metrics.csv")
core=["EF_PD","UF_PD","CAGF","DWR","PDRF_NOQ","PDRF"]
sel=met[(met.subset=="all")&met.method.isin(core)&met.metric.isin(["accuracy","macro_f1","macro_auprc","macro_auroc","nll"])]
s=mean(sel,["fault","method","metric"])
with open(TAB/"major_gas_results.tex","w",encoding="utf-8") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nFault & Method & Accuracy & Macro-F1 & Macro-AUPRC & Macro-AUROC & NLL\\\\\n\\midrule\n")
    for fault in ("natural","silent"):
        for method in core:
            g=s[(s.fault==fault)&(s.method==method)].set_index("metric")
            vals=[g.loc[k,"mean"] for k in ("accuracy","macro_f1","macro_auprc","macro_auroc","nll")]
            f.write(f"{fault.replace('_',' ')} & {tex_escape(method)} & "+" & ".join(f"{v:.3f}" for v in vals)+"\\\\\n")
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

abl=["PD_ONLY","PD_MONO","PD_RANK","NO_LSCORE","LSCORE_NO_LRANK","PDRF"]
a=mean(met[(met.fault=="silent")&(met.subset=="all")&met.method.isin(abl)&
           met.metric.isin(["accuracy","macro_auprc","macro_auroc","nll"])],["method","metric"])
with open(TAB/"major_ablation_table.tex","w",encoding="utf-8") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\nVariant & Accuracy & Macro-AUPRC & Macro-AUROC & NLL\\\\\n\\midrule\n")
    for method in abl:
        g=a[a.method==method].set_index("metric"); vals=[g.loc[k,"mean"] for k in ("accuracy","macro_auprc","macro_auroc","nll")]
        f.write(f"{tex_escape(method)} & "+" & ".join(f"{v:.3f}" for v in vals)+"\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

hyd=pd.read_csv(OUT/"hydraulic_validation_summary.csv")
h=hyd[(hyd.subset if "subset" in hyd else True)] if False else hyd
with open(TAB/"hydraulic_validation_table.tex","w",encoding="utf-8") as f:
    f.write("\\begin{tabular}{llrrr}\n\\toprule\nTask & Method & Natural & Pressure fault & Vibration fault\\\\\n\\midrule\n")
    for task in ("cooler","valve","pump","accumulator"):
        for method in ("CAGF","DWR","PDRF_NOQ"):
            g=hyd[(hyd.task==task)&(hyd.method==method)&(hyd.metric=="macro_auroc")].set_index("fault")
            vals=[g.loc[k,"mean"] for k in ("natural","silent_pressure","silent_vibration")]
            f.write(f"{task} & {tex_escape(method)} & "+" & ".join(f"{v:.3f}" for v in vals)+"\\\\\n")
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# Six-panel quantitative grid.
fig,axs=plt.subplots(2,3,figsize=(12,7.1)); axs=axs.ravel()

# a: per-seed paired effects
seed=pd.read_csv(OUT/"major_seed_paired_effects.csv")
order=["accuracy","macro_f1","macro_auprc","macro_auroc"]
for i,k in enumerate(order):
    v=seed[seed.metric==k].difference.to_numpy()
    x=np.full(len(v),i)+np.linspace(-.12,.12,len(v)); axs[0].scatter(x,v,s=18,color="#459b78",alpha=.8)
    axs[0].plot([i-.18,i+.18],[v.mean(),v.mean()],color="black",lw=1.5)
axs[0].axhline(0,color="#555",ls="--",lw=1); axs[0].set_xticks(range(4),["Accuracy","Macro-F1","Macro-AUPRC","Macro-AUROC"],rotation=20,ha="right")
axs[0].set_ylabel("Paired PDRF - CAGF"); axs[0].set_title("Single-model effects across 10 seeds")

# b: ablation
aa=a[a.metric=="macro_auroc"].set_index("method").loc[abl]
axs[1].barh(range(len(abl)),aa["mean"],xerr=aa["std"],color=["#b8b8b8"]*5+["#459b78"],height=.68,capsize=2)
axs[1].set_yticks(range(len(abl)),[x.replace("_"," ") for x in abl]); axs[1].invert_yaxis(); axs[1].set_xlim(.79,.85)
axs[1].set_xlabel("Macro-AUROC"); axs[1].set_title("Mechanism ablation, silent fault")

# c: hydraulic transfer
tasks=["cooler","valve","pump","accumulator"]; xx=np.arange(4); width=.35
for off,method in ((-.5,"CAGF"),(.5,"PDRF_NOQ")):
    vals=[]; errs=[]
    for task in tasks:
        row=hyd[(hyd.task==task)&(hyd.method==method)&(hyd.fault=="silent_pressure")&(hyd.metric=="macro_auroc")].iloc[0]
        vals.append(row["mean"]);errs.append(row["std"])
    axs[2].bar(xx+off*width,vals,width,yerr=errs,color=COL[method],label=method.replace("_NOQ"," (no q)"),capsize=2)
axs[2].set_xticks(xx,["Cooler","Valve","Pump","Accumulator"],rotation=18,ha="right");axs[2].set_ylim(.68,1.02)
axs[2].set_ylabel("Macro-AUROC");axs[2].set_title("Independent hydraulic test rig");axs[2].legend(loc="lower left")

# d: quality controls
q=pd.read_csv(OUT/"major_quality_controls.csv"); q=mean(q[q.metric=="macro_auroc"],["training","test_q"])
reg=["silent","reported","shuffled","misleading"]
for training,color in (("PDRF","#459b78"),("PDRF_NOQ","#2f7f62")):
    g=q[q.training==training].set_index("test_q").loc[reg]
    axs[3].plot(range(4),g["mean"],marker="o",color=color,label=training.replace("_NOQ"," (no q)"))
axs[3].set_xticks(range(4),["No q update","Reported","Shuffled","Misleading"],rotation=20,ha="right");axs[3].set_ylim(.74,.87)
axs[3].set_ylabel("Macro-AUROC");axs[3].set_title("Diagnostic-quality controls");axs[3].legend()

# e: monotonicity and saturation audit
w=pd.read_csv(OUT/"major_weight_audit.csv")
labels=["Raw score\nviolation","Unnormalized\nviolation","Normalized\nviolation","Score\nsaturation"]
cols=["raw_score_violation","unnormalized_weight_violation","normalized_weight_violation","score_saturation"]
vals=[w[c].mean() for c in cols]; errs=[w[c].std() for c in cols]
axs[4].bar(range(4),vals,yerr=errs,color=["#6597c5"]*3+["#d39b5f"],capsize=2)
axs[4].set_xticks(range(4),labels);axs[4].set_ylim(0,.75);axs[4].set_ylabel("Fraction");axs[4].set_title("Directional audit and boundary occupancy")

# f: fixed-ensemble bootstrap
b=pd.read_csv(OUT/"major_bootstrap_5000.csv");b=b[b.design=="sample"].set_index("metric")
bo=["accuracy","macro_f1","macro_auprc","macro_auroc"]; y=np.arange(len(bo))
est=b.loc[bo,"aligned_difference"].to_numpy();lo=b.loc[bo,"ci_low"].to_numpy();hi=b.loc[bo,"ci_high"].to_numpy()
axs[5].errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt="o",color="#9273b5",capsize=2)
axs[5].axvline(0,color="#555",ls="--",lw=1);axs[5].set_yticks(y,["Accuracy","Macro-F1","Macro-AUPRC","Macro-AUROC"]);axs[5].invert_yaxis();axs[5].set_xlim(-.022,.018)
axs[5].set_xlabel("Aligned PDRF - CAGF (positive favours PDRF)");axs[5].set_title("Fixed-ensemble bootstrap, 5,000 resamples")

for i,ax in enumerate(axs): ax.text(-.16,1.06,chr(97+i),transform=ax.transAxes,fontweight="bold",fontsize=12)
fig.tight_layout(w_pad=2.0,h_pad=2.0)
for ext,kw in (("svg",{}),("pdf",{}),("png",{"dpi":300}),("tiff",{"dpi":600})):
    fig.savefig(FIG/f"figure10_major_revision.{ext}",bbox_inches="tight",**kw)
plt.close(fig)
print("Major-revision tables and figure written")
