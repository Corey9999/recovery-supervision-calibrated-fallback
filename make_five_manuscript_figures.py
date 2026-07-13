"""Create the five Python-only manuscript figures requested in the figure plan.

Figure contracts
----------------
1. Conclusion: bounded weighting and paired degradation form one auditable path.
2. Conclusion: temporal/stratified splits and fixed corruptions prevent leakage.
3. Conclusion: single-model gains transfer from the chemical array to four
   hydraulic tasks under two controlled physical-group corruptions.
4. Conclusion: the gain is not explained by degradation exposure or q alone.
5. Conclusion: directional behavior coexists with saturation and aggregation
   boundaries; targeted remedies are evaluated without hiding negative results.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

ROOT=Path(__file__).resolve().parent; SRC=ROOT/"source_data"; FIG=ROOT/"figures"
FIG.mkdir(exist_ok=True)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans","Liberation Sans"],
 "svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.titlesize":9,"axes.labelsize":8,
 "xtick.labelsize":7,"ytick.labelsize":7,"axes.spines.top":False,"axes.spines.right":False,
 "axes.linewidth":.7,"legend.frameon":False,"legend.fontsize":7})
BLUE="#3F7CAC"; BLUE2="#82A9C8"; GREEN="#2F8F6B"; GREEN2="#82B99F"; ORANGE="#D9853B"
RED="#C94C4C"; GREY="#8A8A8A"; LIGHT="#E9EEF2"; DARK="#263238"; PURPLE="#7A68A6"

def panel(ax,label): ax.text(.01,.98,label,transform=ax.transAxes,fontsize=10,fontweight="bold",va="top",ha="left",zorder=20)
def box(ax,xy,w,h,text,fc=LIGHT,ec=BLUE,lw=1,fs=7,style="round,pad=0.02"):
    p=FancyBboxPatch(xy,w,h,boxstyle=style,facecolor=fc,edgecolor=ec,linewidth=lw)
    ax.add_patch(p); ax.text(xy[0]+w/2,xy[1]+h/2,text,ha="center",va="center",fontsize=fs)
    return p
def arrow(ax,a,b,color=DARK): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=1,color=color))
def elbow_arrow(ax, points, color=DARK, lw=1.0):
    """Orthogonal connector whose final segment carries the arrow head."""
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=1)
    arrow(ax, points[-2], points[-1], color)
def save(fig,name):
    fig.tight_layout(pad=.7)
    for ext,kw in (("svg",{}),("pdf",{}),("png",{"dpi":300}),("tiff",{"dpi":600})):
        fig.savefig(FIG/f"{name}.{ext}",bbox_inches="tight",**kw)
    plt.close(fig)

# Figure 1 contract: the reader must see (i) how the bounded response controls
# weights and (ii) where clean-to-faulted supervision enters. Explanatory prose
# is kept in the caption so no sentence is compressed against the figure edge.
fig,axs=plt.subplots(1,2,figsize=(7.2,3.82),
                      gridspec_kw={"width_ratios":[1.0,1.12]})
fig.subplots_adjust(left=.035,right=.985,top=.87,bottom=.045,wspace=.17)
ax=axs[0]; ax.set_xlim(0,12);ax.set_ylim(0,10);ax.axis("off");panel(ax,"a")
ax.set_title("Bounded responsive fusion",pad=10,fontsize=9.6)
box(ax,(.20,7.0),2.25,1.50,"Sensor\ngroups",fc="#E7F0F7",fs=8.0)
box(ax,(3.05,7.0),2.25,1.50,"Group\nencoders",fc="#E7F0F7",fs=8.0)
arrow(ax,(2.45,7.75),(3.05,7.75))
box(ax,(6.15,7.60),2.25,1.50,"Group\nlogits",fc="#F0F3F5",ec=GREY,fs=8.0)
box(ax,(6.15,4.80),2.25,1.50,"Raw\nresponse",fc="#F0F3F5",ec=GREY,fs=8.0)
arrow(ax,(5.30,8.00),(6.15,8.35))
elbow_arrow(ax,[(4.18,7.00),(4.18,5.55),(6.15,5.55)])
box(ax,(9.25,4.80),2.45,1.50,"Bounded\nresponse",fc="#EAF4EF",ec=GREEN,fs=8.0)
arrow(ax,(8.40,5.55),(9.25,5.55),GREEN)
box(ax,(9.25,2.10),2.45,1.50,"Normalized\nweights",fc="#EAF4EF",ec=GREEN,fs=8.0)
arrow(ax,(10.48,4.80),(10.48,3.60),GREEN)
box(ax,(6.15,.15),2.25,1.50,"Weighted\nsum",fc="#EAF4EF",ec=GREEN,fs=8.0)
box(ax,(9.45,.15),2.25,1.50,"Class\nprobability",fc="#EAF4EF",ec=GREEN,fs=8.0)
# The class-logit and weight paths enter the weighted-sum box at separate
# top ports.  Their orthogonal channels do not intersect.
elbow_arrow(ax,[(8.40,8.35),(8.75,8.35),(8.75,2.55),(6.75,2.55),(6.75,1.65)],GREY)
arrow(ax,(9.25,2.45),(8.05,1.65),GREEN)
arrow(ax,(8.40,.90),(9.45,.90),GREEN)

ax=axs[1]; ax.set_xlim(0,14);ax.set_ylim(0,10);ax.axis("off");panel(ax,"b")
ax.set_title("Paired degradation and recovery supervision",pad=10,fontsize=9.6)
box(ax,(.20,7.35),2.45,1.50,"Clean\nview",fc="#E7F0F7",fs=8.0)
box(ax,(.20,2.35),2.45,1.50,"Faulted\nview",fc="#FCEBDD",ec=ORANGE,fs=8.0)
arrow(ax,(1.43,7.35),(1.43,3.85),ORANGE)
ax.text(1.72,5.60,"corrupt one\navailable group",ha="left",va="center",fontsize=6.8,color=ORANGE)
box(ax,(4.20,4.55),2.70,1.65,"Shared\nmodel",fc="#E7F0F7",ec=BLUE,fs=8.0)
arrow(ax,(2.65,8.10),(4.20,5.85),BLUE)
arrow(ax,(2.65,3.10),(4.20,4.95),ORANGE)
box(ax,(7.95,7.35),2.40,1.50,"Clean\noutput",fc="#F7F7F7",ec=GREY,fs=8.0)
box(ax,(7.95,2.35),2.40,1.50,"Faulted\noutput",fc="#F7F7F7",ec=GREY,fs=8.0)
arrow(ax,(6.90,5.85),(7.95,7.75))
arrow(ax,(6.90,4.95),(7.95,3.45),ORANGE)
box(ax,(11.45,7.35),2.40,1.50,"Task\nloss",fc="#EAF4EF",ec=GREEN,fs=8.0)
box(ax,(11.45,4.50),2.40,1.50,"Recovery\nloss",fc="#EAF4EF",ec=GREEN,fs=8.0)
box(ax,(11.45,1.65),2.40,1.50,"Auxiliary\nlosses",fc="#F5F0FA",ec=PURPLE,fs=8.0)
arrow(ax,(10.35,8.10),(11.45,8.10),GREEN)
# Clean and faulted outputs use separate vertical channels and separate entry
# ports on the recovery-loss box; neither arrow shares a segment.
elbow_arrow(ax,[(10.35,7.75),(10.65,7.75),(10.65,5.75),(11.45,5.75)],GREEN)
elbow_arrow(ax,[(10.35,3.50),(11.02,3.50),(11.02,5.15),(11.45,5.15)],GREEN)
arrow(ax,(10.35,2.85),(11.45,2.40),PURPLE)
save(fig,"figure1_method_framework")

# Figure 2: one integrated two-lane protocol diagram.
fig,ax=plt.subplots(figsize=(7.2,4.6));ax.set_xlim(0,16);ax.set_ylim(0,10);ax.axis("off")
ax.set_title("Experimental protocol, fixed interventions and evidence boundary",fontsize=10,pad=8)
box(ax,(.15,7.35),2.0,1.25,"Chemical array\n13,910 observations\n16 sensors",fc="#E7F0F7",ec=BLUE,fs=6.4)
box(ax,(2.75,7.35),1.8,1.25,"Batches 1--6\ntraining",fc="#E7F0F7",ec=BLUE,fs=6.6)
box(ax,(5.15,7.35),2.0,1.25,"Batch 7\n60% selection\n40% calibration",fc="#EAF4EF",ec=GREEN,fs=6.3)
box(ax,(7.75,7.35),2.0,1.25,"Batches 8--10\nheld-out test",fc="#FCEBDD",ec=ORANGE,fs=6.6)
box(ax,(10.35,7.35),2.3,1.25,"40% group-1 fault\nGaussian scale 3\n20% missingness",fc="#FBE5E5",ec=RED,fs=6.2)
box(ax,(13.25,7.35),2.55,1.25,"Fixed fault indices,\nmasks and noise across\nmethods and seeds",fc="#EAF4EF",ec=GREEN,fs=6.1)
for x0,x1,c in [(2.15,2.75,BLUE),(4.55,5.15,GREEN),(7.15,7.75,ORANGE),(9.75,10.35,RED),(12.65,13.25,GREEN)]:arrow(ax,(x0,7.98),(x1,7.98),c)
box(ax,(.15,5.45),15.65,.8,"Temporal acquisition order over 36 months; no test batch enters training, checkpoint selection or calibration",fc="#FAFAFA",ec=GREY,fs=6.5)

box(ax,(.15,2.6),2.0,1.25,"Hydraulic rig\n2,205 raw\n60-s cycles",fc="#E7F0F7",ec=BLUE,fs=6.4)
box(ax,(2.75,2.6),1.8,1.25,"1,449 stable\ncycles",fc="#E7F0F7",ec=BLUE,fs=6.6)
box(ax,(5.15,2.6),2.0,1.25,"Guarded blocked\nmain split; stratified\nsensitivity",fc="#EAF4EF",ec=GREEN,fs=6.0)
box(ax,(7.75,2.6),2.0,1.25,"14 sensors\n5 physical groups",fc="#EAF4EF",ec=GREEN,fs=6.5)
box(ax,(10.35,2.6),2.3,1.25,"Pressure or vibration\ncontrolled corruption\non 40% of cycles",fc="#FBE5E5",ec=RED,fs=6.2)
box(ax,(13.25,2.6),2.55,1.25,"Native cooler, valve,\npump and accumulator\ncondition labels",fc="#FCEBDD",ec=ORANGE,fs=6.1)
for x0,x1,c in [(2.15,2.75,BLUE),(4.55,5.15,GREEN),(7.15,7.75,GREEN),(9.75,10.35,RED),(12.65,13.25,ORANGE)]:arrow(ax,(x0,3.23),(x1,3.23),c)
box(ax,(.15,.7),15.65,.8,"Physical component labels are native to the rig; sensor faults in both datasets remain controlled interventions",fc="#FAFAFA",ec=GREY,fs=6.5)
save(fig,"figure2_experimental_protocol")

# Shared data for Figures 3--5.
met=pd.read_csv(SRC/"major_ablation_metrics.csv"); hyd=pd.read_csv(SRC/"hydraulic_validation_summary.csv")
seed=pd.read_csv(SRC/"major_seed_paired_effects.csv"); q=pd.read_csv(SRC/"major_quality_controls.csv")
audit=pd.read_csv(SRC/"major_weight_audit.csv"); boot=pd.read_csv(SRC/"major_bootstrap_5000.csv")
q1risk=pd.read_csv(SRC/"q1_risk_metrics.csv")
q1boot=pd.read_csv(SRC/"q1_ensemble_bootstrap.csv")
q1sens=pd.read_csv(SRC/"q1_sensitivity_summary.csv")
q1div=pd.read_csv(SRC/"q1_ensemble_diversity.csv")

# Figure 3: main results.
fig,axs=plt.subplots(2,2,figsize=(7.2,5.7));axs=axs.ravel()
order=["accuracy","macro_f1","macro_auprc","macro_auroc"]
ax=axs[0];panel(ax,"a")
for i,k in enumerate(order):
    v=seed[seed.metric==k].difference.to_numpy();x=np.full(len(v),i)+np.linspace(-.1,.1,len(v));ax.scatter(x,v,s=14,color=GREEN,alpha=.82);ax.plot([i-.15,i+.15],[v.mean()]*2,color=DARK,lw=1.4)
ax.axhline(0,color=GREY,ls="--",lw=.8);ax.set_xticks(range(4),["Accuracy","Macro-F1","Macro-AUPRC","Macro-AUROC"],rotation=18,ha="right");ax.set_ylabel("PDRF - CAGF");ax.set_title("Chemical-array paired effects (10 seeds)")
ax=axs[1];panel(ax,"b")
methods=["EF_PD","UF_PD","CAGF","DWR","PDRF_NOQ","PDRF"];x=np.arange(len(methods));width=.36
for off,metric,c in [(-.5,"macro_auroc",BLUE),(.5,"macro_auprc",GREEN)]:
    g=(met[(met.fault=="silent")&(met.subset=="all")&(met.metric==metric)&met.method.isin(methods)].groupby("method").value.agg(["mean","std"]).reindex(methods))
    ax.bar(x+off*width,g["mean"],width,yerr=g["std"],capsize=1.5,color=c,label=metric.replace("macro_","Macro-").upper())
ax.set_xticks(x,[m.replace("_","-") for m in methods],rotation=25,ha="right");ax.set_ylim(.46,.87);ax.set_ylabel("Score");ax.legend(ncol=2,loc="lower right");ax.set_title("Chemical-array silent-fault discrimination")
for pi,(fault,title) in enumerate([("silent_pressure","Hydraulic pressure-group fault"),("silent_vibration","Hydraulic vibration-group fault")],start=2):
    ax=axs[pi];panel(ax,chr(97+pi));tasks=["cooler","valve","pump","accumulator"];xx=np.arange(4);w=.24
    for j,(method,c) in enumerate([("CAGF",GREY),("DWR",BLUE2),("PDRF_NOQ",GREEN)]):
        vals=[];errs=[]
        for task in tasks:
            r=hyd[(hyd.task==task)&(hyd.method==method)&(hyd.fault==fault)&(hyd.metric=="macro_auroc")].iloc[0];vals.append(r["mean"]);errs.append(r["std"])
        ax.bar(xx+(j-1)*w,vals,w,yerr=errs,capsize=1.3,color=c,label=method.replace("_NOQ"," (no q)"))
    ax.set_xticks(xx,[t.capitalize() for t in tasks],rotation=16,ha="right");ax.set_ylim(.68,1.02);ax.set_ylabel("Macro-AUROC");ax.set_title(title)
    if pi==2: ax.legend(ncol=3,loc="lower center",fontsize=6)
save(fig,"figure3_main_results")

# Figure 4: ablations and q controls.
fig,axs=plt.subplots(2,2,figsize=(7.2,5.5));axs=axs.ravel();abl=["PD_ONLY","PD_MONO","PD_RANK","NO_LSCORE","LSCORE_NO_LRANK","PDRF"]
for pi,metric in enumerate(["macro_auroc","macro_auprc"]):
    ax=axs[pi];panel(ax,chr(97+pi));g=met[(met.fault=="silent")&(met.subset=="all")&(met.metric==metric)&met.method.isin(abl)]
    sm=g.groupby("method").value.agg(["mean","std"]).reindex(abl);colors=[GREY]*5+[GREEN]
    ax.barh(np.arange(6),sm["mean"],xerr=sm["std"],color=colors,capsize=1.5,height=.65)
    for i,m in enumerate(abl): ax.scatter(g[g.method==m].value,np.full((g.method==m).sum(),i),s=7,color=DARK,alpha=.45,zorder=3)
    ax.set_yticks(range(6),[m.replace("_","-") for m in abl]);ax.invert_yaxis();ax.set_xlabel(metric.replace("macro_","Macro-").upper());ax.set_title("Mechanism ablation")
    ax.set_xlim((.79,.85) if metric=="macro_auroc" else (.50,.61))
ax=axs[2];panel(ax,"c");orderq=["silent","reported","shuffled","misleading"]
for method,c in [("PDRF",GREEN),("PDRF_NOQ",BLUE)]:
    g=q[(q.training==method)&(q.metric=="macro_auroc")].groupby("test_q").value.agg(["mean","std"]).reindex(orderq)
    ax.errorbar(range(4),g["mean"],yerr=g["std"],marker="o",capsize=2,color=c,label=method.replace("_NOQ"," (no q)"))
ax.axvspan(2.65,3.35,color=mpl.colors.to_rgba(RED,.08));ax.set_xticks(range(4),["No q update","Reported","Shuffled","Misleading"],rotation=18,ha="right");ax.set_ylim(.74,.87);ax.set_ylabel("Macro-AUROC");ax.legend();ax.set_title("Diagnostic-quality metadata controls")
ax=axs[3];panel(ax,"d")
core=["Q_ONLY","PDRF_NOQ","PDRF"];g=met[(met.fault=="silent")&(met.subset=="all")&(met.metric=="macro_auroc")&met.method.isin(core)].groupby("method").value.agg(["mean","std"]).reindex(core)
ax.bar(range(3),g["mean"],yerr=g["std"],capsize=2,color=[GREY,BLUE,GREEN]);ax.set_xticks(range(3),["Quality only","PDRF (no q)","PDRF"],rotation=15,ha="right");ax.set_ylim(.78,.85);ax.set_ylabel("Macro-AUROC");ax.text(1,.843,"main gain does not require q",ha="center",fontsize=7,color=GREEN);ax.set_title("Quality alone does not explain the gain")
save(fig,"figure4_mechanism_quality")

# Figure 5: boundary evidence and targeted remedies.
fig,axs=plt.subplots(2,3,figsize=(7.2,5.8));axs=axs.ravel();fig.subplots_adjust(wspace=.45,hspace=.50)
ax=axs[0];panel(ax,"a");cols=["raw_score_violation","unnormalized_weight_violation","normalized_weight_violation","score_saturation"];labs=["Raw score\nviolation","Unnorm. weight\nviolation","Norm. weight\nviolation","Outer 5%\noccupancy"]
means=[audit[c].mean() for c in cols];stds=[audit[c].std() for c in cols];ax.bar(range(4),means,yerr=stds,capsize=1.5,color=[BLUE,BLUE2,PURPLE,ORANGE]);ax.set_xticks(range(4),["Score\nviol.","Unnorm.\nviol.","Norm.\nviol.","Outer 5%\noccup."],rotation=22,ha="right");ax.set_ylim(0,.75);ax.set_ylabel("Fraction");ax.set_title("Directional audit and saturation")
ax=axs[1];panel(ax,"b");dist=pd.read_csv(SRC/"representative_score_distribution.csv");bins=np.linspace(-3,3,25);ax.hist(dist.score_clean,bins=bins,histtype="step",lw=1.2,color=BLUE,label="Clean");ax.hist(dist.score_degraded,bins=bins,alpha=.45,color=ORANGE,label="Degraded");ax.axvspan(2.7,3,color=mpl.colors.to_rgba(RED,.12));ax.axvspan(-3,-2.7,color=mpl.colors.to_rgba(RED,.12));ax.set_xlabel("Operational score $r$");ax.set_ylabel("Affected observations");ax.legend();ax.set_title("Representative affected-score distribution")
ax=axs[2];panel(ax,"c");b=q1boot.set_index(["subset","metric"]);bo=[("all","macro_auroc"),("affected","macro_auroc")];y=np.arange(2);est=np.array([b.loc[k,"difference"] for k in bo]);lo=np.array([b.loc[k,"ci_low"] for k in bo]);hi=np.array([b.loc[k,"ci_high"] for k in bo]);ax.errorbar(est,y,xerr=np.vstack([est-lo,hi-est]),fmt="o",color=PURPLE,capsize=2);ax.axvline(0,color=GREY,ls="--",lw=.8);ax.set_yticks(y,["All","Affected"]);ax.invert_yaxis();ax.set_xlabel("RO-PDRF - CAGF macro-AUROC");ax.set_title("Fixed-ensemble bootstrap")
ax=axs[3];panel(ax,"d");sub=q1risk[q1risk.method.isin(["CAGF","PDRF","PDRF_RCIA_R30"])&q1risk.metric.isin(["macro_auroc","macro_f1"])];labels=[];vals=[];errs=[];colors=[]
for subset in ["affected","unaffected"]:
  for metric in ["macro_auroc","macro_f1"]:
    for method,c in [("CAGF",GREY),("PDRF",BLUE),("PDRF_RCIA_R30",GREEN)]:
      z=sub[(sub.subset==subset)&(sub.metric==metric)&(sub.method==method)].value;vals.append(z.mean());errs.append(z.std());colors.append(c)
    labels.append(f"{subset[:3]}.\n{metric.split('_')[1].upper()}")
pos=np.array([0,1,3,4]);w=.23
for j,(lab,c) in enumerate([("CAGF",GREY),("PDRF",BLUE),("RO-PDRF",GREEN)]): ax.bar(pos+(j-1)*w,vals[j::3],w,yerr=errs[j::3],color=c,capsize=1,label=lab)
ax.set_xticks(pos,labels);ax.set_ylim(.35,.91);ax.set_ylabel("Score");ax.legend(ncol=3,fontsize=6);ax.set_title("Affected versus unaffected subsets")
ax=axs[4];panel(ax,"e");g=q1sens[q1sens.setting.str.startswith("B=")].sort_values("bound");ax.plot(g.saturation,g.affected_auroc,marker="o",color=GREEN);[ax.annotate(f"B={int(b)}",(s,a),xytext=(2,3),textcoords="offset points",fontsize=6) for b,s,a in zip(g.bound,g.saturation,g.affected_auroc)];ax.set_xlabel("Outer-5% score occupancy");ax.set_ylabel("Affected macro-AUROC");ax.set_title("Bound-saturation trade-off")
ax=axs[5];panel(ax,"f");methods=["CAGF","PDRF","PDRF_RCIA_R30"];labs=["CAGF","PDRF","RO-PDRF"];g=q1div.set_index("method").reindex(methods);x=np.arange(3);ax.bar(x-.18,g.pairwise_disagreement,.36,color=[GREY,BLUE,GREEN],label="Disagreement");ax.bar(x+.18,g.pairwise_js,.36,color=[GREY,BLUE,GREEN],alpha=.55,label="Jensen-Shannon");ax.set_xticks(x,labs,rotation=12);ax.set_ylabel("Pairwise diversity");ax.legend(fontsize=6);ax.set_title("Ensemble diversity audit")
save(fig,"figure5_boundaries_remedies")
print("Five manuscript figures written in SVG/PDF/PNG/TIFF")
