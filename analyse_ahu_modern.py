"""Detailed AHU diagnostics, conformal sets, tables and supplementary figure."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC, TAB, FIG = ROOT/"source_data", ROOT/"tables", ROOT/"figures"
DISPLAY = {"RC-PDRF-NOQ": "RO-PDRF-NOQ"}
METHODS = ["EF-PD", "HGB-current", "HGB-lag", "ENT-PD", "CAGF", "PDRF-NOQ", "RC-PDRF-NOQ"]
COL = {"EF-PD":"#0072B2", "HGB-current":"#56B4E9", "HGB-lag":"#009E73",
       "ENT-PD":"#CC79A7", "CAGF":"#D55E00", "PDRF-NOQ":"#777777",
       "RC-PDRF-NOQ":"#E69F00"}

mpl.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
    "font.size":7, "axes.titlesize":8, "axes.labelsize":7, "xtick.labelsize":6.5,
    "ytick.labelsize":6.5, "legend.fontsize":6.3, "axes.spines.top":False,
    "axes.spines.right":False, "pdf.fonttype":42, "svg.fonttype":"none"})


def dname(method): return DISPLAY.get(method, method)


def conformal_quantile(scores, alpha):
    n = len(scores)
    rank = min(n, math.ceil((n+1)*(1-alpha)))
    return float(np.partition(scores, rank-1)[rank-1])


metrics = pd.read_csv(SRC/"ahu_temporal_metrics.csv")
pred = pd.read_csv(SRC/"ahu_temporal_predictions.csv")
cal = pd.read_csv(SRC/"ahu_temporal_calibration_predictions.csv")
cross_path = SRC/"ahu_field_metrics.csv"
cross = pd.read_csv(cross_path) if cross_path.exists() else pd.DataFrame()

# Chronological test-set class and native fault-subtype distributions.  The
# ensemble prediction file repeats the same observations by method, so one
# prespecified method is retained solely to count unique test rows.
dist = pred[pred.method == "EF-PD"].groupby(["building", "subtype"]).size().unstack(fill_value=0)
dist.to_csv(SRC/"ahu_test_class_distribution.csv")
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Building & Normal & Return-air fault & Supply-air fault & Total \\", r"\midrule"]
for building in ("auditorium", "hospital", "office"):
    z = dist.loc[building]
    normal = int(z.get("Normal condition", 0))
    ret = int(z.get("Return air temperature fault", 0))
    sup = int(z.get("Supply air temperature fault", 0))
    lines.append(f"{building.capitalize()} & {normal:,} & {ret:,} & {sup:,} & {normal+ret+sup:,}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB/"ahu_class_distribution.tex").write_text("\n".join(lines), encoding="utf-8")

# Fixed five-member ensemble confusion counts for the principal deployment
# comparators.  Counts, unlike seed-averaged rates, make imbalance visible.
conf_lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
              r"Building & Method & TN & FP & FN & TP \\", r"\midrule"]
for building in ("auditorium", "hospital", "office"):
    for method in ("EF-PD", "HGB-current", "HGB-lag", "ENT-PD", "CAGF", "RC-PDRF-NOQ"):
        g = pred[(pred.building == building) & (pred.method == method)]
        y = g.y.to_numpy(); guess = g[["p0", "p1"]].to_numpy().argmax(1)
        tn = int(((y == 0) & (guess == 0)).sum()); fp = int(((y == 0) & (guess == 1)).sum())
        fn = int(((y == 1) & (guess == 0)).sum()); tp = int(((y == 1) & (guess == 1)).sum())
        conf_lines.append(f"{building.capitalize()} & {dname(method)} & {tn:,} & {fp:,} & {fn:,} & {tp:,}" + r" \\")
    conf_lines.append(r"\addlinespace")
conf_lines += [r"\bottomrule", r"\end{tabular}"]
(TAB/"ahu_ensemble_confusion.tex").write_text("\n".join(conf_lines), encoding="utf-8")

# Detailed building/method metrics.
wanted = ["macro_auroc", "macro_auprc", "macro_f1", "balanced_accuracy",
          "class_2_precision", "class_2_recall", "class_2_f1"]
agg = metrics[metrics.metric.isin(wanted)].groupby(["building","method","metric"]).value.agg(["mean","std"]).reset_index()
agg.to_csv(SRC/"ahu_temporal_detailed_summary.csv", index=False)
lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         "Building & Method & AUROC & AUPRC & Macro-F1 & Bal. Acc. & Fault Prec. & Fault Recall & Fault F1 \\\\", r"\midrule"]
for building in ("auditorium","hospital","office"):
    for method in METHODS:
        vals=[]
        for metric in wanted:
            z=agg[(agg.building==building)&(agg.method==method)&(agg.metric==metric)]
            vals.append(z.iloc[0]["mean"] if len(z) else np.nan)
        lines.append(f"{building.capitalize()} & {dname(method)} & "+" & ".join(f"{v:.3f}" for v in vals)+" \\\\")
    lines.append(r"\addlinespace")
lines += [r"\bottomrule",r"\end{tabular}"]
(TAB/"ahu_temporal_detailed.tex").write_text("\n".join(lines),encoding="utf-8")

# Fault-subtype recall.
subtypes = [("recall_return_air_temperature_fault","Return-air"),
            ("recall_supply_air_temperature_fault","Supply-air")]
sub = metrics[metrics.metric.isin([x[0] for x in subtypes])].groupby(["building","method","metric"]).value.agg(["mean","std","count"]).reset_index()
sub.to_csv(SRC/"ahu_subtype_recall.csv",index=False)
lines=[r"\begin{tabular}{lllrr}",r"\toprule","Building & Method & Fault subtype & Recall & Number of seeds \\\\",r"\midrule"]
for building in ("auditorium","hospital","office"):
    for method in METHODS:
        for metric,label in subtypes:
            z=sub[(sub.building==building)&(sub.method==method)&(sub.metric==metric)]
            if len(z): lines.append(f"{building.capitalize()} & {dname(method)} & {label} & {z.iloc[0]['mean']:.3f} $\\pm$ {z.iloc[0]['std']:.3f} & {int(z.iloc[0]['count'])}"+" \\\\")
lines += [r"\bottomrule",r"\end{tabular}"]
(TAB/"ahu_subtype_recall.tex").write_text("\n".join(lines),encoding="utf-8")

# Risk-coverage and split-conformal set diagnostics on fixed five-member ensembles.
risk_rows=[]; conformal_rows=[]
for (building,method),g in pred.groupby(["building","method"]):
    p=g[["p0","p1"]].to_numpy(); y=g.y.to_numpy(); conf=p.max(1); correct=p.argmax(1)==y
    order=np.argsort(-conf)
    for coverage in (.50,.70,.80,.90,.95,1.00):
        n=max(1,int(np.floor(coverage*len(g)))); take=order[:n]
        risk_rows.append({"building":building,"method":method,"target_coverage":coverage,"coverage":n/len(g),
                          "selective_error":float(1-correct[take].mean()),
                          "fault_recall":float((p[take].argmax(1)[y[take]==1]==1).mean()) if np.any(y[take]==1) else np.nan})
    c=cal[(cal.building==building)&(cal.method==method)]
    cp=c[["p0","p1"]].to_numpy(); cy=c.y.to_numpy(); scores=1-cp[np.arange(len(cy)),cy]
    for alpha in (.05,.10,.20):
        q=conformal_quantile(scores,alpha); sets=p >= (1-q)
        covered=sets[np.arange(len(y)),y]
        size=sets.sum(1); single=size==1
        conformal_rows.append({"building":building,"method":method,"alpha":alpha,"threshold":q,
            "empirical_coverage":float(covered.mean()),"mean_set_size":float(size.mean()),
            "singleton_rate":float(single.mean()),"singleton_accuracy":float(correct[single].mean()) if single.any() else np.nan,
            "empty_rate":float((size==0).mean())})
pd.DataFrame(risk_rows).to_csv(SRC/"ahu_risk_coverage.csv",index=False)
pd.DataFrame(conformal_rows).to_csv(SRC/"ahu_conformal_sets.csv",index=False)

# Compact conformal table at alpha=0.10.
cf=pd.DataFrame(conformal_rows);cf=cf[cf.alpha==.10]
lines=[r"\begin{tabular}{llrrrr}",r"\toprule","Building & Method & Coverage & Mean set size & Singleton rate & Singleton accuracy \\\\",r"\midrule"]
for _,z in cf[cf.method.isin(["EF-PD","HGB-lag","ENT-PD","CAGF","RC-PDRF-NOQ"])].iterrows():
    lines.append(f"{z.building.capitalize()} & {dname(z.method)} & {z.empirical_coverage:.3f} & {z.mean_set_size:.3f} & {z.singleton_rate:.3f} & {z.singleton_accuracy:.3f}"+" \\\\")
lines += [r"\bottomrule",r"\end{tabular}"]
(TAB/"ahu_conformal.tex").write_text("\n".join(lines),encoding="utf-8")

# Six-panel field-diagnosis boundary figure.
fig,axs=plt.subplots(2,3,figsize=(7.2,4.8),constrained_layout=True)
buildings=["auditorium","hospital","office"]; shown=["EF-PD","HGB-lag","ENT-PD","CAGF","RC-PDRF-NOQ"]
for pi,metric in enumerate(["macro_auroc","macro_auprc"]):
    ax=axs.flat[pi]
    x=np.arange(3);w=.16
    for j,method in enumerate(shown):
        z=agg[(agg.metric==metric)&(agg.method==method)].set_index("building").reindex(buildings)
        ax.bar(x+(j-2)*w,z["mean"],w,yerr=z["std"],capsize=1,color=COL[method],label=dname(method))
    ax.set_xticks(x,[b.capitalize() for b in buildings],rotation=15,ha="right")
    ax.set_ylim((.88,1.01) if metric=="macro_auroc" else (.60,1.01));ax.set_ylabel(metric.replace("macro_","Macro-").upper())
    ax.set_title(chr(97+pi)+"  Chronological field diagnosis",loc="left",fontweight="bold")
    if pi==0:ax.legend(ncol=2,loc="lower left")

ax=axs.flat[2];x=np.arange(6);w=.25;labels=[]
for bi,building in enumerate(buildings):
    for si,(metric,label) in enumerate(subtypes):
        labels.append(building[:3].capitalize()+"\n"+label[:3]+".")
for j,method in enumerate(["EF-PD","HGB-lag","RC-PDRF-NOQ"]):
    vals=[]
    for building in buildings:
        for metric,_ in subtypes:
            z=sub[(sub.building==building)&(sub.method==method)&(sub.metric==metric)]["mean"]
            vals.append(z.iloc[0] if len(z) else np.nan)
    ax.bar(x+(j-1)*w,vals,w,color=COL[method],label=dname(method))
ax.set_xticks(x,labels);ax.set_ylim(0,1.05);ax.set_ylabel("Fault recall");ax.legend(ncol=1)
ax.annotate("hospital supply-air: $n=8$", xy=(3, .02), xytext=(3, .15),
            ha="center", fontsize=6, arrowprops={"arrowstyle":"-","lw":.6,"color":"#555555"})
ax.set_title("c  Fault-subtype sensitivity",loc="left",fontweight="bold")

ax=axs.flat[3];risk=pd.DataFrame(risk_rows)
for method in ["EF-PD","HGB-lag","ENT-PD","CAGF","RC-PDRF-NOQ"]:
    z=risk[risk.method==method].groupby("target_coverage").selective_error.mean().reset_index()
    ax.plot(z.target_coverage,z.selective_error,marker="o",ms=2.5,color=COL[method],label=dname(method))
ax.set_xlabel("Retained coverage");ax.set_ylabel("Selective error");ax.legend(ncol=2)
ax.set_title("d  Confidence-based abstention",loc="left",fontweight="bold")

ax=axs.flat[4];z=cf[cf.method.isin(shown)].groupby("method")[["empirical_coverage","singleton_rate"]].mean().reindex(shown)
xx=np.arange(len(shown));ax.bar(xx-.18,z.empirical_coverage,.36,color="#56B4E9",label="Empirical coverage")
ax.bar(xx+.18,z.singleton_rate,.36,color="#E69F00",label="Singleton rate")
ax.axhline(.9,color="#555555",ls="--",lw=.7);ax.set_xticks(xx,[dname(m) for m in shown],rotation=22,ha="right");ax.set_ylim(0,1.05);ax.legend()
ax.set_title("e  Split-conformal sets ($\\alpha=0.10$)",loc="left",fontweight="bold")

ax=axs.flat[5]
if len(cross):
    requested=["EF-PD","ENT-PD","CAGF","RC-PDRF-NOQ"]
    cm=cross[(cross.metric=="macro_auroc")&cross.method.isin(requested)].groupby(["held_out_building","method"]).value.mean().unstack("method").reindex(buildings)
    shown_cross=[m for m in requested if m in cm.columns]
    xx=np.arange(3);w=.8/max(1,len(shown_cross))
    for j,method in enumerate(shown_cross):
        ax.bar(xx+(j-(len(shown_cross)-1)/2)*w,cm[method],w,color=COL[method],label=dname(method))
    ax.set_xticks(xx,[b.capitalize() for b in buildings],rotation=15,ha="right");ax.set_ylim(0,1.02);ax.set_ylabel("Macro-AUROC");ax.legend(ncol=2)
    ax.set_title("f  Leave-one-building-out",loc="left",fontweight="bold")
else: ax.axis("off")
for ax in axs.flat: ax.grid(axis="y",color="#E8E8E8",lw=.45,zorder=0)
for ext in ("svg","pdf","png","tiff"):
    fig.savefig(FIG/f"figureS1_ahu_field_diagnostics.{ext}",dpi=600 if ext=="tiff" else 300,bbox_inches="tight")
plt.close(fig)
