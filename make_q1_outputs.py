"""Create the Q1 mechanism figure and compact LaTeX result tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC, FIG, TAB = ROOT / "source_data", ROOT / "figures", ROOT / "tables"
FINAL = "PDRF_RCIA_R30"
LABEL = {FINAL: "RO-PDRF", "PDRF": "PDRF", "CAGF": "CAGF"}
COL = {FINAL: "#0072B2", "PDRF": "#6B6B6B", "CAGF": "#D55E00"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11.5, "axes.titlesize": 12.5,
    "axes.labelsize": 11.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

metrics = pd.read_csv(SRC / "q1_risk_metrics.csv")
audit = pd.read_csv(SRC / "q1_mechanism_audit.csv")
classes = pd.read_csv(SRC / "q1_affected_class_confusion.csv")
ensemble = pd.read_csv(SRC / "q1_ensemble_metrics.csv")
ensemble_pred = pd.read_csv(SRC / "q1_ensemble_predictions.csv")
sensitivity = pd.read_csv(SRC / "q1_sensitivity.csv")
hyd = pd.read_csv(SRC / "q1_hydraulic_extension.csv")

fig, ax = plt.subplots(2, 3, figsize=(10.8, 6.1), constrained_layout=True)

# a: paired affected-subset AUROC across optimization seeds.
a = metrics[(metrics.subset == "affected") & (metrics.metric == "macro_auroc")]
for seed in sorted(a.seed.unique()):
    z = a[(a.seed == seed) & a.method.isin(["PDRF", FINAL])].set_index("method").value
    ax[0, 0].plot([0, 1], [z["PDRF"], z[FINAL]], color="#B7B7B7", lw=.7, zorder=1)
for j, method in enumerate(["PDRF", FINAL]):
    v = a[a.method == method].sort_values("seed").value
    ax[0, 0].scatter(np.full(len(v), j), v, s=18, color=COL[method], zorder=2)
ax[0, 0].set_xticks([0, 1], ["PDRF", "RO-PDRF"])
ax[0, 0].set_ylabel("Affected macro-AUROC")
ax[0, 0].set_title("a  Affected-set recovery", loc="left", fontweight="bold")

# b: mechanism success is not equivalent to prediction recovery.
b = audit[audit.method.isin(["PDRF", FINAL])].groupby("method")[["downweight_success", "failure_despite_downweight"]].mean()
x = np.arange(2); width = .34
for j, method in enumerate(["PDRF", FINAL]):
    ax[0, 1].bar(x + (j-.5)*width, 100*b.loc[method], width, color=COL[method], label=LABEL[method])
ax[0, 1].set_xticks(x, ["Affected weight\ndecreased", "Prediction failed\ndespite decrease"])
ax[0, 1].set_ylabel("Affected observations (%)")
ax[0, 1].legend(frameon=False, ncol=2, loc="upper right")
ax[0, 1].set_title("b  Downweighting is insufficient", loc="left", fontweight="bold")

# c: affected-row reliability diagram for fixed probability ensembles.
edges=np.linspace(0,1,11)
for method in ["CAGF",FINAL]:
    z=ensemble_pred[(ensemble_pred.method==method)&(ensemble_pred.pool=="probability_mean")&ensemble_pred.affected]
    pp=z[[f"p{i}" for i in range(1,7)]].to_numpy(); yy=z.y.to_numpy();conf=pp.max(1);correct=pp.argmax(1)==yy
    xc=[];yc=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        take=(conf>lo)&(conf<=hi)
        if take.any():xc.append(conf[take].mean());yc.append(correct[take].mean())
    ax[0,2].plot(xc,yc,"o-",ms=3,color=COL[method],label=LABEL[method])
ax[0,2].plot([0,1],[0,1],"--",lw=.7,color="#777777")
ax[0,2].set_xlim(.15,1.0);ax[0,2].set_ylim(0,1.0);ax[0,2].set_xlabel("Mean confidence");ax[0,2].set_ylabel("Empirical accuracy")
ax[0,2].legend(frameon=False);ax[0,2].set_title("c  Affected-row reliability",loc="left",fontweight="bold")

# d: aggregation audit.
pools = ["probability_mean", "log_probability_mean", "majority_vote", "validation_selected"]
pool_labels = ["Probability\nmean", "Log-prob.\nmean", "Majority\nvote", "Selected\nsingle"]
d = ensemble[(ensemble.subset == "all") & (ensemble.metric == "macro_auroc")]
for j, method in enumerate(["CAGF", FINAL]):
    vals = [d[(d.method == method) & (d.pool == p)].value.iloc[0] for p in pools]
    ax[1, 0].bar(np.arange(4)+(j-.5)*.34, vals, .34, color=COL[method], label=LABEL[method])
ax[1, 0].set_xticks(np.arange(4), pool_labels)
ax[1, 0].set_ylim(.72, .87)
ax[1, 0].set_ylabel("Macro-AUROC")
ax[1, 0].legend(frameon=False, ncol=2)
ax[1, 0].set_title("d  Aggregation audit", loc="left", fontweight="bold")

# e: explicit saturation/performance trade-off across score bounds.
e = sensitivity[sensitivity.setting.str.startswith("B=")].groupby(["setting", "bound"])[["saturation", "affected_auroc"]].mean().reset_index().sort_values("bound")
ax[1, 1].plot(e.bound, 100*e.saturation, "o-", color="#CC79A7", label="Saturation")
ax[1, 1].set_xlabel("Score bound B")
ax[1, 1].set_ylabel("Outer-5% occupancy (%)", color="#A94E86")
ax[1, 1].tick_params(axis="y", labelcolor="#A94E86")
ax2 = ax[1, 1].twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(e.bound, e.affected_auroc, "s--", color="#009E73", label="Affected AUROC")
ax2.set_ylabel("Affected macro-AUROC", color="#007F5F"); ax2.tick_params(axis="y", labelcolor="#007F5F")
ax[1, 1].set_title("e  Bound--saturation trade-off", loc="left", fontweight="bold")

# f: transfer with and without linear resampling.
f = hyd[(hyd.subset == "affected") & (hyd.metric == "macro_auroc") & (hyd.fault == "pressure")]
means = f.groupby(["representation", "task", "method"]).value.mean().unstack("method")
final_h = "PDRF_RCIA_R30_NOQ"
diff = (means[final_h] - means["CAGF"]).unstack("representation")
tasks = list(diff.index); xpos = np.arange(len(tasks))
for j, rep in enumerate(["linear_resampling", "zero_padding"]):
    ax[1, 2].bar(xpos+(j-.5)*.34, diff[rep], .34,
                 color=["#56B4E9", "#009E73"][j],
                 label=["Linear resampling", "No interpolation"][j])
ax[1, 2].axhline(0, color="black", lw=.7)
ax[1, 2].set_xticks(xpos, [t.capitalize() for t in tasks], rotation=20, ha="right")
ax[1, 2].set_ylabel("Affected AUROC difference\n(RO-PDRF minus CAGF)")
ax[1, 2].legend(frameon=False, fontsize=9.5)
ax[1, 2].set_title("f  Hydraulic representation audit", loc="left", fontweight="bold")

for axy in ax.flat:
    axy.grid(axis="y", color="#E6E6E6", lw=.5, zorder=0)

for ext in ["pdf", "svg", "png", "tiff"]:
    fig.savefig(FIG / f"figure6_recovery_mechanism.{ext}", dpi=600 if ext == "tiff" else 300,
                bbox_inches="tight")
plt.close(fig)

# Compact manuscript table: means over ten matched seeds.
sel = metrics[(metrics.method.isin(["CAGF", "PDRF", FINAL])) &
              (metrics.subset.isin(["all", "affected"])) &
              (metrics.metric.isin(["accuracy", "macro_f1", "macro_auprc", "macro_auroc", "nll", "ece15"]))]
piv = sel.groupby(["method", "subset", "metric"]).value.mean().unstack(["subset", "metric"])
order = [("all", "accuracy"), ("all", "macro_auroc"), ("affected", "accuracy"),
         ("affected", "macro_f1"), ("affected", "macro_auprc"),
         ("affected", "macro_auroc"), ("affected", "nll"), ("affected", "ece15")]
lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Method & All Acc. & All AUROC & Aff. Acc. & Aff. F1 & Aff. AUPRC & Aff. AUROC & Aff. NLL & Aff. ECE \\", r"\midrule"]
for method in ["CAGF", "PDRF", FINAL]:
    vals = [piv.loc[method, k] for k in order]
    lines.append(f"{LABEL[method]} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "q1_recovery_results.tex").write_text("\n".join(lines), encoding="utf-8")

# Sensitivity summary table.
sens = sensitivity.groupby(["setting", "bound", "beta", "rank_weight"])[["saturation", "macro_auroc", "affected_auroc", "nll", "ece15"]].mean().reset_index()
sens.to_csv(SRC / "q1_sensitivity_summary.csv", index=False)
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Setting & Saturation & All AUROC & Affected AUROC & NLL & ECE \\", r"\midrule"]
for _, row in sens.iterrows():
    lines.append(f"{row.setting} & {row.saturation:.3f} & {row.macro_auroc:.3f} & {row.affected_auroc:.3f} & {row.nll:.3f} & {row.ece15:.3f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "q1_sensitivity_table.tex").write_text("\n".join(lines), encoding="utf-8")

# Hydraulic paired effect summary for the supplement.
hmean = hyd.groupby(["representation", "task", "fault", "subset", "metric", "method"]).value.mean().unstack("method")
hmean["final_minus_cagf"] = hmean[final_h] - hmean["CAGF"]
hmean.reset_index().to_csv(SRC / "q1_hydraulic_extension_summary.csv", index=False)
hshow = hmean.reset_index()
hshow = hshow[(hshow.fault == "pressure") & (hshow.subset == "affected") & (hshow.metric == "macro_auroc")]
lines = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Representation & Task & CAGF & RO-PDRF & Difference \\", r"\midrule"]
for _, row in hshow.sort_values(["representation", "task"]).iterrows():
    rep = "Linear resampling" if row.representation == "linear_resampling" else "No interpolation"
    lines.append(f"{rep} & {row.task.capitalize()} & {row.CAGF:.3f} & {row[final_h]:.3f} & {row.final_minus_cagf:+.3f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "q1_hydraulic_extension_table.tex").write_text("\n".join(lines), encoding="utf-8")
