"""Q2-review figures generated exclusively with Python/matplotlib.

Main conclusion: PDRF improves controlled-fault ranking and some hard decisions,
but probability calibration remains worse and AUROC gains over matched augmented
baselines are uncertain. Supplementary conclusion: grouping, unseen sensors and
score-scale comparability remain material boundaries.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
                     "axes.spines.right": False, "axes.spines.top": False,
                     "axes.linewidth": 0.7, "legend.frameon": False})
COL = {"EF": "#8D8D8D", "UF": "#B3B3B3", "CAGF": "#6B9AC4", "DWR": "#9A77B5", "BCRF": "#4F9D79"}
LAB = {"EF": "Early fusion", "UF": "Uniform fusion", "CAGF": "Corruption-aware gate",
       "DWR": "Direct weight ranking", "BCRF": "PDRF"}


def save(fig, stem):
    path = FIG / stem
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


metrics = pd.read_csv(SRC / "q2_metrics_long.csv")
head = metrics[(metrics.calibrated == True) & metrics.metric.isin(["accuracy", "macro_f1", "macro_auprc", "macro_auroc"])]
head = head.groupby(["method", "fault", "metric"], as_index=False).value.mean()
boot = pd.read_csv(SRC / "q2_bootstrap_effects.csv")
cal = pd.read_csv(SRC / "q2_calibration_summary.csv")
util = pd.read_csv(SRC / "q2_deployment_utility.csv")

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
ax = axes[0, 0]
metric_order = ["accuracy", "macro_f1", "macro_auprc", "macro_auroc"]
x = np.arange(len(metric_order)); offsets = np.linspace(-0.24, 0.24, 5)
for method, off in zip(["EF", "UF", "CAGF", "DWR", "BCRF"], offsets):
    d = head[(head.method == method) & (head.fault == "silent_gaussian")].set_index("metric").loc[metric_order]
    ax.plot(x + off, d.value, "o", color=COL[method], ms=4, label=LAB[method])
ax.set_xticks(x, ["Accuracy", "Macro-F1", "Macro-AUPRC", "Macro-AUROC"])
ax.set_ylabel("Metric value")
ax.set_ylim(0.45, 0.87); ax.grid(axis="y", color="#E6E6E6", lw=.6)
ax.legend(ncol=2, fontsize=6.2, loc="lower right")

ax = axes[0, 1]
contrasts = ["PDRF-EF", "PDRF-UF", "PDRF-CAGF", "PDRF-DWR"]
colors = [COL["EF"], COL["UF"], COL["CAGF"], COL["DWR"]]
ypos = np.arange(len(contrasts)); metric_offsets = {"macro_auroc": .22, "accuracy": 0, "nll": -.22}
markers = {"macro_auroc": "o", "accuracy": "s", "nll": "^"}
for metric_name, off in metric_offsets.items():
    d = boot[(boot.fault == "silent_gaussian") & (boot.metric == metric_name)].set_index("contrast").loc[contrasts]
    for i, (contrast, row) in enumerate(d.iterrows()):
        ax.errorbar(row.aligned_difference, i + off,
                    xerr=[[row.aligned_difference-row.sample_ci_low], [row.sample_ci_high-row.aligned_difference]],
                    fmt=markers[metric_name], color=colors[i], ms=3.5, capsize=2,
                    label={"macro_auroc":"AUROC", "accuracy":"Accuracy", "nll":"NLL improvement"}[metric_name] if i == 0 else None)
ax.axvline(0, color="#444444", lw=.8, ls="--")
ax.set_yticks(ypos, ["vs EF", "vs UF", "vs CAGF", "vs DWR"])
ax.set_xlabel("Aligned PDRF improvement")
ax.legend(fontsize=6.2, loc="lower right")
ax.grid(axis="x", color="#E6E6E6", lw=.6)

ax = axes[1, 0]
for method in ("EF", "CAGF", "BCRF"):
    vals = []
    for fault in ("natural", "silent_gaussian"):
        for calibrated in (False, True):
            v = cal[(cal.method == method) & (cal.fault == fault) &
                    (cal.calibrated == calibrated) & (cal.metric == "ece_15")].value.iloc[0]
            vals.append(v)
    ax.plot(np.arange(4), vals, marker="o", color=COL[method], label=LAB[method])
ax.set_xticks(range(4), ["Natural\nraw", "Natural\nscaled", "Silent\nraw", "Silent\nscaled"])
ax.set_ylabel("ECE (15 bins)")
ax.grid(axis="y", color="#E6E6E6", lw=.6); ax.legend(fontsize=6.2)

ax = axes[1, 1]
d = util[(util.cost_scheme == "uniform") & (util.coverage == 1.0) & (util.reject_cost == 0.25)]
wide = d.pivot(index="fault_prevalence", columns="method", values="expected_cost")
for comparator in ("EF", "UF", "CAGF", "DWR"):
    ax.plot(wide.index, wide["BCRF"] - wide[comparator], color=COL[comparator], label=f"PDRF - {LAB[comparator]}")
ax.axhline(0, color="#444444", lw=.8, ls="--")
ax.axvline(.67, color=COL["EF"], lw=.8, ls=":")
ax.set_xlabel("Silent-fault prevalence")
ax.set_ylabel("Expected-cost difference (lower is better)")
ax.legend(fontsize=5.8)

for label, ax in zip("abcd", axes.flat):
    ax.text(-.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
save(fig, "figure9_q2_validation")
plt.close(fig)

# Supplementary boundary figure.
grouping = pd.read_csv(SRC / "q2_grouping_sensitivity.csv")
sampling = pd.read_csv(SRC / "q2_sampling_unseen_sensor.csv")
targets = pd.read_csv(SRC / "q2_target_margin_ablation.csv")
thresholds = pd.read_csv(SRC / "q2_boundary_thresholds.csv")
scores = pd.read_csv(SRC / "score_comparability_summary.csv")
batch = pd.read_csv(SRC / "q2_batch_performance.csv")

fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), constrained_layout=True)
ax = axes[0, 0]
gorder = ["consecutive", "interleaved", "random_1", "random_2", "random_3", "random_4", "random_5"]
for i, g in enumerate(gorder):
    v = grouping[(grouping.grouping == g) & (grouping.fault == "silent_gaussian")].macro_auroc
    ax.scatter(np.full(len(v), i), v, s=10, color="#6B9AC4", alpha=.75)
    ax.plot([i-.22, i+.22], [v.mean(), v.mean()], color="#1F4E79", lw=1.3)
ax.set_xticks(range(len(gorder)), ["Consec.", "Interl.", "R1", "R2", "R3", "R4", "R5"], rotation=25)
ax.set_ylabel("Silent-fault macro-AUROC")

ax = axes[0, 1]
s = sampling.groupby(["sampling", "test_fault_modality"]).macro_auroc.mean().unstack()
for i, name in enumerate(["dominant_prior", "uniform", "exclude_test_sensor"]):
    ax.plot([1, 4], s.loc[name, [1, 4]], marker="o", label=name.replace("_", " "))
ax.set_xticks([1, 4], ["Sensor group 1", "Sensor group 4"]); ax.set_ylabel("Macro-AUROC"); ax.legend(fontsize=5.8)

ax = axes[0, 2]
for name, d in targets.groupby("target"):
    z = d.groupby("fault_scale").macro_auroc.mean()
    ax.plot(z.index, z.values, marker="o", label=name.replace("_", " "))
ax.set_xlabel("Fault scale"); ax.set_ylabel("Macro-AUROC"); ax.legend(fontsize=5.2)

ax = axes[1, 0]
t = thresholds.groupby("threshold").fraction.agg(["mean", "std"])
ax.errorbar(t.index, 100*t["mean"], yerr=100*1.96*t["std"]/np.sqrt(10), marker="o", capsize=2, color=COL["BCRF"])
ax.set_xlabel("Score threshold"); ax.set_ylabel("Fraction above threshold (%)")

ax = axes[1, 1]
names = ["global_spearman", "partial_spearman_controlling_task_ambiguity",
         "cross_sensor_pairwise_accuracy", "fault_sensor_identification_auroc", "severity_delta_spearman"]
labels = ["Global\nSpearman", "Partial\nSpearman", "Pairwise\naccuracy", "Fault-ID\nAUROC", "Severity\nSpearman"]
z = scores.set_index("metric").loc[names]
ax.bar(range(len(names)), z["mean"], yerr=z.ci95, color=["#6B9AC4"]*3+[COL["BCRF"]]*2, capsize=2)
ax.set_xticks(range(len(names)), labels, rotation=20); ax.set_ylim(0,1); ax.set_ylabel("Audit value")

ax = axes[1, 2]
for method in ("EF", "CAGF", "BCRF"):
    d = batch[(batch.method == method) & (batch.fault == "silent_gaussian") & (batch.metric == "macro_auroc")]
    ax.plot(d.batch, d.value, marker="o", color=COL[method], label=LAB[method])
ax.set_xticks([8,9,10]); ax.set_xlabel("Acquisition batch"); ax.set_ylabel("Macro-AUROC"); ax.legend(fontsize=5.8)

for label, ax in zip("abcdef", axes.flat):
    ax.text(-.16, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    ax.grid(axis="y", color="#E6E6E6", lw=.5)
save(fig, "supp_figure1_q2_boundaries")
plt.close(fig)
print("Q2 figures written")
