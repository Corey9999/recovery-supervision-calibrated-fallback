"""Create the revision figure with Python/matplotlib only.

Figure conclusion: counterfactual ranking improves robustness across several
non-Gaussian fault families and rolling temporal origins, while bounded CRF
reduces near-bound occupancy; misleading diagnostics remain a failure case.
Evidence map: panel a compares fault families, panel b audits the reliability
head, and panels c-d test direction across rolling temporal origins.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures" / "figure8_revision_validation"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
})

COLORS = {"UF": "#8C8C8C", "MOME_A": "#6A9BC3", "CRF": "#D98C4A", "BCRF": "#4F9D79"}
LABELS = {"UF": "Uniform fusion", "MOME_A": "Routed experts", "CRF": "CRF", "BCRF": "Bounded CRF"}
FAULTS = ["silent_gaussian", "bias", "clipping", "burst", "correlated", "misleading_quality", "delayed_quality"]
FAULT_LABELS = ["Silent\nGaussian", "Fixed\nbias", "Clipping", "Contiguous\nburst",
                "Correlated\ndual-sensor", "Misleading\ndiagnostic", "Delayed\ndiagnostic"]
METHODS = ["UF", "MOME_A", "CRF", "BCRF"]

faults = pd.read_csv(SRC / "revision_faults_summary.csv")
rolling = pd.read_csv(SRC / "revision_rolling_summary.csv")
audit = pd.read_csv(SRC / "revision_boundary_summary.csv")

fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0])
ax_a = fig.add_subplot(gs[0, :3])
ax_b = fig.add_subplot(gs[0, 3])
ax_c = fig.add_subplot(gs[1, :2])
ax_d = fig.add_subplot(gs[1, 2:])

x = np.arange(len(FAULTS))
offsets = np.linspace(-0.24, 0.24, len(METHODS))
for method, offset in zip(METHODS, offsets):
    d = faults[(faults.method == method) & (faults.metric == "macro_auroc")].set_index("fault").loc[FAULTS]
    ax_a.errorbar(x + offset, d["mean"], yerr=d["ci95"], fmt="o", ms=3.7,
                  lw=0.9, capsize=1.8, color=COLORS[method], label=LABELS[method])
ax_a.set_xticks(x, FAULT_LABELS)
ax_a.set_ylabel("Macro-AUROC")
ax_a.set_ylim(0.72, 0.88)
ax_a.grid(axis="y", color="#E6E6E6", lw=0.6)
ax_a.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17), columnspacing=1.2, handletextpad=0.4)
ax_a.text(0.01, 0.03, "Misleading diagnostics remain a failure case", transform=ax_a.transAxes,
          color="#555555", fontsize=6.5)

audit = audit.set_index("method").loc[["CRF", "BCRF"]]
vals = 100 * audit["near_upper_bound"].to_numpy()
errs = 100 * 1.96 * audit["near_upper_bound_sd"].to_numpy() / np.sqrt(10)
ax_b.bar([0, 1], vals, yerr=errs, width=0.62, capsize=2,
         color=[COLORS["CRF"], COLORS["BCRF"]], edgecolor="none")
ax_b.set_xticks([0, 1], ["CRF", "Bounded\nCRF"])
ax_b.set_ylabel("Near upper bound (%)")
ax_b.set_ylim(0, 100)
ax_b.grid(axis="y", color="#E6E6E6", lw=0.6)
for i, value in enumerate(vals):
    ax_b.text(i, value + 3.5, f"{value:.1f}%", ha="center", fontsize=6.5)

for ax, regime, title in ((ax_c, "natural", "Natural temporal drift"),
                          (ax_d, "silent", "Silent Gaussian fault")):
    for method in METHODS:
        d = rolling[(rolling.method == method) & (rolling.regime == regime) &
                    (rolling.metric == "macro_auroc")].sort_values("origin")
        ax.errorbar(d.origin, d["mean"], yerr=d.ci95, marker="o", ms=3.5,
                    lw=1.2, capsize=2, color=COLORS[method], label=LABELS[method])
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Rolling temporal origin")
    ax.set_ylabel("Macro-AUROC")
    ax.set_title(title, fontsize=7.5, pad=4)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
ax_c.set_ylim(0.85, 1.00)
ax_d.set_ylim(0.76, 0.97)

for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
    ax.text(-0.10 if ax is not ax_b else -0.28, 1.06, label, transform=ax.transAxes,
            fontweight="bold", fontsize=8, va="top")

FIG.parent.mkdir(exist_ok=True)
fig.savefig(FIG.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(FIG.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
fig.savefig(FIG.with_suffix(".png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(FIG)
