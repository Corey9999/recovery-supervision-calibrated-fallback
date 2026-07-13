"""Python-only publication figure for the strict estimand and safety upgrade."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT/"source_data"
FIG = ROOT/"figures"
FIG.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["xtick.labelsize"] = 6.5
plt.rcParams["ytick.labelsize"] = 6.5
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False

NAVY = "#484878"
BLUE = "#7884B4"
PINK = "#D990A4"
TEAL = "#42949E"
GREEN = "#2E8B57"
ORANGE = "#D98256"
GREY = "#A8A8A8"
LIGHT = "#E4E4F0"


def label(ax, text):
    ax.text(-.12, 1.04, text, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom")


def panel_a(ax):
    counts = pd.read_csv(SRC/"major9_estimand_counts.csv").set_index("subset").n
    assigned = int(counts["assigned"])
    applied = int(counts["fault_applied_available"])
    unavailable = int(counts["assigned_unavailable"])
    ax.barh([0], [applied], color=TEAL, height=.42, label="Fault applied and group available")
    ax.barh([0], [unavailable], left=[applied], color=GREY, height=.42,
            label="Assigned group unavailable")
    ax.text(applied/2, 0, f"Applied\n{applied:,}", ha="center", va="center",
            color="white", fontweight="bold")
    ax.text(applied+unavailable/2, 0, f"Masked\n{unavailable:,}", ha="center", va="center")
    ax.set_xlim(0, assigned)
    ax.set_yticks([])
    ax.set_xlabel(f"Fault-assigned observations (n={assigned:,})")
    ax.set_title("Intervention estimand", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=1, fontsize=6)
    ax.spines["left"].set_visible(False)
    label(ax, "a")


def panel_b(ax):
    m = pd.read_csv(SRC/"major9_estimand_metrics.csv")
    m = m[(m.subset == "fault_applied_available") & (m.metric == "macro_auroc")]
    p = m.pivot(index="seed", columns="method", values="value")
    data = [
        p["RO-PDRF"]-p["PDRF"],
        p["RO-PDRF"]-p["RO-CAGF"],
        p["RO-CAGF"]-p["CAGF"],
    ]
    names = ["Full − PDRF", "Full − recovery gate", "Recovery gate − gate"]
    colors = [PINK, NAVY, BLUE]
    for x, (values, color) in enumerate(zip(data, colors)):
        jitter = np.linspace(-.08, .08, len(values))
        ax.scatter(np.full(len(values), x)+jitter, values, s=16,
                   color=color, alpha=.75, edgecolor="white", linewidth=.3, zorder=2)
        ax.plot([x-.17, x+.17], [values.mean(), values.mean()], color="black", lw=1.4, zorder=3)
        ax.text(x, values.max()+.004, f"{values.mean():+.4f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color="#666666", lw=.8, ls="--")
    ax.set_xticks(range(3), names, rotation=18, ha="right")
    ax.set_ylabel("Paired Δ macro-AUROC")
    ax.set_title("Strict applied-fault comparison", loc="left", fontweight="bold")
    label(ax, "b")


def panel_c(ax):
    m = pd.read_csv(SRC/"major9_modern_baseline_metrics.csv")
    m = m[(m.subset == "fault_applied_available") & (m.metric == "macro_auroc")]
    order = ["gaussian", "offset", "drift", "stuck_at"]
    methods = ["RO-CAGF", "RO-MER", "RO-PDRF-Full"]
    colors = [GREY, BLUE, PINK]
    markers = ["o", "s", "D"]
    x = np.arange(len(order))
    for method, color, marker in zip(methods, colors, markers):
        z = m[m.method == method].groupby("fault_type").value.agg(["mean", "std"]).reindex(order)
        ax.errorbar(x, z["mean"], yerr=z["std"], color=color, marker=marker,
                    ms=4, lw=1.2, capsize=2, label=method)
    ax.set_xticks(x, ["Gaussian", "Offset", "Drift", "Stuck-at"])
    ax.set_ylabel("Affected macro-AUROC")
    ax.set_ylim(.74, .86)
    ax.set_title("Modern multi-expert control", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=6)
    label(ax, "c")


def panel_d(ax):
    s = pd.read_csv(SRC/"major9_safe_recovery_summary.csv")
    per_fault = s[s.scope != "four_fault_types"]
    aggregate = s[s.scope == "four_fault_types"]
    colors = {"Balanced": ORANGE, "Safe": GREEN}
    markers = {"Balanced": "o", "Safe": "s"}
    for variant in ["Balanced", "Safe"]:
        z = per_fault[per_fault.variant == variant]
        ax.scatter(z.recovery_preservation_rate, z.negative_transfer_prevented_rate,
                   s=22, color=colors[variant], marker=markers[variant], alpha=.55,
                   label=f"{variant}, individual faults")
        a = aggregate[aggregate.variant == variant].iloc[0]
        ax.scatter(a.recovery_preservation_rate, a.negative_transfer_prevented_rate,
                   s=65, color=colors[variant], marker=markers[variant],
                   edgecolor="black", linewidth=.6, zorder=3)
        dx = .015
        dy = .012 if variant == "Balanced" else -.035
        ax.text(a.recovery_preservation_rate+dx,
                a.negative_transfer_prevented_rate+dy,
                f"{variant}: {100*a.negative_transfer_prevented_rate:.0f}% prevented\n"
                f"{100*a.recovery_preservation_rate:.0f}% recovery retained",
                fontsize=6, va="center")
    ax.set_xlim(0, .55)
    ax.set_ylim(.55, 1.02)
    ax.set_xlabel("Recovery opportunities preserved")
    ax.set_ylabel("Negative transfers prevented")
    ax.set_title("Calibrated fallback trade-off", loc="left", fontweight="bold")
    ax.grid(color="#E5E5E5", lw=.6)
    ax.legend(loc="lower left", fontsize=5.8)
    label(ax, "d")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4),
                             gridspec_kw={"hspace": .52, "wspace": .32})
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.subplots_adjust(left=.10, right=.98, top=.95, bottom=.12)
    base = FIG/"figure8_strict_safe_upgrade"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
