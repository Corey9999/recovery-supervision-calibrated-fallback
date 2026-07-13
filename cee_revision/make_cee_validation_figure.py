"""Draw the C&EE selector-validation figure from frozen source tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)

COLORS = {
    "PDRF": "#6F7682",
    "RO-PDRF-Full": "#4C78A8",
    "SR-PDRF-Safe-CF": "#2A9D8F",
    "proposed": "#2A9D8F",
    "simple": "#A6AAB2",
    "random": "#E9A23B",
    "unseen": "#C65D47",
}


def panel_label(ax, label):
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def mean_se(values):
    values = np.asarray(values, float)
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


def main():
    metrics = pd.read_csv(DATA / "cee_cf10_stream_metrics.csv")
    lofo = pd.read_csv(DATA / "cee_cf10_lofo.csv")
    unseen = pd.read_csv(DATA / "cee_cf10_unseen_faults.csv")
    simple = pd.read_csv(DATA / "cee_cf10_simple_rule_test.csv")
    intervals = pd.read_csv(DATA / "cee_cf10_hierarchical_intervals.csv").set_index("quantity")
    scaling = pd.read_csv(DATA / "cee_cf10_group_scaling.csv")

    fig = plt.figure(figsize=(7.2, 6.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.13], height_ratios=[1.0, 1.12])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a: full deployment stream across fault prevalence.
    prevalence = np.array([0.0, 0.1, 0.4, 0.7])
    for method in ("PDRF", "RO-PDRF-Full", "SR-PDRF-Safe-CF"):
        means, errors = [], []
        for value in prevalence:
            if value == 0:
                part = metrics[
                    (metrics.stream == "clean_no_imposed_fault")
                    & (metrics.prevalence == 0)
                    & (metrics.metric == "macro_auroc")
                    & (metrics.method == method)
                ]
                per_seed = part.groupby("seed").value.mean()
            else:
                part = metrics[
                    (metrics.stream == "full_mixed_stream")
                    & (metrics.prevalence == value)
                    & (metrics.metric == "macro_auroc")
                    & (metrics.method == method)
                ]
                per_seed = part.groupby("seed").value.mean()
            mean, error = mean_se(per_seed)
            means.append(mean)
            errors.append(error)
        label = {"PDRF": "Base", "RO-PDRF-Full": "Full", "SR-PDRF-Safe-CF": "Safe-CF"}[method]
        marker = {"PDRF": "o", "RO-PDRF-Full": "s", "SR-PDRF-Safe-CF": "D"}[method]
        ax_a.errorbar(
            prevalence * 100,
            means,
            yerr=errors,
            marker=marker,
            markersize=4,
            capsize=2,
            color=COLORS[method],
            label=label,
        )
    ax_a.set_xlabel("Imposed-fault prevalence (%)")
    ax_a.set_ylabel("Full-stream macro-AUROC")
    ax_a.set_xticks(prevalence * 100)
    ax_a.legend(loc="lower left", ncol=1)
    ax_a.grid(axis="y", color="#D9DCE1", linewidth=0.5, alpha=0.8)
    ax_a.set_title("Deployment mixture", loc="left", fontweight="bold")
    panel_label(ax_a, "a")

    # b: mechanism transfer effects.
    transfer_rows = []
    label_map = {
        "gaussian": "Gaussian",
        "offset": "Offset",
        "drift": "Drift",
        "stuck_at": "Stuck-at",
        "gain_loss": "Gain loss",
        "clipping": "Clipping",
        "correlated_dual": "Dual corruption",
    }
    for fault, group in lofo.groupby("held_out_fault"):
        diff = group.safe_macro_auroc - group.pdrf_macro_auroc
        mean, se = mean_se(diff)
        transfer_rows.append((label_map[fault], mean, 1.96 * se, "LOFO"))
    for fault, group in unseen.groupby("unseen_fault"):
        diff = group.safe_macro_auroc - group.pdrf_macro_auroc
        mean, se = mean_se(diff)
        transfer_rows.append((label_map[fault], mean, 1.96 * se, "Unseen"))
    y = np.arange(len(transfer_rows))[::-1]
    for yi, (label, mean, error, audit) in zip(y, transfer_rows):
        color = COLORS["proposed"] if audit == "LOFO" else COLORS["unseen"]
        marker = "o" if audit == "LOFO" else "s"
        ax_b.errorbar(mean, yi, xerr=error, fmt=marker, ms=4, capsize=2, color=color)
    ax_b.axvline(0, color="#53565C", linestyle="--", linewidth=0.8)
    ax_b.set_yticks(y, [r[0] for r in transfer_rows])
    ax_b.set_xlabel("Safe-CF minus PDRF macro-AUROC")
    ax_b.grid(axis="x", color="#D9DCE1", linewidth=0.5, alpha=0.8)
    ax_b.set_title("Held-out and unseen mechanisms", loc="left", fontweight="bold")
    ax_b.text(0.99, 0.04, "circles: leave-one-family-out\nsquares: unseen mechanism", transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6, color="#555A63")
    panel_label(ax_b, "b")

    # c: risk-recovery trade-off against simple selectors.
    selected_rules = [
        ("Cross-fitted-logistic-Safe", "proposed", "Safe-CF"),
        ("Higher-confidence", "prevention_matched", "Confidence"),
        ("Lower-entropy", "prevention_matched", "Entropy"),
        ("Lower-LOO-disagreement", "prevention_matched", "LOO"),
        ("Higher-JS-disagreement", "prevention_matched", "JS"),
        ("Random-matched-selection", "selection_rate_matched", "Random"),
    ]
    offsets = {
        "Safe-CF": (6, 8),
        "Confidence": (8, -13),
        "Entropy": (8, 9),
        "LOO": (-23, -16),
        "JS": (-21, 13),
        "Random": (5, 5),
    }
    for rule, matching, label in selected_rules:
        group = simple[(simple.rule == rule) & (simple.matching == matching)]
        x_mean, x_se = mean_se(group.negative_transfer_prevention)
        y_mean, y_se = mean_se(group.recovery_retention)
        if label == "Safe-CF":
            xrow = intervals.loc["negative_transfer_prevention"]
            yrow = intervals.loc["recovery_retention"]
            x_mean, y_mean = xrow.bootstrap_mean, yrow.bootstrap_mean
            xerr = np.array([[x_mean - xrow["ci_2.5"]], [xrow["ci_97.5"] - x_mean]])
            yerr = np.array([[y_mean - yrow["ci_2.5"]], [yrow["ci_97.5"] - y_mean]])
            color, marker, size, zorder = COLORS["proposed"], "D", 36, 5
        else:
            xerr, yerr = 1.96 * x_se, 1.96 * y_se
            color = COLORS["random"] if label == "Random" else COLORS["simple"]
            marker, size, zorder = ("s" if label == "Random" else "o"), 25, 3
        ax_c.errorbar(x_mean, y_mean, xerr=xerr, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.8, capsize=1.8, zorder=zorder)
        ax_c.scatter(x_mean, y_mean, s=size, color=color, marker=marker, edgecolor="white", linewidth=0.5, zorder=zorder + 1)
        dx, dy = offsets[label]
        ax_c.annotate(
            label,
            (x_mean, y_mean),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.0,
            color="#33363B",
            ha="right" if label in {"LOO", "JS"} else "left",
            va="top" if label in {"Confidence", "LOO"} else "bottom",
        )
    ax_c.scatter([1], [1], marker="*", s=55, color="#7A4EAB", edgecolor="white", linewidth=0.4, zorder=6)
    ax_c.annotate("Ideal / oracle", (1, 1), xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=6.2, color="#5D3B85")
    ax_c.set_xlim(-0.03, 1.04)
    ax_c.set_ylim(-0.03, 1.04)
    ax_c.set_xlabel("Negative-transfer prevention")
    ax_c.set_ylabel("Recovery retention")
    ax_c.grid(color="#D9DCE1", linewidth=0.5, alpha=0.8)
    ax_c.set_title("Matched selector baselines", loc="left", fontweight="bold")
    panel_label(ax_c, "c")

    # d: computational scaling.
    x = scaling.groups.to_numpy(int)
    ax_d.plot(x, scaling.pdrf_flops, "o--", color=COLORS["PDRF"], label="PDRF (1 pass)")
    ax_d.plot(x, scaling.inference_flops, "D-", color=COLORS["proposed"], label="Safe-CF (G+2 passes)")
    for xi, yi, passes in zip(x, scaling.inference_flops, scaling.forward_passes):
        ax_d.annotate(f"{int(passes)} passes", (xi, yi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.2, color=COLORS["proposed"])
    ax_d.set_yscale("log")
    ax_d.set_xticks(x)
    ax_d.set_xlabel("Number of sensor groups (G)")
    ax_d.set_ylabel("Counted inference FLOPs / sample (log scale)")
    ax_d.grid(axis="y", which="both", color="#D9DCE1", linewidth=0.5, alpha=0.8)
    ax_d.legend(loc="upper left")
    ax_d.set_title("Deployment cost boundary", loc="left", fontweight="bold")
    panel_label(ax_d, "d")

    base_name = FIG / "figure_cee_crossfitted_deployment"
    fig.savefig(base_name.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_name.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_name.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base_name.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(base_name)


if __name__ == "__main__":
    main()
