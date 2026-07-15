"""Create submission-grade figures for the selective-recovery major revision."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

COLORS = {
    "base": "#4C78A8",
    "recovery": "#E17C39",
    "frozen": "#7B6FD0",
    "gate": "#2A9D6F",
    "allrow": "#D65F5F",
    "twostage": "#C79A2B",
    "scalar": "#6F8F72",
    "random": "#9A9A9A",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13) -> None:
    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def decision_figure() -> None:
    curves = pd.read_csv(DATA / "major_prevention_retention_curve_summary.csv")
    policies = pd.read_csv(DATA / "major_primary_policy_metrics.csv")
    utility = pd.read_csv(DATA / "major_utility_threshold_summary.csv")
    confusion = pd.read_csv(DATA / "major_allrow_confusion.csv")

    fig = plt.figure(figsize=(7.20, 6.30), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0), hspace=0.30, wspace=0.22)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    styles = {
        "Conditional Lite-CF": (COLORS["frozen"], "-", 1.8),
        "All-row multinomial": (COLORS["allrow"], "-", 1.6),
        "Two-stage": (COLORS["twostage"], "-", 1.4),
        "Entropy difference": (COLORS["scalar"], "--", 1.2),
        "Random": (COLORS["random"], ":", 1.2),
    }
    line_handles = []
    for method, (color, linestyle, linewidth) in styles.items():
        part = curves[curves.method.eq(method)].sort_values("recovery_retention")
        handle, = ax_a.plot(
            100 * part.recovery_retention,
            100 * part.negative_transfer_prevention,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=method,
        )
        line_handles.append(handle)
    point_specs = {
        "Frozen Lite-CF (ungated)": (COLORS["frozen"], "o", "Frozen"),
        "Prospective gate Lite-CF": (COLORS["gate"], "s", "Gate"),
        "All-row multinomial (safety)": (COLORS["allrow"], "D", "All-row safety"),
        "Always PDRF": (COLORS["base"], "^", "Always base"),
        "Always RO-PDRF-Lite": (COLORS["recovery"], "v", "Always recovery"),
    }
    point_handles = []
    for method, (color, marker, label) in point_specs.items():
        row = policies[policies.method.eq(method)].mean(numeric_only=True)
        x = 100 * row.recovery_retention
        y = 100 * row.negative_transfer_prevention
        handle = ax_a.scatter(
            x,
            y,
            s=27,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
            label=label,
        )
        point_handles.append(handle)
    ax_a.set_xlim(-2, 102)
    ax_a.set_ylim(-2, 102)
    ax_a.set_xlabel("Recovery retention (%)")
    ax_a.set_ylabel("Negative-transfer prevention (%)")
    ax_a.set_title("Complete fixed-test operating curves")
    ax_a.grid(color="#E8E8E8", linewidth=0.5)
    line_legend = ax_a.legend(
        handles=line_handles,
        loc="lower left", bbox_to_anchor=(0.0, 0.03), ncol=1, handlelength=2.4
    )
    ax_a.add_artist(line_legend)
    ax_a.legend(
        handles=point_handles,
        loc="upper right",
        ncol=1,
        borderaxespad=0.4,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    panel_label(ax_a, "a")

    method_order = [
        "Always PDRF",
        "Frozen Lite-CF (ungated)",
        "Prospective gate Lite-CF",
        "All-row multinomial (safety)",
        "Always RO-PDRF-Lite",
    ]
    short = ["Always\nbase", "Frozen\nLite-CF", "Support\ngate", "All-row\nsafety", "Always\nrecovery"]
    means = policies.groupby("method").mean(numeric_only=True).loc[method_order]
    x = np.arange(len(method_order))
    width = 0.36
    ax_b.bar(
        x - width / 2,
        100 * means.negative_transfer_prevention,
        width,
        label="Prevention",
        color="#8DB7D8",
    )
    ax_b.bar(
        x + width / 2,
        100 * means.recovery_retention,
        width,
        label="Retention",
        color="#E6A06C",
    )
    for index, value in enumerate(means.utility_per_10000):
        ax_b.text(index, 104, f"U={value:+.1f}", ha="center", va="bottom", fontsize=5.5)
    ax_b.set_ylim(0, 118)
    ax_b.set_ylabel("Opportunity-normalized rate (%)")
    ax_b.set_xticks(x, short)
    ax_b.set_title("Deployment context at frozen thresholds", pad=16)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)
    ax_b.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    panel_label(ax_b, "b")

    fixed_methods = {
        "Always recovery": ("fixed_safety_or_endpoint", "Always RO-PDRF-Lite", COLORS["recovery"], "--"),
        "Frozen Lite-CF": ("fixed_safety_or_endpoint", "Frozen Lite-CF (ungated)", COLORS["frozen"], "-"),
        "Support gate": ("fixed_safety_or_endpoint", "Prospective gate Lite-CF", COLORS["gate"], "-"),
        "All-row safety": ("fixed_safety_or_endpoint", "All-row multinomial (safety)", COLORS["allrow"], ":"),
        "All-row utility": ("utility_optimized_on_calibration", "All-row multinomial", COLORS["allrow"], "-"),
        "Two-stage utility": ("utility_optimized_on_calibration", "Two-stage", COLORS["twostage"], "-"),
    }
    for label, (strategy, method, color, linestyle) in fixed_methods.items():
        part = utility[
            utility.threshold_strategy.eq(strategy) & utility.method.eq(method)
        ].sort_values("harm_to_correction_ratio")
        ax_c.plot(
            part.harm_to_correction_ratio,
            part["mean"],
            marker="o",
            markersize=3.4,
            linewidth=1.3,
            linestyle=linestyle,
            color=color,
            label=label,
        )
    ax_c.axhline(0, color="#555555", linewidth=0.7)
    ax_c.set_xscale("log")
    ax_c.set_yscale("symlog", linthresh=25, linscale=1.0)
    ax_c.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax_c.set_xlabel("Harm-to-correction value ratio")
    ax_c.set_ylabel("Net utility per 10,000")
    ax_c.set_title("Threshold objective changes the useful regime")
    ax_c.grid(color="#E8E8E8", linewidth=0.5)
    ax_c.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        columnspacing=0.9,
    )
    panel_label(ax_c, "c")

    matrix = confusion.groupby(["true_outcome", "predicted_outcome"]).n.sum().unstack(fill_value=0)
    normalized = matrix.div(matrix.sum(axis=1), axis=0)
    image = ax_d.imshow(normalized, cmap=mpl.colors.LinearSegmentedColormap.from_list("blue", ["#F7FAFC", "#4C78A8"]), vmin=0, vmax=1)
    labels = ["Base\nbetter", "Equivalent", "Recovery\nbetter"]
    ax_d.set_xticks(range(3), labels)
    ax_d.set_yticks(range(3), labels)
    ax_d.set_xlabel("Predicted all-row outcome")
    ax_d.set_ylabel("Observed outcome")
    ax_d.set_title("All-row router: row-normalized OOF confusion")
    for row in range(3):
        for column in range(3):
            value = normalized.iloc[row, column]
            color = "white" if value > 0.55 else "#222222"
            ax_d.text(column, row, f"{100*value:.1f}%\n(n={matrix.iloc[row, column]:,})", ha="center", va="center", fontsize=6.2, color=color)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.047, pad=0.03)
    colorbar.set_label("Within-observed-outcome fraction")
    panel_label(ax_d, "d", x=-0.20)

    save_figure(fig, "figure3_selective_recovery_decision")


def robustness_figure() -> None:
    severity = pd.read_csv(DATA / "major_severity_transport_summary.csv")
    groups = pd.read_csv(DATA / "major_fault_group_transport_summary.csv")
    batches = pd.read_csv(DATA / "major_batch_robustness_summary.csv")
    mechanism = pd.read_csv(DATA / "major_mechanism_transport_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.65), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    part = severity[
        severity.method.eq("Conditional Lite-CF")
        & severity.metric.eq("utility_per_10000")
    ]
    matrix = part.pivot(index="calibration_severity", columns="test_severity", values="mean").loc[[1, 2, 3], [1, 2, 3]]
    limit = float(np.abs(matrix.to_numpy()).max())
    image = ax_a.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
    ax_a.set_xticks(range(3), ["1", "2", "3"])
    ax_a.set_yticks(range(3), ["1", "2", "3"])
    ax_a.set_xlabel("Test severity")
    ax_a.set_ylabel("Calibration severity")
    ax_a.set_title("Severity transport: equal-cost utility")
    for row in range(3):
        for column in range(3):
            ax_a.text(column, row, f"{matrix.iloc[row, column]:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax_a, fraction=0.047, pad=0.03, label="Utility / 10,000")
    panel_label(ax_a, "a")

    group_part = groups[
        groups.metric.isin(["negative_transfer_prevention", "recovery_retention"])
    ].pivot_table(index=["method", "test_group"], columns="metric", values="mean").reset_index()
    x = np.arange(4)
    for method, color, offset, marker in (
        ("Conditional Lite-CF", COLORS["frozen"], -0.10, "o"),
        ("All-row multinomial", COLORS["allrow"], 0.10, "s"),
    ):
        p = group_part[group_part.method.eq(method)].sort_values("test_group")
        ax_b.plot(
            x + offset,
            100 * p.negative_transfer_prevention,
            color=color,
            marker=marker,
            linewidth=1.3,
            label=f"{method}: prevent",
        )
        ax_b.plot(
            x + offset,
            100 * p.recovery_retention,
            color=color,
            marker=marker,
            linewidth=1.0,
            linestyle="--",
            label=f"{method}: retain",
        )
    ax_b.set_xticks(x, ["1\n(cal.)", "2", "3", "4"])
    ax_b.set_xlabel("Affected sensor group at test")
    ax_b.set_ylabel("Rate (%)")
    ax_b.set_ylim(0, 105)
    ax_b.set_title("Group-1 calibration does not transfer uniformly")
    ax_b.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax_b.legend(loc="center left", bbox_to_anchor=(0.0, 0.45), ncol=1)
    panel_label(ax_b, "b")

    batch_part = batches[
        batches.measure.eq("macro_auroc")
        & batches["metric"].eq("estimate")
        & batches.method.isin(
            ["Always PDRF", "Frozen Lite-CF (ungated)", "Prospective gate Lite-CF", "Always RO-PDRF-Lite"]
        )
    ]
    view_order = ["observation_weighted_overall", "equal_weight_batches", "worst_batch", "B8-9_pooled", "B10"]
    view_label = ["Overall", "Equal\nbatch", "Worst\nbatch", "B8--9", "B10"]
    color_map = {
        "Always PDRF": COLORS["base"],
        "Frozen Lite-CF (ungated)": COLORS["frozen"],
        "Prospective gate Lite-CF": COLORS["gate"],
        "Always RO-PDRF-Lite": COLORS["recovery"],
    }
    label_map = {
        "Always PDRF": "Always base",
        "Frozen Lite-CF (ungated)": "Frozen Lite-CF",
        "Prospective gate Lite-CF": "Support gate",
        "Always RO-PDRF-Lite": "Always recovery",
    }
    for method in color_map:
        p = batch_part[batch_part.method.eq(method)].set_index("view").loc[view_order]
        ax_c.plot(
            range(len(view_order)),
            p["mean"],
            marker="o",
            markersize=3.5,
            linewidth=1.3,
            color=color_map[method],
            label=label_map[method],
        )
    ax_c.set_xticks(range(len(view_order)), view_label)
    ax_c.set_ylabel("Macro-AUROC")
    ax_c.set_title("Batch 10 dominates the observation-weighted result")
    ax_c.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    ax_c.legend(loc="lower left", ncol=2)
    panel_label(ax_c, "c")

    m = mechanism[
        mechanism.calibration_regime.eq("leave_one_mechanism_out")
        & mechanism.method.eq("Conditional Lite-CF")
        & mechanism.metric.eq("utility_per_10000")
    ].sort_values("heldout_or_test_mechanism")
    order = ["gaussian", "offset", "drift", "stuck_at"]
    m = m.set_index("heldout_or_test_mechanism").loc[order]
    yerr = np.vstack([m["mean"] - m.fitting_ci_low, m.fitting_ci_high - m["mean"]])
    ax_d.bar(range(4), m["mean"], color=["#78A9CF", "#E3A164", "#8DBF9E", "#B28AC7"], width=0.66)
    ax_d.errorbar(range(4), m["mean"], yerr=yerr, fmt="none", ecolor="#333333", elinewidth=0.8, capsize=2)
    ax_d.axhline(0, color="#555555", linewidth=0.7)
    ax_d.set_xticks(range(4), ["Gaussian", "Offset", "Drift", "Stuck-at"], rotation=15, ha="right")
    ax_d.set_ylabel("Equal-cost utility / 10,000")
    ax_d.set_title("Leave-one-mechanism-out selector transport")
    ax_d.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    panel_label(ax_d, "d")

    save_figure(fig, "figure4_shift_batch_robustness")


def main() -> None:
    decision_figure()
    robustness_figure()


if __name__ == "__main__":
    main()
