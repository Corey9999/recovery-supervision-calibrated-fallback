"""Create Q1-revision deployment and selector-stability main figures."""

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
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.35,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

BLUE = "#3A78B4"
GREEN = "#2D9B72"
ORANGE = "#E68A3F"
PURPLE = "#7E6BB2"
RED = "#C95B59"
GRAY = "#777777"
LIGHT = "#D9E2EA"
METHOD_COLORS = {
    "Always PDRF": "#8A8A8A",
    "Always RO-PDRF-Full": ORANGE,
    "Safe-CF-12": PURPLE,
    "Cascade-25": BLUE,
    "Shrink-6": "#65A6D9",
    "Three-outcome-6": "#B58BC7",
    "Tree-6": "#B0B0B0",
    "Lite-CF": GREEN,
}


def save_figure(fig, stem):
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(
            FIGURES / f"{stem}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )


def panel_label(ax, label):
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        va="top",
    )


def t_error(values):
    values = np.asarray(values, float)
    return 2.262 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) == 10 else 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def tradeoff_figure():
    lite_safety = pd.read_csv(DATA / "q1_lite_routing_safety.csv")
    lite_safety = lite_safety[
        lite_safety.stream == "strict_fault_applied_available"
    ].copy()
    lite_mean = lite_safety[
        [
            "negative_transfer_prevention",
            "recovery_retention",
            "mean_forward_passes",
        ]
    ].mean()
    summary = pd.DataFrame(
        [
            {
                "method": "PDRF",
                "negative_transfer_prevention": 1.0,
                "recovery_retention": 0.0,
                "mean_forward_passes": 1.0,
            },
            {
                "method": "RO-PDRF-Lite",
                "negative_transfer_prevention": 0.0,
                "recovery_retention": 1.0,
                "mean_forward_passes": 1.0,
            },
            {"method": "Lite-CF", **lite_mean.to_dict()},
        ]
    )

    intervals = pd.read_csv(DATA / "q1_lite_mechanism_fixed_intervals.csv")
    intervals = intervals[
        intervals.comparison == "lite_cf_minus_pdrf_auroc"
    ].copy()
    intervals["fault_type"] = pd.Categorical(
        intervals.fault_type,
        ["gaussian", "offset", "drift", "stuck_at"],
        ordered=True,
    )
    intervals = intervals.sort_values("fault_type")

    lite_utility = pd.read_csv(DATA / "q1_lite_decision_utility.csv")
    lite_utility = lite_utility[
        lite_utility.pass_penalty_equivalent_corrections_per_10000 == 0.5
    ].copy()
    endpoint_utility_rows = []
    for _, row in lite_safety.iterrows():
        corrections = 10000.0 * row.correction_opportunities / row.n
        harms = 10000.0 * row.harm_opportunities / row.n
        for ratio in (1.0, 2.0, 5.0, 10.0):
            endpoint_utility_rows.extend(
                [
                    {
                        "method": "PDRF",
                        "harm_to_correction_cost_ratio": ratio,
                        "net_utility_per_10000": 0.0,
                    },
                    {
                        "method": "RO-PDRF-Lite",
                        "harm_to_correction_cost_ratio": ratio,
                        "net_utility_per_10000": corrections - ratio * harms,
                    },
                ]
            )
    utility = pd.concat(
        [pd.DataFrame(endpoint_utility_rows), lite_utility], ignore_index=True
    )
    utility_summary = (
        utility.groupby(["method", "harm_to_correction_cost_ratio"])
        .net_utility_per_10000.mean()
        .reset_index()
    )

    calibration_size = pd.read_csv(DATA / "q1_lite_selector_calibration_size.csv")
    size_summary = (
        calibration_size.groupby("fraction")
        .agg(
            prevention=("negative_transfer_prevention", "mean"),
            prevention_se=("negative_transfer_prevention", "sem"),
            retention=("recovery_retention", "mean"),
            retention_se=("recovery_retention", "sem"),
            auroc=("macro_auroc", "mean"),
            auroc_se=("macro_auroc", "sem"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.05))
    fig.subplots_adjust(
        left=0.095,
        right=0.96,
        bottom=0.10,
        top=0.94,
        wspace=0.38,
        hspace=0.46,
    )

    ax = axes[0, 0]
    for _, row in summary.iterrows():
        size = 30 + 22 * row.mean_forward_passes
        color = {
            "PDRF": GRAY,
            "RO-PDRF-Lite": ORANGE,
            "Lite-CF": GREEN,
        }[row.method]
        ax.scatter(
            row.recovery_retention,
            row.negative_transfer_prevention,
            s=size,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset = {
            "PDRF": (7, -3),
            "RO-PDRF-Lite": (-58, 8),
            "Lite-CF": (8, -15),
        }[row.method]
        ax.annotate(
            row.method,
            (row.recovery_retention, row.negative_transfer_prevention),
            xytext=offset,
            textcoords="offset points",
            color=color,
            fontsize=7.5,
            fontweight="bold" if row.method == "Lite-CF" else "normal",
        )
    ax.set_xlim(-0.04, 1.05)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Recovery retention")
    ax.set_ylabel("Negative-transfer prevention")
    ax.set_title("Hard-decision safety and compute")
    ax.grid(color="#E8E8E8", linewidth=0.6, zorder=0)
    ax.text(
        0.04,
        0.05,
        "Marker area: forward passes",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=7.0,
    )
    panel_label(ax, "a")

    ax = axes[0, 1]
    y_positions = np.arange(len(intervals))
    xerr = np.vstack(
        [intervals["mean"] - intervals["ci_2.5"], intervals["ci_97.5"] - intervals["mean"]]
    )
    prevention_handle = ax.errorbar(
        intervals["mean"],
        y_positions,
        xerr=xerr,
        fmt="o",
        color=GREEN,
        ecolor=GREEN,
        capsize=2.5,
        markersize=4.5,
    )
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions, [value.replace("_", " ").title() for value in intervals.fault_type.astype(str)])
    ax.invert_yaxis()
    ax.set_xlabel("Lite-CF minus PDRF macro-AUROC")
    ax.set_title("Mechanism-fixed fitted-pair intervals")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.6)
    panel_label(ax, "b")

    ax = axes[1, 0]
    for method in ["PDRF", "RO-PDRF-Lite", "Lite-CF"]:
        group = utility_summary[utility_summary.method == method]
        if group.empty:
            continue
        ax.plot(
            group.harm_to_correction_cost_ratio,
            group.net_utility_per_10000,
            marker="o",
            markersize=3.5,
            label=method,
            color={
                "PDRF": GRAY,
                "RO-PDRF-Lite": ORANGE,
                "Lite-CF": GREEN,
            }[method],
        )
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.set_xlabel("Harm cost / correction value")
    ax.set_ylabel("Net utility per 10,000")
    ax.set_title("Decision utility ($\\lambda=0.5$ per extra pass)")
    ax.legend(ncol=1, loc="lower left", handlelength=1.4)
    ax.grid(color="#E8E8E8", linewidth=0.6)

    # The unconditional-recovery curve occupies the full utility range at
    # high harm cost.  A fixed-range inset preserves the smaller PDRF versus
    # Lite-CF differences without changing the scale of the main axis.
    inset = ax.inset_axes([0.48, 0.52, 0.49, 0.40])
    for method in ["PDRF", "Lite-CF"]:
        group = utility_summary[utility_summary.method == method]
        inset.plot(
            group.harm_to_correction_cost_ratio,
            group.net_utility_per_10000,
            marker="o",
            markersize=2.5,
            color={"PDRF": GRAY, "Lite-CF": GREEN}[method],
        )
    inset.axhline(0, color="#555555", linewidth=0.6, linestyle="--")
    inset.set_xscale("log")
    inset.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    inset.set_ylim(-70, 25)
    inset.set_yticks([-60, -30, 0])
    inset.tick_params(labelsize=5.6, length=2)
    inset.set_title("Zoom: PDRF and Lite-CF", fontsize=6.1, pad=2)
    inset.grid(color="#ECECEC", linewidth=0.45)
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.96)
    panel_label(ax, "c")

    ax = axes[1, 1]
    x = 100 * size_summary.fraction
    prevention_handle = ax.errorbar(
        x,
        size_summary.prevention,
        yerr=1.96 * size_summary.prevention_se,
        marker="o",
        color=GREEN,
        capsize=2,
        label="Prevention",
    )
    retention_handle = ax.errorbar(
        x,
        size_summary.retention,
        yerr=1.96 * size_summary.retention_se,
        marker="s",
        color=ORANGE,
        capsize=2,
        label="Retention",
    )
    ax.set_ylim(0, 1.02)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xlabel("Available calibration groups (%)")
    ax.set_ylabel("Hard-decision rate")
    secondary = ax.twinx()
    auroc_handle = secondary.errorbar(
        x,
        size_summary.auroc,
        yerr=1.96 * size_summary.auroc_se,
        marker="^",
        color=GRAY,
        capsize=2,
        label="Macro-AUROC",
    )
    secondary.set_ylim(0.78, 0.83)
    secondary.set_ylabel("Macro-AUROC", color=GRAY)
    secondary.spines["right"].set_visible(True)
    ax.annotate(
        "Prevention",
        (x.iloc[-1], size_summary.prevention.iloc[-1]),
        xytext=(-55, 7),
        textcoords="offset points",
        color=GREEN,
        fontsize=7.2,
    )
    ax.annotate(
        "Retention",
        (x.iloc[-1], size_summary.retention.iloc[-1]),
        xytext=(-50, 8),
        textcoords="offset points",
        color=ORANGE,
        fontsize=7.2,
    )
    secondary.annotate(
        "Macro-AUROC",
        (x.iloc[-1], size_summary.auroc.iloc[-1]),
        xytext=(-68, -17),
        textcoords="offset points",
        color=GRAY,
        fontsize=7.2,
    )
    ax.set_title("Calibration-set-size sensitivity")
    ax.grid(color="#E8E8E8", linewidth=0.6)
    panel_label(ax, "d")

    save_figure(fig, "figure_cee_q1_tradeoff")
    plt.close(fig)


def reliability_rows(frame, prefix, bins=10):
    rows = []
    probability_columns_ = sorted(
        [column for column in frame if column.startswith(prefix)],
        key=lambda value: int(value.removeprefix(prefix)),
    )
    for (seed, fault), group in frame.groupby(["seed", "fault_type"]):
        p = group[probability_columns_].to_numpy(float)
        y = group.y.to_numpy(int)
        confidence = p.max(axis=1)
        correct = p.argmax(axis=1) == y
        edges = np.linspace(0, 1, bins + 1)
        for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            take = (confidence >= lower) & (
                confidence < upper if upper < 1 else confidence <= upper
            )
            if take.sum() < 20:
                continue
            rows.append(
                {
                    "seed": seed,
                    "fault_type": fault,
                    "bin": bin_index,
                    "n": int(take.sum()),
                    "confidence": confidence[take].mean(),
                    "accuracy": correct[take].mean(),
                }
            )
    return pd.DataFrame(rows)


def stability_calibration_figure():
    repeated = pd.read_csv(DATA / "q1_lite_repeated_grouped_cv.csv")
    coefficients = pd.read_csv(DATA / "q1_lite_coefficient_stability.csv")
    predictions = pd.read_csv(DATA / "q1_lite_routing_predictions.csv.gz")
    agreement = pd.read_csv(DATA / "q1_lite_agreement_probability_quality.csv")

    reliability = {
        "PDRF": reliability_rows(predictions, "base_p"),
        "RO-PDRF-Lite": reliability_rows(predictions, "lite_p"),
        "Lite-CF": reliability_rows(predictions, "selected_p"),
    }
    reliability_colors = {
        "PDRF": GRAY,
        "RO-PDRF-Lite": ORANGE,
        "Lite-CF": GREEN,
    }

    fig = plt.figure(figsize=(7.6, 6.05))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.96,
        bottom=0.10,
        top=0.94,
        wspace=0.40,
        hspace=0.46,
    )
    panel_a = outer[0, 0].subgridspec(1, 2, wspace=0.42)
    ax_auc = fig.add_subplot(panel_a[0, 0])
    ax_threshold = fig.add_subplot(panel_a[0, 1])
    ax_coefficient = fig.add_subplot(outer[0, 1])
    ax_reliability = fig.add_subplot(outer[1, 0])
    ax_probability = fig.add_subplot(outer[1, 1])

    pair_seeds = sorted(repeated.seed.unique())
    rng = np.random.default_rng(2718)
    for axis, metric, ylabel, title, color in (
        (ax_auc, "auroc", "OOF AUROC", "Preference AUROC", BLUE),
        (ax_threshold, "threshold", "Threshold", "Frozen threshold", GREEN),
    ):
        values = [
            repeated.loc[repeated.seed == seed, metric].to_numpy()
            for seed in pair_seeds
        ]
        axis.boxplot(
            values,
            positions=np.arange(1, len(pair_seeds) + 1),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#333333", "linewidth": 0.9},
            boxprops={"facecolor": color, "alpha": 0.22, "edgecolor": color},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
        for position, seed_values in enumerate(values, start=1):
            jitter = rng.uniform(-0.13, 0.13, size=len(seed_values))
            axis.scatter(
                position + jitter,
                seed_values,
                s=8,
                color=color,
                alpha=0.55,
                edgecolor="none",
                zorder=3,
            )
        axis.set_xlim(0.4, 10.6)
        axis.set_xticks([1, 3, 5, 7, 9], ["1", "3", "5", "7", "9"])
        axis.set_xlabel("Fitted pair")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    panel_label(ax_auc, "a")

    ax = ax_coefficient
    display = {
        "base_confidence": "Base confidence",
        "recovery_confidence": "Lite confidence",
        "confidence_delta": "Confidence change",
        "base_entropy": "Base entropy",
        "recovery_entropy": "Lite entropy",
        "entropy_delta": "Entropy change",
    }
    coefficients["label"] = coefficients.feature.map(display)
    coefficients = coefficients.sort_values("median")
    y_positions = np.arange(len(coefficients))
    xerr = np.vstack(
        [coefficients["median"] - coefficients["q1"], coefficients["q3"] - coefficients["median"]]
    )
    ax.errorbar(
        coefficients["median"],
        y_positions,
        xerr=xerr,
        fmt="o",
        color=GREEN,
        ecolor=GREEN,
        capsize=2,
        markersize=4,
    )
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.32, 0.36)
    ax.set_yticks(y_positions, coefficients.label)
    ax.set_xlabel("Standardized coefficient, median [IQR]")
    ax.set_title("Coefficient and sign stability")
    for _, row in coefficients.iterrows():
        ax.text(
            0.35,
            np.where(coefficients.index == row.name)[0][0],
            f"{100 * row.dominant_sign_fraction:.0f}%",
            va="center",
            ha="right",
            color=GRAY,
            fontsize=7.0,
        )
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.6)
    panel_label(ax, "b")

    ax = ax_reliability
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=0.8, label="Ideal")
    for method, rows in reliability.items():
        summary = rows.groupby("bin").agg(
            confidence=("confidence", "mean"),
            accuracy=("accuracy", "mean"),
            accuracy_se=("accuracy", "sem"),
        )
        ax.errorbar(
            summary.confidence,
            summary.accuracy,
            yerr=1.96 * summary.accuracy_se,
            marker="o",
            markersize=3.2,
            capsize=2,
            label=method,
            color=reliability_colors[method],
        )
    ax.set_xlim(0.15, 1.02)
    ax.set_ylim(0.15, 1.02)
    ax.set_xlabel("Mean emitted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Strict-subset reliability")
    ax.legend(loc="upper left")
    ax.grid(color="#E8E8E8", linewidth=0.6)
    panel_label(ax, "c")

    ax = ax_probability
    plotted = agreement[
        agreement.method.isin(["PDRF", "RO-PDRF-Lite", "Lite-CF"])
    ]
    aggregate = (
        plotted.groupby(["method", "subset"])[["nll", "ece15_equal_width"]]
        .mean()
        .reset_index()
    )
    methods = ["PDRF", "RO-PDRF-Lite", "Lite-CF"]
    x = np.arange(len(methods))
    width = 0.35
    agree_nll = [aggregate[(aggregate.method == method) & (aggregate.subset == "endpoint_class_agreement")].nll.iloc[0] for method in methods]
    disagree_nll = [aggregate[(aggregate.method == method) & (aggregate.subset == "endpoint_class_disagreement")].nll.iloc[0] for method in methods]
    agree_bar = ax.bar(
        x - width / 2,
        agree_nll,
        width,
        color=LIGHT,
        edgecolor="#666666",
        label="Class agreement",
    )
    disagree_bar = ax.bar(
        x + width / 2,
        disagree_nll,
        width,
        color="#AFC9DC",
        edgecolor="#666666",
        label="Class disagreement",
    )
    ax.set_xticks(x, ["PDRF", "Lite", "Lite-CF"])
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Probability quality by endpoint agreement")
    ax.set_ylim(0, 3.35)
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    agree_ece = [aggregate[(aggregate.method == method) & (aggregate.subset == "endpoint_class_agreement")].ece15_equal_width.iloc[0] for method in methods]
    disagree_ece = [aggregate[(aggregate.method == method) & (aggregate.subset == "endpoint_class_disagreement")].ece15_equal_width.iloc[0] for method in methods]
    secondary = ax.twinx()
    agree_ece_handle, = secondary.plot(
        x, agree_ece, marker="o", color=GRAY, linewidth=1.2, label="ECE: agreement"
    )
    disagree_ece_handle, = secondary.plot(
        x, disagree_ece, marker="s", color=BLUE, linewidth=1.2, label="ECE: disagreement"
    )
    secondary.set_ylim(0.18, 0.43)
    secondary.set_ylabel("Expected calibration error")
    secondary.spines["right"].set_visible(True)
    ax.legend(
        [agree_bar, disagree_bar, agree_ece_handle, disagree_ece_handle],
        [
            "NLL: agreement",
            "NLL: disagreement",
            "ECE: agreement",
            "ECE: disagreement",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    panel_label(ax, "d")

    save_figure(fig, "figure_cee_q1_stability_calibration")
    plt.close(fig)


def main():
    tradeoff_figure()
    stability_calibration_figure()


if __name__ == "__main__":
    main()
