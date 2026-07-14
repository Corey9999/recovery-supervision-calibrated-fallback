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
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.5,
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
        fontsize=9,
        va="top",
    )


def t_error(values):
    values = np.asarray(values, float)
    return 2.262 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) == 10 else 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def tradeoff_figure():
    learned = pd.read_csv(DATA / "q1_learned_routing_comparison.csv")
    learned = learned[
        (learned.analysis_set == "strict")
        & (learned.prevalence == 0.40)
    ]
    lite_safety = pd.read_csv(DATA / "q1_lite_routing_safety.csv")
    lite_safety = lite_safety[
        lite_safety.stream == "strict_fault_applied_available"
    ].copy()
    lite_safety["method"] = "Lite-CF"
    common = [
        "method",
        "negative_transfer_prevention",
        "recovery_retention",
        "mean_forward_passes",
    ]
    points = pd.concat([learned[common], lite_safety[common]], ignore_index=True)
    summary = points.groupby("method")[common[1:]].mean().reset_index()
    order = [
        "Lite-CF",
        "Cascade-25",
        "Safe-CF-12",
        "Three-outcome-6",
        "Tree-6",
        "Always PDRF",
        "Always RO-PDRF-Full",
    ]
    summary = summary[summary.method.isin(order)]

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

    utility = pd.read_csv(DATA / "q1_decision_utility.csv")
    utility = utility[
        (utility.stream == "strict_fault_applied_available")
        & (utility.pass_penalty_equivalent_corrections_per_10000 == 0.5)
        & utility.method.isin(["Always PDRF", "Always RO-PDRF-Full", "Safe-CF-12", "Cascade-25"])
    ]
    lite_utility = pd.read_csv(DATA / "q1_lite_decision_utility.csv")
    lite_utility = lite_utility[
        lite_utility.pass_penalty_equivalent_corrections_per_10000 == 0.5
    ].copy()
    utility = pd.concat([utility, lite_utility], ignore_index=True)
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

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.55))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.10, top=0.94, wspace=0.34, hspace=0.42)

    ax = axes[0, 0]
    handles = []
    labels = []
    for _, row in summary.iterrows():
        size = 30 + 22 * row.mean_forward_passes
        handle = ax.scatter(
            row.recovery_retention,
            row.negative_transfer_prevention,
            s=size,
            color=METHOD_COLORS.get(row.method, GRAY),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        handles.append(handle)
        labels.append(row.method)
    ax.set_xlim(-0.04, 1.05)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Recovery retention")
    ax.set_ylabel("Negative-transfer prevention")
    ax.set_title("Hard-decision safety and compute")
    ax.grid(color="#E8E8E8", linewidth=0.6, zorder=0)
    ax.text(0.02, 0.03, "Marker area scales with forward passes", transform=ax.transAxes, color=GRAY, fontsize=6.1)
    ax.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.55, 0.52),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
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
    ax.set_xlabel("Lite-CF − PDRF macro-AUROC")
    ax.set_title("Mechanism-fixed fitted-pair intervals")
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.6)
    panel_label(ax, "b")

    ax = axes[1, 0]
    for method in ["Always PDRF", "Always RO-PDRF-Full", "Safe-CF-12", "Cascade-25", "Lite-CF"]:
        group = utility_summary[utility_summary.method == method]
        if group.empty:
            continue
        ax.plot(
            group.harm_to_correction_cost_ratio,
            group.net_utility_per_10000,
            marker="o",
            markersize=3.5,
            label=method,
            color=METHOD_COLORS[method],
        )
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.set_xlabel("Harm cost / correction value")
    ax.set_ylabel("Net utility per 10,000")
    ax.set_title("Decision utility with a pass-cost proxy")
    ax.legend(ncol=2, loc="lower left", handlelength=1.4, columnspacing=0.8)
    ax.grid(color="#E8E8E8", linewidth=0.6)
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
    ax.legend(
        [prevention_handle, retention_handle, auroc_handle],
        ["Prevention", "Retention", "Macro-AUROC"],
        loc="lower right",
        ncol=1,
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

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.55))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.10, top=0.94, wspace=0.36, hspace=0.42)

    ax = axes[0, 0]
    scatter = ax.scatter(
        repeated.threshold,
        repeated.auroc,
        c=repeated.seed,
        cmap="viridis",
        s=17,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    ax.axvline(repeated.threshold.median(), color=GRAY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Selected threshold")
    ax.set_ylabel("Out-of-fold AUROC")
    ax.set_title("Repeated grouped calibration splits")
    ax.grid(color="#E8E8E8", linewidth=0.6)
    ax.text(
        0.02,
        0.03,
        "Color identifies the fitted-pair seed",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=6.0,
    )
    panel_label(ax, "a")

    ax = axes[0, 1]
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
            fontsize=6.1,
        )
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.6)
    panel_label(ax, "b")

    ax = axes[1, 0]
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

    ax = axes[1, 1]
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
    ax.bar(x - width / 2, agree_nll, width, color=LIGHT, edgecolor="#666666", label="Class agreement")
    ax.bar(x + width / 2, disagree_nll, width, color="#AFC9DC", edgecolor="#666666", label="Class disagreement")
    ax.set_xticks(x, ["PDRF", "Lite", "Lite-CF"])
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Probability quality by endpoint agreement")
    ax.set_ylim(0, 3.35)
    nll_legend = ax.legend(loc="upper left")
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
    secondary.legend(
        [agree_ece_handle, disagree_ece_handle],
        ["ECE: agreement", "ECE: disagreement"],
        loc="lower right",
    )
    ax.add_artist(nll_legend)
    panel_label(ax, "d")

    save_figure(fig, "figure_cee_q1_stability_calibration")
    plt.close(fig)


def main():
    tradeoff_figure()
    stability_calibration_figure()


if __name__ == "__main__":
    main()
