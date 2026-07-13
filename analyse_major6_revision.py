"""Analyse and visualize the strong-baseline, teacher and calibration controls."""
from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FIG = ROOT / "figures"
TAB = ROOT / "tables"

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 8,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.linewidth": .65,
})


def load_long():
    m3 = pd.read_csv(SRC / "major3_objective_matched_metrics.csv")
    m3 = m3[(m3.dataset == "chemical") &
            (m3.environment == "type:silent_gaussian")].copy()
    m5 = pd.read_csv(SRC / "major5_consistency_metrics.csv")
    m5 = m5[(m5.fault_realization == 70001) &
            (m5.batch.astype(str) == "all")].copy()
    m6 = pd.read_csv(SRC / "major6_strong_control_metrics.csv")
    ema = pd.read_csv(SRC / "major6_ema_metrics.csv")
    uncertainty = pd.read_csv(SRC / "uncertainty_baseline_metrics.csv")
    keep = ["method", "seed", "subset", "metric", "value"]
    return pd.concat([m3[keep], m5[keep], m6[keep], ema[keep],
                      uncertainty[keep]], ignore_index=True).drop_duplicates(
                          keep, keep="last")


def mean_ci(values):
    x = np.asarray(values, float)
    sem = x.std(ddof=1) / np.sqrt(len(x))
    half = t.ppf(.975, len(x) - 1) * sem
    return x.mean(), x.mean() - half, x.mean() + half


def paired_test(df, method, reference, metric, lower_better=False):
    q = df[(df.subset == "affected") & (df.metric == metric) &
           df.method.isin([method, reference])]
    p = q.pivot_table(index="seed", columns="method", values="value")
    delta = p[method] - p[reference]
    aligned = -delta if lower_better else delta
    return {
        "method": method, "reference": reference, "metric": metric,
        "mean_raw_difference": delta.mean(),
        "favourable_seeds": int((aligned > 0).sum()), "n_seeds": len(delta),
        "wilcoxon_two_sided_p": wilcoxon(aligned).pvalue,
    }


def build_outputs():
    df = load_long()
    methods = ["ENT-PD", "ModDrop-SD", "RO-AT-GATE", "RO-CAGF",
               "PDRF+CD", "PDRF+EMA-CD", "RO-PDRF",
               "RO-PDRF-EMA", "RO-PDRF-CAL"]
    metrics = ["macro_auroc", "macro_auprc", "nll", "brier", "ece15"]
    summary_rows = []
    for method in methods:
        for metric in metrics:
            x = df[(df.method == method) & (df.subset == "affected") &
                   (df.metric == metric)].value
            if len(x):
                mean, lo, hi = mean_ci(x)
                summary_rows.append({"method": method, "metric": metric,
                                     "mean": mean, "ci_low": lo,
                                     "ci_high": hi, "n_seeds": len(x)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SRC / "major6_control_summary.csv", index=False)

    tests = [
        paired_test(df, "PDRF+EMA-CD", "PDRF+CD", "macro_auroc"),
        paired_test(df, "PDRF+Soft-CD", "PDRF+CD", "macro_auroc"),
        paired_test(df, "PDRF+CD-T07", "PDRF+CD", "macro_auroc"),
        paired_test(df, "RO-PDRF-EMA", "RO-PDRF", "macro_auroc"),
        paired_test(df, "RO-PDRF-EMA", "RO-PDRF", "nll", True),
        paired_test(df, "RO-PDRF-EMA", "RO-PDRF", "brier", True),
        paired_test(df, "RO-PDRF-EMA", "RO-PDRF", "ece15", True),
        paired_test(df, "RO-PDRF-CAL", "RO-PDRF", "macro_auroc"),
        paired_test(df, "RO-PDRF-CAL", "RO-PDRF", "nll", True),
    ]
    pd.DataFrame(tests).to_csv(SRC / "major6_paired_tests.csv", index=False)

    wide = summary.pivot(index="method", columns="metric", values="mean")
    costs = pd.concat([
        pd.read_csv(SRC / "major6_strong_control_costs.csv"),
        pd.read_csv(SRC / "major6_ema_costs.csv")], ignore_index=True)
    cost_mean = costs.groupby("method").agg(
        parameters=("parameters", "mean"), train_seconds=("train_seconds", "mean"))
    table_methods = ["RO-AT-GATE", "PDRF+CD", "PDRF+EMA-CD",
                     "RO-PDRF", "RO-PDRF-EMA", "RO-PDRF-CAL"]
    # RO-PDRF cost is identical in parameter count to the other PDRF variants;
    # its historical timing is not mixed with the new same-run timing audit.
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & Params & Aff. AUROC & Aff. AUPRC & Aff. NLL & Aff. Brier & Aff. ECE \\",
             r"\midrule"]
    for method in table_methods:
        params = int(cost_mean.loc[method, "parameters"]) if method in cost_mean.index else 19740
        row = wide.loc[method]
        lines.append(f"{method} & {params:,} & {row.macro_auroc:.3f} & "
                     f"{row.macro_auprc:.3f} & {row.nll:.3f} & "
                     f"{row.brier:.3f} & {row.ece15:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "major6_strong_controls.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot(df)
    (SRC / "major6_revision_summary.json").write_text(json.dumps({
        "ro_pdrf_ema_vs_ro_pdrf": {
            "affected_auroc_difference": float(
                wide.loc["RO-PDRF-EMA", "macro_auroc"] - wide.loc["RO-PDRF", "macro_auroc"]),
            "affected_nll_difference": float(
                wide.loc["RO-PDRF-EMA", "nll"] - wide.loc["RO-PDRF", "nll"]),
            "affected_ece_difference": float(
                wide.loc["RO-PDRF-EMA", "ece15"] - wide.loc["RO-PDRF", "ece15"]),
        },
        "interpretation": "EMA is an exploratory principal-fault remedy; cross-platform claims remain based on the prespecified non-EMA model."
    }, indent=2), encoding="utf-8")


def jitter(n, width=.055):
    return np.linspace(-width, width, n)


def panel_points(ax, df, methods, metric, ylabel, ylim=None):
    colors = ["#6B7280", "#D97706", "#7C3AED", "#2563EB", "#059669", "#DC2626"]
    for i, (method, color) in enumerate(zip(methods, colors)):
        x = df[(df.method == method) & (df.subset == "affected") &
               (df.metric == metric)].sort_values("seed").value.to_numpy()
        ax.scatter(i + jitter(len(x)), x, s=9, facecolors="white",
                   edgecolors=color, linewidths=.65, zorder=2)
        mean, lo, hi = mean_ci(x)
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                    ms=3.1, color=color, capsize=2, lw=1.0, zorder=3)
    ax.set_xticks(range(len(methods)), [m.replace("RO-", "RO-\n") for m in methods], rotation=0)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#E5E7EB", lw=.55)
    ax.spines[["top", "right"]].set_visible(False)


def plot(df):
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=True)
    panel_points(axes[0, 0], df,
                 ["ENT-PD", "ModDrop-SD", "RO-AT-GATE", "RO-CAGF", "RO-PDRF", "RO-PDRF-EMA"],
                 "macro_auroc", "Affected macro-AUROC", (.67, .85))
    axes[0, 0].set_title("Strong corruption-aware fusion controls", loc="left", fontweight="bold")

    comparisons = [("PDRF+CD-T07", "PDRF+CD", "Threshold 0.70"),
                   ("PDRF+Soft-CD", "PDRF+CD", "Soft confidence"),
                   ("PDRF+EMA-CD", "PDRF+CD", "EMA teacher"),
                   ("RO-PDRF-EMA", "RO-PDRF", "EMA + full")]
    ax = axes[0, 1]
    colors = ["#D97706", "#7C3AED", "#2563EB", "#059669"]
    for i, ((method, ref, label), color) in enumerate(zip(comparisons, colors)):
        q = df[(df.subset == "affected") & (df.metric == "macro_auroc") &
               df.method.isin([method, ref])]
        p = q.pivot_table(index="seed", columns="method", values="value")
        delta = (p[method] - p[ref]).to_numpy()
        ax.scatter(i + jitter(len(delta)), delta, s=9, facecolors="white",
                   edgecolors=color, linewidths=.65)
        mean, lo, hi = mean_ci(delta)
        ax.errorbar(i, mean, yerr=[[mean-lo], [hi-mean]], fmt="o", ms=3.1,
                    color=color, capsize=2, lw=1)
    ax.axhline(0, color="#374151", lw=.7, ls="--")
    ax.set_xticks(range(4), [x[2] for x in comparisons])
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel(r"Paired $\Delta$ affected AUROC")
    ax.set_title("Teacher-quality controls", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", lw=.55)
    ax.spines[["top", "right"]].set_visible(False)

    probability_methods = ["RO-CAGF", "RO-AT-GATE", "RO-PDRF", "RO-PDRF-EMA", "RO-PDRF-CAL"]
    panel_points(axes[1, 0], df, probability_methods, "nll", "Affected NLL", (1.2, 3.45))
    axes[1, 0].set_title("Temperature-scaled probability quality", loc="left", fontweight="bold")
    panel_points(axes[1, 1], df, probability_methods, "ece15", "Affected ECE (15 bins)", (.15, .42))
    axes[1, 1].set_title("Calibration after scaling", loc="left", fontweight="bold")

    for label, ax in zip("abcd", axes.ravel()):
        ax.text(-.15, 1.07, label, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
    fig.suptitle("Modern baselines and teacher/calibration controls", fontsize=9, fontweight="bold")
    base = FIG / "figureS7_strong_teacher_controls"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    build_outputs()
