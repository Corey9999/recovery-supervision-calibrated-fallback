"""Generate submission figures from the experiment source data (Python only)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

METHODS = ["EF", "MD", "UF", "GF", "CRF"]
LABELS = {
    "EF": "Early fusion", "MD": "Modality dropout", "UF": "Uniform fusion",
    "GF": "Gated fusion", "CRF": "Counterfactual reliability fusion",
}
COLORS = {
    "EF": "#484878", "MD": "#7884B4", "UF": "#B4C0E4",
    "GF": "#D4C8E0", "CRF": "#B64342",
}
MARKERS = {"EF": "o", "MD": "s", "UF": "^", "GF": "D", "CRF": "o"}


def panel(ax, label):
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")


def save(fig, name):
    for ext in ("svg", "pdf", "tiff"):
        kw = {"bbox_inches": "tight"}
        if ext == "tiff":
            kw["dpi"] = 600
        fig.savefig(FIG / f"{name}.{ext}", **kw)
    plt.close(fig)


def rounded(ax, xy, w, h, text, fc, ec="#4D4D4D", fontsize=7):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.025",
                       facecolor=fc, edgecolor=ec, linewidth=0.8)
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, a, b, color="#606060"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color=color))


def figure1():
    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.35, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel(ax, "a")
    ax.text(0.02, 0.92, "Controlled multimodal sensor benchmark", fontweight="bold", fontsize=8)
    rounded(ax, (0.02, 0.32), 0.12, 0.34, "Latent\nstate", "#E4E4F0")
    for j, y in enumerate([0.72, 0.54, 0.36, 0.18]):
        rounded(ax, (0.20, y - 0.07), 0.13, 0.12, f"Sensor {j+1}\nquality $q_{j+1}$", "#E0F0F0")
        arrow(ax, (0.14, 0.49), (0.20, y))
        rounded(ax, (0.38, y - 0.07), 0.12, 0.12, f"Encoder {j+1}", "#B4C0E4")
        arrow(ax, (0.33, y), (0.38, y))
    rounded(ax, (0.57, 0.37), 0.16, 0.24, "Quality × learned\nprecision weights", "#F6CFCB")
    for y in [0.72, 0.54, 0.36, 0.18]:
        arrow(ax, (0.50, y), (0.57, 0.49))
    rounded(ax, (0.80, 0.39), 0.16, 0.20, "Calibrated\nprediction", "#E9A6A1")
    arrow(ax, (0.73, 0.49), (0.80, 0.49), "#B64342")
    rounded(ax, (0.56, 0.78), 0.20, 0.10, "Counterfactual degradation + ranking", "#F0E0D0", fontsize=6.2)
    arrow(ax, (0.66, 0.78), (0.66, 0.62), "#B64342")
    ax.text(0.20, 0.04, "Missing sensors are masked; noisy sensors retain an observed diagnostic quality channel.",
            fontsize=6.5, color="#4D4D4D")

    titles = ["Missing-not-at-random", "Sensor noise shift", "Gross outliers", "Silent failure"]
    subtitles = ["quality-dependent dropout", "dominant sensor degraded",
                 "35% heavy-tailed corruption", "diagnostic remains optimistic"]
    for i in range(4):
        a = fig.add_subplot(gs[1, i])
        panel(a, chr(ord("b") + i))
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
        rounded(a, (0.05, 0.35), 0.90, 0.34, titles[i],
                ["#E0E0F0", "#E0F0F0", "#F0E0D0", "#F6CFCB"][i], fontsize=7.2)
        a.text(0.5, 0.22, subtitles[i], ha="center", fontsize=6.3, color="#606060")
    save(fig, "figure1_workflow")


def figure2():
    s = pd.read_csv(DATA / "results_summary.csv")
    regimes = ["ID_missing", "noise_shift", "outlier_shift", "silent_failure"]
    rlabels = ["ID missing", "Reported shift", "Outliers", "Silent failure"]
    metrics = [("auroc", "AUROC ↑"), ("nll", "NLL ↓"), ("ece", "ECE ↓"), ("aurc", "AURC ↓")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    x = np.arange(len(regimes)); width = 0.15
    for idx, (ax, (metric, ylabel)) in enumerate(zip(axes.ravel(), metrics)):
        for j, method in enumerate(METHODS):
            q = s[(s.model == method) & (s.metric == metric)].set_index("regime").loc[regimes]
            ax.bar(x + (j - 2) * width, q["mean"], width=width, color=COLORS[method],
                   edgecolor="#333333", linewidth=0.35, yerr=q["ci95"], capsize=1.6,
                   label=LABELS[method])
        ax.set_xticks(x); ax.set_xticklabels(rlabels)
        ax.set_ylabel(ylabel); ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        panel(ax, chr(ord("a") + idx))
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01), fontsize=6.4)
    fig.subplots_adjust(top=0.88, hspace=0.42, wspace=0.32)
    save(fig, "figure2_main_performance")


def figure3():
    s = pd.read_csv(DATA / "stress_results_summary.csv")
    specs = [
        ("missingness", "auroc", "Missingness multiplier", "AUROC ↑"),
        ("missingness", "ece", "Missingness multiplier", "ECE ↓"),
        ("sensor_noise", "auroc", "Extra noise s.d.", "AUROC ↑"),
        ("sensor_noise", "ece", "Extra noise s.d.", "ECE ↓"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for idx, (ax, spec) in enumerate(zip(axes.ravel(), specs)):
        stress, metric, xlabel, ylabel = spec
        for method in METHODS:
            q = s[(s.model == method) & (s.stress_type == stress) & (s.metric == metric)].sort_values("level")
            ax.plot(q.level, q["mean"], marker=MARKERS[method], ms=3.4, lw=1.2,
                    color=COLORS[method], label=LABELS[method])
            ax.fill_between(q.level, q["mean"] - q.ci95, q["mean"] + q.ci95,
                            color=COLORS[method], alpha=0.10, linewidth=0)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(color="#E5E5E5", linewidth=0.6)
        panel(ax, chr(ord("a") + idx))
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01), fontsize=6.4)
    fig.subplots_adjust(top=0.87, hspace=0.40, wspace=0.32)
    save(fig, "figure3_stress_tests")


def figure4():
    d = pd.read_csv(DATA / "crf_example_predictions.csv")
    audit = pd.read_csv(DATA / "counterfactual_audit.csv")
    d["predictive_uncertainty"] = 1 - np.maximum(d.probability, 1 - d.probability)
    d["error"] = ((d.probability >= 0.5).astype(int) != d.y.astype(int)).astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axes = axes.ravel()

    ax = axes[0]
    audit_methods = ("LWF", "RWF", "CFA", "CRF_NR", "CRF")
    groups = [audit[audit.model == m].delta_log_variance.values for m in audit_methods]
    bp = ax.boxplot(groups, tick_labels=["Learned\nonly", "Factorized", "CF-Aug", "No\nranking", "Full CRF"],
                    showfliers=False, patch_artist=True, widths=0.62)
    for patch, color in zip(bp["boxes"], ["#B4C0E4", "#D4C8E0", "#E9A6A1", "#D78C86", "#B64342"]):
        patch.set_facecolor(color); patch.set_edgecolor("#333333")
    ax.axhline(0, color="#767676", linestyle="--", lw=0.9)
    ax.set_ylabel("Change in log variance after\nsilent degradation")
    panel(ax, "a")

    ax = axes[1]
    bins = pd.cut(d.probability, np.linspace(0, 1, 11), include_lowest=True, labels=False)
    cal = d.assign(bin=bins).groupby("bin", as_index=False).agg(
        predicted=("probability", "mean"), observed=("y", "mean"), n=("y", "size"))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8F8F8F", lw=1, label="Ideal")
    ax.plot(cal.predicted, cal.observed, color=COLORS["CRF"], marker="o", lw=1.4,
            ms=3.5, label="CRF")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event frequency")
    ax.legend(loc="upper left", fontsize=6)
    panel(ax, "b")

    ax = axes[2]
    d["uq"] = pd.qcut(d.predictive_uncertainty, 5, labels=False, duplicates="drop")
    q = d.groupby("uq", as_index=False).agg(error=("error", "mean"), n=("error", "size"))
    ax.bar(q.uq + 1, q.error, color=["#B4C0E4", "#A8B6D8", "#E4CCD8", "#E9A6A1", "#B64342"][:len(q)],
           edgecolor="#333333", linewidth=0.4)
    ax.set_xlabel("Uncertainty quintile (low → high)")
    ax.set_ylabel("Error rate")
    panel(ax, "c")

    ax = axes[3]
    vals, ns = [], []
    levels = sorted(d.available_modalities.unique())
    for level in levels:
        z = d[d.available_modalities == level]
        vals.append(roc_auc_score(z.y, z.probability) if z.y.nunique() == 2 else np.nan)
        ns.append(len(z))
    ax.plot(levels, vals, color=COLORS["CRF"], marker="o", lw=1.5)
    for x, y, n in zip(levels, vals, ns):
        ax.text(x, y + 0.008, f"n={n}", ha="center", fontsize=5.8)
    ax.set_xticks(levels)
    ax.set_xlabel("Available modalities")
    ax.set_ylabel("AUROC ↑")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    panel(ax, "d")
    fig.subplots_adjust(wspace=0.38, hspace=0.42)
    save(fig, "figure4_uncertainty_analysis")


def figure5():
    """Longitudinal real-data evaluation on UCI gas-sensor measurements."""
    s = pd.read_csv(DATA / "uci_results_summary.csv")
    tests = pd.read_csv(DATA / "uci_paired_tests.csv")
    regimes = ["complete_drift", "missing_drift", "reported_fault", "silent_fault"]
    rlabels = ["Natural\ndrift", "Drift +\nmissingness", "Reported\nfault", "Silent\nfault"]
    methods = ["EF", "MD", "UF", "GF", "RWF", "CRF"]
    labels = {**LABELS, "RWF": "Factorized reliability"}
    colors = {**COLORS, "RWF": "#D78C86"}
    metrics = [("macro_auroc", "Macro-AUROC ↑"), ("nll", "NLL ↓"), ("aurc", "AURC ↓")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    x = np.arange(len(regimes)); width = 0.125
    for idx, (ax, (metric, ylabel)) in enumerate(zip(axes.ravel()[:3], metrics)):
        for j, method in enumerate(methods):
            q = s[(s.model == method) & (s.metric == metric)].set_index("regime").loc[regimes]
            ax.bar(x + (j - 2.5) * width, q["mean"], width=width, color=colors[method],
                   edgecolor="#333333", linewidth=0.3, yerr=q["ci95"], capsize=1.2,
                   label=labels[method])
        ax.set_xticks(x); ax.set_xticklabels(rlabels)
        ax.set_ylabel(ylabel); ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
        panel(ax, chr(ord("a") + idx))

    ax = axes.ravel()[3]
    q = tests[tests.metric == "macro_auroc"].set_index("regime").loc[regimes]
    gains = q.aligned_improvement.to_numpy()
    cis = q.improvement_ci95.to_numpy()
    bar_colors = ["#7884B4" if v < 0 else "#B64342" for v in gains]
    ax.bar(x, gains, yerr=cis, color=bar_colors, edgecolor="#333333", linewidth=0.4, capsize=2)
    ax.axhline(0, color="#606060", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(rlabels)
    ax.set_ylabel("CRF − uniform fusion\nmacro-AUROC")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    panel(ax, "d")

    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.01), fontsize=6.1)
    fig.subplots_adjust(top=0.87, hspace=0.45, wspace=0.34)
    save(fig, "figure5_uci_validation")


def figure6():
    """Strict ablations and robustness geometry on real sensor measurements."""
    s = pd.read_csv(DATA / "uci_robustness_summary.csv")
    audit = pd.read_csv(DATA / "uci_counterfactual_audit.csv")
    methods = ["UF", "RWF", "CFA", "CRF_NR", "CRF"]
    labels = {"UF": "Uniform fusion", "RWF": "No intervention", "CFA": "CF-Aug",
              "CRF_NR": "CRF without ranking", "CRF": "Full CRF"}
    colors = {"UF": "#B4C0E4", "RWF": "#D4C8E0", "CFA": "#E9A6A1",
              "CRF_NR": "#D78C86", "CRF": "#B64342"}
    markers = {"UF": "^", "RWF": "D", "CFA": "s", "CRF_NR": "P", "CRF": "o"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))

    specs = [
        ("severity", "Fault noise s.d.", None),
        ("fraction", "Affected observations", lambda x: f"{int(100*x)}%"),
        ("fault_modality", "Faulted modality", lambda x: str(int(x) + 1)),
        ("reported_quality", "Reported quality after fault", None),
    ]
    for idx, (ax, (stress, xlabel, formatter)) in enumerate(zip(axes.ravel()[:4], specs)):
        for method in methods:
            q = s[(s.model == method) & (s.stress_type == stress) &
                  (s.metric == "macro_auroc")].sort_values("level")
            ax.plot(q.level, q["mean"], color=colors[method], marker=markers[method],
                    ms=3.2, lw=1.2, label=labels[method])
            ax.fill_between(q.level, q["mean"] - q.ci95, q["mean"] + q.ci95,
                            color=colors[method], alpha=0.09, linewidth=0)
        levels = sorted(s[s.stress_type == stress].level.unique())
        ax.set_xticks(levels)
        if formatter:
            ax.set_xticklabels([formatter(v) for v in levels])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Macro-AUROC ↑")
        ax.grid(color="#E5E5E5", linewidth=0.6)
        panel(ax, chr(ord("a") + idx))
    axes.ravel()[3].annotate("diagnostic becomes stale →", xy=(0.98, 0.04), xycoords="axes fraction",
                             ha="right", va="bottom", fontsize=6, color="#606060")

    mechanism_methods = ["RWF", "CFA", "CRF_NR", "CRF"]
    x = np.arange(len(mechanism_methods))
    means = audit.groupby("model").mean(numeric_only=True)
    cis = audit.groupby("model").std(numeric_only=True) * 1.96 / np.sqrt(
        audit.groupby("model").size())

    ax = axes.ravel()[4]
    vals = [means.loc[m, "delta_clipped_log_variance"] for m in mechanism_methods]
    errs = [cis.loc[m, "delta_clipped_log_variance"] for m in mechanism_methods]
    ax.bar(x, vals, yerr=errs, color=[colors[m] for m in mechanism_methods],
           edgecolor="#333333", linewidth=0.4, capsize=2)
    ax.axhline(0, color="#606060", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["No\ninterv.", "CF-Aug", "No\nranking", "Full CRF"])
    ax.set_ylabel("Δ clipped log variance ↑")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    panel(ax, "e")

    ax = axes.ravel()[5]
    vals = [means.loc[m, "delta_normalized_weight"] for m in mechanism_methods]
    errs = [cis.loc[m, "delta_normalized_weight"] for m in mechanism_methods]
    ax.bar(x, vals, yerr=errs, color=[colors[m] for m in mechanism_methods],
           edgecolor="#333333", linewidth=0.4, capsize=2, label="Δ normalized weight")
    ax.axhline(0, color="#606060", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["No\ninterv.", "CF-Aug", "No\nranking", "Full CRF"])
    ax.set_ylabel("Δ faulted-modality weight ↓")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax2 = ax.twinx()
    sat = [100 * means.loc[m, "post_upper_clip_fraction"] for m in mechanism_methods]
    ax2.plot(x, sat, color="#202020", marker="o", ms=3.3, lw=1.0, label="Post-fault saturation")
    ax2.set_ylabel("At upper variance clip (%)")
    ax2.set_ylim(0, 100)
    panel(ax, "f")

    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01), fontsize=6.2)
    fig.subplots_adjust(top=0.87, hspace=0.44, wspace=0.42)
    save(fig, "figure6_robustness_mechanism")


def figure7():
    """Rank-loss sensitivity and clean--fault trade-off."""
    s = pd.read_csv(DATA / "uci_rank_sensitivity_summary.csv")
    audit = pd.read_csv(DATA / "uci_rank_sensitivity_audit.csv")
    weights = sorted(s.rank_weight.unique())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    regimes = [("complete_drift", "Natural drift", "#7884B4", "o"),
               ("silent_fault", "Silent fault", "#B64342", "s")]
    for idx, metric in enumerate(("macro_auroc", "nll")):
        ax = axes[idx]
        for regime, label, color, marker in regimes:
            q = s[(s.regime == regime) & (s.metric == metric)].sort_values("rank_weight")
            ax.plot(q.rank_weight, q["mean"], color=color, marker=marker, ms=3.5, lw=1.3,
                    label=label)
            ax.fill_between(q.rank_weight, q["mean"] - q.ci95, q["mean"] + q.ci95,
                            color=color, alpha=0.12, linewidth=0)
        ax.axvline(0.15, color="#777777", linestyle=":", linewidth=0.9)
        ax.set_xticks(weights)
        ax.set_xticklabels(["0", ".03", ".07", ".15", ".30"], rotation=30, ha="right")
        ax.set_xlabel(r"Ranking weight $\lambda_{rank}$")
        ax.set_ylabel("Macro-AUROC ↑" if metric == "macro_auroc" else "NLL ↓")
        ax.grid(color="#E5E5E5", linewidth=0.6)
        panel(ax, chr(ord("a") + idx))
    axes[0].legend(loc="lower right", fontsize=6.2)
    axes[1].text(0.15, axes[1].get_ylim()[1], " main", fontsize=6, color="#606060", va="top")

    ax = axes[2]
    agg = audit.groupby("rank_weight").agg(
        clip=("delta_clipped_log_variance", "mean"),
        clip_sd=("delta_clipped_log_variance", "std"),
        sat=("post_upper_clip_fraction", "mean"),
        sat_sd=("post_upper_clip_fraction", "std"),
        n=("seed", "size")).reset_index()
    clip_ci = 1.96 * agg["clip_sd"] / np.sqrt(agg["n"])
    sat_ci = 100 * 1.96 * agg["sat_sd"] / np.sqrt(agg["n"])
    ax.errorbar(agg["rank_weight"], agg["clip"], yerr=clip_ci, color="#B64342", marker="o",
                ms=3.5, lw=1.3, capsize=2, label="Δ clipped log variance")
    ax.set_xticks(weights)
    ax.set_xticklabels(["0", ".03", ".07", ".15", ".30"], rotation=30, ha="right")
    ax.set_xlabel(r"Ranking weight $\lambda_{rank}$")
    ax.set_ylabel("Δ clipped log variance ↑", color="#B64342")
    ax.tick_params(axis="y", labelcolor="#B64342")
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.axvline(0.15, color="#777777", linestyle=":", linewidth=0.9)
    ax2 = ax.twinx()
    ax2.errorbar(agg["rank_weight"], 100 * agg["sat"], yerr=sat_ci, color="#202020", marker="s",
                 ms=3.2, lw=1.1, capsize=2, label="Post-fault saturation")
    ax2.set_ylabel("At upper variance clip (%)")
    ax2.set_ylim(0, 100)
    panel(ax, "c")
    fig.subplots_adjust(wspace=0.48)
    save(fig, "figure7_rank_sensitivity")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    print("Figures written to", FIG)
