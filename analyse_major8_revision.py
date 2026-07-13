"""Analyse the manuscript-17 QMF-fairness and teacher-error experiments.

The outputs in this file are descriptive repeated-fit audits on one chemical
instrument.  No test metric is used for model or hyperparameter selection.
"""
from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyse_major7_revision import two_level_interval, three_level_interval


ROOT = Path(__file__).resolve().parent
SRC, TAB, FIG = ROOT / "source_data", ROOT / "tables", ROOT / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.2,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

BLUE = "#4C78A8"
GREEN = "#2F8F6B"
ORANGE = "#D98256"
PURPLE = "#7A68A6"
GREY = "#777777"
RED = "#C94C4C"
TEAL = "#1B9E77"
DISPLAY_METHOD = {
    "RO-PDRF-CEG": "CE-conflict training audit",
    "RO-PDRF-CRA": "RO-PDRF-Full-CRA",
    "RO-PDRF-ECA": "RO-PDRF-Full-ECA",
}
ROLE = {
    "QMF-PD": "Published-method adaptation",
    "RO-AT-GATE": "Strong learned baseline",
    "RO-CAGF": "Architecture-matched baseline",
    "RO-PDRF-Lite": "Practical default",
    "RO-PDRF-Full": "Mechanistic full model",
    "RO-PDRF-EMA": "Calibration sensitivity",
    "RO-PDRF-CRA": "Diagnostic audit",
    "RO-PDRF-ECA": "Diagnostic audit",
    "RO-PDRF-CEG": "Diagnostic audit",
}


def save(fig, name):
    fig.tight_layout(pad=0.9)
    for ext, kwargs in (("svg", {}), ("pdf", {}),
                        ("png", {"dpi": 300}), ("tiff", {"dpi": 600})):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def metric_mean(frame, method, subset, metric):
    z = frame[(frame.method == method) & (frame.subset == subset) &
              (frame.metric == metric)].value
    if z.empty:
        raise ValueError(f"Missing {method}/{subset}/{metric}")
    return float(z.mean()), float(z.std(ddof=1)), int(z.size)


def qmf_audit():
    sensitivity = pd.read_csv(SRC / "major8_qmf_sensitivity_metrics.csv")
    extension = pd.read_csv(SRC / "major8_qmf_faithful_extension_metrics.csv")
    faithful = pd.concat([
        sensitivity[sensitivity.method == "QMF-PD-faithful"], extension
    ], ignore_index=True)
    faithful["subset"] = faithful["subset"].replace({
        "fault_all": "all", "fault_affected": "affected",
        "fault_unaffected": "unaffected"})
    sensitivity["subset"] = sensitivity["subset"].replace({
        "fault_all": "all", "fault_affected": "affected",
        "fault_unaffected": "unaffected"})
    all_metrics = pd.concat([
        sensitivity[sensitivity.method != "QMF-PD-faithful"], faithful
    ], ignore_index=True)

    costs = pd.concat([
        pd.read_csv(SRC / "major8_qmf_sensitivity_costs.csv"),
        pd.read_csv(SRC / "major8_qmf_faithful_extension_costs.csv")
    ], ignore_index=True)
    costs = costs.drop_duplicates(["method", "seed"], keep="last")

    order = [
        "QMF-PD-current", "QMF-PD-scale003", "QMF-PD-scale030",
        "QMF-PD-normalized", "QMF-PD-rank0", "QMF-PD-rank1",
        "QMF-PD-faithful"
    ]
    labels = {
        "QMF-PD-current": "Previous adaptation",
        "QMF-PD-scale003": "Confidence scale 0.03",
        "QMF-PD-scale030": "Confidence scale 0.30",
        "QMF-PD-normalized": "Normalized fusion",
        "QMF-PD-rank0": "Ranking weight 0",
        "QMF-PD-rank1": "Ranking weight 1",
        "QMF-PD-faithful": "Official-rule-aligned adaptation",
    }
    rows = []
    for method in order:
        row = {"method": method, "setting": labels[method]}
        for subset, metric, key in (
                ("clean_matched_mask", "macro_auroc", "clean_auroc"),
                ("affected", "macro_auroc", "affected_auroc"),
                ("affected", "nll", "affected_nll"),
                ("affected", "ece15", "affected_ece")):
            mean, sd, n = metric_mean(all_metrics, method, subset, metric)
            row[key], row[key + "_sd"], row["seeds"] = mean, sd, n
        row["train_seconds"] = float(
            costs[costs.method == method].train_seconds.mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(SRC / "major8_qmf_sensitivity_summary.csv", index=False)

    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"QMF sensor-group setting & Seeds & Clean AUROC & Affected AUROC & Affected NLL & Affected ECE \\",
             r"\midrule"]
    for _, x in out.iterrows():
        lines.append(
            f"{x.setting} & {int(x.seeds)} & {x.clean_auroc:.3f} & "
            f"{x.affected_auroc:.3f} & {x.affected_nll:.3f} & "
            f"{x.affected_ece:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "major8_qmf_sensitivity.tex").write_text(
        "\n".join(lines), encoding="utf-8")

    rows = [
        ("Original QMF", "Text--image; RGB--D", r"$\operatorname{LSE}(\boldsymbol\mu)/10$",
         "Unnormalized; confidence detached", "Cumulative CE history; normalized margin; weight 1"),
        ("Primary adaptation", "Four chemical sensor groups",
         r"$\max(10^{-6},\bar c)$", "Unnormalized; confidence detached",
         "Cumulative CE history; normalized margin; weight 1"),
        ("Sensitivity variants", "Same groups and paired fault",
         "Scales 0.03, 0.10 or 0.30", "Normalized or unnormalized",
         "Weights 0, 0.10 or 1; instantaneous or cumulative history"),
    ]
    pd.DataFrame(rows, columns=["implementation", "task", "confidence",
                                "fusion", "ranking"]).to_csv(
        SRC / "major8_qmf_implementation_audit.csv", index=False)
    tex = [r"\begin{tabular}{p{0.14\textwidth}p{0.13\textwidth}p{0.18\textwidth}p{0.16\textwidth}p{0.21\textwidth}}",
           r"\toprule",
           r"Version & Task/input & Confidence & Fusion & Ranking objective \\",
           r"\midrule"]
    for r in rows:
        tex.append(" & ".join(r) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "major8_qmf_implementation.tex").write_text(
        "\n".join(tex), encoding="utf-8")
    return faithful, out


def principal_methods(qmf_faithful):
    head = pd.read_csv(SRC / "major3_objective_matched_metrics.csv")
    head = head[(head.dataset == "chemical") &
                (head.environment == "type:silent_gaussian")]
    lite = pd.read_csv(SRC / "major5_consistency_metrics.csv")
    lite = lite[(lite.fault_realization == 70001) &
                (lite.batch.astype(str) == "all")]
    strong = pd.read_csv(SRC / "major6_strong_control_metrics.csv")
    ema = pd.read_csv(SRC / "major6_ema_metrics.csv")
    ceg = pd.read_csv(SRC / "major8_ce_conflict_metrics.csv")
    agreement = pd.read_csv(SRC / "major7_teacher_agreement_metrics.csv")

    sources = {
        "RO-CAGF": (head, "RO-CAGF"),
        "RO-PDRF-Lite": (lite, "PDRF+CD"),
        "RO-PDRF-Full": (head, "RO-PDRF"),
        "RO-AT-GATE": (strong, "RO-AT-GATE"),
        "RO-PDRF-EMA": (ema, "RO-PDRF-EMA"),
        "QMF-PD": (qmf_faithful, "QMF-PD-faithful"),
        "RO-PDRF-CEG": (ceg, "RO-PDRF-CEG"),
        "RO-PDRF-CRA": (agreement, "RO-PDRF-CRA"),
        "RO-PDRF-ECA": (agreement, "RO-PDRF-ECA"),
    }
    rows = []
    for label, (frame, raw_method) in sources.items():
        row = {"method": label}
        for metric in ("macro_auroc", "nll", "brier", "ece15"):
            mean, sd, n = metric_mean(frame, raw_method, "affected", metric)
            row[metric], row[metric + "_sd"], row["seeds"] = mean, sd, n
        rows.append(row)
    out = pd.DataFrame(rows)

    cost_sources = []
    for file in ("major3_objective_matched_costs.csv",
                 "major5_consistency_costs.csv",
                 "major6_strong_control_costs.csv",
                 "major6_ema_costs.csv",
                 "major7_teacher_agreement_costs.csv",
                 "major8_qmf_sensitivity_costs.csv",
                 "major8_qmf_faithful_extension_costs.csv",
                 "major8_ce_conflict_costs.csv"):
        x = pd.read_csv(SRC / file)
        if "dataset" in x.columns:
            x = x[x.dataset == "chemical"]
        cost_sources.append(x)
    costs = pd.concat(cost_sources, ignore_index=True)
    costs = costs.drop_duplicates(["method", "seed"], keep="last")
    raw_names = {
        "RO-CAGF": "RO-CAGF", "RO-PDRF-Lite": "PDRF+CD",
        "RO-PDRF-Full": "RO-PDRF", "RO-AT-GATE": "RO-AT-GATE",
        "RO-PDRF-EMA": "RO-PDRF-EMA", "QMF-PD": "QMF-PD-faithful",
        "RO-PDRF-CEG": "RO-PDRF-CEG", "RO-PDRF-CRA": "RO-PDRF-CRA",
        "RO-PDRF-ECA": "RO-PDRF-ECA"
    }
    for i, row in out.iterrows():
        z = costs[costs.method == raw_names[row.method]]
        out.loc[i, "parameters"] = z.parameters.iloc[0]
        out.loc[i, "train_seconds"] = z.train_seconds.mean()
        out.loc[i, "train_seconds_sd"] = z.train_seconds.std(ddof=1)
        inference = z["inference_ms_per_observation"].dropna()
        if inference.empty:
            inference = z["fault_inference_ms_per_observation"].dropna()
        out.loc[i, "inference_ms"] = inference.mean()
    out.to_csv(SRC / "major8_principal_method_summary.csv", index=False)

    order = ["QMF-PD", "RO-AT-GATE", "RO-CAGF", "RO-PDRF-Lite",
             "RO-PDRF-Full", "RO-PDRF-EMA", "RO-PDRF-CRA",
             "RO-PDRF-ECA", "RO-PDRF-CEG"]
    view = out.set_index("method").loc[order].reset_index()
    lines = [r"\begin{tabular}{lp{0.17\textwidth}rrrrrr}", r"\toprule",
             r"Method & Role & Parameters & Seeds & Affected AUROC & NLL & Brier & ECE \\",
             r"\midrule"]
    for _, x in view.iterrows():
        lines.append(
            f"{DISPLAY_METHOD.get(x.method, x.method)} & {ROLE[x.method]} & "
            f"{int(x.parameters):,} & {int(x.seeds)} & "
            f"{x.macro_auroc:.3f} & {x.nll:.3f} & {x.brier:.3f} & "
            f"{x.ece15:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "major8_literature_teacher_controls.tex").write_text(
        "\n".join(lines), encoding="utf-8")

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & Parameters & Training time, mean $\pm$ SD (s) & Epochs represented & Inference (ms/observation) \\",
             r"\midrule"]
    for _, x in view.iterrows():
        raw = raw_names[x.method]
        z = costs[costs.method == raw]
        lines.append(
            f"{DISPLAY_METHOD.get(x.method, x.method)} & {int(x.parameters):,} & "
            f"{x.train_seconds:.1f} $\pm$ {x.train_seconds_sd:.1f} & "
            f"{int(z.epochs.min())}--{int(z.epochs.max())} & {x.inference_ms:.4f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "major8_training_costs.tex").write_text(
        "\n".join(lines), encoding="utf-8")
    return out


def fault_type_architecture_audit():
    raw = pd.read_csv(SRC / "major7_fault_type_ema_metrics.csv")
    z = raw[(raw.subset == "affected") & (raw.metric == "macro_auroc") &
            raw.method.isin(["RO-CAGF", "RO-PDRF"])]
    fault_order = ["gaussian", "offset", "drift", "stuck_at"]
    rows = []
    for j, fault in enumerate(fault_order):
        zz = z[z.fault_type == fault]
        effect, lo, hi = two_level_interval(
            zz, ("RO-PDRF", "RO-CAGF"), seed=16800 + j)
        means = zz.groupby("method").value.mean()
        rows.append({"fault_type": fault,
                     "ro_cagf": means["RO-CAGF"],
                     "ro_pdrf": means["RO-PDRF"],
                     "difference": effect, "ci_low": lo, "ci_high": hi})
    all_effect, all_lo, all_hi = three_level_interval(
        z, ("RO-PDRF", "RO-CAGF"), seed=16810)
    out = pd.DataFrame(rows)
    out.to_csv(SRC / "major8_fault_type_architecture_effects.csv", index=False)

    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Controlled fault type & RO-CAGF AUROC & RO-PDRF-Full AUROC & Difference (95\% hierarchical interval) \\",
           r"\midrule"]
    for _, x in out.iterrows():
        label = x.fault_type.replace("_", "-").capitalize()
        tex.append(f"{label} & {x.ro_cagf:.3f} & {x.ro_pdrf:.3f} & "
                   f"{x.difference:+.3f} [{x.ci_low:+.3f}, {x.ci_high:+.3f}]" + r" \\")
    tex.append(r"\midrule")
    tex.append(f"Four-type descriptive aggregate & -- & -- & {all_effect:+.3f} "
               f"[{all_lo:+.3f}, {all_hi:+.3f}]" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "major8_fault_type_architecture.tex").write_text(
        "\n".join(tex), encoding="utf-8")

    lofo = []
    for j, omitted in enumerate(fault_order):
        zz = z[z.fault_type != omitted]
        effect, lo, hi = three_level_interval(
            zz, ("RO-PDRF", "RO-CAGF"), seed=16820 + j)
        lofo.append({"omitted": omitted, "remaining_types": 3,
                     "difference": effect, "ci_low": lo, "ci_high": hi})
    lofo = pd.DataFrame(lofo)
    lofo.to_csv(SRC / "major8_leave_one_fault_type_out.csv", index=False)
    tex = [r"\begin{tabular}{lrr}", r"\toprule",
           r"Omitted controlled fault type & Remaining top-level clusters & RO-PDRF-Full--RO-CAGF AUROC (95\% interval) \\",
           r"\midrule"]
    for _, x in lofo.iterrows():
        tex.append(f"{x.omitted.replace('_','-').capitalize()} & 3 & "
                   f"{x.difference:+.3f} [{x.ci_low:+.3f}, {x.ci_high:+.3f}]" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "major8_leave_one_fault_type_out.tex").write_text(
        "\n".join(tex), encoding="utf-8")
    return out, lofo, (all_effect, all_lo, all_hi)


def teacher_error_audit():
    prior = pd.read_csv(SRC / "major7_teacher_conditional_rates.csv")
    old = pd.read_csv(SRC / "major5_teacher_fault_predictions.csv")
    old = old[(old.fault_realization == 70001) & old.seed.isin(range(101, 106))]
    baseline = old[old.method == "PDRF"].set_index(["seed", "sample"])
    ceg = pd.read_csv(SRC / "major8_ce_conflict_predictions.csv")
    ceg = ceg[ceg.seed.isin(range(101, 106))].set_index(["seed", "sample"])
    base = baseline.reindex(ceg.index)
    teacher_correct = ceg.clean_correct.astype(bool)
    baseline_correct = base.fault_correct.astype(bool)
    model_correct = ceg.fault_correct.astype(bool)
    recovery = teacher_correct & ~baseline_correct
    transfer = ~teacher_correct & baseline_correct
    ceg_row = {
        "teacher_rule": "CE-conflict training audit",
        "active_coverage": np.nan,
        "recovery_eligible": int(recovery.sum()),
        "recovered": int((recovery & model_correct).sum()),
        "recovery_rate": float((recovery & model_correct).sum() / recovery.sum()),
        "transfer_eligible": int(transfer.sum()),
        "lost": int((transfer & ~model_correct).sum()),
        "error_transfer_rate": float((transfer & ~model_correct).sum() / transfer.sum()),
        "heldout_label_gate_fraction": float(ceg.ce_conflict_keep.mean()),
    }
    prior["heldout_label_gate_fraction"] = np.nan
    out = pd.concat([prior, pd.DataFrame([ceg_row])], ignore_index=True)
    out.to_csv(SRC / "major8_teacher_conditional_rates.csv", index=False)

    tex = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Teacher rule or training audit & Active target coverage & Conditional recovery & Conditional error transfer \\",
           r"\midrule"]
    for _, x in out.iterrows():
        coverage = ("Training-only; not observable at test" if pd.isna(x.active_coverage)
                    else f"{100*x.active_coverage:.1f}\\%")
        tex.append(f"{x.teacher_rule} & {coverage} & "
                   f"{int(x.recovered)}/{int(x.recovery_eligible)} ({100*x.recovery_rate:.1f}\\%) & "
                   f"{int(x.lost)}/{int(x.transfer_eligible)} ({100*x.error_transfer_rate:.1f}\\%)" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "major8_teacher_safeguards.tex").write_text(
        "\n".join(tex), encoding="utf-8")
    return out


def make_figures(methods, teacher):
    # Main-figure contract:
    # Conclusion: a stronger QMF adaptation narrows the probability-quality
    # gap, but no tested method dominates ranking and calibration; teacher
    # safeguards do not eliminate conditional error transfer.
    # Evidence: mean affected AUROC/ECE and paired conditional rates.
    # Review risk: means are repeated fits on one instrument, not devices.
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.15),
                            gridspec_kw={"width_ratios": [1.18, 1.0]})
    ax = axs[0]
    methods = methods[methods.method.isin(
        ["RO-CAGF", "RO-PDRF-Lite", "RO-PDRF-Full", "RO-AT-GATE",
         "RO-PDRF-EMA", "QMF-PD", "RO-PDRF-CEG"])].copy()
    colors = {"RO-CAGF": ORANGE, "RO-PDRF-Lite": "#59A14F",
              "RO-PDRF-Full": GREEN, "RO-AT-GATE": PURPLE,
              "RO-PDRF-EMA": BLUE, "QMF-PD": GREY,
              "RO-PDRF-CEG": RED}
    offsets = {"QMF-PD": (5, 5), "RO-AT-GATE": (5, 5),
               "RO-CAGF": (-55, 8), "RO-PDRF-Lite": (-74, -12),
               "RO-PDRF-Full": (-78, 5), "RO-PDRF-EMA": (5, -12),
               "RO-PDRF-CEG": (5, 5)}
    for _, r in methods.iterrows():
        ax.scatter(r.macro_auroc, r.ece15, s=31, color=colors[r.method],
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.annotate(DISPLAY_METHOD.get(r.method, r.method),
                    (r.macro_auroc, r.ece15),
                    xytext=offsets[r.method], textcoords="offset points",
                    fontsize=5.8)
    frontier = []
    for _, r in methods.iterrows():
        dominated = ((methods.macro_auroc >= r.macro_auroc) &
                     (methods.ece15 <= r.ece15) &
                     ((methods.macro_auroc > r.macro_auroc) |
                      (methods.ece15 < r.ece15))).any()
        if not dominated:
            frontier.append((r.macro_auroc, r.ece15))
    frontier = sorted(frontier)
    if len(frontier) > 1:
        ax.plot(*zip(*frontier), color="#444444", ls="--", lw=0.8, zorder=1)
    ax.set_xlabel("Affected macro-AUROC (higher is better)")
    ax.set_ylabel("Affected ECE (lower is better)")
    ax.set_xlim(0.768, 0.813)
    ax.set_ylim(0.195, 0.355)
    ax.set_title("a  Ranking--calibration audit", loc="left", fontweight="bold")

    ax = axs[1]
    order = ["Clean teacher", "Clean/removal agreement",
             "Student/EMA agreement", "CE-conflict training audit"]
    labels = ["Clean", "Clean/removal\nagreement", "Student/EMA\nagreement",
              "CE-conflict\ntraining audit"]
    view = teacher.set_index("teacher_rule").loc[order]
    x = np.arange(len(order))
    width = 0.34
    recovery = view.recovery_rate.to_numpy()
    transfer = view.error_transfer_rate.to_numpy()
    ax.bar(x - width/2, recovery, width, color=TEAL, label="Conditional recovery")
    ax.bar(x + width/2, transfer, width, color=RED,
           label="Conditional error transfer")
    for k, val in enumerate(recovery):
        ax.text(x[k] - width/2, val + 0.012, f"{100*val:.1f}%",
                ha="center", va="bottom", fontsize=5.6)
    for k, val in enumerate(transfer):
        ax.text(x[k] + width/2, val + 0.012, f"{100*val:.1f}%",
                ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.49)
    ax.set_ylabel("Conditional fraction")
    ax.set_title("b  Teacher-error controls", loc="left", fontweight="bold")
    ax.legend(loc="upper left", ncol=1, fontsize=5.6)
    save(fig, "figure5_pareto_teacher")

    # Supplementary training-curve contract: optimization is stable for the
    # three QMF implementations; the final ranking difference is not explained
    # by an obvious non-converged run. Curves are calibration-selection CE.
    curves = pd.concat([
        pd.read_csv(SRC / "major8_qmf_training_curves.csv"),
        pd.read_csv(SRC / "major8_qmf_faithful_extension_curves.csv")
    ], ignore_index=True)
    curves = curves[curves.method.isin(
        ["QMF-PD-current", "QMF-PD-normalized", "QMF-PD-faithful"])]
    curves = curves.drop_duplicates(["method", "seed", "epoch"], keep="last")
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    entries = [("QMF-PD-current", "Previous adaptation", GREY),
               ("QMF-PD-normalized", "Normalized fusion", ORANGE),
               ("QMF-PD-faithful", "Official-rule-aligned adaptation", BLUE)]
    for method, label, color in entries:
        z = curves[curves.method == method]
        s = z.groupby("epoch").selection_ce.agg(["mean", "std", "count"])
        s = s[s["count"] >= 3]
        ax.plot(s.index, s["mean"], color=color, lw=1.4, label=label)
        ax.fill_between(s.index.to_numpy(), (s["mean"]-s["std"]).to_numpy(),
                        (s["mean"]+s["std"]).to_numpy(), color=color, alpha=0.13,
                        linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Clean calibration-split cross-entropy")
    ax.set_title("QMF sensor-group adaptation training audit", loc="left",
                 fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    save(fig, "figureS9_qmf_training")


def main():
    qmf, qmf_summary = qmf_audit()
    methods = principal_methods(qmf)
    fault, lofo, aggregate = fault_type_architecture_audit()
    teacher = teacher_error_audit()
    make_figures(methods, teacher)
    print(json.dumps({
        "qmf_faithful": qmf_summary[
            qmf_summary.method == "QMF-PD-faithful"].to_dict("records"),
        "methods": methods.to_dict("records"),
        "fault_type_effects": fault.to_dict("records"),
        "leave_one_fault_type_out": lofo.to_dict("records"),
        "four_type_descriptive_aggregate": aggregate,
        "teacher_safeguards": teacher.to_dict("records"),
    }, indent=2))


if __name__ == "__main__":
    main()
