"""Create paired statistical tests and manuscript tables from experiment outputs."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def p_macro(value):
    """Format a p value as a LaTeX macro body valid in text or math mode."""
    if value < 0.001:
        exponent = int(np.floor(np.log10(value)))
        coefficient = value / (10 ** exponent)
        body = f"{coefficient:.3g}\\times 10^{{{exponent}}}"
    else:
        body = f"{value:.3g}"
    return f"\\ensuremath{{{body}}}"


def signed_rank(diff):
    """Two-sided paired Wilcoxon sensitivity test with a safe all-zero case."""
    diff = np.asarray(diff, dtype=float)
    if np.allclose(diff, 0):
        return 0.0, 1.0
    result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def main():
    r = pd.read_csv(DATA / "results_long.csv")
    metrics = ["auroc", "nll", "ece", "aurc"]
    regimes = ["ID_missing", "noise_shift", "outlier_shift", "silent_failure"]
    tests = []
    for regime in regimes:
        for metric in metrics:
            q = r[(r.regime == regime) & (r.metric == metric)].pivot(index="seed", columns="model", values="value")
            proposed, baseline = q["CRF"], q["UF"]
            raw_diff = proposed - baseline
            aligned = raw_diff if metric == "auroc" else -raw_diff
            stat, p = ttest_rel(proposed, baseline)
            wstat, wp = signed_rank(raw_diff)
            tests.append({
                "regime": regime, "metric": metric, "comparison": "CRF vs UF",
                "crf_mean": proposed.mean(), "uf_mean": baseline.mean(),
                "aligned_improvement": aligned.mean(),
                "improvement_ci95": 1.96 * aligned.std(ddof=1) / np.sqrt(len(aligned)),
                "paired_t": stat, "p_unadjusted": p,
                "wilcoxon_stat": wstat, "wilcoxon_p_unadjusted": wp,
                "cohen_dz": aligned.mean() / aligned.std(ddof=1) if aligned.std(ddof=1) > 0 else np.nan,
                "n_seeds": len(aligned),
            })
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p_unadjusted)
    tests["wilcoxon_p_holm"] = holm(tests.wilcoxon_p_unadjusted)
    tests.to_csv(DATA / "paired_tests.csv", index=False)

    # Prespecified mechanism contrasts: same architecture and counterfactual
    # exposure, with CRF uniquely adding the variance-ranking objective.
    ablation_tests = []
    for baseline in ("CFA", "CRF_NR"):
        for metric in ("auroc", "nll", "aurc"):
            q = r[(r.regime == "silent_failure") & (r.metric == metric)].pivot(
                index="seed", columns="model", values="value")
            proposed, control = q["CRF"], q[baseline]
            raw_diff = proposed - control
            aligned = raw_diff if metric == "auroc" else -raw_diff
            stat, p = ttest_rel(proposed, control)
            wstat, wp = signed_rank(raw_diff)
            ablation_tests.append({
                "regime": "silent_failure", "metric": metric,
                "comparison": f"CRF vs {baseline}", "crf_mean": proposed.mean(),
                "baseline_mean": control.mean(), "aligned_improvement": aligned.mean(),
                "improvement_ci95": 1.96 * aligned.std(ddof=1) / np.sqrt(len(aligned)),
                "paired_t": stat, "p_unadjusted": p,
                "wilcoxon_stat": wstat, "wilcoxon_p_unadjusted": wp,
                "cohen_dz": aligned.mean() / aligned.std(ddof=1), "n_seeds": len(aligned),
            })
    ablation_tests = pd.DataFrame(ablation_tests)
    ablation_tests["p_holm"] = holm(ablation_tests.p_unadjusted)
    ablation_tests["wilcoxon_p_holm"] = holm(ablation_tests.wilcoxon_p_unadjusted)
    ablation_tests.to_csv(DATA / "ablation_paired_tests.csv", index=False)

    s = pd.read_csv(DATA / "results_summary.csv")
    rows = []
    for regime in regimes:
        for model in ["EF", "MD", "UF", "GF", "CRF"]:
            row = {"regime": regime, "model": model}
            for metric in metrics:
                z = s[(s.regime == regime) & (s.model == model) & (s.metric == metric)].iloc[0]
                row[metric] = z["mean"]
                row[f"{metric}_ci95"] = z["ci95"]
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(DATA / "main_table.csv", index=False)

    def cell(model, regime, metric, candidates):
        z = s[(s.regime == regime) & (s.model == model) & (s.metric == metric)].iloc[0]
        vals = s[(s.regime == regime) & (s.model.isin(candidates)) & (s.metric == metric)]["mean"]
        best = vals.max() if metric == "auroc" else vals.min()
        txt = f"{z['mean']:.3f} $\\pm$ {z['ci95']:.3f}"
        return f"\\textbf{{{txt}}}" if np.isclose(z["mean"], best) else txt

    main_methods = ["EF", "MD", "UF", "GF", "CRF"]
    lines = [
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Regime & Method & AUROC $\uparrow$ & NLL $\downarrow$ & ECE $\downarrow$ & AURC $\downarrow$ \\",
        r"\midrule",
    ]
    labels = {"EF": "Early fusion", "MD": "Modality dropout", "UF": "Uniform fusion",
              "GF": "Gated fusion", "CRF": "CRF"}
    rlabels = {"ID_missing": "ID missingness", "noise_shift": "Noise shift",
               "outlier_shift": "Outlier shift", "silent_failure": "Silent failure"}
    for regime in regimes:
        for j, model in enumerate(main_methods):
            prefix = rlabels[regime] if j == 0 else ""
            vals = [cell(model, regime, metric, main_methods) for metric in metrics]
            lines.append(f"{prefix} & {labels[model]} & " + " & ".join(vals) + r" \\")
        if regime != regimes[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "main_results.tex").write_text("\n".join(lines), encoding="utf-8")

    ablation_methods = ["UF", "QWF", "LWF", "RWF", "CFA", "CRF_NR", "CRF"]
    alabels = {"UF": "Uniform", "QWF": "Quality only", "LWF": "Learned precision only",
               "RWF": "Factorized reliability", "CFA": "CF-Aug",
               "CRF_NR": "CRF without ranking", "CRF": "Full CRF"}
    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r"Regime & Variant & AUROC $\uparrow$ & NLL $\downarrow$ \\", r"\midrule"]
    for regime in regimes:
        for j, model in enumerate(ablation_methods):
            prefix = rlabels[regime] if j == 0 else ""
            vals = [cell(model, regime, metric, ablation_methods) for metric in ("auroc", "nll")]
            lines.append(f"{prefix} & {alabels[model]} & " + " & ".join(vals) + r" \\")
        if regime != regimes[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "ablation_results.tex").write_text("\n".join(lines), encoding="utf-8")

    macros = []
    rcode = {"ID_missing": "ID", "noise_shift": "NS", "outlier_shift": "OS", "silent_failure": "SF"}
    mcode = {"auroc": "AUC", "nll": "NLL", "ece": "ECE", "aurc": "AURC"}
    for regime in regimes:
        for metric in metrics:
            for model in ("UF", "QWF", "LWF", "RWF", "CFA", "CRF_NR", "CRF"):
                z = s[(s.regime == regime) & (s.model == model) & (s.metric == metric)].iloc[0]
                macro_model = "CRFNoRank" if model == "CRF_NR" else model
                macros.append(f"\\newcommand{{\\{macro_model}{rcode[regime]}{mcode[metric]}}}{{{z['mean']:.3f}}}")
                macros.append(f"\\newcommand{{\\{macro_model}{rcode[regime]}{mcode[metric]}CI}}{{{z['ci95']:.3f}}}")
    for regime in regimes:
        for metric in metrics:
            z = tests[(tests.regime == regime) & (tests.metric == metric)].iloc[0]
            macros.append(f"\\newcommand{{\\Gain{rcode[regime]}{mcode[metric]}}}{{{z['aligned_improvement']:.3f}}}")
            macros.append(f"\\newcommand{{\\PHolm{rcode[regime]}{mcode[metric]}}}{{{p_macro(z['p_holm'])}}}")
    for baseline, code in (("CFA", "CFA"), ("CRF_NR", "NoRank")):
        for metric in ("auroc", "nll", "aurc"):
            z = ablation_tests[(ablation_tests.comparison == f"CRF vs {baseline}") &
                               (ablation_tests.metric == metric)].iloc[0]
            macros.append(f"\\newcommand{{\\GainSF{code}{mcode[metric]}}}{{{z['aligned_improvement']:.3f}}}")
            macros.append(f"\\newcommand{{\\PHolmSF{code}{mcode[metric]}}}{{{p_macro(z['p_holm'])}}}")
    stress = pd.read_csv(DATA / "stress_results_summary.csv")
    for stress_type, level, code in (("missingness", 2.0, "HighMiss"), ("sensor_noise", 3.0, "HighNoise")):
        for model in ("UF", "CRF"):
            for metric in ("auroc", "ece"):
                z = stress[(stress.model == model) & (stress.stress_type == stress_type) &
                           (stress.level == level) & (stress.metric == metric)].iloc[0]
                macros.append(f"\\newcommand{{\\{model}{code}{mcode[metric]}}}{{{z['mean']:.3f}}}")

    # Longitudinal real-data evaluation on the UCI gas-sensor dataset.
    uci = pd.read_csv(DATA / "uci_results_long.csv")
    uci_summary = pd.read_csv(DATA / "uci_results_summary.csv")
    uci_regimes = ["complete_drift", "missing_drift", "reported_fault", "silent_fault"]
    uci_metrics = ["accuracy", "macro_f1", "macro_auroc", "nll", "aurc"]
    uci_tests = []
    for regime in uci_regimes:
        for metric in ("macro_auroc", "nll", "aurc"):
            q = uci[(uci.regime == regime) & (uci.metric == metric)].pivot(
                index="seed", columns="model", values="value"
            )
            proposed, baseline = q["CRF"], q["UF"]
            raw_diff = proposed - baseline
            aligned = raw_diff if metric == "macro_auroc" else -raw_diff
            stat, p = ttest_rel(proposed, baseline)
            wstat, wp = signed_rank(raw_diff)
            uci_tests.append({
                "regime": regime, "metric": metric, "comparison": "CRF vs UF",
                "crf_mean": proposed.mean(), "uf_mean": baseline.mean(),
                "aligned_improvement": aligned.mean(),
                "improvement_ci95": 1.96 * aligned.std(ddof=1) / np.sqrt(len(aligned)),
                "paired_t": stat, "p_unadjusted": p,
                "wilcoxon_stat": wstat, "wilcoxon_p_unadjusted": wp,
                "cohen_dz": aligned.mean() / aligned.std(ddof=1) if aligned.std(ddof=1) > 0 else np.nan,
                "n_seeds": len(aligned),
            })
    uci_tests = pd.DataFrame(uci_tests)
    uci_tests["p_holm"] = holm(uci_tests.p_unadjusted)
    uci_tests["wilcoxon_p_holm"] = holm(uci_tests.wilcoxon_p_unadjusted)
    uci_tests.to_csv(DATA / "uci_paired_tests.csv", index=False)

    uci_ablation_tests = []
    for baseline in ("CFA", "CRF_NR"):
        for metric in ("macro_auroc", "nll", "aurc"):
            q = uci[(uci.regime == "silent_fault") & (uci.metric == metric)].pivot(
                index="seed", columns="model", values="value")
            proposed, control = q["CRF"], q[baseline]
            raw_diff = proposed - control
            aligned = raw_diff if metric == "macro_auroc" else -raw_diff
            stat, p = ttest_rel(proposed, control)
            wstat, wp = signed_rank(raw_diff)
            uci_ablation_tests.append({
                "regime": "silent_fault", "metric": metric,
                "comparison": f"CRF vs {baseline}", "crf_mean": proposed.mean(),
                "baseline_mean": control.mean(), "aligned_improvement": aligned.mean(),
                "improvement_ci95": 1.96 * aligned.std(ddof=1) / np.sqrt(len(aligned)),
                "paired_t": stat, "p_unadjusted": p,
                "wilcoxon_stat": wstat, "wilcoxon_p_unadjusted": wp,
                "cohen_dz": aligned.mean() / aligned.std(ddof=1), "n_seeds": len(aligned),
            })
    uci_ablation_tests = pd.DataFrame(uci_ablation_tests)
    uci_ablation_tests["p_holm"] = holm(uci_ablation_tests.p_unadjusted)
    uci_ablation_tests["wilcoxon_p_holm"] = holm(uci_ablation_tests.wilcoxon_p_unadjusted)
    uci_ablation_tests.to_csv(DATA / "uci_ablation_paired_tests.csv", index=False)

    audit = pd.read_csv(DATA / "uci_counterfactual_audit.csv")
    mechanism_tests = []
    for baseline in ("RWF", "CFA", "CRF_NR"):
        for endpoint in ("delta_clipped_log_variance", "delta_normalized_weight"):
            q = audit.pivot(index="seed", columns="model", values=endpoint)
            proposed, control = q["CRF"], q[baseline]
            raw_diff = proposed - control
            aligned = raw_diff if endpoint == "delta_clipped_log_variance" else -raw_diff
            stat, p = ttest_rel(proposed, control)
            wstat, wp = signed_rank(raw_diff)
            mechanism_tests.append({
                "endpoint": endpoint, "comparison": f"CRF vs {baseline}",
                "crf_mean": proposed.mean(), "baseline_mean": control.mean(),
                "aligned_improvement": aligned.mean(),
                "improvement_ci95": 1.96 * aligned.std(ddof=1) / np.sqrt(len(aligned)),
                "paired_t": stat, "p_unadjusted": p,
                "wilcoxon_stat": wstat, "wilcoxon_p_unadjusted": wp,
                "cohen_dz": aligned.mean() / aligned.std(ddof=1), "n_seeds": len(aligned),
            })
    mechanism_tests = pd.DataFrame(mechanism_tests)
    mechanism_tests["p_holm"] = holm(mechanism_tests.p_unadjusted)
    mechanism_tests["wilcoxon_p_holm"] = holm(mechanism_tests.wilcoxon_p_unadjusted)
    mechanism_tests.to_csv(DATA / "uci_mechanism_tests.csv", index=False)

    uci_methods = ["EF", "MD", "UF", "GF", "RWF", "CRF"]
    uci_labels = {"EF": "Early fusion", "MD": "Modality dropout", "UF": "Uniform fusion",
                  "GF": "Gated fusion", "RWF": "Factorized reliability", "CRF": "CRF"}
    uci_rlabels = {"complete_drift": "Natural temporal drift", "missing_drift": "Drift + missingness",
                   "reported_fault": "Reported sensor fault", "silent_fault": "Silent sensor fault"}

    def uci_cell(model, regime, metric):
        z = uci_summary[(uci_summary.regime == regime) & (uci_summary.model == model) &
                        (uci_summary.metric == metric)].iloc[0]
        vals = uci_summary[(uci_summary.regime == regime) & (uci_summary.model.isin(uci_methods)) &
                           (uci_summary.metric == metric)]["mean"]
        best = vals.max() if metric in ("accuracy", "macro_f1", "macro_auroc") else vals.min()
        txt = f"{z['mean']:.3f} $\\pm$ {z['ci95']:.3f}"
        return f"\\textbf{{{txt}}}" if np.isclose(z["mean"], best) else txt

    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Regime & Method & Accuracy $\uparrow$ & Macro-F1 $\uparrow$ & Macro-AUROC $\uparrow$ & NLL $\downarrow$ & AURC $\downarrow$ \\",
             r"\midrule"]
    for regime in uci_regimes:
        for j, model in enumerate(uci_methods):
            prefix = uci_rlabels[regime] if j == 0 else ""
            vals = [uci_cell(model, regime, metric) for metric in uci_metrics]
            lines.append(f"{prefix} & {uci_labels[model]} & " + " & ".join(vals) + r" \\")
        if regime != uci_regimes[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "uci_results.tex").write_text("\n".join(lines), encoding="utf-8")

    # Focused real-data ablation table; all variants share architecture.
    focused = ["RWF", "CFA", "CRF_NR", "CRF"]
    focused_labels = {"RWF": "No intervention", "CFA": "CF-Aug",
                      "CRF_NR": "CRF without ranking", "CRF": "Full CRF"}
    audit_summary = audit.groupby("model", as_index=True).agg(
        delta_clip=("delta_clipped_log_variance", "mean"),
        delta_clip_ci=("delta_clipped_log_variance", lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))),
        post_sat=("post_upper_clip_fraction", "mean"),
        delta_weight=("delta_normalized_weight", "mean"),
        delta_weight_ci=("delta_normalized_weight", lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))))
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Variant & Natural AUROC $\uparrow$ & Silent AUROC $\uparrow$ & Silent NLL $\downarrow$ & $\Delta$ clipped log variance $\uparrow$ & $\Delta$ weight $\downarrow$ \\",
             r"\midrule"]
    for model in focused:
        natural = uci_summary[(uci_summary.model == model) & (uci_summary.regime == "complete_drift") &
                              (uci_summary.metric == "macro_auroc")].iloc[0]
        silent_auc = uci_summary[(uci_summary.model == model) & (uci_summary.regime == "silent_fault") &
                                 (uci_summary.metric == "macro_auroc")].iloc[0]
        silent_nll = uci_summary[(uci_summary.model == model) & (uci_summary.regime == "silent_fault") &
                                 (uci_summary.metric == "nll")].iloc[0]
        a = audit_summary.loc[model]
        values = [f"{natural['mean']:.3f} $\\pm$ {natural['ci95']:.3f}",
                  f"{silent_auc['mean']:.3f} $\\pm$ {silent_auc['ci95']:.3f}",
                  f"{silent_nll['mean']:.3f} $\\pm$ {silent_nll['ci95']:.3f}",
                  f"{a.delta_clip:.3f} $\\pm$ {a.delta_clip_ci:.3f}",
                  f"{a.delta_weight:.3f} $\\pm$ {a.delta_weight_ci:.3f}"]
        if model == "CRF":
            values = [f"\\textbf{{{v}}}" for v in values]
        lines.append(f"{focused_labels[model]} & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "uci_mechanism_results.tex").write_text("\n".join(lines), encoding="utf-8")

    syn_cost = pd.read_csv(DATA / "training_costs.csv")
    uci_cost = pd.read_csv(DATA / "uci_training_costs.csv")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & Synthetic parameters & Synthetic time (s) & UCI parameters & UCI time (s) \\",
             r"\midrule"]
    cost_labels = {"UF": "Uniform fusion", "RWF": "Factorized reliability", "CFA": "CF-Aug",
                   "CRF_NR": "CRF without ranking", "CRF": "Full CRF"}
    for model in ("UF", "RWF", "CFA", "CRF_NR", "CRF"):
        sc = syn_cost[syn_cost.model == model]
        uc = uci_cost[uci_cost.model == model]
        lines.append(f"{cost_labels[model]} & {int(sc.parameters.iloc[0]):,} & {sc.train_seconds.median():.2f} & "
                     f"{int(uc.parameters.iloc[0]):,} & {uc.train_seconds.median():.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "computational_costs.tex").write_text("\n".join(lines), encoding="utf-8")

    rank_summary = pd.read_csv(DATA / "uci_rank_sensitivity_summary.csv")
    rank_audit = pd.read_csv(DATA / "uci_rank_sensitivity_audit.csv")
    rank_audit_summary = rank_audit.groupby("rank_weight", as_index=True).mean(numeric_only=True)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"$\lambda_{rank}$ & Natural AUROC & Natural NLL & Silent AUROC & Silent NLL & Post-fault saturation & $\Delta$ weight \\",
             r"\midrule"]
    for weight in sorted(rank_summary.rank_weight.unique()):
        def rank_value(regime, metric):
            return rank_summary[(rank_summary.rank_weight == weight) &
                                (rank_summary.regime == regime) &
                                (rank_summary.metric == metric)].iloc[0]["mean"]
        a = rank_audit_summary.loc[weight]
        label = f"{weight:.2f}" + (" (main)" if np.isclose(weight, 0.15) else "")
        lines.append(f"{label} & {rank_value('complete_drift', 'macro_auroc'):.3f} & "
                     f"{rank_value('complete_drift', 'nll'):.3f} & "
                     f"{rank_value('silent_fault', 'macro_auroc'):.3f} & "
                     f"{rank_value('silent_fault', 'nll'):.3f} & "
                     f"{100 * a.post_upper_clip_fraction:.1f}\\% & {a.delta_normalized_weight:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "rank_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")

    urcode = {"complete_drift": "Natural", "missing_drift": "Missing",
              "reported_fault": "Reported", "silent_fault": "Silent"}
    umcode = {"accuracy": "Acc", "macro_f1": "FOne", "macro_auroc": "AUC",
              "nll": "NLL", "aurc": "AURC"}
    for regime in uci_regimes:
        for metric in uci_metrics:
            for model in ("UF", "RWF", "CFA", "CRF_NR", "CRF"):
                z = uci_summary[(uci_summary.regime == regime) & (uci_summary.model == model) &
                                (uci_summary.metric == metric)].iloc[0]
                macro_model = "CRFNoRank" if model == "CRF_NR" else model
                macros.append(f"\\newcommand{{\\UCI{macro_model}{urcode[regime]}{umcode[metric]}}}{{{z['mean']:.3f}}}")
        for metric in ("macro_auroc", "nll", "aurc"):
            z = uci_tests[(uci_tests.regime == regime) & (uci_tests.metric == metric)].iloc[0]
            macros.append(f"\\newcommand{{\\UCIGain{urcode[regime]}{umcode[metric]}}}{{{z['aligned_improvement']:.3f}}}")
            macros.append(f"\\newcommand{{\\UCIPHolm{urcode[regime]}{umcode[metric]}}}{{{p_macro(z['p_holm'])}}}")
    for baseline, code in (("CFA", "CFA"), ("CRF_NR", "NoRank")):
        for metric in ("macro_auroc", "nll", "aurc"):
            z = uci_ablation_tests[(uci_ablation_tests.comparison == f"CRF vs {baseline}") &
                                   (uci_ablation_tests.metric == metric)].iloc[0]
            macros.append(f"\\newcommand{{\\UCIGainSilent{code}{umcode[metric]}}}{{{z['aligned_improvement']:.3f}}}")
            macros.append(f"\\newcommand{{\\UCIPHolmSilent{code}{umcode[metric]}}}{{{p_macro(z['p_holm'])}}}")

    robust = pd.read_csv(DATA / "uci_robustness_summary.csv")
    for model, macro_model in (("CFA", "CFA"), ("CRF_NR", "CRFNoRank"), ("CRF", "CRF")):
        for stress_type, level, code in (("severity", 5.0, "SeverityFive"),
                                         ("fraction", 0.8, "FractionHigh"),
                                         ("reported_quality", 1.0, "SilentMetadata")):
            z = robust[(robust.model == model) & (robust.stress_type == stress_type) &
                       (robust.level == level) & (robust.metric == "macro_auroc")].iloc[0]
            macros.append(f"\\newcommand{{\\UCI{macro_model}{code}AUC}}{{{z['mean']:.3f}}}")
    location = robust[(robust.stress_type == "fault_modality") & (robust.metric == "macro_auroc")]
    loc = location.pivot(index="level", columns="model", values="mean")
    macros.append(f"\\newcommand{{\\UCIMinLocationGainCFA}}{{{(loc['CRF'] - loc['CFA']).min():.3f}}}")
    macros.append(f"\\newcommand{{\\UCIMinLocationGainNoRank}}{{{(loc['CRF'] - loc['CRF_NR']).min():.3f}}}")

    crf_audit = audit_summary.loc["CRF"]
    macros.append(f"\\newcommand{{\\UCICRFDeltaRawVar}}{{{audit[audit.model == 'CRF'].delta_log_variance.mean():.2f}}}")
    macros.append(f"\\newcommand{{\\UCICRFDeltaClipVar}}{{{crf_audit.delta_clip:.2f}}}")
    macros.append(f"\\newcommand{{\\UCICRFPostSaturation}}{{{100 * crf_audit.post_sat:.1f}\\%}}")
    macros.append(f"\\newcommand{{\\UCICRFDeltaWeight}}{{{crf_audit.delta_weight:.3f}}}")
    for baseline, code in (("CFA", "CFA"), ("CRF_NR", "NoRank")):
        for endpoint, endpoint_code in (("delta_clipped_log_variance", "Var"),
                                        ("delta_normalized_weight", "Weight")):
            z = mechanism_tests[(mechanism_tests.comparison == f"CRF vs {baseline}") &
                                (mechanism_tests.endpoint == endpoint)].iloc[0]
            macros.append(f"\\newcommand{{\\UCIMechGain{code}{endpoint_code}}}{{{z['aligned_improvement']:.3f}}}")
            macros.append(f"\\newcommand{{\\UCIMechPHolm{code}{endpoint_code}}}{{{p_macro(z['p_holm'])}}}")
    syn_ratio = (syn_cost[syn_cost.model == "CRF"].train_seconds.median() /
                 syn_cost[syn_cost.model == "RWF"].train_seconds.median())
    uci_ratio = (uci_cost[uci_cost.model == "CRF"].train_seconds.median() /
                 uci_cost[uci_cost.model == "RWF"].train_seconds.median())
    macros.append(f"\\newcommand{{\\SyntheticTrainRatio}}{{{syn_ratio:.2f}}}")
    macros.append(f"\\newcommand{{\\UCITrainRatio}}{{{uci_ratio:.2f}}}")
    macros.append(f"\\newcommand{{\\MaxWilcoxonSynAblation}}{{{p_macro(ablation_tests.wilcoxon_p_holm.max())}}}")
    macros.append(f"\\newcommand{{\\MaxWilcoxonUCIAblation}}{{{p_macro(uci_ablation_tests.wilcoxon_p_holm.max())}}}")
    macros.append(f"\\newcommand{{\\MaxWilcoxonUCIMechanism}}{{{p_macro(mechanism_tests.wilcoxon_p_holm.max())}}}")
    rank_codes = {0.0: "Zero", 0.03: "Low", 0.07: "Mid", 0.15: "Main", 0.30: "High"}
    for weight, code in rank_codes.items():
        for regime, regime_code in (("complete_drift", "Natural"), ("silent_fault", "Silent")):
            for metric, metric_code in (("macro_auroc", "AUC"), ("nll", "NLL")):
                z = rank_summary[(np.isclose(rank_summary.rank_weight, weight)) &
                                 (rank_summary.regime == regime) &
                                 (rank_summary.metric == metric)].iloc[0]
                macros.append(f"\\newcommand{{\\Rank{code}{regime_code}{metric_code}}}{{{z['mean']:.3f}}}")
        a = rank_audit_summary.loc[weight]
        macros.append(f"\\newcommand{{\\Rank{code}Saturation}}{{{100 * a.post_upper_clip_fraction:.1f}\\%}}")
        macros.append(f"\\newcommand{{\\Rank{code}DeltaWeight}}{{{a.delta_normalized_weight:.3f}}}")
    (TABLES / "result_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print("Synthetic paired tests:\n", tests.to_string(index=False))
    print("\nSynthetic mechanism ablations:\n", ablation_tests.to_string(index=False))
    print("\nUCI paired tests:\n", uci_tests.to_string(index=False))
    print("\nUCI mechanism ablations:\n", uci_ablation_tests.to_string(index=False))
    print("\nUCI mechanism tests:\n", mechanism_tests.to_string(index=False))


if __name__ == "__main__":
    main()
