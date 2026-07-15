"""Build Strict-88 reviewer-audit tables from frozen prediction outputs."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
TABLES = ROOT / "tables"


def paired_interval(values: pd.Series) -> tuple[float, float, float]:
    values = pd.Series(values, dtype=float).dropna()
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan, np.nan
    half = float(
        t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def signed(value: float, digits: int = 4) -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        return f"{0.0:.{digits}f}"
    return f"{value:+.{digits}f}"


def interval_cell(mean: float, low: float, high: float, digits: int = 4) -> str:
    return (
        f"${signed(mean, digits)}$ "
        f"$[{signed(low, digits)},{signed(high, digits)}]$"
    )


def count_cell(values: pd.Series) -> str:
    return f"{values.mean():.1f} ({int(values.min())}--{int(values.max())})"


def write_latex(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deployment_support() -> tuple[pd.DataFrame, dict]:
    diagnostics = pd.read_csv(DATA / "q1_lite_selector_diagnostics.csv")
    repeated = pd.read_csv(DATA / "q1_lite_repeated_grouped_cv.csv")
    repeated_summary = repeated.groupby("seed").agg(
        repeated_auroc_median=("auroc", "median"),
        repeated_auroc_min=("auroc", "min"),
        threshold_min=("threshold", "min"),
        threshold_max=("threshold", "max"),
    )
    repeated_summary["threshold_range"] = (
        repeated_summary.threshold_max - repeated_summary.threshold_min
    )
    result = diagnostics.merge(repeated_summary.reset_index(), on="seed")
    result["event_count_gate"] = (
        (result.base_better_rows >= 25) & (result.lite_better_rows >= 25)
    )
    result["discrimination_gate"] = (
        (result.repeated_auroc_median >= 0.65)
        & (result.repeated_auroc_min >= 0.60)
    )
    result["threshold_stability_gate"] = result.threshold_range <= 0.10 + 1e-12
    result["deployment_supported"] = result[
        ["event_count_gate", "discrimination_gate", "threshold_stability_gate"]
    ].all(axis=1)
    result.to_csv(DATA / "strict88_deployment_support.csv", index=False)

    lines = [
        r"\begin{tabular}{rrrrrrc}",
        r"\toprule",
        r"Seed & Base-better & Recovery-better & Median AUROC & Minimum AUROC & Threshold range & Supported \\",
        r"\midrule",
    ]
    for row in result.itertuples():
        lines.append(
            f"{row.seed} & {row.base_better_rows} & {row.lite_better_rows} & "
            f"{row.repeated_auroc_median:.3f} & {row.repeated_auroc_min:.3f} & "
            f"{row.threshold_min:.3f}--{row.threshold_max:.3f} & "
            f"{'Yes' if row.deployment_supported else 'No'} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex(TABLES / "strict88_deployment_support.tex", lines)
    summary = {
        "minimum_events_per_preference_class": 25,
        "minimum_median_repeated_oof_auroc": 0.65,
        "minimum_each_repeat_oof_auroc": 0.60,
        "maximum_threshold_range": 0.10,
        "supported_pairs": int(result.deployment_supported.sum()),
        "total_pairs": int(len(result)),
        "supported_seeds": result.loc[
            result.deployment_supported, "seed"
        ].astype(int).tolist(),
    }
    return result, summary


def opportunity_counts() -> pd.DataFrame:
    safety = pd.read_csv(DATA / "q1_lite_routing_safety.csv")
    strict = safety[
        safety.stream.eq("strict_fault_applied_available")
        & safety.fault_prevalence.eq(0.40)
    ].copy()
    rows = []
    for fault, frame in strict.groupby("fault_type", sort=False):
        rows.append(
            {
                "fault_type": fault,
                "n": int(frame.n.iloc[0]),
                "correction_opportunities_mean": frame.correction_opportunities.mean(),
                "correction_opportunities_min": frame.correction_opportunities.min(),
                "correction_opportunities_max": frame.correction_opportunities.max(),
                "harm_opportunities_mean": frame.harm_opportunities.mean(),
                "harm_opportunities_min": frame.harm_opportunities.min(),
                "harm_opportunities_max": frame.harm_opportunities.max(),
                "corrections_retained_mean": frame.corrections_retained.mean(),
                "corrections_retained_min": frame.corrections_retained.min(),
                "corrections_retained_max": frame.corrections_retained.max(),
                "harms_realized_mean": frame.harms_realized.mean(),
                "harms_realized_min": frame.harms_realized.min(),
                "harms_realized_max": frame.harms_realized.max(),
                "recovery_selection_rate": frame.recovery_selection_rate.mean(),
                "negative_transfer_prevention": frame.negative_transfer_prevention.mean(),
                "recovery_retention": frame.recovery_retention.mean(),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "strict88_opportunity_counts_by_mechanism.csv", index=False)
    raw = {fault: frame for fault, frame in strict.groupby("fault_type", sort=False)}
    labels = {
        "gaussian": "Gaussian",
        "offset": "Offset",
        "drift": "Drift",
        "stuck_at": "Stuck-at",
    }
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Mechanism & $N_C$ & $N_H$ & $n_C$ & $n_H$ & Select (\%) & Prevent (\%) & Retain (\%) \\",
        r"\midrule",
    ]
    for fault in ("gaussian", "offset", "drift", "stuck_at"):
        frame = raw[fault]
        lines.append(
            f"{labels[fault]} & {count_cell(frame.correction_opportunities)} & "
            f"{count_cell(frame.harm_opportunities)} & "
            f"{count_cell(frame.corrections_retained)} & "
            f"{count_cell(frame.harms_realized)} & "
            f"{100 * frame.recovery_selection_rate.mean():.1f} & "
            f"{100 * frame.negative_transfer_prevention.mean():.1f} & "
            f"{100 * frame.recovery_retention.mean():.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex(TABLES / "strict88_opportunity_counts.tex", lines)
    return result


def endpoint_seed_dispersion() -> pd.DataFrame:
    metrics = pd.read_csv(DATA / "q1_lite_routing_metrics.csv")
    metrics = metrics[
        metrics.stream.eq("strict_fault_applied_available")
        & metrics.fault_prevalence.eq(0.40)
    ]
    per_seed = metrics.groupby(["seed", "method"], as_index=False)[
        [
            "accuracy",
            "macro_auroc",
            "macro_auprc",
            "nll",
            "brier",
            "ece15_equal_width",
        ]
    ].mean()
    summary = per_seed.groupby("method", sort=False).agg(
        n_fitted_pairs=("seed", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        macro_auroc_mean=("macro_auroc", "mean"),
        macro_auroc_sd=("macro_auroc", "std"),
        macro_auprc_mean=("macro_auprc", "mean"),
        nll_mean=("nll", "mean"),
        brier_mean=("brier", "mean"),
        ece_mean=("ece15_equal_width", "mean"),
    ).reset_index()
    summary.to_csv(DATA / "strict88_endpoint_seed_dispersion.csv", index=False)
    labels = {"Lite-CF-6": "Lite-CF", "RO-PDRF-Lite": "RO-PDRF-Lite", "PDRF": "PDRF"}
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Accuracy & Macro-AUROC & Macro-AUPRC & NLL & Brier & ECE \\",
        r"\midrule",
    ]
    for method in ("PDRF", "RO-PDRF-Lite", "Lite-CF-6"):
        row = summary[summary.method.eq(method)].iloc[0]
        lines.append(
            f"{labels[method]} & {row.accuracy_mean:.4f}$\\pm${row.accuracy_sd:.4f} & "
            f"{row.macro_auroc_mean:.4f}$\\pm${row.macro_auroc_sd:.4f} & "
            f"{row.macro_auprc_mean:.4f} & {row.nll_mean:.4f} & "
            f"{row.brier_mean:.4f} & {row.ece_mean:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex(TABLES / "strict88_endpoint_seed_dispersion.tex", lines)
    return summary


def support_gated_policy(support: pd.DataFrame) -> pd.DataFrame:
    support_map = support.set_index("seed").deployment_supported
    metrics = pd.read_csv(DATA / "q1_lite_routing_metrics.csv")
    metrics = metrics[
        metrics.stream.eq("strict_fault_applied_available")
        & metrics.fault_prevalence.eq(0.40)
    ]
    selected_rows = []
    for (seed, fault), frame in metrics.groupby(["seed", "fault_type"]):
        method = "Lite-CF-6" if bool(support_map.loc[seed]) else "PDRF"
        selected_rows.append(frame[frame.method.eq(method)].iloc[0])
    selected = pd.DataFrame(selected_rows)

    safety = pd.read_csv(DATA / "q1_lite_routing_safety.csv")
    safety = safety[
        safety.stream.eq("strict_fault_applied_available")
        & safety.fault_prevalence.eq(0.40)
    ].copy()
    fallback = ~safety.seed.map(support_map)
    safety.loc[fallback, ["corrections_retained", "harms_realized"]] = 0
    safety.loc[fallback, "recovery_selection_rate"] = 0.0
    safety.loc[fallback, "negative_transfer_prevention"] = 1.0
    safety.loc[fallback, "recovery_retention"] = 0.0
    safety.loc[fallback, "net_correct_change_per_10000"] = 0.0

    row = {
        "supported_pairs": int(support_map.sum()),
        "total_pairs": int(len(support_map)),
        "accuracy": selected.accuracy.mean(),
        "macro_auroc": selected.macro_auroc.mean(),
        "macro_auprc": selected.macro_auprc.mean(),
        "nll": selected.nll.mean(),
        "brier": selected.brier.mean(),
        "ece15_equal_width": selected.ece15_equal_width.mean(),
        "recovery_selection_rate": safety.recovery_selection_rate.mean(),
        "negative_transfer_prevention": safety.negative_transfer_prevention.mean(),
        "recovery_retention": safety.recovery_retention.mean(),
        "utility_per_10000": safety.net_correct_change_per_10000.mean(),
    }
    result = pd.DataFrame([row])
    result.to_csv(DATA / "strict88_support_gated_policy.csv", index=False)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & Select (\%) & Prevent (\%) & Retain (\%) & Accuracy & Macro-AUROC & $U/10^4$ \\",
        r"\midrule",
        r"Ungated Lite-CF & 52.2 & 94.7 & 9.7 & 0.4885 & 0.8151 & +15.84 \\",
        f"Support-gated Lite-CF & {100 * row['recovery_selection_rate']:.1f} & "
        f"{100 * row['negative_transfer_prevention']:.1f} & "
        f"{100 * row['recovery_retention']:.1f} & {row['accuracy']:.4f} & "
        f"{row['macro_auroc']:.4f} & {row['utility_per_10000']:+.2f} \\\\ ",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_latex(TABLES / "strict88_support_gated_policy.tex", lines)
    return result


def severity_comparison() -> pd.DataFrame:
    main_metrics = pd.read_csv(DATA / "q1_lite_routing_metrics.csv")
    main_safety = pd.read_csv(DATA / "q1_lite_routing_safety.csv")
    mild_metrics = pd.read_csv(DATA / "strict88_mild_scale1_metrics.csv")
    mild_safety = pd.read_csv(DATA / "strict88_mild_scale1_safety.csv")
    main_metrics = main_metrics[
        main_metrics.stream.eq("strict_fault_applied_available")
        & main_metrics.fault_prevalence.eq(0.40)
    ]
    main_safety = main_safety[
        main_safety.stream.eq("strict_fault_applied_available")
        & main_safety.fault_prevalence.eq(0.40)
    ]
    rows = []
    for scale, metrics, safety in (
        (1.0, mild_metrics, mild_safety),
        (3.0, main_metrics, main_safety),
    ):
        row = {"fault_scale": scale}
        for method in ("PDRF", "RO-PDRF-Lite", "Lite-CF-6"):
            frame = metrics[metrics.method.eq(method)]
            key = method.lower().replace("-", "_")
            row[f"{key}_accuracy"] = frame.accuracy.mean()
            row[f"{key}_macro_auroc"] = frame.macro_auroc.mean()
        row.update(
            {
                "recovery_selection_rate": safety.recovery_selection_rate.mean(),
                "negative_transfer_prevention": safety.negative_transfer_prevention.mean(),
                "recovery_retention": safety.recovery_retention.mean(),
                "utility_per_10000": safety.net_correct_change_per_10000.mean(),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "strict88_severity_comparison.csv", index=False)
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"Scale & PDRF acc. & Lite-CF acc. & PDRF AUROC & Lite-CF AUROC & Prevent (\%) & Retain (\%) & $U/10^4$ \\",
        r"\midrule",
    ]
    for row in result.itertuples():
        lines.append(
            f"{row.fault_scale:.0f} & {row.pdrf_accuracy:.4f} & "
            f"{row.lite_cf_6_accuracy:.4f} & {row.pdrf_macro_auroc:.4f} & "
            f"{row.lite_cf_6_macro_auroc:.4f} & "
            f"{100 * row.negative_transfer_prevention:.1f} & "
            f"{100 * row.recovery_retention:.1f} & "
            f"{row.utility_per_10000:+.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex(TABLES / "strict88_severity_comparison.tex", lines)
    return result


def mechanism_paired_effects() -> pd.DataFrame:
    metrics = pd.read_csv(DATA / "q1_lite_routing_metrics.csv")
    safety = pd.read_csv(DATA / "q1_lite_routing_safety.csv")
    metrics = metrics[
        metrics.stream.eq("strict_fault_applied_available")
        & metrics.fault_prevalence.eq(0.40)
    ]
    safety = safety[
        safety.stream.eq("strict_fault_applied_available")
        & safety.fault_prevalence.eq(0.40)
    ]
    pivot = metrics.pivot_table(
        index=["seed", "fault_type"],
        columns="method",
        values=["accuracy", "macro_auroc"],
    )
    rows = []
    for fault in ("gaussian", "offset", "drift", "stuck_at"):
        frame = pivot.xs(fault, level="fault_type")
        util = safety[safety.fault_type.eq(fault)].set_index("seed")
        measures = {
            "accuracy_difference": frame[("accuracy", "Lite-CF-6")]
            - frame[("accuracy", "PDRF")],
            "macro_auroc_difference": frame[("macro_auroc", "Lite-CF-6")]
            - frame[("macro_auroc", "PDRF")],
            "utility_per_10000": util.net_correct_change_per_10000,
        }
        row = {"fault_type": fault}
        for name, values in measures.items():
            mean, low, high = paired_interval(values)
            row[f"{name}_mean"] = mean
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "strict88_mechanism_paired_effects.csv", index=False)
    labels = {
        "gaussian": "Gaussian",
        "offset": "Offset",
        "drift": "Drift",
        "stuck_at": "Stuck-at",
    }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Fixed mechanism & $\Delta$ accuracy & $\Delta$ macro-AUROC & Equal-cost $U/10^4$ \\",
        r"\midrule",
    ]
    for row in result.itertuples():
        lines.append(
            f"{labels[row.fault_type]} & "
            f"{interval_cell(row.accuracy_difference_mean, row.accuracy_difference_ci_low, row.accuracy_difference_ci_high)} & "
            f"{interval_cell(row.macro_auroc_difference_mean, row.macro_auroc_difference_ci_low, row.macro_auroc_difference_ci_high)} & "
            f"{interval_cell(row.utility_per_10000_mean, row.utility_per_10000_ci_low, row.utility_per_10000_ci_high, 2)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex(TABLES / "strict88_mechanism_paired_effects.tex", lines)
    return result


def cpu_thread_sensitivity() -> pd.DataFrame:
    latency = pd.read_csv(DATA / "strict88_cpu_thread_sensitivity.csv")
    summary = latency.groupby(
        ["requested_intraop_threads", "method"], sort=False
    ).agg(
        n_fitted_pairs=("seed", "nunique"),
        batch1_ms_mean=("latency_batch1_ms_median", "mean"),
        batch1_ms_sd=("latency_batch1_ms_median", "std"),
        full_batch_ms_mean=("latency_batch_ms_median", "mean"),
        full_batch_ms_sd=("latency_batch_ms_median", "std"),
        throughput_mean=("throughput_observations_per_second", "mean"),
        throughput_sd=("throughput_observations_per_second", "std"),
    ).reset_index()
    summary.to_csv(DATA / "strict88_cpu_thread_summary.csv", index=False)
    labels = {"Lite-CF-6": "Lite-CF", "RO-PDRF-Lite": "RO-PDRF-Lite", "PDRF": "PDRF"}
    lines = [
        r"\begin{tabular}{rlrrr}",
        r"\toprule",
        r"Threads & Method & Batch-one (ms) & 4,364-row batch (ms) & Throughput (obs/s) \\",
        r"\midrule",
    ]
    for row in summary.itertuples():
        lines.append(
            f"{row.requested_intraop_threads} & {labels[row.method]} & "
            f"{row.batch1_ms_mean:.3f}$\\pm${row.batch1_ms_sd:.3f} & "
            f"{row.full_batch_ms_mean:.2f}$\\pm${row.full_batch_ms_sd:.2f} & "
            f"{row.throughput_mean:,.0f}$\\pm${row.throughput_sd:,.0f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex(TABLES / "strict88_cpu_thread_summary.tex", lines)
    return summary


def main() -> None:
    support, support_summary = deployment_support()
    opportunity = opportunity_counts()
    endpoint_dispersion = endpoint_seed_dispersion()
    gated_policy = support_gated_policy(support)
    severity = severity_comparison()
    effects = mechanism_paired_effects()
    cpu = cpu_thread_sensitivity()
    summary = {
        "deployment_support": support_summary,
        "mild_scale1": severity[severity.fault_scale.eq(1.0)].iloc[0].to_dict(),
        "scale3": severity[severity.fault_scale.eq(3.0)].iloc[0].to_dict(),
        "files": {
            "support_rows": len(support),
            "opportunity_rows": len(opportunity),
            "endpoint_dispersion_rows": len(endpoint_dispersion),
            "support_gated_policy_rows": len(gated_policy),
            "mechanism_effect_rows": len(effects),
            "cpu_summary_rows": len(cpu),
        },
    }
    (DATA / "strict88_revision_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
