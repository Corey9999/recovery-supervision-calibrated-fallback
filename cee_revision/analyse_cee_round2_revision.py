from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
TABLES = ROOT / "tables"


def write_latex(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def t_interval(values: pd.Series) -> tuple[float, float, float]:
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


def safe_ratio(numerator: float, denominator: float, empty: float) -> float:
    return float(numerator / denominator) if denominator else empty


def support_mask(
    support: pd.DataFrame,
    minimum_events: int,
    minimum_median_auroc: float,
    maximum_threshold_range: float,
) -> pd.Series:
    return (
        (support.base_better_rows >= minimum_events)
        & (support.lite_better_rows >= minimum_events)
        & (support.repeated_auroc_median >= minimum_median_auroc)
        & (support.repeated_auroc_min >= 0.60)
        & (support.threshold_range <= maximum_threshold_range + 1e-12)
    )


def decision_metrics(frame: pd.DataFrame, choose: pd.Series, ratio: float = 1.0) -> dict:
    choose = np.asarray(choose, dtype=bool)
    correction = frame.base_wrong_lite_correct.to_numpy(dtype=bool)
    harm = frame.base_correct_lite_wrong.to_numpy(dtype=bool)
    retained = int(np.sum(choose & correction))
    realized = int(np.sum(choose & harm))
    n_c = int(correction.sum())
    n_h = int(harm.sum())
    n = len(frame)
    return {
        "n": n,
        "correction_opportunities": n_c,
        "corrections_retained": retained,
        "harm_opportunities": n_h,
        "harms_realized": realized,
        "recovery_selection_rate": float(choose.mean()),
        "negative_transfer_prevention": 1.0 - safe_ratio(realized, n_h, 0.0),
        "recovery_retention": safe_ratio(retained, n_c, 0.0),
        "utility_per_10000": 10000.0 * (retained - ratio * realized) / n,
    }


def gate_sensitivity() -> pd.DataFrame:
    support = pd.read_csv(DATA / "strict88_deployment_support.csv")
    pred = pd.read_csv(DATA / "q1_lite_routing_predictions.csv.gz")
    base_cols = [f"base_p{k}" for k in range(6)]
    lite_cols = [f"lite_p{k}" for k in range(6)]
    pred["base_correct"] = pred[base_cols].to_numpy().argmax(axis=1) == pred.y.to_numpy()
    pred["lite_correct"] = pred[lite_cols].to_numpy().argmax(axis=1) == pred.y.to_numpy()
    pred["base_wrong_lite_correct"] = ~pred.base_correct & pred.lite_correct
    pred["base_correct_lite_wrong"] = pred.base_correct & ~pred.lite_correct

    rows: list[dict] = []
    for minimum_events in (15, 25, 40):
        for median_auroc in (0.60, 0.65, 0.70):
            for threshold_range in (0.05, 0.10, 0.15):
                passed = support_mask(
                    support, minimum_events, median_auroc, threshold_range
                )
                passed_map = dict(zip(support.seed.astype(int), passed.astype(bool)))
                cell_rows = []
                batch_rows = []
                for (seed, fault), frame in pred.groupby(["seed", "fault_type"], sort=False):
                    choose = frame.choose_lite if passed_map[int(seed)] else np.zeros(len(frame), dtype=bool)
                    cell_rows.append(decision_metrics(frame, choose))
                    for batch, batch_frame in frame.groupby("batch", sort=True):
                        batch_choose = (
                            batch_frame.choose_lite
                            if passed_map[int(seed)]
                            else np.zeros(len(batch_frame), dtype=bool)
                        )
                        metric = decision_metrics(batch_frame, batch_choose)
                        metric["seed"] = int(seed)
                        metric["batch"] = int(batch)
                        batch_rows.append(metric)
                cells = pd.DataFrame(cell_rows)
                batches = pd.DataFrame(batch_rows)
                seed_batch_utility = (
                    batches.groupby(["seed", "batch"]).utility_per_10000.mean()
                )
                per_seed_worst_batch = seed_batch_utility.groupby("seed").min()
                rows.append(
                    {
                        "minimum_events_per_preference_class": minimum_events,
                        "minimum_median_repeated_auroc": median_auroc,
                        "maximum_threshold_range": threshold_range,
                        "minimum_each_repeat_auroc_fixed": 0.60,
                        "passed_fitted_pairs": int(passed.sum()),
                        "passed_seeds": ";".join(
                            str(int(seed)) for seed in support.loc[passed, "seed"]
                        ),
                        "recovery_selection_rate": cells.recovery_selection_rate.mean(),
                        "negative_transfer_prevention": cells.negative_transfer_prevention.mean(),
                        "recovery_retention": cells.recovery_retention.mean(),
                        "utility_per_10000": cells.utility_per_10000.mean(),
                        "worst_batch_utility_per_10000": per_seed_worst_batch.mean(),
                        "worst_batch_definition": "mean of each fitted pair's worst acquisition batch",
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "round2_gate_sensitivity_full.csv", index=False)

    reference = (25, 0.65, 0.10)
    one_factor = []
    for row in result.itertuples(index=False):
        config = (
            row.minimum_events_per_preference_class,
            row.minimum_median_repeated_auroc,
            row.maximum_threshold_range,
        )
        differences = sum(a != b for a, b in zip(config, reference))
        if config == reference or differences == 1:
            if config[0] not in (15, 25, 40):
                continue
            if config[1] not in (0.60, 0.65, 0.70):
                continue
            if config[2] not in (0.05, 0.10, 0.15):
                continue
            one_factor.append(row._asdict())
    one_factor_df = pd.DataFrame(one_factor)
    one_factor_df.to_csv(DATA / "round2_gate_sensitivity_onefactor.csv", index=False)

    def gate_lines(frame: pd.DataFrame) -> list[str]:
        lines = [
            r"\begin{tabular}{rrrcrrrrr}",
            r"\toprule",
            r"Min. events & Median AUROC & Max. $\Delta t$ & Passed & Select (\%) & Prevent (\%) & Retain (\%) & $U_1/10^4$ & Worst-batch $U_1/10^4$ \\",
            r"\midrule",
        ]
        for row in frame.itertuples(index=False):
            ref = (
                row.minimum_events_per_preference_class == 25
                and abs(row.minimum_median_repeated_auroc - 0.65) < 1e-9
                and abs(row.maximum_threshold_range - 0.10) < 1e-9
            )
            passed_cell = rf"\textbf{{{row.passed_fitted_pairs}/10}}" if ref else f"{row.passed_fitted_pairs}/10"
            lines.append(
                f"{row.minimum_events_per_preference_class} & "
                f"{row.minimum_median_repeated_auroc:.2f} & "
                f"{row.maximum_threshold_range:.2f} & {passed_cell} & "
                f"{100 * row.recovery_selection_rate:.1f} & "
                f"{100 * row.negative_transfer_prevention:.1f} & "
                f"{100 * row.recovery_retention:.1f} & "
                f"{row.utility_per_10000:+.2f} & "
                f"{row.worst_batch_utility_per_10000:+.2f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return lines

    write_latex(TABLES / "round2_gate_sensitivity_main.tex", gate_lines(one_factor_df))
    write_latex(TABLES / "round2_gate_sensitivity_full.tex", gate_lines(result))
    return result


def fitted_pair_utility() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DATA / "major_utility_thresholds.csv")
    fixed = raw[raw.analysis_set.eq("fixed_test")].copy()
    pair_rows = []
    keys = ["seed", "method", "harm_to_correction_ratio"]
    for (seed, method, ratio), frame in fixed.groupby(keys, sort=False):
        n = int(frame.n.sum())
        n_c = float(frame.correction_opportunities.sum())
        retained = float(frame.corrections_retained.sum())
        n_h = float(frame.harm_opportunities.sum())
        harms = float(frame.harms_realized.sum())
        selected = float((frame.recovery_selection_rate * frame.n).sum())
        pair_rows.append(
            {
                "seed": int(seed),
                "method": method,
                "harm_to_correction_ratio": ratio,
                "n": n,
                "correction_opportunities": n_c,
                "corrections_retained": retained,
                "harm_opportunities": n_h,
                "harms_realized": harms,
                "recovery_selection_rate": selected / n,
                "negative_transfer_prevention": 1.0 - safe_ratio(harms, n_h, 0.0),
                "recovery_retention": safe_ratio(retained, n_c, 0.0),
                "test_utility_per_10000": 10000.0 * (retained - ratio * harms) / n,
            }
        )
    pair = pd.DataFrame(pair_rows)

    calibration = raw[raw.analysis_set.eq("calibration_oof")].copy()
    calibration["calibration_utility_per_10000"] = (
        10000.0 * calibration.utility_events / calibration.n
    )
    calibration = calibration[
        ["seed", "method", "harm_to_correction_ratio", "calibration_utility_per_10000"]
    ]
    pair = pair.merge(
        calibration,
        how="left",
        on=["seed", "method", "harm_to_correction_ratio"],
    )
    pair["test_minus_calibration_utility"] = (
        pair.test_utility_per_10000 - pair.calibration_utility_per_10000
    )
    # The upstream frozen file retains the historical machine key
    # "Prospective gate Lite-CF".  It is a post hoc, retrospectively audited
    # policy, so all new round-two outputs use the scientifically accurate
    # display key below while leaving the frozen input untouched.
    pair["method"] = pair.method.replace(
        {"Prospective gate Lite-CF": "Retrospective support gate Lite-CF"}
    )
    pair.to_csv(DATA / "round2_utility_fitted_pair_metrics.csv", index=False)

    summary_rows = []
    for (method, ratio), frame in pair.groupby(
        ["method", "harm_to_correction_ratio"], sort=False
    ):
        mean, low, high = t_interval(frame.test_utility_per_10000)
        summary_rows.append(
            {
                "method": method,
                "harm_to_correction_ratio": ratio,
                "n_fitted_pairs": int(frame.seed.nunique()),
                "calibration_utility_mean": frame.calibration_utility_per_10000.mean(),
                "test_utility_mean": mean,
                "test_utility_ci_low": low,
                "test_utility_ci_high": high,
                "test_utility_median": frame.test_utility_per_10000.median(),
                "test_utility_q1": frame.test_utility_per_10000.quantile(0.25),
                "test_utility_q3": frame.test_utility_per_10000.quantile(0.75),
                "test_utility_min": frame.test_utility_per_10000.min(),
                "test_utility_max": frame.test_utility_per_10000.max(),
                "positive_utility_pairs": int((frame.test_utility_per_10000 > 0).sum()),
                "test_minus_calibration_mean": frame.test_minus_calibration_utility.mean(),
                "recovery_selection_rate": frame.recovery_selection_rate.mean(),
                "negative_transfer_prevention": frame.negative_transfer_prevention.mean(),
                "recovery_retention": frame.recovery_retention.mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA / "round2_utility_fitted_pair_summary.csv", index=False)

    labels = {
        "Conditional Lite-CF": "Conditional",
        "All-row multinomial": "All-row",
        "Two-stage": "Two-stage",
        "Frozen Lite-CF (ungated)": "Frozen safety",
        "Retrospective support gate Lite-CF": "Support gate",
        "All-row multinomial (safety)": "All-row safety",
        "Always PDRF": "Always base",
        "Always RO-PDRF-Lite": "Always recovery",
    }

    def utility_lines(frame: pd.DataFrame) -> list[str]:
        lines = [
            r"\begin{tabular}{rlrrrrrrrr}",
            r"\toprule",
            r"Ratio & Policy & Cal. $U$ & Test $U$ [95\% CI] & Median [IQR] & Test$-$cal. & Positive & Select (\%) & Prevent (\%) & Retain (\%) \\",
            r"\midrule",
        ]
        for row in frame.itertuples(index=False):
            cal = "--" if pd.isna(row.calibration_utility_mean) else f"{row.calibration_utility_mean:+.1f}"
            transport = "--" if pd.isna(row.test_minus_calibration_mean) else f"{row.test_minus_calibration_mean:+.1f}"
            lines.append(
                f"{row.harm_to_correction_ratio:g}:1 & {labels[row.method]} & {cal} & "
                f"{row.test_utility_mean:+.1f} [{row.test_utility_ci_low:+.1f},{row.test_utility_ci_high:+.1f}] & "
                f"{row.test_utility_median:+.1f} [{row.test_utility_q1:+.1f},{row.test_utility_q3:+.1f}] & "
                f"{transport} & {row.positive_utility_pairs}/10 & "
                f"{100 * row.recovery_selection_rate:.1f} & "
                f"{100 * row.negative_transfer_prevention:.1f} & "
                f"{100 * row.recovery_retention:.1f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return lines

    main = summary[
        summary.method.isin(["Conditional Lite-CF", "All-row multinomial", "Two-stage"])
    ].copy()
    order = {"Conditional Lite-CF": 0, "All-row multinomial": 1, "Two-stage": 2}
    main["method_order"] = main.method.map(order)
    main = main.sort_values(["harm_to_correction_ratio", "method_order"])
    write_latex(TABLES / "round2_utility_fitted_pair_main.tex", utility_lines(main))

    full_order = {
        "Always PDRF": 0,
        "Always RO-PDRF-Lite": 1,
        "Frozen Lite-CF (ungated)": 2,
        "Retrospective support gate Lite-CF": 3,
        "All-row multinomial (safety)": 4,
        "Conditional Lite-CF": 5,
        "All-row multinomial": 6,
        "Two-stage": 7,
    }
    full = summary.copy()
    full["method_order"] = full.method.map(full_order)
    full = full.sort_values(["harm_to_correction_ratio", "method_order"])
    write_latex(TABLES / "round2_utility_fitted_pair_full.tex", utility_lines(full))
    return pair, summary


def primary_policy_table() -> None:
    raw = pd.read_csv(DATA / "major_utility_thresholds.csv")
    raw = raw[
        raw.analysis_set.eq("fixed_test")
        & raw.harm_to_correction_ratio.eq(1.0)
    ]
    methods = [
        ("Always PDRF", "Always PDRF"),
        ("Always RO-PDRF-Lite", "Always RO-PDRF-Lite"),
        ("Frozen Lite-CF (ungated)", "Frozen Lite-CF (ungated)"),
        ("Prospective gate Lite-CF", "Retrospective support-gated Lite-CF"),
        ("All-row multinomial (safety)", "All-row multinomial (safety control)"),
        ("All-row multinomial", "All-row multinomial (utility 1:1)"),
        ("Two-stage", "Two-stage (utility 1:1)"),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & Accuracy & Macro-AUROC & Prevent (\%) & Retain (\%) & Select (\%) & $U_1/10^4$ \\",
        r"\midrule",
    ]
    for method, label in methods:
        frame = raw[raw.method.eq(method)]
        lines.append(
            f"{label} & {frame.accuracy.mean():.4f} & {frame.macro_auroc.mean():.4f} & "
            f"{100 * frame.negative_transfer_prevention.mean():.1f} & "
            f"{100 * frame.recovery_retention.mean():.1f} & "
            f"{100 * frame.recovery_selection_rate.mean():.1f} & "
            f"{frame.utility_per_10000.mean():+.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_latex(TABLES / "round2_primary_policy_main.tex", lines)


def endpoint_benchmark_table() -> None:
    lines = [
        r"\begin{tabular}{lp{0.25\textwidth}rrc}",
        r"\toprule",
        r"Method & Role in endpoint benchmark & Parameters & Affected AUROC & Router endpoint \\",
        r"\midrule",
        r"PDRF & stable base reference & 19,740 & 0.791 & Base \\",
        r"RO-PDRF-Lite & practical paired-recovery model & 19,740 & 0.802 & Recovery \\",
        r"RO-PDRF-Full & mechanistic recovery audit & 19,740 & 0.804 & No \\",
        r"RO-CAGF & architecture-matched learned baseline & 26,588 & 0.797 & No \\",
        r"RO-AT-GATE & attention-gated learned baseline & 26,809 & 0.784 & No \\",
        r"QMF-PD & published-method adaptation & 17,560 & 0.775 & No \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_latex(TABLES / "round2_endpoint_benchmark.tex", lines)


def main() -> None:
    gate = gate_sensitivity()
    pair, utility = fitted_pair_utility()
    primary_policy_table()
    endpoint_benchmark_table()
    reference = gate[
        gate.minimum_events_per_preference_class.eq(25)
        & gate.minimum_median_repeated_auroc.eq(0.65)
        & gate.maximum_threshold_range.eq(0.10)
    ].iloc[0]
    print(
        "Reference gate:",
        f"{int(reference.passed_fitted_pairs)}/10 pairs,",
        f"prevention={100 * reference.negative_transfer_prevention:.1f}%,",
        f"retention={100 * reference.recovery_retention:.1f}%,",
        f"U={reference.utility_per_10000:+.2f}",
    )
    print(f"Utility pair rows: {len(pair)}; summary rows: {len(utility)}")


if __name__ == "__main__":
    main()
