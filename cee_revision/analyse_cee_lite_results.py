"""Summarize RO-PDRF-Lite routing with fixed-mechanism and batch intervals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from analyse_cee_q1_scores import (
    FEATURES_6,
    calibration_size_audit,
    classwise_metrics,
    grouped_binary_fit,
    probability_metrics,
    probability_columns,
    selection_metrics,
    t_interval,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
TABLES = ROOT / "tables"


def main():
    predictions = pd.read_csv(DATA / "q1_lite_routing_predictions.csv.gz")
    effect_rows = []
    batch_rows = []
    agreement_rows = []
    class_rows = []
    for (seed, fault), frame in predictions.groupby(["seed", "fault_type"]):
        pb = frame[probability_columns(frame, "base_p")].to_numpy(float)
        pl = frame[probability_columns(frame, "lite_p")].to_numpy(float)
        ps = frame[probability_columns(frame, "selected_p")].to_numpy(float)
        y = frame.y.to_numpy(int)
        base_metrics = probability_metrics(y, pb)
        lite_metrics = probability_metrics(y, pl)
        selected_metrics = probability_metrics(y, ps)
        effect_rows.append(
            {
                "seed": seed,
                "fault_type": fault,
                "lite_minus_pdrf_auroc": lite_metrics["macro_auroc"] - base_metrics["macro_auroc"],
                "lite_cf_minus_pdrf_auroc": selected_metrics["macro_auroc"] - base_metrics["macro_auroc"],
                "lite_cf_minus_lite_auroc": selected_metrics["macro_auroc"] - lite_metrics["macro_auroc"],
            }
        )
        agreement = pb.argmax(axis=1) == pl.argmax(axis=1)
        for method, probability in (
            ("PDRF", pb),
            ("RO-PDRF-Lite", pl),
            ("Lite-CF", ps),
        ):
            for subset, take in (
                ("endpoint_class_agreement", agreement),
                ("endpoint_class_disagreement", ~agreement),
            ):
                agreement_rows.append(
                    {
                        "seed": seed,
                        "fault_type": fault,
                        "method": method,
                        "subset": subset,
                        "n": int(take.sum()),
                        **probability_metrics(y[take], probability[take]),
                    }
                )
            for row in classwise_metrics(y, probability):
                class_rows.append(
                    {
                        "seed": seed,
                        "fault_type": fault,
                        "method": method,
                        **row,
                    }
                )
        for batch, batch_frame in frame.groupby("batch"):
            batch_pb = batch_frame[probability_columns(batch_frame, "base_p")].to_numpy(float)
            batch_pl = batch_frame[probability_columns(batch_frame, "lite_p")].to_numpy(float)
            batch_ps = batch_frame[probability_columns(batch_frame, "selected_p")].to_numpy(float)
            batch_y = batch_frame.y.to_numpy(int)
            base_auc = probability_metrics(batch_y, batch_pb)["macro_auroc"]
            lite_auc = probability_metrics(batch_y, batch_pl)["macro_auroc"]
            selected_auc = probability_metrics(batch_y, batch_ps)["macro_auroc"]
            batch_rows.append(
                {
                    "seed": seed,
                    "fault_type": fault,
                    "batch": batch,
                    "n": len(batch_frame),
                    "lite_minus_pdrf_auroc": lite_auc - base_auc,
                    "lite_cf_minus_pdrf_auroc": selected_auc - base_auc,
                    "lite_cf_minus_lite_auroc": selected_auc - lite_auc,
                }
            )

    effects = pd.DataFrame(effect_rows)
    batches = pd.DataFrame(batch_rows)
    effects.to_csv(DATA / "q1_lite_seed_effects.csv", index=False)
    batches.to_csv(DATA / "q1_lite_batch_effects.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(
        DATA / "q1_lite_agreement_probability_quality.csv", index=False
    )
    pd.DataFrame(class_rows).to_csv(
        DATA / "q1_lite_classwise_metrics.csv", index=False
    )

    interval_rows = []
    for fault, group in effects.groupby("fault_type"):
        for comparison in (
            "lite_minus_pdrf_auroc",
            "lite_cf_minus_pdrf_auroc",
            "lite_cf_minus_lite_auroc",
        ):
            mean, lower, upper = t_interval(group[comparison].to_numpy(float))
            interval_rows.append(
                {
                    "fault_type": fault,
                    "comparison": comparison,
                    "mean": mean,
                    "ci_2.5": lower,
                    "ci_97.5": upper,
                    "n_fitted_pairs": len(group),
                    "interval_unit": "fitted model pair; mechanism fixed",
                }
            )
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(DATA / "q1_lite_mechanism_fixed_intervals.csv", index=False)

    batch_interval_rows = []
    for (fault, batch), group in batches.groupby(["fault_type", "batch"]):
        for comparison in (
            "lite_minus_pdrf_auroc",
            "lite_cf_minus_pdrf_auroc",
            "lite_cf_minus_lite_auroc",
        ):
            mean, lower, upper = t_interval(group[comparison].to_numpy(float))
            batch_interval_rows.append(
                {
                    "fault_type": fault,
                    "batch": batch,
                    "comparison": comparison,
                    "mean": mean,
                    "ci_2.5": lower,
                    "ci_97.5": upper,
                    "n_fitted_pairs": len(group),
                    "interval_unit": "fitted model pair; mechanism and batch fixed",
                }
            )
    pd.DataFrame(batch_interval_rows).to_csv(
        DATA / "q1_lite_batch_fixed_intervals.csv", index=False
    )

    metrics = pd.read_csv(DATA / "q1_lite_routing_metrics.csv")
    safety = pd.read_csv(DATA / "q1_lite_routing_safety.csv")
    strict_metrics = (
        metrics[metrics.stream == "strict_fault_applied_available"]
        .groupby("method")[[
            "accuracy",
            "macro_auroc",
            "macro_auprc",
            "nll",
            "brier",
            "ece15_equal_width",
        ]]
        .mean()
    )
    strict_safety = (
        safety[safety.stream == "strict_fault_applied_available"]
        .groupby("method")[[
            "negative_transfer_prevention",
            "recovery_retention",
            "recovery_selection_rate",
            "net_correct_change_per_10000",
            "mean_forward_passes",
        ]]
        .mean()
    )
    summary = strict_metrics.join(strict_safety, how="left").reset_index()
    summary.to_csv(DATA / "q1_lite_main_summary.csv", index=False)
    (TABLES / "q1_lite_main_summary.tex").write_text(
        summary.round(4).to_latex(index=False, escape=False), encoding="utf-8"
    )
    utility_rows = []
    strict_safety_rows = safety[
        safety.stream == "strict_fault_applied_available"
    ]
    for _, row in strict_safety_rows.iterrows():
        corrections = 10000.0 * row.corrections_retained / row.n
        harms = 10000.0 * row.harms_realized / row.n
        for harm_cost in (1.0, 2.0, 5.0, 10.0):
            for pass_penalty in (0.0, 0.5, 2.0):
                utility_rows.append(
                    {
                        "seed": row.seed,
                        "fault_type": row.fault_type,
                        "method": "Lite-CF",
                        "harm_to_correction_cost_ratio": harm_cost,
                        "pass_penalty_equivalent_corrections_per_10000": pass_penalty,
                        "corrections_per_10000": corrections,
                        "harms_per_10000": harms,
                        "mean_forward_passes": row.mean_forward_passes,
                        "net_utility_per_10000": corrections
                        - harm_cost * harms
                        - pass_penalty * (row.mean_forward_passes - 1.0),
                        "energy_note": "forward-pass/FLOP penalty proxy; energy was not measured",
                    }
                )
    pd.DataFrame(utility_rows).to_csv(
        DATA / "q1_lite_decision_utility.csv", index=False
    )

    calibration = pd.read_csv(DATA / "q1_lite_selector_calibration.csv.gz")
    repeated_rows = []
    coefficient_rows = []
    regularization_rows = []
    for seed, frame in calibration.groupby("seed"):
        frame = frame.reset_index(drop=True)
        informative = frame.base_correct != frame.recovery_correct
        target = frame.loc[informative, "recovery_correct"].to_numpy(int)
        for repeat in range(10):
            fit = grouped_binary_fit(
                frame,
                FEATURES_6,
                int(seed) * 100 + repeat,
                c_value=0.05,
            )
            probability = fit["oof"][informative.to_numpy()]
            repeated_rows.append(
                {
                    "seed": seed,
                    "repeat": repeat + 1,
                    "auroc": float(roc_auc_score(target, probability)),
                    "auprc": float(average_precision_score(target, probability)),
                    "brier": float(np.square(probability - target).mean()),
                    "threshold": fit["threshold"],
                    "threshold_tie_count": fit["tie_count"],
                }
            )
            for fold, coefficient in enumerate(fit["coefficients"], start=1):
                for feature, value in zip(FEATURES_6, coefficient):
                    coefficient_rows.append(
                        {
                            "seed": seed,
                            "repeat": repeat + 1,
                            "fold": fold,
                            "feature": feature,
                            "standardized_coefficient": value,
                        }
                    )
        for c_value in (0.01, 0.05, 0.10, 0.50, 1.00, 5.00):
            for repeat in range(5):
                fit = grouped_binary_fit(
                    frame,
                    FEATURES_6,
                    int(seed) * 1000 + repeat + int(100 * c_value),
                    c_value=c_value,
                )
                probability = fit["oof"][informative.to_numpy()]
                regularization_rows.append(
                    {
                        "seed": seed,
                        "repeat": repeat + 1,
                        "C": c_value,
                        "auroc": float(roc_auc_score(target, probability)),
                        "auprc": float(average_precision_score(target, probability)),
                        "brier": float(np.square(probability - target).mean()),
                        "threshold": fit["threshold"],
                        "threshold_tie_count": fit["tie_count"],
                    }
                )
    repeated = pd.DataFrame(repeated_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    repeated.to_csv(DATA / "q1_lite_repeated_grouped_cv.csv", index=False)
    coefficients.to_csv(DATA / "q1_lite_repeated_coefficients.csv", index=False)
    pd.DataFrame(regularization_rows).to_csv(
        DATA / "q1_lite_regularization_path.csv", index=False
    )
    coefficient_stability = (
        coefficients.groupby("feature")
        .standardized_coefficient.agg(
            n="size",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            positive_fraction=lambda values: (values > 0).mean(),
        )
        .reset_index()
    )
    coefficient_stability["dominant_sign_fraction"] = np.maximum(
        coefficient_stability.positive_fraction,
        1 - coefficient_stability.positive_fraction,
    )
    coefficient_stability.to_csv(
        DATA / "q1_lite_coefficient_stability.csv", index=False
    )
    lite_strict = predictions.rename(
        columns={
            f"lite_p{class_index}": f"recovery_p{class_index}"
            for class_index in range(6)
        }
    ).copy()
    lite_strict["prevalence"] = 0.40
    lite_strict["stream"] = "strict_fault_applied_available"
    calibration_size_audit(
        calibration,
        lite_strict,
        output_name="q1_lite_selector_calibration_size.csv",
    )


if __name__ == "__main__":
    main()
