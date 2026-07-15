"""Analyse severity, sensor-group and probability-calibration transport."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from analyse_cee_q1_scores import (
    FEATURES_6,
    grouped_binary_fit,
    grouped_three_outcome_fit,
    model_probability_three,
    normalized_entropy,
    selection_metrics,
)
from analyse_cee_major_router_audits import (
    FAULTS,
    choose_utility_threshold,
    endpoint_arrays,
    fitting_summary,
    policy_metrics,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
TABLES = ROOT / "tables"
SEVERITIES = (1.0, 2.0, 3.0)


def probability_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame if column.startswith(prefix)]
    return sorted(columns, key=lambda value: int(value.removeprefix(prefix)))


def restore_uncalibrated(probability: np.ndarray, temperature: float) -> np.ndarray:
    raw = np.power(np.clip(probability, 1e-12, 1.0), float(temperature))
    return raw / raw.sum(axis=1, keepdims=True)


def add_endpoint_features(
    frame: pd.DataFrame,
    base_temperature: float | None = None,
    lite_temperature: float | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    pb = output[probability_columns(output, "base_p")].to_numpy(float)
    pr = output[probability_columns(output, "lite_p")].to_numpy(float)
    if base_temperature is not None:
        pb = restore_uncalibrated(pb, base_temperature)
        pr = restore_uncalibrated(pr, lite_temperature)
    base_sorted = np.sort(pb, axis=1)
    recovery_sorted = np.sort(pr, axis=1)
    base_entropy = normalized_entropy(pb)
    recovery_entropy = normalized_entropy(pr)
    output["base_confidence"] = pb.max(axis=1)
    output["recovery_confidence"] = pr.max(axis=1)
    output["confidence_delta"] = output.recovery_confidence - output.base_confidence
    output["base_entropy"] = base_entropy
    output["recovery_entropy"] = recovery_entropy
    output["entropy_delta"] = recovery_entropy - base_entropy
    output["margin_delta"] = (
        recovery_sorted[:, -1] - recovery_sorted[:, -2]
        - base_sorted[:, -1] + base_sorted[:, -2]
    )
    base_class = pb.argmax(axis=1)
    recovery_class = pr.argmax(axis=1)
    for class_index in range(pb.shape[1]):
        output[f"base_class_{class_index}"] = (base_class == class_index).astype(float)
        output[f"recovery_class_{class_index}"] = (recovery_class == class_index).astype(float)
    return output


def safety_threshold(score: np.ndarray, frame: pd.DataFrame) -> float:
    base_correct = frame.base_correct.to_numpy(bool)
    recovery_correct = frame.recovery_correct.to_numpy(bool)
    rows = []
    for threshold in np.linspace(0.50, 0.90, 17):
        metrics = selection_metrics(base_correct, recovery_correct, score >= threshold)
        rows.append(
            (
                metrics["harms_realized"],
                -metrics["selected_accuracy"],
                -metrics["corrections_retained"],
                -threshold,
                threshold,
            )
        )
    return float(min(rows)[4])


def rank_grouped_fit(frame: pd.DataFrame, seed: int) -> dict:
    frame = frame.reset_index(drop=True)
    informative = frame.base_correct.to_numpy(bool) != frame.recovery_correct.to_numpy(bool)
    y = frame.recovery_correct.to_numpy(int)
    groups = frame["sample"].to_numpy(int)
    stratify = np.where(informative, y, 2)
    counts = np.bincount(stratify)
    n_splits = max(2, min(5, int(counts[counts > 0].min())))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(frame), np.nan)
    for train, valid in splitter.split(frame[FEATURES_6], stratify, groups):
        fit_rows = train[informative[train]]
        model = make_pipeline(
            QuantileTransformer(
                n_quantiles=min(100, len(fit_rows)),
                output_distribution="normal",
                random_state=0,
            ),
            LogisticRegression(
                C=0.05,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(frame.iloc[fit_rows][FEATURES_6], y[fit_rows])
        oof[valid] = model.predict_proba(frame.iloc[valid][FEATURES_6])[:, 1]
    final = make_pipeline(
        QuantileTransformer(
            n_quantiles=min(100, int(informative.sum())),
            output_distribution="normal",
            random_state=0,
        ),
        LogisticRegression(
            C=0.05,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ),
    )
    final.fit(frame.loc[informative, FEATURES_6], y[informative])
    return {
        "model": final,
        "oof": oof,
        "threshold": safety_threshold(oof, frame),
    }


def binary_ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        take = (probability >= lower) & (
            probability < upper if upper < 1 else probability <= upper
        )
        if take.any():
            result += take.mean() * abs(y[take].mean() - probability[take].mean())
    return float(result)


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    calibration_raw = pd.read_csv(DATA / "major_shift_selector_calibration.csv.gz")
    test_raw = pd.read_csv(DATA / "major_shift_test_predictions.csv.gz")
    temperatures = pd.read_csv(DATA / "major_shift_temperatures.csv").set_index("seed")

    severity_rows = []
    group_rows = []
    endpoint_calibration_rows = []
    score_calibration_rows = []
    feature_control_rows = []

    for seed in sorted(calibration_raw.seed.unique()):
        seed = int(seed)
        temperature = temperatures.loc[seed]
        calibration_by_scale = {}
        test_by_condition = {}
        for severity in SEVERITIES:
            calibration_by_scale[severity] = add_endpoint_features(
                calibration_raw[
                    (calibration_raw.seed.eq(seed))
                    & (calibration_raw.severity.eq(severity))
                ].reset_index(drop=True)
            )
        for (severity, group), part in test_raw[test_raw.seed.eq(seed)].groupby(
            ["severity", "affected_group"]
        ):
            test_by_condition[(float(severity), int(group))] = add_endpoint_features(
                part.reset_index(drop=True)
            )

        # Severity source-to-target matrix.
        for source_scale in SEVERITIES:
            cal = calibration_by_scale[source_scale]
            conditional = grouped_binary_fit(
                cal, FEATURES_6, seed + 5000, c_value=0.05
            )
            allrow = grouped_three_outcome_fit(
                cal, FEATURES_6, seed + 12000, c_value=0.05
            )
            for target_scale in SEVERITIES:
                test = test_by_condition[(target_scale, 1)]
                for fault, part in test.groupby("fault_type"):
                    part = part.reset_index(drop=True)
                    conditional_score = conditional["model"].predict_proba(
                        part[FEATURES_6]
                    )[:, 1]
                    probability = model_probability_three(allrow["model"], part[FEATURES_6])
                    allrow_score = probability[:, 2] - probability[:, 0]
                    for method, score, threshold in (
                        ("Conditional Lite-CF", conditional_score, conditional["threshold"]),
                        ("All-row multinomial", allrow_score, allrow["threshold"]),
                    ):
                        severity_rows.append(
                            {
                                "seed": seed,
                                "calibration_severity": source_scale,
                                "test_severity": target_scale,
                                "fault_type": fault,
                                "threshold": threshold,
                                **policy_metrics(part, score >= threshold, method, 2.0),
                            }
                        )

        # Group transport: calibrate only on group 1, scale 3; test groups 1--4.
        cal = calibration_by_scale[3.0]
        conditional = grouped_binary_fit(cal, FEATURES_6, seed + 5000, c_value=0.05)
        allrow = grouped_three_outcome_fit(cal, FEATURES_6, seed + 12000, c_value=0.05)
        for group in (1, 2, 3, 4):
            test = test_by_condition[(3.0, group)]
            for fault, part in test.groupby("fault_type"):
                part = part.reset_index(drop=True)
                conditional_score = conditional["model"].predict_proba(part[FEATURES_6])[:, 1]
                probability = model_probability_three(allrow["model"], part[FEATURES_6])
                allrow_score = probability[:, 2] - probability[:, 0]
                for method, score, threshold in (
                    ("Conditional Lite-CF", conditional_score, conditional["threshold"]),
                    ("All-row multinomial", allrow_score, allrow["threshold"]),
                ):
                    group_rows.append(
                        {
                            "seed": seed,
                            "calibration_group": 1,
                            "test_group": group,
                            "fault_type": fault,
                            "threshold": threshold,
                            **policy_metrics(part, score >= threshold, method, 2.0),
                        }
                    )

        # Temperature-scaled versus uncalibrated endpoint features.
        cal_scaled = calibration_by_scale[3.0]
        cal_uncal = add_endpoint_features(
            calibration_raw[
                (calibration_raw.seed.eq(seed))
                & (calibration_raw.severity.eq(3.0))
            ].reset_index(drop=True),
            float(temperature.base_temperature),
            float(temperature.lite_temperature),
        )
        test_scaled = test_by_condition[(3.0, 1)]
        test_uncal = add_endpoint_features(
            test_raw[
                (test_raw.seed.eq(seed))
                & (test_raw.severity.eq(3.0))
                & (test_raw.affected_group.eq(1))
            ].reset_index(drop=True),
            float(temperature.base_temperature),
            float(temperature.lite_temperature),
        )
        for feature_regime, cal_frame, test_frame in (
            ("temperature_scaled", cal_scaled, test_scaled),
            ("uncalibrated_endpoints", cal_uncal, test_uncal),
        ):
            fit = grouped_binary_fit(
                cal_frame, FEATURES_6, seed + 5000, c_value=0.05
            )
            for fault, part in test_frame.groupby("fault_type"):
                part = part.reset_index(drop=True)
                score = fit["model"].predict_proba(part[FEATURES_6])[:, 1]
                pb, pr, y = endpoint_arrays(part)
                base_correct = pb.argmax(axis=1) == y
                recovery_correct = pr.argmax(axis=1) == y
                informative = base_correct != recovery_correct
                endpoint_calibration_rows.append(
                    {
                        "seed": seed,
                        "fault_type": fault,
                        "feature_regime": feature_regime,
                        "threshold": fit["threshold"],
                        "preference_auroc": roc_auc_score(
                            recovery_correct[informative].astype(int), score[informative]
                        ),
                        **policy_metrics(
                            part,
                            score >= fit["threshold"],
                            "Conditional Lite-CF",
                            2.0,
                        ),
                    }
                )

        # Conditional-score calibration: raw, Platt and isotonic.  Monotone
        # transformations map the frozen safety threshold; they do not use test
        # outcomes to choose a new operating point.
        fit = grouped_binary_fit(cal_scaled, FEATURES_6, seed + 5000, c_value=0.05)
        informative_cal = cal_scaled.base_correct.ne(cal_scaled.recovery_correct).to_numpy()
        y_cal = cal_scaled.loc[informative_cal, "recovery_correct"].to_numpy(int)
        oof = fit["oof"]
        platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0)
        platt.fit(logit(oof[informative_cal]).reshape(-1, 1), y_cal)
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        isotonic.fit(oof[informative_cal], y_cal)
        mapped_thresholds = {
            "raw_logistic": fit["threshold"],
            "platt": float(platt.predict_proba(logit(np.array([fit["threshold"]])).reshape(-1, 1))[0, 1]),
            "isotonic": float(isotonic.predict([fit["threshold"]])[0]),
        }
        for fault, part in test_scaled.groupby("fault_type"):
            part = part.reset_index(drop=True)
            raw_score = fit["model"].predict_proba(part[FEATURES_6])[:, 1]
            transformed = {
                "raw_logistic": raw_score,
                "platt": platt.predict_proba(logit(raw_score).reshape(-1, 1))[:, 1],
                "isotonic": isotonic.predict(raw_score),
            }
            pb, pr, y = endpoint_arrays(part)
            base_correct = pb.argmax(axis=1) == y
            recovery_correct = pr.argmax(axis=1) == y
            informative = base_correct != recovery_correct
            preference_y = recovery_correct[informative].astype(int)
            for calibration_method, score in transformed.items():
                probability = np.clip(score[informative], 1e-6, 1 - 1e-6)
                score_calibration_rows.append(
                    {
                        "seed": seed,
                        "fault_type": fault,
                        "score_calibration": calibration_method,
                        "threshold": mapped_thresholds[calibration_method],
                        "preference_n": int(informative.sum()),
                        "preference_auroc": roc_auc_score(preference_y, probability),
                        "preference_nll": log_loss(preference_y, probability, labels=[0, 1]),
                        "preference_brier": float(np.square(probability - preference_y).mean()),
                        "preference_ece10": binary_ece(preference_y, probability),
                        **policy_metrics(
                            part,
                            score >= mapped_thresholds[calibration_method],
                            "Conditional Lite-CF",
                            2.0,
                        ),
                    }
                )

        # Rank-normalized and class-conditional controls, plus a direct margin.
        class_features = FEATURES_6 + [
            *(f"base_class_{index}" for index in range(6)),
            *(f"recovery_class_{index}" for index in range(6)),
        ]
        rank_fit = rank_grouped_fit(cal_scaled, seed + 35000)
        class_fit = grouped_binary_fit(
            cal_scaled, class_features, seed + 35100, c_value=0.05
        )
        margin_threshold, _ = choose_utility_threshold(
            cal_scaled.margin_delta.to_numpy(float),
            cal_scaled.base_correct.to_numpy(bool),
            cal_scaled.recovery_correct.to_numpy(bool),
            1.0,
        )
        for fault, part in test_scaled.groupby("fault_type"):
            part = part.reset_index(drop=True)
            controls = (
                (
                    "rank_normalized_logistic",
                    rank_fit["model"].predict_proba(part[FEATURES_6])[:, 1],
                    rank_fit["threshold"],
                ),
                (
                    "class_conditional_logistic",
                    class_fit["model"].predict_proba(part[class_features])[:, 1],
                    class_fit["threshold"],
                ),
                ("logit_margin_difference", part.margin_delta.to_numpy(float), margin_threshold),
            )
            pb, pr, y = endpoint_arrays(part)
            base_correct = pb.argmax(axis=1) == y
            recovery_correct = pr.argmax(axis=1) == y
            informative = base_correct != recovery_correct
            for control, score, threshold in controls:
                feature_control_rows.append(
                    {
                        "seed": seed,
                        "fault_type": fault,
                        "control": control,
                        "threshold": threshold,
                        "preference_auroc": roc_auc_score(
                            recovery_correct[informative].astype(int), score[informative]
                        ),
                        **policy_metrics(part, score >= threshold, control, 2.0),
                    }
                )

    outputs = {
        "major_severity_transport.csv": pd.DataFrame(severity_rows),
        "major_fault_group_transport.csv": pd.DataFrame(group_rows),
        "major_endpoint_calibration_router_effect.csv": pd.DataFrame(endpoint_calibration_rows),
        "major_score_calibration_effect.csv": pd.DataFrame(score_calibration_rows),
        "major_feature_router_controls.csv": pd.DataFrame(feature_control_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(DATA / name, index=False)

    fitting_summary(
        outputs["major_severity_transport.csv"],
        ["method", "calibration_severity", "test_severity"],
        ["negative_transfer_prevention", "recovery_retention", "utility_per_10000", "macro_auroc"],
    ).to_csv(DATA / "major_severity_transport_summary.csv", index=False)
    fitting_summary(
        outputs["major_fault_group_transport.csv"],
        ["method", "test_group"],
        ["negative_transfer_prevention", "recovery_retention", "utility_per_10000", "macro_auroc"],
    ).to_csv(DATA / "major_fault_group_transport_summary.csv", index=False)
    fitting_summary(
        outputs["major_endpoint_calibration_router_effect.csv"],
        ["feature_regime"],
        ["preference_auroc", "negative_transfer_prevention", "recovery_retention", "utility_per_10000", "macro_auroc"],
    ).to_csv(DATA / "major_endpoint_calibration_router_summary.csv", index=False)
    fitting_summary(
        outputs["major_score_calibration_effect.csv"],
        ["score_calibration"],
        ["preference_auroc", "preference_nll", "preference_brier", "preference_ece10", "negative_transfer_prevention", "recovery_retention", "utility_per_10000"],
    ).to_csv(DATA / "major_score_calibration_summary.csv", index=False)
    fitting_summary(
        outputs["major_feature_router_controls.csv"],
        ["control"],
        ["preference_auroc", "negative_transfer_prevention", "recovery_retention", "utility_per_10000", "macro_auroc"],
    ).to_csv(DATA / "major_feature_router_control_summary.csv", index=False)

    print("severity transport")
    print(
        pd.read_csv(DATA / "major_severity_transport_summary.csv")
        .query("metric in ['negative_transfer_prevention','recovery_retention','utility_per_10000']")
        .pivot_table(
            index=["method", "calibration_severity", "test_severity"],
            columns="metric",
            values="mean",
        )
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
