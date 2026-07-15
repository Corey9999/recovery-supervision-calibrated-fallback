"""Reviewer-driven selective-recovery audits from frozen endpoint predictions.

This script never refits the PDRF or RO-PDRF-Lite endpoints.  It fits routing
controls only on the original Batch-7 selector partition and evaluates them on
the unchanged Batches 8--10 strict observations.  The resulting intervals are
therefore fitted-pair/fitting-variation summaries on one fixed test set.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyse_cee_q1_scores import (
    FEATURES_6,
    grouped_binary_fit,
    grouped_three_outcome_fit,
    model_probability_three,
    normalized_entropy,
    probability_metrics,
    selection_metrics,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)

FAULTS = ("gaussian", "offset", "drift", "stuck_at")
RATIOS = (1.0, 2.0, 5.0, 10.0)
SELECTION_GRID = np.linspace(0.0, 1.0, 41)


def probability_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame if column.startswith(prefix)]
    return sorted(columns, key=lambda value: int(value.removeprefix(prefix)))


def endpoint_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pb = frame[probability_columns(frame, "base_p")].to_numpy(float)
    pr = frame[probability_columns(frame, "lite_p")].to_numpy(float)
    return pb, pr, frame.y.to_numpy(int)


def endpoint_features(frame: pd.DataFrame) -> pd.DataFrame:
    pb, pr, _ = endpoint_arrays(frame)
    base_sorted = np.sort(pb, axis=1)
    recovery_sorted = np.sort(pr, axis=1)
    base_entropy = normalized_entropy(pb)
    recovery_entropy = normalized_entropy(pr)
    return pd.DataFrame(
        {
            "base_confidence": pb.max(axis=1),
            "recovery_confidence": pr.max(axis=1),
            "confidence_delta": pr.max(axis=1) - pb.max(axis=1),
            "base_entropy": base_entropy,
            "recovery_entropy": recovery_entropy,
            "entropy_delta": recovery_entropy - base_entropy,
            "margin_delta": (
                recovery_sorted[:, -1] - recovery_sorted[:, -2]
                - base_sorted[:, -1] + base_sorted[:, -2]
            ),
        },
        index=frame.index,
    )


def outcome_code(base_correct: np.ndarray, recovery_correct: np.ndarray) -> np.ndarray:
    outcome = np.ones(len(base_correct), dtype=int)
    outcome[base_correct & ~recovery_correct] = 0
    outcome[~base_correct & recovery_correct] = 2
    return outcome


def candidate_thresholds(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, float)
    finite = score[np.isfinite(score)]
    if not len(finite):
        return np.array([0.0])
    quantiles = np.unique(np.quantile(finite, np.linspace(0, 1, 101)))
    epsilon = max(1e-12, 1e-9 * max(1.0, np.abs(finite).max()))
    return np.r_[finite.max() + epsilon, quantiles[::-1], finite.min() - epsilon]


def choose_utility_threshold(
    score: np.ndarray,
    base_correct: np.ndarray,
    recovery_correct: np.ndarray,
    ratio: float,
) -> tuple[float, dict]:
    rows = []
    for threshold in candidate_thresholds(score):
        metrics = selection_metrics(base_correct, recovery_correct, score >= threshold)
        utility = metrics["corrections_retained"] - ratio * metrics["harms_realized"]
        rows.append((utility, -metrics["harms_realized"], -metrics["recovery_selection_rate"], threshold, metrics))
    selected = max(rows, key=lambda row: row[:4])
    result = dict(selected[4])
    result["utility_events"] = float(selected[0])
    return float(selected[3]), result


def choose_matched_threshold(
    score: np.ndarray,
    base_correct: np.ndarray,
    recovery_correct: np.ndarray,
    target: float,
    metric: str,
) -> tuple[float, dict]:
    rows = []
    for threshold in candidate_thresholds(score):
        metrics = selection_metrics(base_correct, recovery_correct, score >= threshold)
        distance = abs(metrics[metric] - target)
        secondary = (
            -metrics["recovery_retention"]
            if metric == "negative_transfer_prevention"
            else -metrics["negative_transfer_prevention"]
        )
        rows.append((distance, secondary, metrics["harms_realized"], -threshold, threshold, metrics))
    selected = min(rows, key=lambda row: row[:4])
    return float(selected[4]), dict(selected[5])


def grouped_stage1(frame: pd.DataFrame, seed: int) -> dict:
    y = (frame.base_correct.to_numpy(bool) != frame.recovery_correct.to_numpy(bool)).astype(int)
    groups = frame["sample"].to_numpy(int)
    n_splits = max(2, min(5, int(np.bincount(y).min())))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(frame), np.nan)
    for train, valid in splitter.split(frame[FEATURES_6], y, groups):
        model = make_pipeline(
            StandardScaler(),
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
            model.fit(frame.iloc[train][FEATURES_6], y[train])
        oof[valid] = model.predict_proba(frame.iloc[valid][FEATURES_6])[:, 1]
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.05,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ),
    )
    final.fit(frame[FEATURES_6], y)
    return {"model": final, "oof": oof}


def two_stage_fit(frame: pd.DataFrame, seed: int) -> dict:
    stage1 = grouped_stage1(frame, seed + 9000)
    stage2 = grouped_binary_fit(frame, FEATURES_6, seed + 9100, c_value=0.05)
    p_disagreement = stage1["oof"]
    p_recovery_given_disagreement = stage2["oof"]
    p_recovery = p_disagreement * p_recovery_given_disagreement
    p_base = p_disagreement * (1.0 - p_recovery_given_disagreement)
    score = p_recovery - p_base
    return {
        "stage1": stage1,
        "stage2": stage2,
        "oof_probability": np.column_stack([p_base, 1.0 - p_disagreement, p_recovery]),
        "oof_score": score,
    }


def two_stage_probability(fit: dict, frame: pd.DataFrame) -> np.ndarray:
    p_disagreement = fit["stage1"]["model"].predict_proba(frame[FEATURES_6])[:, 1]
    p_recovery_given_disagreement = fit["stage2"]["model"].predict_proba(frame[FEATURES_6])[:, 1]
    return np.column_stack(
        [
            p_disagreement * (1.0 - p_recovery_given_disagreement),
            1.0 - p_disagreement,
            p_disagreement * p_recovery_given_disagreement,
        ]
    )


def policy_metrics(
    frame: pd.DataFrame,
    choose: np.ndarray,
    method: str,
    passes: float,
    ratio: float = 1.0,
) -> dict:
    pb, pr, y = endpoint_arrays(frame)
    base_correct = pb.argmax(axis=1) == y
    recovery_correct = pr.argmax(axis=1) == y
    emitted = np.where(np.asarray(choose)[:, None], pr, pb)
    safety = selection_metrics(base_correct, recovery_correct, choose)
    return {
        "method": method,
        "mean_forward_passes": passes,
        **safety,
        "utility_ratio": ratio,
        "utility_per_10000": 10000.0
        * (safety["corrections_retained"] - ratio * safety["harms_realized"])
        / len(frame),
        **probability_metrics(y, emitted),
    }


def top_fraction(score: np.ndarray, fraction: float) -> np.ndarray:
    n = len(score)
    k = int(round(fraction * n))
    choose = np.zeros(n, dtype=bool)
    if k:
        order = np.lexsort((np.arange(n), -np.asarray(score, float)))
        choose[order[:k]] = True
    return choose


def fitting_summary(frame: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(group, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group, keys))
        if "metric" in base:
            base["measure"] = base.pop("metric")
        for metric in metrics:
            values = part[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            mean = float(values.mean())
            if len(values) > 1:
                half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
            else:
                half = np.nan
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_fitted_cells": len(values),
                    "mean": mean,
                    "fitting_ci_low": mean - half,
                    "fitting_ci_high": mean + half,
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    calibration = pd.read_csv(DATA / "q1_lite_selector_calibration.csv.gz")
    predictions = pd.read_csv(DATA / "q1_lite_routing_predictions.csv.gz")
    support = pd.read_csv(DATA / "strict88_deployment_support.csv").set_index("seed")

    test_score_rows = []
    calibration_score_rows = []
    policy_rows = []
    utility_rows = []
    matched_rows = []
    outcome_rows = []
    confusion_rows = []
    reliability_rows = []
    fit_by_seed: dict[int, dict] = {}

    for seed, cal in calibration.groupby("seed", sort=True):
        seed = int(seed)
        cal = cal.reset_index(drop=True).copy()
        conditional = grouped_binary_fit(cal, FEATURES_6, seed + 5000, c_value=0.05)
        tree = grouped_binary_fit(cal, FEATURES_6, seed + 8000, c_value=0.05, kind="tree")
        allrow = grouped_three_outcome_fit(cal, FEATURES_6, seed + 12000, c_value=0.05)
        two_stage = two_stage_fit(cal, seed)
        base_cal = cal.base_correct.to_numpy(bool)
        recovery_cal = cal.recovery_correct.to_numpy(bool)
        true_outcome = outcome_code(base_cal, recovery_cal)
        allrow_predicted = np.argmax(allrow["oof"], axis=1)

        for truth in range(3):
            for predicted in range(3):
                confusion_rows.append(
                    {
                        "seed": seed,
                        "true_outcome": truth,
                        "predicted_outcome": predicted,
                        "n": int(((true_outcome == truth) & (allrow_predicted == predicted)).sum()),
                    }
                )
        outcome_rows.append(
            {
                "seed": seed,
                "base_better": int((true_outcome == 0).sum()),
                "equivalent": int((true_outcome == 1).sum()),
                "recovery_better": int((true_outcome == 2).sum()),
                "oof_accuracy": float((allrow_predicted == true_outcome).mean()),
                "oof_nll": float(log_loss(true_outcome, allrow["oof"], labels=[0, 1, 2])),
                "oof_brier": float(
                    np.square(allrow["oof"] - np.eye(3)[true_outcome]).sum(axis=1).mean()
                ),
            }
        )
        for outcome_index, label in ((0, "base_better"), (2, "recovery_better")):
            probability = allrow["oof"][:, outcome_index]
            for bin_index, chunk in enumerate(np.array_split(np.argsort(probability), 10), start=1):
                reliability_rows.append(
                    {
                        "seed": seed,
                        "outcome": label,
                        "bin": bin_index,
                        "n": len(chunk),
                        "mean_probability": float(probability[chunk].mean()),
                        "observed_frequency": float((true_outcome[chunk] == outcome_index).mean()),
                    }
                )

        cal_scores = {
            "Conditional Lite-CF": conditional["oof"],
            "All-row multinomial": allrow["score"],
            "Two-stage": two_stage["oof_score"],
            "Shallow tree": tree["oof"],
            "Confidence difference": cal.confidence_delta.to_numpy(float),
            "Base uncertainty": -cal.base_confidence.to_numpy(float),
            "Entropy difference": -cal.entropy_delta.to_numpy(float),
        }
        cal_rng = np.random.default_rng(seed * 100003 + 17)
        cal_scores["Random"] = cal_rng.random(len(cal))
        target = selection_metrics(
            base_cal,
            recovery_cal,
            conditional["oof"] >= conditional["threshold"],
        )
        thresholds = {
            "Conditional Lite-CF": conditional["threshold"],
            "All-row multinomial": allrow["threshold"],
            "Shallow tree": tree["threshold"],
        }
        for ratio in RATIOS:
            for method in ("Conditional Lite-CF", "All-row multinomial", "Two-stage"):
                threshold, cal_metric = choose_utility_threshold(
                    cal_scores[method], base_cal, recovery_cal, ratio
                )
                utility_rows.append(
                    {
                        "analysis_set": "calibration_oof",
                        "seed": seed,
                        "fault_type": "all_calibration_families",
                        "method": method,
                        "threshold_strategy": "utility_optimized_on_calibration",
                        "harm_to_correction_ratio": ratio,
                        "threshold": threshold,
                        **cal_metric,
                    }
                )
                thresholds[f"{method}|utility|{ratio:g}"] = threshold

        for method, score in cal_scores.items():
            for criterion, metric_name in (
                ("matched_prevention", "negative_transfer_prevention"),
                ("matched_retention", "recovery_retention"),
            ):
                threshold, cal_metric = choose_matched_threshold(
                    score, base_cal, recovery_cal, target[metric_name], metric_name
                )
                thresholds[f"{method}|{criterion}"] = threshold
                matched_rows.append(
                    {
                        "analysis_set": "calibration_oof",
                        "seed": seed,
                        "fault_type": "all_calibration_families",
                        "method": method,
                        "criterion": criterion,
                        "threshold": threshold,
                        **cal_metric,
                    }
                )

        fit_by_seed[seed] = {
            "conditional": conditional,
            "tree": tree,
            "allrow": allrow,
            "two_stage": two_stage,
            "thresholds": thresholds,
        }
        for index in range(len(cal)):
            calibration_score_rows.append(
                {
                    "seed": seed,
                    "fault_type": cal.loc[index, "fault_type"],
                    "sample": int(cal.loc[index, "sample"]),
                    "base_correct": bool(base_cal[index]),
                    "recovery_correct": bool(recovery_cal[index]),
                    "outcome": int(true_outcome[index]),
                    **{f"score_{key.lower().replace(' ', '_')}": float(value[index]) for key, value in cal_scores.items()},
                }
            )

    for (seed, fault), frame in predictions.groupby(["seed", "fault_type"], sort=True):
        seed = int(seed)
        frame = frame.reset_index(drop=True).copy()
        derived = endpoint_features(frame)
        for column in derived:
            frame[column] = derived[column]
        fit = fit_by_seed[seed]
        allrow_probability = model_probability_three(fit["allrow"]["model"], frame[FEATURES_6])
        two_stage_probability_test = two_stage_probability(fit["two_stage"], frame)
        scores = {
            "Conditional Lite-CF": fit["conditional"]["model"].predict_proba(frame[FEATURES_6])[:, 1],
            "All-row multinomial": allrow_probability[:, 2] - allrow_probability[:, 0],
            "Two-stage": two_stage_probability_test[:, 2] - two_stage_probability_test[:, 0],
            "Shallow tree": fit["tree"]["model"].predict_proba(frame[FEATURES_6])[:, 1],
            "Confidence difference": frame.confidence_delta.to_numpy(float),
            "Base uncertainty": -frame.base_confidence.to_numpy(float),
            "Entropy difference": -frame.entropy_delta.to_numpy(float),
            "Logit-margin difference": frame.margin_delta.to_numpy(float),
        }
        rng = np.random.default_rng(seed * 100 + FAULTS.index(fault))
        scores["Random"] = rng.random(len(frame))
        supported = bool(support.loc[seed, "deployment_supported"])
        policies = {
            "Always PDRF": (np.zeros(len(frame), bool), 1.0),
            "Always RO-PDRF-Lite": (np.ones(len(frame), bool), 1.0),
            "Frozen Lite-CF (ungated)": (frame.choose_lite.to_numpy(bool), 2.0),
            "Prospective gate Lite-CF": (frame.choose_lite.to_numpy(bool) if supported else np.zeros(len(frame), bool), 2.0 if supported else 1.0),
            "All-row multinomial (safety)": (
                scores["All-row multinomial"] >= fit["thresholds"]["All-row multinomial"],
                2.0,
            ),
        }
        threshold_u1 = fit["thresholds"]["All-row multinomial|utility|1"]
        policies["All-row multinomial (utility 1:1)"] = (
            scores["All-row multinomial"] >= threshold_u1,
            2.0,
        )
        for method, (choose, passes) in policies.items():
            policy_rows.append(
                {
                    "seed": seed,
                    "fault_type": fault,
                    **policy_metrics(frame, choose, method, passes),
                }
            )

        pb, pr, y = endpoint_arrays(frame)
        base_correct = pb.argmax(axis=1) == y
        recovery_correct = pr.argmax(axis=1) == y
        for ratio in RATIOS:
            for method in ("Conditional Lite-CF", "All-row multinomial", "Two-stage"):
                threshold = fit["thresholds"][f"{method}|utility|{ratio:g}"]
                choose = scores[method] >= threshold
                utility_rows.append(
                    {
                        "analysis_set": "fixed_test",
                        "seed": seed,
                        "fault_type": fault,
                        "threshold": threshold,
                        "threshold_strategy": "utility_optimized_on_calibration",
                        **policy_metrics(frame, choose, method, 2.0, ratio),
                        "harm_to_correction_ratio": ratio,
                    }
                )
            fixed_policies = (
                (
                    "Frozen Lite-CF (ungated)",
                    frame.choose_lite.to_numpy(bool),
                    float(fit["thresholds"]["Conditional Lite-CF"]),
                ),
                (
                    "Prospective gate Lite-CF",
                    frame.choose_lite.to_numpy(bool) if supported else np.zeros(len(frame), bool),
                    float(fit["thresholds"]["Conditional Lite-CF"]),
                ),
                (
                    "All-row multinomial (safety)",
                    scores["All-row multinomial"] >= fit["thresholds"]["All-row multinomial"],
                    float(fit["thresholds"]["All-row multinomial"]),
                ),
                ("Always PDRF", np.zeros(len(frame), bool), np.nan),
                ("Always RO-PDRF-Lite", np.ones(len(frame), bool), np.nan),
            )
            for method, choose, threshold in fixed_policies:
                utility_rows.append(
                    {
                        "analysis_set": "fixed_test",
                        "seed": seed,
                        "fault_type": fault,
                        "threshold": threshold,
                        "threshold_strategy": "fixed_safety_or_endpoint",
                        **policy_metrics(
                            frame,
                            choose,
                            method,
                            1.0 if method.startswith("Always") else (2.0 if supported or method != "Prospective gate Lite-CF" else 1.0),
                            ratio,
                        ),
                        "harm_to_correction_ratio": ratio,
                    }
                )
        for method in (
            "Conditional Lite-CF",
            "All-row multinomial",
            "Two-stage",
            "Shallow tree",
            "Confidence difference",
            "Base uncertainty",
            "Entropy difference",
            "Random",
        ):
            for criterion in ("matched_prevention", "matched_retention"):
                threshold = fit["thresholds"][f"{method}|{criterion}"]
                matched_rows.append(
                    {
                        "analysis_set": "fixed_test",
                        "seed": seed,
                        "fault_type": fault,
                        "criterion": criterion,
                        "threshold": threshold,
                        **policy_metrics(frame, scores[method] >= threshold, method, 2.0),
                    }
                )
        for index in range(len(frame)):
            test_score_rows.append(
                {
                    "seed": seed,
                    "fault_type": fault,
                    "sample": int(frame.loc[index, "sample"]),
                    "batch": int(frame.loc[index, "batch"]),
                    "y": int(frame.loc[index, "y"]),
                    "formal_choose_lite": bool(frame.loc[index, "choose_lite"]),
                    "deployment_supported": supported,
                    **{
                        column: float(frame.loc[index, column])
                        for column in [*FEATURES_6, "margin_delta"]
                    },
                    **{f"score_{key.lower().replace(' ', '_').replace('-', '_')}": float(value[index]) for key, value in scores.items()},
                }
            )

    test_scores = pd.DataFrame(test_score_rows)
    cal_scores_output = pd.DataFrame(calibration_score_rows)
    policies = pd.DataFrame(policy_rows)
    utility = pd.DataFrame(utility_rows)
    matched = pd.DataFrame(matched_rows)
    outcomes = pd.DataFrame(outcome_rows)
    confusion = pd.DataFrame(confusion_rows)
    reliability = pd.DataFrame(reliability_rows)
    test_scores.to_csv(DATA / "major_router_test_scores.csv.gz", index=False, compression="gzip")
    cal_scores_output.to_csv(DATA / "major_router_calibration_scores.csv.gz", index=False, compression="gzip")
    policies.to_csv(DATA / "major_primary_policy_metrics.csv", index=False)
    utility.to_csv(DATA / "major_utility_thresholds.csv", index=False)
    matched.to_csv(DATA / "major_matched_control_metrics.csv", index=False)
    outcomes.to_csv(DATA / "major_allrow_outcome_diagnostics.csv", index=False)
    confusion.to_csv(DATA / "major_allrow_confusion.csv", index=False)
    reliability.to_csv(DATA / "major_allrow_probability_calibration.csv", index=False)

    # Full prevention--retention and utility--selection curves, ranked within
    # each fixed mechanism--seed cell.  Curves are descriptive test audits;
    # only the marked operating points use calibration-frozen thresholds.
    curve_rows = []
    score_columns = {
        "Conditional Lite-CF": "score_conditional_lite_cf",
        "All-row multinomial": "score_all_row_multinomial",
        "Two-stage": "score_two_stage",
        "Shallow tree": "score_shallow_tree",
        "Confidence difference": "score_confidence_difference",
        "Entropy difference": "score_entropy_difference",
        "Logit-margin difference": "score_logit_margin_difference",
        "Random": "score_random",
    }
    merged = predictions.merge(
        test_scores,
        on=["seed", "fault_type", "sample", "batch", "y"],
        how="left",
        validate="one_to_one",
    )
    for (seed, fault), frame in merged.groupby(["seed", "fault_type"], sort=True):
        pb, pr, y = endpoint_arrays(frame)
        base_correct = pb.argmax(axis=1) == y
        recovery_correct = pr.argmax(axis=1) == y
        for method, column in score_columns.items():
            for fraction in SELECTION_GRID:
                choose = top_fraction(frame[column].to_numpy(float), float(fraction))
                metrics = selection_metrics(base_correct, recovery_correct, choose)
                curve_rows.append(
                    {
                        "seed": seed,
                        "fault_type": fault,
                        "method": method,
                        "selection_fraction": fraction,
                        **metrics,
                        "utility_ratio1_per_10000": 10000.0
                        * (metrics["corrections_retained"] - metrics["harms_realized"])
                        / len(frame),
                    }
                )
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(DATA / "major_prevention_retention_curves.csv", index=False)
    curve_summary = (
        curves.groupby(["method", "selection_fraction"], as_index=False)[
            ["negative_transfer_prevention", "recovery_retention", "utility_ratio1_per_10000"]
        ]
        .mean()
    )
    curve_summary.to_csv(DATA / "major_prevention_retention_curve_summary.csv", index=False)

    # Full mechanism x batch x method matrix.
    batch_rows = []
    batch_contrast_rows = []
    batch_methods = [
        "Always PDRF",
        "Always RO-PDRF-Lite",
        "Frozen Lite-CF (ungated)",
        "Prospective gate Lite-CF",
        "All-row multinomial (utility 1:1)",
    ]
    for (seed, fault), frame in merged.groupby(["seed", "fault_type"], sort=True):
        fit = fit_by_seed[int(seed)]
        supported = bool(support.loc[int(seed), "deployment_supported"])
        choose_map = {
            "Always PDRF": np.zeros(len(frame), bool),
            "Always RO-PDRF-Lite": np.ones(len(frame), bool),
            "Frozen Lite-CF (ungated)": frame.choose_lite.to_numpy(bool),
            "Prospective gate Lite-CF": frame.choose_lite.to_numpy(bool) if supported else np.zeros(len(frame), bool),
            "All-row multinomial (utility 1:1)": frame.score_all_row_multinomial.to_numpy(float)
            >= fit["thresholds"]["All-row multinomial|utility|1"],
        }
        for batch, take_index in frame.groupby("batch").groups.items():
            positions = frame.index.get_indexer(take_index)
            part = frame.loc[take_index].reset_index(drop=True)
            for method in batch_methods:
                batch_rows.append(
                    {
                        "seed": int(seed),
                        "fault_type": fault,
                        "batch": int(batch),
                        **policy_metrics(
                            part,
                            choose_map[method][positions],
                            method,
                            1.0 if method.startswith("Always") else (2.0 if supported or method != "Prospective gate Lite-CF" else 1.0),
                        ),
                    }
                )
        for view, take in (
            ("B8-9_pooled", frame.batch.isin([8, 9]).to_numpy()),
            ("B10", frame.batch.eq(10).to_numpy()),
        ):
            positions = np.flatnonzero(take)
            part = frame.iloc[positions].reset_index(drop=True)
            for method in batch_methods:
                batch_contrast_rows.append(
                    {
                        "seed": int(seed),
                        "fault_type": fault,
                        "view": view,
                        **policy_metrics(
                            part,
                            choose_map[method][positions],
                            method,
                            1.0 if method.startswith("Always") else (2.0 if supported or method != "Prospective gate Lite-CF" else 1.0),
                        ),
                    }
                )
    batch_matrix = pd.DataFrame(batch_rows)
    batch_matrix.to_csv(DATA / "major_mechanism_batch_method_matrix.csv", index=False)
    batch_contrasts = pd.DataFrame(batch_contrast_rows)
    batch_contrasts.to_csv(DATA / "major_batch_contrast_metrics.csv", index=False)
    batch_summary = fitting_summary(
        batch_matrix,
        ["method", "fault_type", "batch"],
        [
            "accuracy",
            "macro_auroc",
            "correction_opportunities",
            "harm_opportunities",
            "negative_transfer_prevention",
            "recovery_retention",
            "utility_per_10000",
        ],
    )
    batch_summary.to_csv(DATA / "major_mechanism_batch_fitting_summary.csv", index=False)

    # Observation-weighted, equal-batch, worst-batch and B8--9/B10 views.
    robustness_rows = []
    for method, method_frame in batch_matrix.groupby("method"):
        for seed, seed_frame in method_frame.groupby("seed"):
            for metric in ("accuracy", "macro_auroc", "utility_per_10000"):
                overall = policies[
                    policies.method.eq(method) & policies.seed.eq(seed)
                ][metric].mean()
                contrast = batch_contrasts[
                    batch_contrasts.method.eq(method)
                    & batch_contrasts.seed.eq(seed)
                ]
                robustness_rows.extend(
                    [
                        {"method": method, "seed": seed, "view": "observation_weighted_overall", "metric": metric, "value": float(overall)},
                        {"method": method, "seed": seed, "view": "equal_weight_batches", "metric": metric, "value": float(seed_frame.groupby("batch")[metric].mean().mean())},
                        {"method": method, "seed": seed, "view": "worst_batch", "metric": metric, "value": float(seed_frame.groupby("batch")[metric].mean().min())},
                        {"method": method, "seed": seed, "view": "B8-9_pooled", "metric": metric, "value": float(contrast[contrast.view.eq("B8-9_pooled")][metric].mean())},
                        {"method": method, "seed": seed, "view": "B10", "metric": metric, "value": float(contrast[contrast.view.eq("B10")][metric].mean())},
                    ]
                )
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(DATA / "major_batch_robustness_views.csv", index=False)
    robustness_summary = fitting_summary(robustness, ["method", "view"], ["value"])
    robustness_summary = robustness_summary.merge(
        robustness[["method", "view", "metric"]].drop_duplicates(),
        on=["method", "view"],
        how="left",
    )
    # The merge above is many-to-many when several metrics share a view; make
    # the correct metric-specific summary directly instead.
    robustness_summary = fitting_summary(
        robustness.rename(columns={"value": "estimate"}),
        ["method", "view", "metric"],
        ["estimate"],
    )
    robustness_summary.to_csv(DATA / "major_batch_robustness_summary.csv", index=False)

    # Sensitivity to the mechanism mixture used for selector calibration.
    mixture_rows = []
    mixture_scenarios = {
        "uniform": dict.fromkeys(FAULTS, 0.25),
        "gaussian_70pct": {"gaussian": 0.70, "offset": 0.10, "drift": 0.10, "stuck_at": 0.10},
        "offset_70pct": {"gaussian": 0.10, "offset": 0.70, "drift": 0.10, "stuck_at": 0.10},
        "drift_70pct": {"gaussian": 0.10, "offset": 0.10, "drift": 0.70, "stuck_at": 0.10},
        "stuck_at_70pct": {"gaussian": 0.10, "offset": 0.10, "drift": 0.10, "stuck_at": 0.70},
    }
    for seed, cal in calibration.groupby("seed", sort=True):
        seed = int(seed)
        n_total = len(cal)
        for scenario_index, (scenario, weights) in enumerate(mixture_scenarios.items()):
            sampled = []
            for fault_index, fault in enumerate(FAULTS):
                n_fault = int(round(n_total * weights[fault]))
                sampled.append(
                    cal[cal.fault_type.eq(fault)].sample(
                        n=n_fault,
                        replace=True,
                        random_state=seed * 1000 + scenario_index * 10 + fault_index,
                    )
                )
            train = pd.concat(sampled, ignore_index=True)
            fit = grouped_binary_fit(train, FEATURES_6, seed + 24000 + scenario_index, c_value=0.05)
            for fault in FAULTS:
                test = merged[(merged.seed.eq(seed)) & (merged.fault_type.eq(fault))].reset_index(drop=True)
                score = fit["model"].predict_proba(test[FEATURES_6])[:, 1]
                mixture_rows.append(
                    {
                        "mixture": scenario,
                        "test_mechanism": fault,
                        "seed": seed,
                        "threshold": fit["threshold"],
                        **policy_metrics(test, score >= fit["threshold"], "Conditional Lite-CF", 2.0),
                    }
                )
    mixtures = pd.DataFrame(mixture_rows)
    mixtures.to_csv(DATA / "major_calibration_mixture_sensitivity.csv", index=False)
    fitting_summary(
        mixtures,
        ["mixture", "test_mechanism"],
        ["negative_transfer_prevention", "recovery_retention", "utility_per_10000", "macro_auroc"],
    ).to_csv(DATA / "major_calibration_mixture_summary.csv", index=False)

    # Leave-one-mechanism-out and Gaussian-only selector calibration.
    transport_rows = []
    for seed, cal in calibration.groupby("seed", sort=True):
        seed = int(seed)
        for heldout in FAULTS:
            for model_name in ("Conditional Lite-CF", "All-row multinomial"):
                train = cal[cal.fault_type.ne(heldout)].reset_index(drop=True)
                if model_name == "Conditional Lite-CF":
                    fit = grouped_binary_fit(train, FEATURES_6, seed + 20000 + FAULTS.index(heldout), c_value=0.05)
                else:
                    fit = grouped_three_outcome_fit(train, FEATURES_6, seed + 21000 + FAULTS.index(heldout), c_value=0.05)
                test = merged[(merged.seed.eq(seed)) & (merged.fault_type.eq(heldout))].reset_index(drop=True)
                if model_name == "Conditional Lite-CF":
                    score = fit["model"].predict_proba(test[FEATURES_6])[:, 1]
                else:
                    prob = model_probability_three(fit["model"], test[FEATURES_6])
                    score = prob[:, 2] - prob[:, 0]
                transport_rows.append(
                    {
                        "calibration_regime": "leave_one_mechanism_out",
                        "heldout_or_test_mechanism": heldout,
                        "seed": seed,
                        **policy_metrics(test, score >= fit["threshold"], model_name, 2.0),
                    }
                )
        gaussian = cal[cal.fault_type.eq("gaussian")].reset_index(drop=True)
        for model_name in ("Conditional Lite-CF", "All-row multinomial"):
            if model_name == "Conditional Lite-CF":
                fit = grouped_binary_fit(gaussian, FEATURES_6, seed + 22000, c_value=0.05)
            else:
                fit = grouped_three_outcome_fit(gaussian, FEATURES_6, seed + 23000, c_value=0.05)
            for fault in FAULTS:
                test = merged[(merged.seed.eq(seed)) & (merged.fault_type.eq(fault))].reset_index(drop=True)
                if model_name == "Conditional Lite-CF":
                    score = fit["model"].predict_proba(test[FEATURES_6])[:, 1]
                else:
                    prob = model_probability_three(fit["model"], test[FEATURES_6])
                    score = prob[:, 2] - prob[:, 0]
                transport_rows.append(
                    {
                        "calibration_regime": "gaussian_only",
                        "heldout_or_test_mechanism": fault,
                        "seed": seed,
                        **policy_metrics(test, score >= fit["threshold"], model_name, 2.0),
                    }
                )
    transport = pd.DataFrame(transport_rows)
    transport.to_csv(DATA / "major_mechanism_transport.csv", index=False)
    fitting_summary(
        transport,
        ["calibration_regime", "method", "heldout_or_test_mechanism"],
        ["negative_transfer_prevention", "recovery_retention", "utility_per_10000", "macro_auroc"],
    ).to_csv(DATA / "major_mechanism_transport_summary.csv", index=False)

    # Engineering quantities from the policy event ledger and fixed-thread timing.
    engineering_rows = []
    for _, row in policies.iterrows():
        extra_passes = row.n * max(0.0, row.mean_forward_passes - 1.0)
        net = row.corrections_retained - row.harms_realized
        selected = row.recovery_selection_rate * row.n
        engineering_rows.append(
            {
                "seed": row.seed,
                "fault_type": row.fault_type,
                "method": row.method,
                "extra_passes": extra_passes,
                "extra_passes_per_retained_correction": extra_passes / row.corrections_retained if row.corrections_retained else np.inf,
                "extra_passes_per_net_correction": extra_passes / net if net > 0 else np.inf,
                "number_needed_to_route_per_retained_correction": selected / row.corrections_retained if row.corrections_retained else np.inf,
                "number_needed_to_route_per_net_correction": selected / net if net > 0 else np.inf,
            }
        )
    engineering = pd.DataFrame(engineering_rows)
    engineering.to_csv(DATA / "major_engineering_event_efficiency.csv", index=False)

    timing = pd.read_csv(DATA / "strict88_cpu_thread_sensitivity.csv")
    timing_wide = timing.pivot_table(
        index=["seed", "requested_intraop_threads", "batch_size"],
        columns="method",
        values=["latency_batch1_ms_median", "latency_batch_ms_median"],
    )
    timing_rows = []
    for index, row in timing_wide.iterrows():
        seed, threads, batch_size = index
        for latency_field, batch_label in (("latency_batch1_ms_median", 1), ("latency_batch_ms_median", int(batch_size))):
            base_ms = row[(latency_field, "PDRF")]
            recovery_ms = row[(latency_field, "RO-PDRF-Lite")]
            routed_ms = row[(latency_field, "Lite-CF-6")]
            timing_rows.append(
                {
                    "seed": seed,
                    "threads": threads,
                    "batch_size": batch_label,
                    "base_ms": base_ms,
                    "recovery_ms": recovery_ms,
                    "routed_ms": routed_ms,
                    "incremental_vs_base_ms": routed_ms - base_ms,
                    "router_orchestration_residual_ms": routed_ms - base_ms - recovery_ms,
                }
            )
    timing_audit = pd.DataFrame(timing_rows)
    timing_audit.to_csv(DATA / "major_router_latency_decomposition.csv", index=False)

    # Compact manuscript tables.
    primary_summary = fitting_summary(
        policies,
        ["method"],
        ["accuracy", "macro_auroc", "negative_transfer_prevention", "recovery_retention", "recovery_selection_rate", "utility_per_10000"],
    )
    primary_summary.to_csv(DATA / "major_primary_policy_summary.csv", index=False)
    primary_wide = primary_summary.pivot(index="method", columns="metric", values="mean").reset_index()
    # The compact publication table is maintained separately so that labels,
    # percentages and policy ordering remain journal-readable.  The complete
    # machine-generated values are in major_primary_policy_summary.csv.
    matched_test = matched[matched.analysis_set.eq("fixed_test")]
    matched_summary = fitting_summary(
        matched_test,
        ["criterion", "method"],
        ["negative_transfer_prevention", "recovery_retention", "utility_per_10000", "macro_auroc"],
    )
    matched_summary.to_csv(DATA / "major_matched_control_summary.csv", index=False)
    utility_test = utility[utility.analysis_set.eq("fixed_test")]
    utility_summary = fitting_summary(
        utility_test,
        ["threshold_strategy", "method", "harm_to_correction_ratio"],
        ["negative_transfer_prevention", "recovery_retention", "utility_per_10000", "macro_auroc"],
    )
    utility_summary.to_csv(DATA / "major_utility_threshold_summary.csv", index=False)

    compute_summary = (
        policies.groupby("method", as_index=False)
        .agg(
            macro_auroc=("macro_auroc", "mean"),
            accuracy=("accuracy", "mean"),
            mean_forward_passes=("mean_forward_passes", "mean"),
            utility_per_10000=("utility_per_10000", "mean"),
        )
    )
    compute_summary.to_csv(DATA / "major_auroc_compute_summary.csv", index=False)

    # Determine empirical Pareto dominance of the frozen operating point using
    # averaged descriptive test curves.  This is not used to tune a policy.
    frozen = policies[policies.method.eq("Frozen Lite-CF (ungated)")]
    frozen_prevention = float(frozen.negative_transfer_prevention.mean())
    frozen_retention = float(frozen.recovery_retention.mean())
    dominance = []
    for method, part in curve_summary.groupby("method"):
        dominates = part[
            (part.negative_transfer_prevention >= frozen_prevention)
            & (part.recovery_retention >= frozen_retention)
            & (
                (part.negative_transfer_prevention > frozen_prevention)
                | (part.recovery_retention > frozen_retention)
            )
        ]
        if len(dominates):
            best = dominates.sort_values(
                ["negative_transfer_prevention", "recovery_retention"], ascending=False
            ).iloc[0]
            dominance.append(
                {
                    "comparator": method,
                    "dominates_frozen_operating_point": True,
                    "selection_fraction": best.selection_fraction,
                    "prevention": best.negative_transfer_prevention,
                    "retention": best.recovery_retention,
                }
            )
        else:
            dominance.append(
                {
                    "comparator": method,
                    "dominates_frozen_operating_point": False,
                    "selection_fraction": np.nan,
                    "prevention": np.nan,
                    "retention": np.nan,
                }
            )
    pd.DataFrame(dominance).to_csv(DATA / "major_pareto_dominance.csv", index=False)

    design = {
        "run": "CEE-CF10-R2-LITE-MAJOR-ROUTER-AUDIT",
        "endpoint_predictions": "frozen; no endpoint refitting",
        "selector_partition": "Batch-7 second chronological half",
        "test_partition": "Batches 8--10 unchanged",
        "fault_mechanisms": list(FAULTS),
        "utility_harm_to_correction_ratios": list(RATIOS),
        "curve_status": "descriptive fixed-test curves; no test-selected operating point",
        "interval_interpretation": "fitted-pair/fitting variation on one fixed test set",
        "primary_deployment_policy": "post hoc support gate audited retrospectively; prospective use would freeze the rule and unsupported fits would revert to PDRF",
    }
    (DATA / "major_router_audit_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(primary_wide.round(4).to_string(index=False))
    print("\nPareto dominance of frozen point")
    print(pd.DataFrame(dominance).to_string(index=False))


if __name__ == "__main__":
    main()
