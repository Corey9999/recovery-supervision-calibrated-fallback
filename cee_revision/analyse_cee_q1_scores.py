"""Major-revision analyses requested in Q1_Reviewer_Report_CEE5_with_Scores.

The script reuses the frozen CEE-CF10-R2 endpoint predictions. It does not
retrain the PDRF endpoints. New routing models are fitted only on the original
Batch-7 selector partition and are evaluated on the unchanged Batches 8--10
test observations.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)

FEATURES_12 = [
    "base_confidence",
    "recovery_confidence",
    "confidence_delta",
    "base_entropy",
    "recovery_entropy",
    "entropy_delta",
    "base_recovery_js",
    "base_consensus_js",
    "recovery_consensus_js",
    "base_consensus_agreement",
    "recovery_consensus_agreement",
    "removal_fraction",
]
FEATURES_6 = FEATURES_12[:6]
CLASSES = np.arange(6)
SAFE_GRID = np.linspace(0.50, 0.90, 17)


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1) / np.log(q.shape[1])


def js_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.clip(a, 1e-12, 1.0)
    b = np.clip(b, 1e-12, 1.0)
    midpoint = 0.5 * (a + b)
    return 0.5 * (
        (a * np.log(a / midpoint)).sum(axis=1)
        + (b * np.log(b / midpoint)).sum(axis=1)
    )


def probability_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame if column.startswith(prefix)]
    return sorted(columns, key=lambda value: int(value.removeprefix(prefix)))


def endpoint_features(frame: pd.DataFrame) -> pd.DataFrame:
    pb = frame[probability_columns(frame, "base_p")].to_numpy(float)
    pr = frame[probability_columns(frame, "recovery_p")].to_numpy(float)
    base_confidence = pb.max(axis=1)
    recovery_confidence = pr.max(axis=1)
    base_entropy = normalized_entropy(pb)
    recovery_entropy = normalized_entropy(pr)
    return pd.DataFrame(
        {
            "base_confidence": base_confidence,
            "recovery_confidence": recovery_confidence,
            "confidence_delta": recovery_confidence - base_confidence,
            "base_entropy": base_entropy,
            "recovery_entropy": recovery_entropy,
            "entropy_delta": recovery_entropy - base_entropy,
            "base_recovery_js": js_rows(pb, pr),
        },
        index=frame.index,
    )


def fixed_ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        take = (confidence >= lower) & (
            confidence < upper if upper < 1.0 else confidence <= upper
        )
        if take.any():
            value += take.mean() * abs(correct[take].mean() - confidence[take].mean())
    return float(value)


def adaptive_ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    chunks = np.array_split(np.argsort(confidence), bins)
    return float(
        sum(
            len(chunk) / len(y)
            * abs(correct[chunk].mean() - confidence[chunk].mean())
            for chunk in chunks
            if len(chunk)
        )
    )


def classwise_ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    values = []
    for class_index in range(p.shape[1]):
        target = y == class_index
        confidence = p[:, class_index]
        value = 0.0
        for lower, upper in zip(edges[:-1], edges[1:]):
            take = (confidence >= lower) & (
                confidence < upper if upper < 1.0 else confidence <= upper
            )
            if take.any():
                value += take.mean() * abs(target[take].mean() - confidence[take].mean())
        values.append(value)
    return float(np.mean(values))


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-12, None)
    p = p / p.sum(axis=1, keepdims=True)
    y = np.asarray(y, int)
    onehot = label_binarize(y, classes=CLASSES)
    auroc_values = []
    auprc_values = []
    for class_index in range(p.shape[1]):
        target = onehot[:, class_index]
        if target.min() != target.max():
            auroc_values.append(roc_auc_score(target, p[:, class_index]))
        if target.sum() > 0:
            auprc_values.append(average_precision_score(target, p[:, class_index]))
    return {
        "accuracy": float((p.argmax(axis=1) == y).mean()),
        "macro_auroc": float(np.mean(auroc_values)),
        "macro_auprc": float(np.mean(auprc_values)),
        "nll": float(log_loss(y, p, labels=CLASSES)),
        "brier": float(np.square(p - onehot).sum(axis=1).mean()),
        "ece15_equal_width": fixed_ece(y, p, 15),
        "ece15_equal_mass": adaptive_ece(y, p, 15),
        "classwise_ece15": classwise_ece(y, p, 15),
    }


def classwise_metrics(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for class_index in CLASSES:
        target = y == class_index
        predicted = p.argmax(axis=1) == class_index
        rows.append(
            {
                "class": int(class_index + 1),
                "n": int(target.sum()),
                "auroc": float(roc_auc_score(target, p[:, class_index]))
                if target.min() != target.max()
                else np.nan,
                "auprc": float(average_precision_score(target, p[:, class_index]))
                if target.sum() > 0
                else np.nan,
                "recall": float((predicted & target).sum() / max(1, target.sum())),
            }
        )
    return rows


def selection_metrics(
    base_correct: np.ndarray,
    recovery_correct: np.ndarray,
    choose_recovery: np.ndarray,
) -> dict[str, float]:
    base_correct = np.asarray(base_correct, bool)
    recovery_correct = np.asarray(recovery_correct, bool)
    choose_recovery = np.asarray(choose_recovery, bool)
    selected_correct = np.where(choose_recovery, recovery_correct, base_correct)
    corrections = ~base_correct & recovery_correct
    harms = base_correct & ~recovery_correct
    retained = corrections & choose_recovery
    realized_harms = harms & choose_recovery
    return {
        "n": int(len(base_correct)),
        "correction_opportunities": int(corrections.sum()),
        "corrections_retained": int(retained.sum()),
        "recovery_retention": float(retained.sum() / max(1, corrections.sum())),
        "harm_opportunities": int(harms.sum()),
        "harms_realized": int(realized_harms.sum()),
        "negative_transfer_prevention": float(1.0 - realized_harms.sum() / max(1, harms.sum())),
        "recovery_selection_rate": float(choose_recovery.mean()),
        "selected_accuracy": float(selected_correct.mean()),
        "net_correct_change_per_10000": float(
            10000.0 * (retained.sum() - realized_harms.sum()) / len(base_correct)
        ),
    }


def tradeoff(score, base_correct, recovery_correct, grid=SAFE_GRID):
    rows = []
    for threshold in grid:
        row = selection_metrics(base_correct, recovery_correct, score >= threshold)
        row["threshold"] = float(threshold)
        rows.append(row)
    audit = pd.DataFrame(rows)
    ranked = audit.sort_values(
        [
            "harms_realized",
            "selected_accuracy",
            "corrections_retained",
            "threshold",
        ],
        ascending=[True, False, False, False],
    )
    selected = ranked.iloc[0]
    primary_ties = audit[
        (audit.harms_realized == selected.harms_realized)
        & np.isclose(audit.selected_accuracy, selected.selected_accuracy)
        & (audit.corrections_retained == selected.corrections_retained)
    ]
    return float(selected.threshold), audit, int(len(primary_ties))


def binary_model(c_value: float = 0.05):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            random_state=0,
        ),
    )


def tree_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=35,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def grouped_binary_fit(frame, features, seed, c_value=0.05, kind="logistic"):
    frame = frame.reset_index(drop=True)
    informative = frame.base_correct.to_numpy(bool) != frame.recovery_correct.to_numpy(bool)
    y = frame.recovery_correct.to_numpy(int)
    groups = frame["sample"].to_numpy(int)
    stratify = np.where(informative, y, 2)
    class_counts = np.bincount(stratify)
    nonzero_counts = class_counts[class_counts > 0]
    n_splits = max(2, min(5, int(nonzero_counts.min())))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(frame), np.nan)
    coefficients = []
    for fold, (train, valid) in enumerate(
        splitter.split(frame[features], stratify, groups), start=1
    ):
        fit_rows = train[informative[train]]
        model = binary_model(c_value) if kind == "logistic" else tree_model(seed + fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(frame.loc[fit_rows, features], y[fit_rows])
        oof[valid] = model.predict_proba(frame.loc[valid, features])[:, 1]
        if kind == "logistic":
            coefficients.append(model.named_steps["logisticregression"].coef_[0].copy())
    threshold, audit, tie_count = tradeoff(
        oof,
        frame.base_correct.to_numpy(bool),
        frame.recovery_correct.to_numpy(bool),
    )
    final = binary_model(c_value) if kind == "logistic" else tree_model(seed)
    final.fit(frame.loc[informative, features], y[informative])
    return {
        "model": final,
        "oof": oof,
        "threshold": threshold,
        "audit": audit,
        "tie_count": tie_count,
        "informative": informative,
        "coefficients": coefficients,
    }


def outcome_code(frame: pd.DataFrame) -> np.ndarray:
    base = frame.base_correct.to_numpy(bool)
    recovery = frame.recovery_correct.to_numpy(bool)
    outcome = np.ones(len(frame), dtype=int)
    outcome[base & ~recovery] = 0
    outcome[~base & recovery] = 2
    return outcome


def grouped_three_outcome_fit(frame, features, seed, c_value=0.05):
    frame = frame.reset_index(drop=True)
    y = outcome_code(frame)
    groups = frame["sample"].to_numpy(int)
    class_counts = np.bincount(y)
    nonzero_counts = class_counts[class_counts > 0]
    n_splits = max(2, min(5, int(nonzero_counts.min())))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full((len(frame), 3), np.nan)
    for train, valid in splitter.split(frame[features], y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=2000,
                tol=1e-4,
                random_state=0,
            ),
        )
        model.fit(frame.loc[train, features], y[train])
        probability = model.predict_proba(frame.loc[valid, features])
        for local, class_index in enumerate(model.classes_):
            oof[valid, int(class_index)] = probability[:, local]
    score = oof[:, 2] - oof[:, 0]
    threshold, audit, tie_count = tradeoff(
        score,
        frame.base_correct.to_numpy(bool),
        frame.recovery_correct.to_numpy(bool),
        grid=np.linspace(-0.20, 0.80, 41),
    )
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            random_state=0,
        ),
    )
    final.fit(frame[features], y)
    return {
        "model": final,
        "oof": oof,
        "score": score,
        "threshold": threshold,
        "audit": audit,
        "tie_count": tie_count,
    }


def model_probability_three(model, x):
    raw = model.predict_proba(x)
    probability = np.zeros((len(x), 3))
    for local, class_index in enumerate(model.named_steps["logisticregression"].classes_):
        probability[:, int(class_index)] = raw[:, local]
    return probability


def t_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan, np.nan
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def latex_table(frame: pd.DataFrame, path: Path, **kwargs):
    path.write_text(frame.to_latex(index=False, escape=False, **kwargs), encoding="utf-8")


def selector_stability(calibration: pd.DataFrame):
    repeated_rows = []
    coefficient_rows = []
    regularization_rows = []
    for seed, frame in calibration.groupby("seed"):
        frame = frame.reset_index(drop=True)
        informative = frame.base_correct != frame.recovery_correct
        y_inf = frame.loc[informative, "recovery_correct"].to_numpy(int)
        for repeat in range(10):
            fit = grouped_binary_fit(
                frame,
                FEATURES_12,
                seed=int(seed) * 100 + repeat,
                c_value=0.5,
            )
            p = fit["oof"][informative.to_numpy()]
            repeated_rows.append(
                {
                    "seed": seed,
                    "repeat": repeat + 1,
                    "features": "all_12",
                    "C": 0.5,
                    "auroc": roc_auc_score(y_inf, p),
                    "auprc": average_precision_score(y_inf, p),
                    "brier": np.square(p - y_inf).mean(),
                    "threshold": fit["threshold"],
                    "threshold_tie_count": fit["tie_count"],
                }
            )
            for fold, coefficient in enumerate(fit["coefficients"], start=1):
                for feature, value in zip(FEATURES_12, coefficient):
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
                    seed=int(seed) * 1000 + 10 * repeat + int(round(100 * c_value)),
                    c_value=c_value,
                )
                p = fit["oof"][informative.to_numpy()]
                regularization_rows.append(
                    {
                        "seed": seed,
                        "repeat": repeat + 1,
                        "features": "confidence_entropy_6",
                        "C": c_value,
                        "auroc": roc_auc_score(y_inf, p),
                        "auprc": average_precision_score(y_inf, p),
                        "brier": np.square(p - y_inf).mean(),
                        "threshold": fit["threshold"],
                    }
                )
    repeated = pd.DataFrame(repeated_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    regularization = pd.DataFrame(regularization_rows)
    stability = (
        coefficients.groupby("feature")
        .standardized_coefficient.agg(
            n="size",
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            positive_fraction=lambda x: (x > 0).mean(),
        )
        .reset_index()
    )
    stability["dominant_sign_fraction"] = np.maximum(
        stability.positive_fraction, 1 - stability.positive_fraction
    )
    repeated.to_csv(DATA / "q1_selector_repeated_grouped_cv.csv", index=False)
    coefficients.to_csv(DATA / "q1_selector_repeated_coefficients.csv", index=False)
    stability.to_csv(DATA / "q1_selector_coefficient_stability.csv", index=False)
    regularization.to_csv(DATA / "q1_selector_regularization_path.csv", index=False)
    return repeated, stability, regularization


def fit_routing_models(calibration: pd.DataFrame):
    models = {}
    diagnostics = []
    for seed, frame in calibration.groupby("seed"):
        frame = frame.reset_index(drop=True)
        full = grouped_binary_fit(frame, FEATURES_12, int(seed), c_value=0.5)
        shrink6 = grouped_binary_fit(frame, FEATURES_6, int(seed) + 2000, c_value=0.05)
        tree6 = grouped_binary_fit(
            frame, FEATURES_6, int(seed) + 3000, c_value=0.05, kind="tree"
        )
        outcome3 = grouped_three_outcome_fit(
            frame, FEATURES_6, int(seed) + 4000, c_value=0.05
        )
        distance = np.abs(shrink6["oof"] - shrink6["threshold"])
        ambiguity_cutoff = float(np.quantile(distance, 0.25))
        models[int(seed)] = {
            "full": full,
            "shrink6": shrink6,
            "tree6": tree6,
            "outcome3": outcome3,
            "ambiguity_cutoff": ambiguity_cutoff,
        }
        for name, fit in (
            ("Safe-CF-12", full),
            ("Shrink-6", shrink6),
            ("Tree-6", tree6),
            ("Three-outcome-6", outcome3),
        ):
            diagnostics.append(
                {
                    "seed": seed,
                    "model": name,
                    "threshold": fit["threshold"],
                    "threshold_tie_count": fit["tie_count"],
                    "calibration_rows": len(frame),
                    "informative_rows": int((frame.base_correct != frame.recovery_correct).sum()),
                }
            )
    pd.DataFrame(diagnostics).to_csv(DATA / "q1_routing_model_diagnostics.csv", index=False)
    return models


def decisions_for_frame(frame: pd.DataFrame, fit_bundle: dict):
    full_score = fit_bundle["full"]["model"].predict_proba(frame[FEATURES_12])[:, 1]
    shrink_score = fit_bundle["shrink6"]["model"].predict_proba(frame[FEATURES_6])[:, 1]
    tree_score = fit_bundle["tree6"]["model"].predict_proba(frame[FEATURES_6])[:, 1]
    outcome_probability = model_probability_three(
        fit_bundle["outcome3"]["model"], frame[FEATURES_6]
    )
    outcome_score = outcome_probability[:, 2] - outcome_probability[:, 0]
    full_choose = full_score >= fit_bundle["full"]["threshold"]
    shrink_choose = shrink_score >= fit_bundle["shrink6"]["threshold"]
    ambiguity = (
        np.abs(shrink_score - fit_bundle["shrink6"]["threshold"])
        <= fit_bundle["ambiguity_cutoff"]
    )
    return {
        "Safe-CF-12": (full_choose, np.full(len(frame), 6.0)),
        "Shrink-6": (shrink_choose, np.full(len(frame), 2.0)),
        "Tree-6": (
            tree_score >= fit_bundle["tree6"]["threshold"],
            np.full(len(frame), 2.0),
        ),
        "Three-outcome-6": (
            outcome_score >= fit_bundle["outcome3"]["threshold"],
            np.full(len(frame), 2.0),
        ),
        "Cascade-25": (
            np.where(ambiguity, full_choose, shrink_choose),
            2.0 + 4.0 * ambiguity.astype(float),
        ),
    }


def evaluate_routing_frames(predictions: pd.DataFrame, models: dict, label: str):
    result_rows = []
    probability_rows = []
    class_rows = []
    grouping = ["seed", "fault_type", "prevalence", "stream"]
    for keys, frame in predictions.groupby(grouping):
        seed, fault, prevalence, stream = keys
        frame = frame.reset_index(drop=True)
        if not set(FEATURES_12).issubset(frame.columns):
            derived = endpoint_features(frame)
            for column in derived:
                frame[column] = derived[column]
        pb = frame[probability_columns(frame, "base_p")].to_numpy(float)
        pr = frame[probability_columns(frame, "recovery_p")].to_numpy(float)
        y = frame.y.to_numpy(int)
        base_correct = pb.argmax(axis=1) == y
        recovery_correct = pr.argmax(axis=1) == y
        decisions = decisions_for_frame(frame, models[int(seed)])
        decisions.update(
            {
                "Always PDRF": (np.zeros(len(frame), bool), np.ones(len(frame))),
                "Always RO-PDRF-Full": (np.ones(len(frame), bool), np.ones(len(frame))),
            }
        )
        for method, (choose, passes) in decisions.items():
            emitted = np.where(choose[:, None], pr, pb)
            safety = selection_metrics(base_correct, recovery_correct, choose)
            metrics = probability_metrics(y, emitted)
            result_rows.append(
                {
                    "analysis_set": label,
                    "seed": seed,
                    "fault_type": fault,
                    "prevalence": prevalence,
                    "stream": stream,
                    "method": method,
                    "mean_forward_passes": float(np.mean(passes)),
                    **safety,
                    **metrics,
                }
            )
            agreement = pb.argmax(axis=1) == pr.argmax(axis=1)
            for subset, take in (
                ("endpoint_class_agreement", agreement),
                ("endpoint_class_disagreement", ~agreement),
            ):
                if take.sum() < 2 or len(np.unique(y[take])) < 2:
                    continue
                subset_metrics = probability_metrics(y[take], emitted[take])
                probability_rows.append(
                    {
                        "analysis_set": label,
                        "seed": seed,
                        "fault_type": fault,
                        "prevalence": prevalence,
                        "stream": stream,
                        "method": method,
                        "subset": subset,
                        "n": int(take.sum()),
                        **subset_metrics,
                    }
                )
            for row in classwise_metrics(y, emitted):
                class_rows.append(
                    {
                        "analysis_set": label,
                        "seed": seed,
                        "fault_type": fault,
                        "prevalence": prevalence,
                        "stream": stream,
                        "method": method,
                        **row,
                    }
                )
    return pd.DataFrame(result_rows), pd.DataFrame(probability_rows), pd.DataFrame(class_rows)


def calibration_size_audit(
    calibration,
    strict,
    fractions=(0.25, 0.50, 0.75, 1.00),
    output_name="q1_selector_calibration_size.csv",
):
    rows = []
    for seed, calibration_seed in calibration.groupby("seed"):
        strict_seed = strict[strict.seed == seed].reset_index(drop=True)
        if not set(FEATURES_6).issubset(strict_seed.columns):
            derived = endpoint_features(strict_seed)
            for column in FEATURES_6:
                strict_seed[column] = derived[column]
        pb = strict_seed[probability_columns(strict_seed, "base_p")].to_numpy(float)
        pr = strict_seed[probability_columns(strict_seed, "recovery_p")].to_numpy(float)
        y = strict_seed.y.to_numpy(int)
        base_correct = pb.argmax(axis=1) == y
        recovery_correct = pr.argmax(axis=1) == y
        groups = np.array(sorted(calibration_seed["sample"].unique()))
        for fraction in fractions:
            for repeat in range(10):
                rng = np.random.default_rng(int(seed) * 10000 + int(100 * fraction) + repeat)
                selected_groups = rng.choice(
                    groups,
                    max(25, int(round(fraction * len(groups)))),
                    replace=False,
                )
                subset = calibration_seed[
                    calibration_seed["sample"].isin(selected_groups)
                ].reset_index(drop=True)
                informative = subset.base_correct != subset.recovery_correct
                if informative.sum() < 20 or subset.loc[informative, "recovery_correct"].nunique() < 2:
                    continue
                try:
                    fit = grouped_binary_fit(
                        subset,
                        FEATURES_6,
                        int(seed) * 100 + repeat,
                        c_value=0.05,
                    )
                except ValueError:
                    continue
                score = fit["model"].predict_proba(strict_seed[FEATURES_6])[:, 1]
                choose = score >= fit["threshold"]
                emitted = np.where(choose[:, None], pr, pb)
                rows.append(
                    {
                        "seed": seed,
                        "fraction": fraction,
                        "repeat": repeat + 1,
                        "calibration_groups": len(selected_groups),
                        "calibration_rows": len(subset),
                        "informative_rows": int(informative.sum()),
                        "threshold": fit["threshold"],
                        **selection_metrics(base_correct, recovery_correct, choose),
                        **probability_metrics(y, emitted),
                    }
                )
    output = pd.DataFrame(rows)
    output.to_csv(DATA / output_name, index=False)
    return output


def batch_and_mechanism_audit(strict: pd.DataFrame):
    rows = []
    count_rows = []
    for (seed, fault, batch), frame in strict.groupby(["seed", "fault_type", "batch"]):
        pb = frame[probability_columns(frame, "base_p")].to_numpy(float)
        pr = frame[probability_columns(frame, "recovery_p")].to_numpy(float)
        ps = frame[probability_columns(frame, "safe_p")].to_numpy(float)
        y = frame.y.to_numpy(int)
        base = probability_metrics(y, pb)
        full = probability_metrics(y, pr)
        safe = probability_metrics(y, ps)
        rows.append(
            {
                "seed": seed,
                "fault_type": fault,
                "batch": batch,
                "n": len(frame),
                "full_minus_pdrf_auroc": full["macro_auroc"] - base["macro_auroc"],
                "safe_minus_pdrf_auroc": safe["macro_auroc"] - base["macro_auroc"],
                "safe_minus_full_auroc": safe["macro_auroc"] - full["macro_auroc"],
            }
        )
        for class_index, count in frame.y.value_counts().sort_index().items():
            count_rows.append(
                {
                    "fault_type": fault,
                    "batch": batch,
                    "class": int(class_index + 1),
                    "seed": seed,
                    "n": int(count),
                }
            )
    batch = pd.DataFrame(rows)
    batch.to_csv(DATA / "q1_batch_specific_effects.csv", index=False)
    pd.DataFrame(count_rows).to_csv(DATA / "q1_strict_class_counts_by_batch.csv", index=False)
    interval_rows = []
    for keys, group in batch.groupby(["fault_type", "batch"]):
        for comparison in (
            "full_minus_pdrf_auroc",
            "safe_minus_pdrf_auroc",
            "safe_minus_full_auroc",
        ):
            mean, lower, upper = t_interval(group[comparison].to_numpy(float))
            interval_rows.append(
                {
                    "fault_type": keys[0],
                    "batch": keys[1],
                    "comparison": comparison,
                    "mean": mean,
                    "ci_2.5": lower,
                    "ci_97.5": upper,
                    "n_fitted_pairs": len(group),
                    "interval_unit": "fitted model pair; mechanism and batch fixed",
                }
            )
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(DATA / "q1_batch_fixed_intervals.csv", index=False)

    mechanism_rows = []
    strict_seed = []
    for (seed, fault), frame in strict.groupby(["seed", "fault_type"]):
        pb = frame[probability_columns(frame, "base_p")].to_numpy(float)
        pr = frame[probability_columns(frame, "recovery_p")].to_numpy(float)
        ps = frame[probability_columns(frame, "safe_p")].to_numpy(float)
        y = frame.y.to_numpy(int)
        strict_seed.append(
            {
                "seed": seed,
                "fault_type": fault,
                "full_minus_pdrf_auroc": probability_metrics(y, pr)["macro_auroc"]
                - probability_metrics(y, pb)["macro_auroc"],
                "safe_minus_pdrf_auroc": probability_metrics(y, ps)["macro_auroc"]
                - probability_metrics(y, pb)["macro_auroc"],
                "safe_minus_full_auroc": probability_metrics(y, ps)["macro_auroc"]
                - probability_metrics(y, pr)["macro_auroc"],
            }
        )
    strict_seed = pd.DataFrame(strict_seed)
    for fault, group in strict_seed.groupby("fault_type"):
        for comparison in (
            "full_minus_pdrf_auroc",
            "safe_minus_pdrf_auroc",
            "safe_minus_full_auroc",
        ):
            mean, lower, upper = t_interval(group[comparison].to_numpy(float))
            mechanism_rows.append(
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
    mechanism = pd.DataFrame(mechanism_rows)
    mechanism.to_csv(DATA / "q1_mechanism_fixed_intervals.csv", index=False)
    return batch, intervals, mechanism


def utility_audit(routing: pd.DataFrame):
    rows = []
    for _, result in routing.iterrows():
        n = max(1, result["n"])
        corrections_per_10000 = 10000 * result["corrections_retained"] / n
        harms_per_10000 = 10000 * result["harms_realized"] / n
        extra_passes = max(0.0, result["mean_forward_passes"] - 1.0)
        for harm_cost in (1.0, 2.0, 5.0, 10.0):
            for pass_penalty in (0.0, 0.5, 2.0):
                utility = (
                    corrections_per_10000
                    - harm_cost * harms_per_10000
                    - pass_penalty * extra_passes
                )
                rows.append(
                    {
                        "seed": result["seed"],
                        "fault_type": result["fault_type"],
                        "prevalence": result["prevalence"],
                        "stream": result["stream"],
                        "method": result["method"],
                        "harm_to_correction_cost_ratio": harm_cost,
                        "pass_penalty_equivalent_corrections_per_10000": pass_penalty,
                        "corrections_per_10000": corrections_per_10000,
                        "harms_per_10000": harms_per_10000,
                        "mean_forward_passes": result["mean_forward_passes"],
                        "net_utility_per_10000": utility,
                        "energy_note": "forward-pass/FLOP penalty proxy; energy was not measured",
                    }
                )
    output = pd.DataFrame(rows)
    output.to_csv(DATA / "q1_decision_utility.csv", index=False)
    return output


def prevalence_utility_from_frozen_safety():
    """Decision utility at 10%, 40% and 70% using frozen full-stream counts."""
    safety = pd.read_csv(DATA / "cee_cf10_stream_safety.csv")
    safety = safety[
        (safety.stream == "full_mixed_stream")
        & (safety.variant == "Safe-CF")
        & safety.prevalence.isin([0.10, 0.40, 0.70])
    ]
    method_rows = []
    for _, row in safety.iterrows():
        endpoints = (
            ("Always PDRF", 0, 0, 1.0),
            (
                "Always RO-PDRF-Full",
                row.recovery_opportunities,
                row.negative_transfer_opportunities,
                1.0,
            ),
            (
                "Safe-CF-12",
                row.recovery_retained,
                row.negative_transfer_remaining,
                6.0,
            ),
        )
        for method, corrections, harms, passes in endpoints:
            for harm_cost in (1.0, 2.0, 5.0, 10.0):
                for pass_penalty in (0.0, 0.5, 2.0):
                    corrections_per_10000 = 10000.0 * corrections / row.n
                    harms_per_10000 = 10000.0 * harms / row.n
                    method_rows.append(
                        {
                            "seed": row.seed,
                            "fault_type": row.fault_type,
                            "prevalence": row.prevalence,
                            "stream": row.stream,
                            "method": method,
                            "harm_to_correction_cost_ratio": harm_cost,
                            "pass_penalty_equivalent_corrections_per_10000": pass_penalty,
                            "corrections_per_10000": corrections_per_10000,
                            "harms_per_10000": harms_per_10000,
                            "mean_forward_passes": passes,
                            "net_utility_per_10000": corrections_per_10000
                            - harm_cost * harms_per_10000
                            - pass_penalty * max(0.0, passes - 1.0),
                            "energy_note": "forward-pass/FLOP penalty proxy; energy was not measured",
                        }
                    )
    output = pd.DataFrame(method_rows)
    output.to_csv(DATA / "q1_decision_utility_all_prevalence.csv", index=False)
    return output


def calibration_curves(predictions: pd.DataFrame, methods=("Safe-CF-12", "Shrink-6")):
    # Curves are generated later from emitted probabilities in the routing output figure script.
    # This manifest fixes the binning definitions used in manuscript tables.
    manifest = {
        "equal_width_ece": {"bins": 15, "range": [0.0, 1.0]},
        "equal_mass_ece": {"bins": 15, "ordering": "maximum class probability"},
        "classwise_ece": {"bins": 15, "aggregation": "unweighted mean over six one-vs-rest classes"},
        "post_routing_recalibration": False,
        "emitted_probabilities": "temperature-scaled endpoint probabilities selected without interpolation",
    }
    (DATA / "q1_probability_protocol.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_main_tables(routing, stability, regularization, size, mechanism):
    summary = (
        routing[
            (routing.analysis_set == "strict")
            & (routing.prevalence == 0.40)
        ]
        .groupby("method")
        .agg(
            prevention=("negative_transfer_prevention", "mean"),
            retention=("recovery_retention", "mean"),
            selection=("recovery_selection_rate", "mean"),
            passes=("mean_forward_passes", "mean"),
            macro_AUROC=("macro_auroc", "mean"),
            NLL=("nll", "mean"),
            Brier=("brier", "mean"),
            ECE15=("ece15_equal_width", "mean"),
        )
        .reset_index()
    )
    latex_table(
        summary.round(4),
        TABLES / "q1_learned_routing_main.tex",
        column_format="lrrrrrrrr",
    )
    mechanism_table = mechanism.pivot_table(
        index="fault_type",
        columns="comparison",
        values=["mean", "ci_2.5", "ci_97.5"],
    ).reset_index()
    mechanism_table.to_csv(DATA / "q1_mechanism_fixed_intervals_wide.csv", index=False)
    regularization_summary = (
        regularization.groupby("C")[["auroc", "auprc", "brier", "threshold"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    regularization_summary.to_csv(DATA / "q1_regularization_path_summary.csv", index=False)
    size_summary = (
        size.groupby("fraction")[[
            "negative_transfer_prevention",
            "recovery_retention",
            "macro_auroc",
            "nll",
        ]]
        .agg(["mean", "std"])
        .reset_index()
    )
    size_summary.to_csv(DATA / "q1_calibration_size_summary.csv", index=False)
    latex_table(
        stability[["feature", "median", "q1", "q3", "dominant_sign_fraction"]].round(3),
        TABLES / "q1_coefficient_stability.tex",
        column_format="lrrrr",
    )
    return summary


def response_range_audit():
    source = ROOT.parent / "paper_package_q1_upgrade" / "source_data" / "major3_objective_matched_mechanism.csv"
    if not source.exists():
        return None
    frame = pd.read_csv(source)
    values = pd.concat([frame.clean_score, frame.fault_score], ignore_index=True)
    output = pd.DataFrame(
        [
            {
                "model": "RO-PDRF-Full",
                "clean_min": frame.clean_score.min(),
                "clean_max": frame.clean_score.max(),
                "fault_min": frame.fault_score.min(),
                "fault_max": frame.fault_score.max(),
                "combined_min": values.min(),
                "combined_max": values.max(),
                "n_seed_observations": len(frame),
                "n_fitted_pairs": frame.seed.nunique(),
                "theoretical_open_lower": -3.0,
                "theoretical_open_upper": 3.0,
                "implementation": "Linear(32,16,bias=True)-ReLU-Linear(16,1,bias=True)-3*tanh(u/3)",
            }
        ]
    )
    output.to_csv(DATA / "q1_response_range_audit.csv", index=False)
    return output


def main():
    calibration = pd.read_csv(DATA / "cee_cf10_selector_calibration.csv.gz")
    strict = pd.read_csv(DATA / "cee_cf10_strict_predictions.csv.gz")
    strict["prevalence"] = 0.40
    strict["stream"] = "strict_fault_applied_available"
    if not set(FEATURES_12).issubset(strict.columns):
        derived = endpoint_features(strict)
        for column in derived:
            strict[column] = derived[column]

    repeated, stability, regularization = selector_stability(calibration)
    models = fit_routing_models(calibration)
    strict_routing, strict_probability, strict_class = evaluate_routing_frames(
        strict, models, "strict"
    )
    stream_path = DATA / "cee_cf10_stream_predictions.csv.gz"
    if stream_path.exists():
        stream = pd.read_csv(stream_path)
        stream_routing, stream_probability, stream_class = evaluate_routing_frames(
            stream, models, "complete_stream"
        )
    else:
        stream_routing = pd.DataFrame()
        stream_probability = pd.DataFrame()
        stream_class = pd.DataFrame()
    routing = pd.concat([strict_routing, stream_routing], ignore_index=True)
    probability = pd.concat([strict_probability, stream_probability], ignore_index=True)
    classwise = pd.concat([strict_class, stream_class], ignore_index=True)
    routing.to_csv(DATA / "q1_learned_routing_comparison.csv", index=False)
    probability.to_csv(DATA / "q1_agreement_probability_quality.csv", index=False)
    classwise.to_csv(DATA / "q1_routed_classwise_metrics.csv", index=False)

    size = calibration_size_audit(calibration, strict)
    batch, batch_intervals, mechanism = batch_and_mechanism_audit(strict)
    utility = utility_audit(routing)
    prevalence_utility = prevalence_utility_from_frozen_safety()
    calibration_curves(strict)
    response_range = response_range_audit()
    main_table = write_main_tables(
        routing, stability, regularization, size, mechanism
    )

    facts = {
        "run": "CEE-CF10-R2-Q1S",
        "source_run": "CEE-CF10-R2",
        "routing_rows": len(routing),
        "repeated_grouped_cv_rows": len(repeated),
        "coefficient_stability": stability.to_dict("records"),
        "regularization_summary": (
            regularization.groupby("C")[["auroc", "auprc", "brier"]].mean().reset_index().to_dict("records")
        ),
        "routing_summary": main_table.to_dict("records"),
        "mechanism_fixed_intervals": mechanism.to_dict("records"),
        "response_range": None if response_range is None else response_range.iloc[0].to_dict(),
        "inference_boundary": "mechanism-fixed fitted-pair intervals; no four-mechanism population bootstrap used for new claims",
        "energy_boundary": "energy not measured; utility uses a transparent forward-pass/FLOP penalty proxy",
    }
    (DATA / "q1_scores_facts.json").write_text(
        json.dumps(facts, indent=2, default=float), encoding="utf-8"
    )
    print(main_table.to_string(index=False))


if __name__ == "__main__":
    main()
