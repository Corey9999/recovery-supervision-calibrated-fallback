"""Independent selector and deployment validation for the C&EE submission.

This script does not modify the earlier manuscript package. It imports the
frozen training code, repeats the same 10 optimization seeds, and writes all
new artefacts below this C&EE working directory.

Additions requested by the pre-submission review:
1. Group-wise cross-fitting of selector probabilities before threshold tuning.
2. Clean, mixed-stream, assigned-but-masked, and prevalence-shift evaluation.
3. Leave-one-fault-family-out and genuinely unseen-mechanism audits.
4. Simple selective-fallback baselines at matched calibration trade-offs.
5. Crossed fault x seed cluster-bootstrap intervals.
6. Frozen-run identifiers, selector coefficients, reliability, and CPU cost.
7. Selector feature ablations, convergence diagnostics, and batch-1 latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import platform
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parent
IMPORT_CANDIDATES = (ROOT.parent, ROOT.parent / "paper_package_q1_upgrade")
UPGRADE = next(
    (candidate for candidate in IMPORT_CANDIDATES if (candidate / "run_major_revision_experiments.py").exists()),
    None,
)
if UPGRADE is None:
    raise FileNotFoundError(
        "The frozen base scripts were not found. Place cee_revision inside the public repository "
        "or beside paper_package_q1_upgrade."
    )
sys.path.insert(0, str(UPGRADE))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

import run_major_revision_experiments as major
import run_major3_objective_matched as m3
import run_major7_fault_types_ema as typed
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base


FAST = os.getenv("CEE_FAST") == "1"
OUT = ROOT / ("source_data_fast" if FAST else "source_data")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (101,) if FAST else tuple(range(101, 111))
EPOCHS = 5 if FAST else 60
FROZEN_RUN = "CEE-CF10-R2"
CAL_FAULTS = ("gaussian", "offset", "drift", "stuck_at")
UNSEEN_FAULTS = ("gain_loss", "clipping", "correlated_dual")
PREVALENCES = (0.10, 0.40, 0.70)
TEST_REALIZATION = 70001
N_BOOTSTRAP = 10 if FAST else 1000


FEATURES = [
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

CONFIDENCE_ENTROPY_FEATURES = FEATURES[:6]
DISAGREEMENT_FEATURES = [
    "base_recovery_js",
    "base_consensus_js",
    "recovery_consensus_js",
    "base_consensus_agreement",
    "recovery_consensus_agreement",
]
FEATURE_SETS = {
    "confidence_entropy_6": CONFIDENCE_ENTROPY_FEATURES,
    "disagreement_5": DISAGREEMENT_FEATURES,
    "all_12": FEATURES,
}

SELECTOR_C = 0.5
SELECTOR_SOLVER = "lbfgs"
SELECTOR_MAX_ITER = 2000
SELECTOR_TOL = 1e-4
SELECTOR_CLASS_WEIGHT = "balanced"
SELECTOR_RANDOM_STATE = 0


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-8, 1.0)
    return -(q * np.log(q)).sum(axis=1) / np.log(q.shape[1])


def js_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.clip(a, 1e-8, 1.0)
    b = np.clip(b, 1e-8, 1.0)
    m = 0.5 * (a + b)
    return 0.5 * (
        (a * np.log(a / m)).sum(axis=1)
        + (b * np.log(b / m)).sum(axis=1)
    )


def leave_one_group_consensus(model, x, mask, quality, temperature):
    candidates, valid_columns = [], []
    for group in range(mask.shape[1]):
        valid = (mask[:, group] > 0) & (mask.sum(axis=1) > 1)
        mm = mask.copy()
        mm[valid, group] = 0
        xx = x.copy()
        xx[valid, group] = 0
        qq = quality * mm
        p, _, _, _, _ = major.predict(model, xx, mm, qq, temperature)
        candidates.append(p)
        valid_columns.append(valid)
    stack = np.stack(candidates, axis=1)
    valid = np.stack(valid_columns, axis=1)
    denom = valid.sum(axis=1, keepdims=True)
    consensus = (stack * valid[:, :, None]).sum(axis=1) / np.maximum(denom, 1)
    return consensus, valid


def selector_features(base_p, recovery_p, consensus, valid_removals):
    no_removal = valid_removals.sum(axis=1) == 0
    consensus = consensus.copy()
    consensus[no_removal] = 0.5 * (
        base_p[no_removal] + recovery_p[no_removal]
    )
    base_conf = base_p.max(axis=1)
    recovery_conf = recovery_p.max(axis=1)
    base_entropy = normalized_entropy(base_p)
    recovery_entropy = normalized_entropy(recovery_p)
    return np.column_stack(
        [
            base_conf,
            recovery_conf,
            recovery_conf - base_conf,
            base_entropy,
            recovery_entropy,
            recovery_entropy - base_entropy,
            js_rows(base_p, recovery_p),
            js_rows(base_p, consensus),
            js_rows(recovery_p, consensus),
            (base_p.argmax(axis=1) == consensus.argmax(axis=1)).astype(float),
            (recovery_p.argmax(axis=1) == consensus.argmax(axis=1)).astype(float),
            valid_removals.sum(axis=1) / valid_removals.shape[1],
        ]
    ).astype(np.float64)


def predict_pair(base_model, recovery_model, x, mask, quality, tb, tr):
    pb, _, _, _, _ = major.predict(base_model, x, mask, quality, tb)
    pr, _, _, _, _ = major.predict(recovery_model, x, mask, quality, tr)
    consensus, valid = leave_one_group_consensus(
        recovery_model, x, mask, quality, tr
    )
    features = selector_features(pb, pr, consensus, valid)
    return pb, pr, consensus, features


def new_selector() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=SELECTOR_C,
            solver=SELECTOR_SOLVER,
            class_weight=SELECTOR_CLASS_WEIGHT,
            max_iter=SELECTOR_MAX_ITER,
            tol=SELECTOR_TOL,
            random_state=SELECTOR_RANDOM_STATE,
        ),
    )


def tradeoff_rows(probability, base_correct, recovery_correct):
    candidates = []
    for threshold in np.linspace(0.50, 0.90, 17):
        choose = probability >= threshold
        selected = np.where(choose, recovery_correct, base_correct)
        transfer = base_correct & ~recovery_correct
        recovery = ~base_correct & recovery_correct
        transfer_rate = float((transfer & ~selected).sum() / max(1, transfer.sum()))
        recovery_rate = float((recovery & selected).sum() / max(1, recovery.sum()))
        candidates.append(
            {
                "threshold": float(threshold),
                "accuracy": float(selected.mean()),
                "negative_transfer_rate": transfer_rate,
                "negative_transfer_prevention": 1.0 - transfer_rate,
                "recovery_retention": recovery_rate,
                "full_selection_rate": float(choose.mean()),
            }
        )
    return pd.DataFrame(candidates)


@dataclass
class SelectorFit:
    model: object
    oof_probability: np.ndarray
    safe_threshold: float
    balanced_threshold: float
    audit: pd.DataFrame
    fold_rows: pd.DataFrame
    feature_names: tuple[str, ...]
    final_n_iter: int
    final_converged: bool
    final_convergence_warning: bool


def fit_crossfitted_selector(
    frame: pd.DataFrame,
    seed: int,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> SelectorFit:
    frame = frame.reset_index(drop=True)
    feature_names = tuple(feature_names or FEATURES)
    informative = frame.base_correct.to_numpy(bool) != frame.recovery_correct.to_numpy(bool)
    y_all = frame.recovery_correct.to_numpy(int)
    groups = frame["sample"].to_numpy(int)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(frame), np.nan, dtype=float)
    fold_records = []
    # Stratification is defined by the disagreement target. Non-informative
    # rows remain in held-out folds for threshold accounting but never fit the
    # preference model.
    stratify = np.where(informative, y_all, 2)
    for fold, (train_idx, valid_idx) in enumerate(
        sgkf.split(frame[list(feature_names)], stratify, groups), start=1
    ):
        fit_idx = train_idx[informative[train_idx]]
        fit_positive = int(y_all[fit_idx].sum())
        fit_negative = int(len(fit_idx) - fit_positive)
        valid_informative = valid_idx[informative[valid_idx]]
        valid_positive = int(y_all[valid_informative].sum())
        valid_negative = int(len(valid_informative) - valid_positive)
        if fit_positive == 0 or fit_negative == 0:
            raise RuntimeError(f"Fold {fold} has a one-class preference target")
        model = new_selector()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(
                frame.loc[fit_idx, list(feature_names)].to_numpy(float),
                y_all[fit_idx],
            )
        convergence_warning = any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        n_iter = int(model.named_steps["logisticregression"].n_iter_[0])
        oof[valid_idx] = model.predict_proba(
            frame.loc[valid_idx, list(feature_names)].to_numpy(float)
        )[:, 1]
        fold_records.append(
            {
                "fold": fold,
                "feature_set": "+".join(feature_names),
                "n_features": len(feature_names),
                "fit_rows": int(len(fit_idx)),
                "fit_recovery_positive": fit_positive,
                "fit_negative_transfer": fit_negative,
                "validation_rows": int(len(valid_idx)),
                "validation_groups": int(len(np.unique(groups[valid_idx]))),
                "validation_informative": int(len(valid_informative)),
                "validation_recovery_positive": valid_positive,
                "validation_negative_transfer": valid_negative,
                "one_class_fit": False,
                "n_iter": n_iter,
                "convergence_warning": convergence_warning,
                "converged": bool((not convergence_warning) and n_iter < SELECTOR_MAX_ITER),
            }
        )
    if np.isnan(oof).any():
        raise RuntimeError("Cross-fitting left calibration probabilities missing")
    audit = tradeoff_rows(
        oof,
        frame.base_correct.to_numpy(bool),
        frame.recovery_correct.to_numpy(bool),
    )
    safe = audit.sort_values(
        ["negative_transfer_rate", "accuracy", "recovery_retention", "threshold"],
        ascending=[True, False, False, False],
    ).iloc[0]
    balanced = audit.sort_values(
        ["accuracy", "negative_transfer_rate", "recovery_retention", "threshold"],
        ascending=[False, True, False, False],
    ).iloc[0]
    final_model = new_selector()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final_model.fit(
            frame.loc[informative, list(feature_names)].to_numpy(float),
            y_all[informative],
        )
    final_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    final_n_iter = int(final_model.named_steps["logisticregression"].n_iter_[0])
    return SelectorFit(
        model=final_model,
        oof_probability=oof,
        safe_threshold=float(safe.threshold),
        balanced_threshold=float(balanced.threshold),
        audit=audit,
        fold_rows=pd.DataFrame(fold_records),
        feature_names=feature_names,
        final_n_iter=final_n_iter,
        final_converged=bool((not final_warning) and final_n_iter < SELECTOR_MAX_ITER),
        final_convergence_warning=final_warning,
    )


def binary_selector_diagnostics(y, probability):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(probability, float), 1e-8, 1 - 1e-8)
    return {
        "selector_auroc": float(roc_auc_score(y, p)),
        "selector_auprc": float(average_precision_score(y, p)),
        "selector_brier": float(brier_score_loss(y, p)),
    }


def reliability_rows(seed, y, p, bins=10):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        take = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        rows.append(
            {
                "seed": seed,
                "bin": index,
                "lower": lo,
                "upper": hi,
                "n": int(take.sum()),
                "mean_conditional_preference_score": float(p[take].mean()) if take.any() else np.nan,
                "observed_recovery_preference": float(y[take].mean()) if take.any() else np.nan,
            }
        )
    return rows


def selector_parameters(seed, fit: SelectorFit):
    scaler = fit.model.named_steps["standardscaler"]
    logistic = fit.model.named_steps["logisticregression"]
    rows = []
    for idx, feature in enumerate(fit.feature_names):
        rows.append(
            {
                "seed": seed,
                "feature": feature,
                "standardization_mean": float(scaler.mean_[idx]),
                "standardization_scale": float(scaler.scale_[idx]),
                "logistic_coefficient": float(logistic.coef_[0, idx]),
                "logistic_intercept": float(logistic.intercept_[0]),
            }
        )
    return rows


def selection_summary(base_correct, recovery_correct, selected_correct, choose):
    transfer = base_correct & ~recovery_correct
    recovery = ~base_correct & recovery_correct
    remaining = transfer & ~selected_correct
    retained = recovery & selected_correct
    return {
        "n": int(len(base_correct)),
        "recovery_opportunities": int(recovery.sum()),
        "recovery_retained": int(retained.sum()),
        "recovery_retention": float(retained.sum() / max(1, recovery.sum())),
        "negative_transfer_opportunities": int(transfer.sum()),
        "negative_transfer_remaining": int(remaining.sum()),
        "negative_transfer_rate": float(remaining.sum() / max(1, transfer.sum())),
        "negative_transfer_prevention": float(1.0 - remaining.sum() / max(1, transfer.sum())),
        "full_selection_rate": float(choose.mean()),
        "selected_accuracy": float(selected_correct.mean()),
    }


def metric_rows(seed, fault, prevalence, stream, method, y, p, n):
    return [
        {
            "run_id": FROZEN_RUN,
            "seed": seed,
            "fault_type": fault,
            "prevalence": prevalence,
            "stream": stream,
            "method": method,
            "metric": metric,
            "value": value,
            "n": n,
        }
        for metric, value in risk.all_metrics(y, p).items()
    ]


def simple_scores(features: np.ndarray):
    position = {name: i for i, name in enumerate(FEATURES)}
    return {
        "Higher-confidence": features[:, position["confidence_delta"]],
        "Lower-entropy": -features[:, position["entropy_delta"]],
        "Lower-LOO-disagreement": (
            features[:, position["base_consensus_js"]]
            - features[:, position["recovery_consensus_js"]]
        ),
        "Higher-JS-disagreement": features[:, position["base_recovery_js"]],
    }


def score_audit(score, base_correct, recovery_correct):
    quantiles = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 101)))
    rows = []
    for threshold in quantiles:
        choose = score >= threshold
        selected = np.where(choose, recovery_correct, base_correct)
        row = selection_summary(base_correct, recovery_correct, selected, choose)
        row["threshold"] = float(threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def matched_rule_threshold(audit, target, field):
    ranked = audit.assign(distance=(audit[field] - target).abs()).sort_values(
        ["distance", "selected_accuracy", "negative_transfer_prevention", "recovery_retention"],
        ascending=[True, False, False, False],
    )
    return float(ranked.iloc[0].threshold)


def evaluate_selection_method(pb, pr, choose, y):
    p = np.where(choose[:, None], pr, pb)
    base_correct = pb.argmax(axis=1) == y
    recovery_correct = pr.argmax(axis=1) == y
    selected_correct = p.argmax(axis=1) == y
    return p, selection_summary(base_correct, recovery_correct, selected_correct, choose)


def selector_thresholds_for_simple_rules(calibration, fit):
    base_correct = calibration.base_correct.to_numpy(bool)
    recovery_correct = calibration.recovery_correct.to_numpy(bool)
    safe_choose = fit.oof_probability >= fit.safe_threshold
    safe_selected = np.where(safe_choose, recovery_correct, base_correct)
    safe = selection_summary(base_correct, recovery_correct, safe_selected, safe_choose)
    rows = []
    thresholds = {}
    for name, score in simple_scores(calibration[FEATURES].to_numpy(float)).items():
        audit = score_audit(score, base_correct, recovery_correct)
        for match_name, field in (
            ("prevention_matched", "negative_transfer_prevention"),
            ("retention_matched", "recovery_retention"),
        ):
            threshold = matched_rule_threshold(audit, safe[field], field)
            thresholds[(name, match_name)] = threshold
            chosen = audit.iloc[(audit.threshold - threshold).abs().argmin()].to_dict()
            rows.append(
                {
                    "rule": name,
                    "matching": match_name,
                    "threshold": threshold,
                    "target": safe[field],
                    **{f"calibration_{k}": v for k, v in chosen.items() if k != "threshold"},
                }
            )
    return thresholds, rows, safe


def strict_prediction_records(
    seed,
    fault,
    y,
    pb,
    pr,
    conditional_preference_score,
    choose_safe,
    choose_balanced,
    simple_choices,
    batch,
    sample_ids,
):
    ps = np.where(choose_safe[:, None], pr, pb)
    pbal = np.where(choose_balanced[:, None], pr, pb)
    records = []
    for local, sample in enumerate(sample_ids):
        row = {
            "run_id": FROZEN_RUN,
            "seed": seed,
            "fault_type": fault,
            "sample": int(sample),
            "batch": int(batch[local]),
            "y": int(y[local]),
            "conditional_preference_score": float(conditional_preference_score[local]),
            "choose_safe": bool(choose_safe[local]),
            "choose_balanced": bool(choose_balanced[local]),
            "base_correct": bool(pb[local].argmax() == y[local]),
            "recovery_correct": bool(pr[local].argmax() == y[local]),
            "safe_correct": bool(ps[local].argmax() == y[local]),
            "balanced_correct": bool(pbal[local].argmax() == y[local]),
        }
        for c in range(pb.shape[1]):
            row[f"base_p{c}"] = float(pb[local, c])
            row[f"recovery_p{c}"] = float(pr[local, c])
            row[f"safe_p{c}"] = float(ps[local, c])
            row[f"balanced_p{c}"] = float(pbal[local, c])
        for name, choose in simple_choices.items():
            row[f"choose_{name}"] = bool(choose[local])
        records.append(row)
    return records


def macro_auroc(y, p):
    classes = np.arange(p.shape[1])
    yb = label_binarize(y, classes=classes)
    if p.shape[1] == 2 and yb.shape[1] == 1:
        yb = np.column_stack([1 - yb[:, 0], yb[:, 0]])
    return float(roc_auc_score(yb, p, average="macro", multi_class="ovr"))


def crossed_cluster_bootstrap(frame: pd.DataFrame, rng_seed=20260714):
    """Resample crossed fault and fitted-model identities, then observations.

    Each optimization seed identifies one fitted PDRF/RO-PDRF-Full pair reused
    across all fault mechanisms.  Seed identities are therefore drawn once per
    replicate and retained across every sampled fault, rather than sampled
    independently within faults.
    """
    rng = np.random.default_rng(rng_seed)
    faults = np.array(sorted(frame.fault_type.unique()))
    seeds = np.array(sorted(frame.seed.unique()))
    classes = len([c for c in frame.columns if c.startswith("base_p")])
    cells = {
        (fault, int(seed)): group.reset_index(drop=True)
        for (fault, seed), group in frame.groupby(["fault_type", "seed"])
    }
    boot = []
    for replicate in range(N_BOOTSTRAP):
        sampled_cells = []
        sampled_faults = rng.choice(faults, len(faults), replace=True)
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        for fault in sampled_faults:
            for seed in sampled_seeds:
                cell = cells[(fault, int(seed))]
                idx = rng.integers(0, len(cell), len(cell))
                sampled_cells.append(cell.iloc[idx])
        sampled = pd.concat(sampled_cells, ignore_index=True)
        y = sampled.y.to_numpy(int)
        base_correct = sampled.base_correct.to_numpy(bool)
        recovery_correct = sampled.recovery_correct.to_numpy(bool)
        safe_correct = sampled.safe_correct.to_numpy(bool)
        choose_safe = sampled.choose_safe.to_numpy(bool)
        safety = selection_summary(
            base_correct, recovery_correct, safe_correct, choose_safe
        )
        pb = sampled[[f"base_p{c}" for c in range(classes)]].to_numpy(float)
        pr = sampled[[f"recovery_p{c}" for c in range(classes)]].to_numpy(float)
        ps = sampled[[f"safe_p{c}" for c in range(classes)]].to_numpy(float)
        boot.append(
            {
                "replicate": replicate + 1,
                "negative_transfer_prevention": safety["negative_transfer_prevention"],
                "recovery_retention": safety["recovery_retention"],
                "full_selection_rate": safety["full_selection_rate"],
                "safe_minus_pdrf_auroc": macro_auroc(y, ps) - macro_auroc(y, pb),
                "safe_minus_full_auroc": macro_auroc(y, ps) - macro_auroc(y, pr),
            }
        )
    return pd.DataFrame(boot)


def interval_summary(frame: pd.DataFrame):
    rows = []
    for column in frame.columns:
        if column == "replicate":
            continue
        rows.append(
            {
                "quantity": column,
                "bootstrap_mean": float(frame[column].mean()),
                "ci_2.5": float(frame[column].quantile(0.025)),
                "ci_97.5": float(frame[column].quantile(0.975)),
                "replicates": len(frame),
            }
        )
    return pd.DataFrame(rows)


def benchmark(callable_, repeats=7):
    for _ in range(3):
        callable_()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        times.append(time.perf_counter() - started)
    return float(np.median(times)), float(np.quantile(times, 0.25)), float(np.quantile(times, 0.75))


def cpu_model_name():
    """Return a human-readable CPU identifier without adding a dependency."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (OSError, ImportError):
            pass
    return platform.processor() or platform.machine() or "unavailable"


def main():
    started = time.perf_counter()
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i + 4)) for i in range(0, 16, 4)]
    )
    testx, testy, test_batches = split["test"]
    cx, cy, _ = split["calibration"]
    cut = len(cx) // 2
    temperature_x, temperature_y = cx[:cut], cy[:cut]
    selector_x, selector_y = cx[cut:], cy[cut:]

    metric_records = []
    safety_records = []
    strict_records = []
    selector_audits = []
    selector_folds = []
    selector_parameter_rows = []
    selector_diagnostic_rows = []
    selector_calibration_records = []
    feature_ablation_calibration_rows = []
    feature_ablation_test_rows = []
    feature_ablation_fold_rows = []
    recovery_eligibility_rows = []
    reliability = []
    simple_threshold_rows = []
    simple_test_rows = []
    lofo_rows = []
    unseen_rows = []
    cost_rows = []

    for seed in SEEDS:
        seed_started = time.perf_counter()
        train, select, _, class_weights = q2.train_views(split, seed)
        train_mask = np.asarray(train[1])
        recovery_eligible = train_mask.sum(axis=1) > 1
        recovery_eligibility_rows.append(
            {
                "seed": seed,
                "training_rows": int(len(train_mask)),
                "eligible_rows": int(recovery_eligible.sum()),
                "excluded_rows": int((~recovery_eligible).sum()),
                "eligible_fraction": float(recovery_eligible.mean()),
                "excluded_fraction": float((~recovery_eligible).mean()),
            }
        )
        temp_view = (
            *base.make_view(
                temperature_x, 25000 + seed, "missing_drift", train=True
            ),
            temperature_y,
        )
        base_model, base_cost = major.fit(
            major.SPECS["PDRF"],
            train,
            select,
            class_weights,
            seed,
            prior=(0.45, 0.25, 0.18, 0.12),
            epochs=EPOCHS,
        )
        recovery_model, recovery_cost = major.fit(
            m3.RO_PDRF,
            train,
            select,
            class_weights,
            seed,
            prior=(0.45, 0.25, 0.18, 0.12),
            epochs=EPOCHS,
        )
        tb = major.fit_temperature(base_model, temp_view)
        tr = major.fit_temperature(recovery_model, temp_view)

        calibration_rows = []
        for fidx, fault_type in enumerate(CAL_FAULTS):
            for realization in range(2):
                realization_id = 96000 + 100 * fidx + 10 * realization + seed
                faultx, mask, quality, assigned = typed.fixed_typed_fault(
                    selector_x, fault_type, seed=realization_id
                )
                applied = assigned & (mask[:, 0] > 0)
                pb, pr, _, features = predict_pair(
                    base_model, recovery_model, faultx, mask, quality, tb, tr
                )
                for sample in np.flatnonzero(applied):
                    row = {
                        "seed": seed,
                        "fault_type": fault_type,
                        "realization": realization,
                        "realization_id": realization_id,
                        "sample": int(sample),
                        "base_correct": bool(pb[sample].argmax() == selector_y[sample]),
                        "recovery_correct": bool(pr[sample].argmax() == selector_y[sample]),
                    }
                    row.update(dict(zip(FEATURES, features[sample])))
                    calibration_rows.append(row)
        calibration = pd.DataFrame(calibration_rows)
        selector_calibration_records.extend(calibration.to_dict("records"))
        fit = fit_crossfitted_selector(calibration, seed)
        fit.audit.insert(0, "seed", seed)
        selector_audits.extend(fit.audit.to_dict("records"))
        fold = fit.fold_rows.copy()
        fold.insert(0, "seed", seed)
        selector_folds.extend(fold.to_dict("records"))
        selector_parameter_rows.extend(selector_parameters(seed, fit))
        informative = calibration.base_correct != calibration.recovery_correct
        recovery_positive = int(
            ((~calibration.base_correct) & calibration.recovery_correct).sum()
        )
        negative_transfer = int(
            (calibration.base_correct & (~calibration.recovery_correct)).sum()
        )
        selector_diagnostic_rows.append(
            {
                "seed": seed,
                "calibration_rows": int(len(calibration)),
                "correctness_disagreements": int(informative.sum()),
                "recovery_positive_rows": recovery_positive,
                "negative_transfer_rows": negative_transfer,
                "minority_events_per_variable": float(
                    min(recovery_positive, negative_transfer) / len(FEATURES)
                ),
                "safe_threshold": fit.safe_threshold,
                "balanced_threshold": fit.balanced_threshold,
                "final_n_iter": fit.final_n_iter,
                "final_converged": fit.final_converged,
                "final_convergence_warning": fit.final_convergence_warning,
                **binary_selector_diagnostics(
                    calibration.loc[informative, "recovery_correct"].to_numpy(int),
                    fit.oof_probability[informative.to_numpy()],
                ),
            }
        )
        ablation_fits = {"all_12": fit}
        for feature_set, feature_names in FEATURE_SETS.items():
            if feature_set != "all_12":
                ablation_fits[feature_set] = fit_crossfitted_selector(
                    calibration, seed, feature_names
                )
            ablation_fit = ablation_fits[feature_set]
            ablation_informative_y = calibration.loc[
                informative, "recovery_correct"
            ].to_numpy(int)
            feature_ablation_calibration_rows.append(
                {
                    "seed": seed,
                    "feature_set": feature_set,
                    "n_features": len(feature_names),
                    "informative_rows": int(informative.sum()),
                    "recovery_positive_rows": recovery_positive,
                    "negative_transfer_rows": negative_transfer,
                    "safe_threshold": ablation_fit.safe_threshold,
                    "balanced_threshold": ablation_fit.balanced_threshold,
                    "final_n_iter": ablation_fit.final_n_iter,
                    "final_converged": ablation_fit.final_converged,
                    "final_convergence_warning": ablation_fit.final_convergence_warning,
                    **binary_selector_diagnostics(
                        ablation_informative_y,
                        ablation_fit.oof_probability[informative.to_numpy()],
                    ),
                }
            )
            fold_rows = ablation_fit.fold_rows.copy()
            fold_rows["seed"] = seed
            fold_rows["feature_set"] = feature_set
            feature_ablation_fold_rows.extend(fold_rows.to_dict("records"))
        reliability.extend(
            reliability_rows(
                seed,
                calibration.loc[informative, "recovery_correct"].to_numpy(int),
                fit.oof_probability[informative.to_numpy()],
            )
        )
        simple_thresholds, simple_calibration, safe_calibration = (
            selector_thresholds_for_simple_rules(calibration, fit)
        )
        for row in simple_calibration:
            row["seed"] = seed
            simple_threshold_rows.append(row)

        # Strict primary and full mixed streams use the same model pair,
        # test realization, temperature protocol, and selector fit.
        for prevalence in PREVALENCES:
            for fault_type in CAL_FAULTS:
                faultx, mask, quality, assigned = typed.fixed_typed_fault(
                    testx,
                    fault_type,
                    prevalence=prevalence,
                    seed=TEST_REALIZATION,
                )
                applied = assigned & (mask[:, 0] > 0)
                assigned_masked = assigned & (mask[:, 0] == 0)
                unaffected = ~assigned
                pb, pr, _, features = predict_pair(
                    base_model, recovery_model, faultx, mask, quality, tb, tr
                )
                conditional_preference_score = fit.model.predict_proba(features)[:, 1]
                choose_safe = conditional_preference_score >= fit.safe_threshold
                choose_balanced = conditional_preference_score >= fit.balanced_threshold
                p_safe, safe_summary = evaluate_selection_method(
                    pb, pr, choose_safe, testy
                )
                p_balanced, balanced_summary = evaluate_selection_method(
                    pb, pr, choose_balanced, testy
                )
                streams = (
                    ("full_mixed_stream", np.ones(len(testy), bool)),
                    ("strict_fault_applied_available", applied),
                    ("assigned_but_masked", assigned_masked),
                    ("unaffected", unaffected),
                )
                for stream, take in streams:
                    if take.sum() < 2:
                        continue
                    for method, probability in (
                        ("PDRF", pb),
                        ("RO-PDRF-Full", pr),
                        ("SR-PDRF-Balanced-CF", p_balanced),
                        ("SR-PDRF-Safe-CF", p_safe),
                    ):
                        metric_records.extend(
                            metric_rows(
                                seed,
                                fault_type,
                                prevalence,
                                stream,
                                method,
                                testy[take],
                                probability[take],
                                int(take.sum()),
                            )
                        )
                for variant, summary in (
                    ("Balanced-CF", balanced_summary),
                    ("Safe-CF", safe_summary),
                ):
                    safety_records.append(
                        {
                            "seed": seed,
                            "fault_type": fault_type,
                            "prevalence": prevalence,
                            "stream": "full_mixed_stream",
                            "variant": variant,
                            **summary,
                        }
                    )
                if math.isclose(prevalence, 0.40):
                    strict = applied
                    strict_features = features[strict]
                    strict_simple_choices = {}
                    for (rule, matching), threshold in simple_thresholds.items():
                        score = simple_scores(strict_features)[rule]
                        strict_simple_choices[f"{rule}_{matching}"] = score >= threshold
                        selected_p, selected_summary = evaluate_selection_method(
                            pb[strict], pr[strict], score >= threshold, testy[strict]
                        )
                        simple_test_rows.append(
                            {
                                "seed": seed,
                                "fault_type": fault_type,
                                "rule": rule,
                                "matching": matching,
                                "threshold": threshold,
                                **selected_summary,
                                "macro_auroc": macro_auroc(testy[strict], selected_p),
                            }
                        )
                    random_rng = np.random.default_rng(
                        500000 + 100 * seed + CAL_FAULTS.index(fault_type)
                    )
                    random_choose = random_rng.random(strict.sum()) < safe_calibration["full_selection_rate"]
                    random_p, random_summary = evaluate_selection_method(
                        pb[strict], pr[strict], random_choose, testy[strict]
                    )
                    simple_test_rows.append(
                        {
                            "seed": seed,
                            "fault_type": fault_type,
                            "rule": "Random-matched-selection",
                            "matching": "selection_rate_matched",
                            "threshold": safe_calibration["full_selection_rate"],
                            **random_summary,
                            "macro_auroc": macro_auroc(testy[strict], random_p),
                        }
                    )
                    base_correct = pb[strict].argmax(axis=1) == testy[strict]
                    recovery_correct = pr[strict].argmax(axis=1) == testy[strict]
                    for feature_set, ablation_fit in ablation_fits.items():
                        indices = [FEATURES.index(name) for name in ablation_fit.feature_names]
                        ablation_score = ablation_fit.model.predict_proba(
                            strict_features[:, indices]
                        )[:, 1]
                        ablation_choose = ablation_score >= ablation_fit.safe_threshold
                        ablation_p, ablation_summary = evaluate_selection_method(
                            pb[strict], pr[strict], ablation_choose, testy[strict]
                        )
                        ablation_metrics = risk.all_metrics(testy[strict], ablation_p)
                        feature_ablation_test_rows.append(
                            {
                                "seed": seed,
                                "fault_type": fault_type,
                                "feature_set": feature_set,
                                "n_features": len(ablation_fit.feature_names),
                                "safe_threshold": ablation_fit.safe_threshold,
                                **ablation_summary,
                                "accuracy": ablation_metrics["accuracy"],
                                "macro_f1": ablation_metrics["macro_f1"],
                                "macro_auroc": ablation_metrics["macro_auroc"],
                            }
                        )
                    oracle_choose = ~base_correct & recovery_correct
                    oracle_p, oracle_summary = evaluate_selection_method(
                        pb[strict], pr[strict], oracle_choose, testy[strict]
                    )
                    simple_test_rows.append(
                        {
                            "seed": seed,
                            "fault_type": fault_type,
                            "rule": "Correctness-disagreement-oracle",
                            "matching": "opportunity_boundary",
                            "threshold": np.nan,
                            **oracle_summary,
                            "macro_auroc": macro_auroc(testy[strict], oracle_p),
                        }
                    )
                    simple_test_rows.extend(
                        [
                            {
                                "seed": seed,
                                "fault_type": fault_type,
                                "rule": "Always-PDRF",
                                "matching": "endpoint",
                                "threshold": np.nan,
                                **selection_summary(
                                    base_correct,
                                    recovery_correct,
                                    base_correct,
                                    np.zeros(strict.sum(), bool),
                                ),
                                "macro_auroc": macro_auroc(testy[strict], pb[strict]),
                            },
                            {
                                "seed": seed,
                                "fault_type": fault_type,
                                "rule": "Always-RO-PDRF-Full",
                                "matching": "endpoint",
                                "threshold": np.nan,
                                **selection_summary(
                                    base_correct,
                                    recovery_correct,
                                    recovery_correct,
                                    np.ones(strict.sum(), bool),
                                ),
                                "macro_auroc": macro_auroc(testy[strict], pr[strict]),
                            },
                            {
                                "seed": seed,
                                "fault_type": fault_type,
                                "rule": "Safe-CF",
                                "matching": "proposed",
                                "threshold": fit.safe_threshold,
                                **selection_summary(
                                    base_correct,
                                    recovery_correct,
                                    p_safe[strict].argmax(axis=1) == testy[strict],
                                    choose_safe[strict],
                                ),
                                "macro_auroc": macro_auroc(testy[strict], p_safe[strict]),
                            },
                        ]
                    )
                    strict_records.extend(
                        strict_prediction_records(
                            seed,
                            fault_type,
                            testy[strict],
                            pb[strict],
                            pr[strict],
                            conditional_preference_score[strict],
                            choose_safe[strict],
                            choose_balanced[strict],
                            strict_simple_choices,
                            test_batches[strict],
                            np.flatnonzero(strict),
                        )
                    )

        # Clean stream with natural 20% group missingness but no imposed fault.
        clean_rng = np.random.default_rng(TEST_REALIZATION)
        clean_mask = base.random_mask(len(testx), clean_rng, 0.20)
        clean_x = testx.copy() * clean_mask[:, :, None]
        clean_quality = np.ones((len(testx), clean_mask.shape[1]), np.float32)
        clean_pb, clean_pr, _, clean_features = predict_pair(
            base_model,
            recovery_model,
            clean_x,
            clean_mask,
            clean_quality,
            tb,
            tr,
        )
        clean_probability = fit.model.predict_proba(clean_features)[:, 1]
        for variant, threshold in (
            ("Balanced-CF", fit.balanced_threshold),
            ("Safe-CF", fit.safe_threshold),
        ):
            choose = clean_probability >= threshold
            selected, summary = evaluate_selection_method(
                clean_pb, clean_pr, choose, testy
            )
            false_switch = float(choose.mean())
            harmful_switch = float(
                (
                    choose
                    & (clean_pb.argmax(axis=1) == testy)
                    & (clean_pr.argmax(axis=1) != testy)
                ).mean()
            )
            safety_records.append(
                {
                    "seed": seed,
                    "fault_type": "none",
                    "prevalence": 0.0,
                    "stream": "clean_no_imposed_fault",
                    "variant": variant,
                    "false_switch_rate": false_switch,
                    "harmful_switch_rate": harmful_switch,
                    **summary,
                }
            )
            metric_records.extend(
                metric_rows(
                    seed,
                    "none",
                    0.0,
                    "clean_no_imposed_fault",
                    f"SR-PDRF-{variant}",
                    testy,
                    selected,
                    len(testy),
                )
            )
        for method, probability in (
            ("PDRF", clean_pb),
            ("RO-PDRF-Full", clean_pr),
        ):
            metric_records.extend(
                metric_rows(
                    seed,
                    "none",
                    0.0,
                    "clean_no_imposed_fault",
                    method,
                    testy,
                    probability,
                    len(testy),
                )
            )

        # Leave-one-fault-family-out selector transfer.
        for held_out in CAL_FAULTS:
            train_cal = calibration[calibration.fault_type != held_out].copy()
            lofo_fit = fit_crossfitted_selector(train_cal, 1000 + seed)
            faultx, mask, quality, assigned = typed.fixed_typed_fault(
                testx, held_out, prevalence=0.40, seed=TEST_REALIZATION
            )
            strict = assigned & (mask[:, 0] > 0)
            pb, pr, _, features = predict_pair(
                base_model, recovery_model, faultx, mask, quality, tb, tr
            )
            probability = lofo_fit.model.predict_proba(features[strict])[:, 1]
            choose = probability >= lofo_fit.safe_threshold
            selected, summary = evaluate_selection_method(
                pb[strict], pr[strict], choose, testy[strict]
            )
            lofo_rows.append(
                {
                    "seed": seed,
                    "held_out_fault": held_out,
                    "calibration_faults": "+".join(
                        f for f in CAL_FAULTS if f != held_out
                    ),
                    "safe_threshold": lofo_fit.safe_threshold,
                    "pdrf_macro_auroc": macro_auroc(testy[strict], pb[strict]),
                    "full_macro_auroc": macro_auroc(testy[strict], pr[strict]),
                    "safe_macro_auroc": macro_auroc(testy[strict], selected),
                    **summary,
                }
            )

        # Fault mechanisms absent from selector calibration.
        for unseen in UNSEEN_FAULTS:
            faultx, mask, quality, assigned = q2.make_fault(
                testx, TEST_REALIZATION, unseen
            )
            strict = assigned & (mask[:, 0] > 0)
            pb, pr, _, features = predict_pair(
                base_model, recovery_model, faultx, mask, quality, tb, tr
            )
            probability = fit.model.predict_proba(features[strict])[:, 1]
            choose = probability >= fit.safe_threshold
            selected, summary = evaluate_selection_method(
                pb[strict], pr[strict], choose, testy[strict]
            )
            unseen_rows.append(
                {
                    "seed": seed,
                    "unseen_fault": unseen,
                    "safe_threshold": fit.safe_threshold,
                    "pdrf_macro_auroc": macro_auroc(testy[strict], pb[strict]),
                    "full_macro_auroc": macro_auroc(testy[strict], pr[strict]),
                    "safe_macro_auroc": macro_auroc(testy[strict], selected),
                    **summary,
                }
            )

        # CPU latency is reported both for a complete held-out batch and for
        # batch size one.  The active thread settings are recorded rather than
        # silently changed during timing.
        bench_x = clean_x
        bench_mask = clean_mask
        bench_quality = clean_quality
        base_time = benchmark(
            lambda: major.predict(
                base_model, bench_x, bench_mask, bench_quality, tb
            )
        )
        recovery_time = benchmark(
            lambda: major.predict(
                recovery_model, bench_x, bench_mask, bench_quality, tr
            )
        )
        end_to_end_time = benchmark(
            lambda: fit.model.predict_proba(
                predict_pair(
                    base_model,
                    recovery_model,
                    bench_x,
                    bench_mask,
                    bench_quality,
                    tb,
                    tr,
                )[3]
            )
        )
        base_time_one = benchmark(
            lambda: major.predict(
                base_model, bench_x[:1], bench_mask[:1], bench_quality[:1], tb
            )
        )
        recovery_time_one = benchmark(
            lambda: major.predict(
                recovery_model,
                bench_x[:1],
                bench_mask[:1],
                bench_quality[:1],
                tr,
            )
        )
        end_to_end_time_one = benchmark(
            lambda: fit.model.predict_proba(
                predict_pair(
                    base_model,
                    recovery_model,
                    bench_x[:1],
                    bench_mask[:1],
                    bench_quality[:1],
                    tb,
                    tr,
                )[3]
            )
        )
        for method, timing, timing_one, params, forward_passes in (
            ("PDRF", base_time, base_time_one, base_cost["parameters"], 1),
            (
                "RO-PDRF-Full",
                recovery_time,
                recovery_time_one,
                recovery_cost["parameters"],
                1,
            ),
            (
                "SR-PDRF-Safe-CF",
                end_to_end_time,
                end_to_end_time_one,
                base_cost["parameters"] + recovery_cost["parameters"] + 13,
                6,
            ),
        ):
            median, q1, q3 = timing
            median_one, q1_one, q3_one = timing_one
            cost_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "batch_size": len(bench_x),
                    "cpu_model": cpu_model_name(),
                    "cpu_threads": torch.get_num_threads(),
                    "cpu_interop_threads": torch.get_num_interop_threads(),
                    "torch_version": torch.__version__,
                    "latency_batch_ms_median": 1000 * median,
                    "latency_batch_ms_q1": 1000 * q1,
                    "latency_batch_ms_q3": 1000 * q3,
                    "latency_ms_per_observation": 1000 * median / len(bench_x),
                    "throughput_observations_per_second": len(bench_x) / median,
                    "latency_batch1_ms_median": 1000 * median_one,
                    "latency_batch1_ms_q1": 1000 * q1_one,
                    "latency_batch1_ms_q3": 1000 * q3_one,
                    "forward_passes_per_observation": forward_passes,
                    "stored_parameters": params,
                    "model_state_kib_fp32": 4 * params / 1024,
                    "base_train_seconds": base_cost["train_seconds"],
                    "recovery_train_seconds": recovery_cost["train_seconds"],
                    "base_temperature": tb,
                    "recovery_temperature": tr,
                    "elapsed_seed_seconds": time.perf_counter() - seed_started,
                }
            )

    metrics = pd.DataFrame(metric_records)
    safety = pd.DataFrame(safety_records)
    strict_predictions = pd.DataFrame(strict_records)
    simple_test = pd.DataFrame(simple_test_rows)
    lofo = pd.DataFrame(lofo_rows)
    unseen = pd.DataFrame(unseen_rows)
    costs = pd.DataFrame(cost_rows)

    metrics.to_csv(OUT / "cee_cf10_stream_metrics.csv", index=False)
    safety.to_csv(OUT / "cee_cf10_stream_safety.csv", index=False)
    strict_predictions.to_csv(
        OUT / "cee_cf10_strict_predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(selector_audits).to_csv(
        OUT / "cee_cf10_threshold_audit.csv", index=False
    )
    pd.DataFrame(selector_folds).to_csv(
        OUT / "cee_cf10_crossfit_folds.csv", index=False
    )
    pd.DataFrame(selector_parameter_rows).to_csv(
        OUT / "cee_cf10_selector_parameters.csv", index=False
    )
    pd.DataFrame(selector_diagnostic_rows).to_csv(
        OUT / "cee_cf10_selector_diagnostics.csv", index=False
    )
    pd.DataFrame(selector_calibration_records).to_csv(
        OUT / "cee_cf10_selector_calibration.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(feature_ablation_calibration_rows).to_csv(
        OUT / "cee_cf10_feature_ablation_calibration.csv", index=False
    )
    pd.DataFrame(feature_ablation_test_rows).to_csv(
        OUT / "cee_cf10_feature_ablation_test.csv", index=False
    )
    pd.DataFrame(feature_ablation_fold_rows).to_csv(
        OUT / "cee_cf10_feature_ablation_folds.csv", index=False
    )
    pd.DataFrame(recovery_eligibility_rows).to_csv(
        OUT / "cee_cf10_recovery_eligibility.csv", index=False
    )
    pd.DataFrame(reliability).to_csv(
        OUT / "cee_cf10_selector_reliability.csv", index=False
    )
    pd.DataFrame(simple_threshold_rows).to_csv(
        OUT / "cee_cf10_simple_rule_thresholds.csv", index=False
    )
    simple_test.to_csv(OUT / "cee_cf10_simple_rule_test.csv", index=False)
    lofo.to_csv(OUT / "cee_cf10_lofo.csv", index=False)
    unseen.to_csv(OUT / "cee_cf10_unseen_faults.csv", index=False)
    costs.to_csv(OUT / "cee_cf10_cpu_cost.csv", index=False)

    for obsolete in (
        "cee_cf10_hierarchical_bootstrap.csv",
        "cee_cf10_hierarchical_intervals.csv",
    ):
        (OUT / obsolete).unlink(missing_ok=True)
    bootstrap = crossed_cluster_bootstrap(strict_predictions)
    bootstrap.to_csv(OUT / "cee_cf10_crossed_bootstrap.csv", index=False)
    intervals = interval_summary(bootstrap)
    intervals.to_csv(OUT / "cee_cf10_crossed_intervals.csv", index=False)

    strict_metric = metrics[
        (metrics.prevalence == 0.40)
        & (metrics.stream == "strict_fault_applied_available")
        & (metrics.metric == "macro_auroc")
    ]
    pivot = strict_metric.pivot_table(
        index=["fault_type", "seed"], columns="method", values="value"
    ).reset_index()
    paired_rows = []
    for fault, group in pivot.groupby("fault_type"):
        for comparison, left, right in (
            ("Full_minus_PDRF", "RO-PDRF-Full", "PDRF"),
            ("Safe_minus_PDRF", "SR-PDRF-Safe-CF", "PDRF"),
            ("Safe_minus_Full", "SR-PDRF-Safe-CF", "RO-PDRF-Full"),
        ):
            difference = group[left] - group[right]
            paired_rows.append(
                {
                    "fault_type": fault,
                    "comparison": comparison,
                    "mean_difference": float(difference.mean()),
                    "positive_seeds": int((difference > 0).sum()),
                    "n_seeds": len(difference),
                    "wilcoxon_p_two_sided": float(
                        wilcoxon(difference, alternative="two-sided").pvalue
                    ),
                }
            )
    pd.DataFrame(paired_rows).to_csv(
        OUT / "cee_cf10_paired_effects.csv", index=False
    )

    scaling_source = pd.read_csv(UPGRADE / "source_data" / "major5_complexity_scaling.csv")
    pdrf_scaling = scaling_source[scaling_source.method == "PDRF"].copy()
    deployment_scaling = pd.DataFrame(
        {
            "groups": pdrf_scaling.groups,
            "forward_passes": pdrf_scaling.groups + 2,
            "stored_parameters": 2 * pdrf_scaling.parameters + 13,
            "model_state_kib_fp32": 2 * pdrf_scaling.state_kib + 13 * 4 / 1024,
            "inference_flops": (pdrf_scaling.groups + 2)
            * pdrf_scaling.inference_flops,
            "pdrf_flops": pdrf_scaling.inference_flops,
        }
    )
    deployment_scaling.to_csv(
        OUT / "cee_cf10_group_scaling.csv", index=False
    )

    design = {
        "run_id": FROZEN_RUN,
        "mode": "fast smoke test" if FAST else "formal frozen rerun",
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "test_realization": TEST_REALIZATION,
        "training_rerun_id": "major.fit-v2026-07-14-clean-teacher",
        "temperature_protocol": "first chronological half of Batch 7 calibration partition",
        "selector_protocol": (
            "second chronological half of Batch 7; 5-fold stratified group cross-fitting by original sample; "
            "OOF probabilities tune thresholds; coefficients refit on all informative calibration rows"
        ),
        "calibration_faults": list(CAL_FAULTS),
        "unseen_faults": list(UNSEEN_FAULTS),
        "prevalences": list(PREVALENCES),
        "primary_estimand": "fault actually applied while the target group remains available",
        "deployment_streams": [
            "clean/no imposed fault",
            "full mixed stream",
            "strict fault-applied-and-available",
            "assigned-but-masked",
            "unaffected",
        ],
        "bootstrap": (
            f"{N_BOOTSTRAP} crossed-cluster replicates: independently resample fault mechanisms and "
            "model-pair seed identities, retain the sampled seed draw across all sampled faults, "
            "then resample observations within each fault-by-seed cell"
        ),
        "selector_model": {
            "penalty": "l2",
            "C": SELECTOR_C,
            "solver": SELECTOR_SOLVER,
            "max_iter": SELECTOR_MAX_ITER,
            "tolerance": SELECTOR_TOL,
            "class_weight": SELECTOR_CLASS_WEIGHT,
            "random_state": SELECTOR_RANDOM_STATE,
        },
        "selector_uses_test_labels": False,
        "threshold_uses_in_fold_probabilities": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "cee_cf10_design.json").write_text(
        json.dumps(design, indent=2), encoding="utf-8"
    )

    summary = {
        "design": design,
        "selector_diagnostics_mean": pd.DataFrame(selector_diagnostic_rows)
        .mean(numeric_only=True)
        .to_dict(),
        "thresholds": {
            "safe_median": float(pd.DataFrame(selector_diagnostic_rows).safe_threshold.median()),
            "safe_range": [
                float(pd.DataFrame(selector_diagnostic_rows).safe_threshold.min()),
                float(pd.DataFrame(selector_diagnostic_rows).safe_threshold.max()),
            ],
            "balanced_median": float(pd.DataFrame(selector_diagnostic_rows).balanced_threshold.median()),
            "balanced_range": [
                float(pd.DataFrame(selector_diagnostic_rows).balanced_threshold.min()),
                float(pd.DataFrame(selector_diagnostic_rows).balanced_threshold.max()),
            ],
        },
        "crossed_intervals": intervals.to_dict("records"),
    }
    (OUT / "cee_cf10_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
