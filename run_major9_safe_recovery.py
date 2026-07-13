"""Calibration-only safe recovery selector for the Q1-upgrade manuscript.

The selector never sees test labels.  For each optimization seed it compares a
base PDRF model with RO-PDRF-Full using confidence, disagreement and leave-one-
group-out consensus features.  A conservative threshold is fitted on a
separate half of Batch 7 with controlled faults.  At inference the system uses
RO-PDRF only when the selector supports it; otherwise it falls back to PDRF.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_major_revision_experiments as major
import run_major3_objective_matched as m3
import run_major7_fault_types_ema as typed
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
FAST = os.getenv("M9_FAST") == "1"
SEEDS = (101, 102, 103) if FAST else tuple(range(101, 111))
EPOCHS = 30 if FAST else 60
CAL_FAULTS = ("gaussian", "offset", "drift", "stuck_at")
TEST_FAULTS = ("gaussian", "offset", "drift", "stuck_at")


def normalized_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-8, 1.0)
    return -(q*np.log(q)).sum(axis=1)/np.log(q.shape[1])


def js_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.clip(a, 1e-8, 1.0)
    b = np.clip(b, 1e-8, 1.0)
    m = .5*(a+b)
    return .5*((a*np.log(a/m)).sum(axis=1) + (b*np.log(b/m)).sum(axis=1))


def leave_one_group_consensus(model, x, mask, quality, temperature):
    candidates = []
    valid_columns = []
    for group in range(mask.shape[1]):
        valid = (mask[:, group] > 0) & (mask.sum(axis=1) > 1)
        mm = mask.copy()
        mm[valid, group] = 0
        xx = x.copy()
        xx[valid, group] = 0
        qq = quality*mm
        p, _, _, _, _ = major.predict(model, xx, mm, qq, temperature)
        candidates.append(p)
        valid_columns.append(valid)
    stack = np.stack(candidates, axis=1)
    valid = np.stack(valid_columns, axis=1)
    denom = valid.sum(axis=1, keepdims=True)
    # Every evaluated chemical observation retains at least one group, but a
    # one-group row has no legal removal.  Its consensus is filled later.
    consensus = (stack*valid[:, :, None]).sum(axis=1)/np.maximum(denom, 1)
    return consensus, valid


def selector_features(base_p, recovery_p, consensus, valid_removals):
    no_removal = valid_removals.sum(axis=1) == 0
    consensus = consensus.copy()
    consensus[no_removal] = .5*(base_p[no_removal]+recovery_p[no_removal])
    base_conf = base_p.max(axis=1)
    recovery_conf = recovery_p.max(axis=1)
    base_entropy = normalized_entropy(base_p)
    recovery_entropy = normalized_entropy(recovery_p)
    return np.column_stack([
        base_conf,
        recovery_conf,
        recovery_conf-base_conf,
        base_entropy,
        recovery_entropy,
        recovery_entropy-base_entropy,
        js_rows(base_p, recovery_p),
        js_rows(base_p, consensus),
        js_rows(recovery_p, consensus),
        (base_p.argmax(axis=1) == consensus.argmax(axis=1)).astype(float),
        (recovery_p.argmax(axis=1) == consensus.argmax(axis=1)).astype(float),
        valid_removals.sum(axis=1)/valid_removals.shape[1],
    ]).astype(np.float64)


FEATURES = [
    "base_confidence", "recovery_confidence", "confidence_delta",
    "base_entropy", "recovery_entropy", "entropy_delta",
    "base_recovery_js", "base_consensus_js", "recovery_consensus_js",
    "base_consensus_agreement", "recovery_consensus_agreement",
    "removal_fraction",
]


def predict_pair(base_model, recovery_model, x, mask, quality, tb, tr):
    pb, _, _, _, _ = major.predict(base_model, x, mask, quality, tb)
    pr, _, _, _, _ = major.predict(recovery_model, x, mask, quality, tr)
    consensus, valid = leave_one_group_consensus(
        recovery_model, x, mask, quality, tr)
    features = selector_features(pb, pr, consensus, valid)
    return pb, pr, consensus, features


def fit_selector(calibration_rows: list[dict]):
    frame = pd.DataFrame(calibration_rows)
    # Only disagreements in correctness identify which model should be used.
    informative = frame.base_correct != frame.recovery_correct
    x = frame.loc[informative, FEATURES].to_numpy(float)
    y = frame.loc[informative, "recovery_correct"].to_numpy(int)
    if len(np.unique(y)) != 2:
        raise RuntimeError("Calibration selector requires both recovery and fallback examples")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.5, class_weight="balanced", max_iter=2000,
                           random_state=0),
    )
    model.fit(x, y)
    probability = model.predict_proba(frame[FEATURES].to_numpy(float))[:, 1]
    base_correct = frame.base_correct.to_numpy(bool)
    recovery_correct = frame.recovery_correct.to_numpy(bool)
    candidates = []
    for threshold in np.linspace(.50, .90, 17):
        choose_recovery = probability >= threshold
        selected_correct = np.where(choose_recovery, recovery_correct, base_correct)
        transfer_eligible = base_correct & ~recovery_correct
        recovery_eligible = ~base_correct & recovery_correct
        candidates.append({
            "threshold": float(threshold),
            "accuracy": float(selected_correct.mean()),
            "transfer_rate": float((transfer_eligible & ~selected_correct).sum()/
                                   max(1, transfer_eligible.sum())),
            "recovery_rate": float((recovery_eligible & selected_correct).sum()/
                                   max(1, recovery_eligible.sum())),
            "recovery_usage": float(choose_recovery.mean()),
        })
    audit = pd.DataFrame(candidates)
    # Safety first: lowest transfer; among ties retain the highest accuracy and
    # recovery.  This rule is fixed before the test set is evaluated.
    safe = audit.sort_values(
        ["transfer_rate", "accuracy", "recovery_rate", "threshold"],
        ascending=[True, False, False, False]).iloc[0]
    balanced = audit.sort_values(
        ["accuracy", "transfer_rate", "recovery_rate", "threshold"],
        ascending=[False, True, False, False]).iloc[0]
    return model, float(safe.threshold), float(balanced.threshold), audit


def append_metric_rows(rows, seed, fault_type, method, subset, y, p, n):
    for metric, value in risk.all_metrics(y, p).items():
        rows.append({
            "seed": seed, "fault_type": fault_type, "method": method,
            "subset": subset, "metric": metric, "value": value, "n": n,
        })


def main():
    OUT.mkdir(exist_ok=True)
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i+4)) for i in range(0, 16, 4)])
    testx, testy, test_batches = split["test"]
    cx, cy, _ = split["calibration"]
    cut = len(cx)//2
    temperature_x, temperature_y = cx[:cut], cy[:cut]
    gate_x, gate_y = cx[cut:], cy[cut:]

    metric_rows, prediction_rows, selector_rows, cost_rows = [], [], [], []
    started = time.perf_counter()
    for seed in SEEDS:
        train, select, _, class_weights = q2.train_views(split, seed)
        temp_view = (*base.make_view(
            temperature_x, 25000+seed, "missing_drift", train=True),
            temperature_y)
        seed_started = time.perf_counter()
        base_model, base_cost = major.fit(
            major.SPECS["PDRF"], train, select, class_weights, seed,
            prior=(.45, .25, .18, .12), epochs=EPOCHS)
        recovery_model, recovery_cost = major.fit(
            m3.RO_PDRF, train, select, class_weights, seed,
            prior=(.45, .25, .18, .12), epochs=EPOCHS)
        tb = major.fit_temperature(base_model, temp_view)
        tr = major.fit_temperature(recovery_model, temp_view)

        calibration_rows = []
        for fidx, fault_type in enumerate(CAL_FAULTS):
            for ridx in range(2):
                faultx, mask, quality, assigned = typed.fixed_typed_fault(
                    gate_x, fault_type, seed=96000+100*fidx+10*ridx+seed)
                applied = assigned & (mask[:, 0] > 0)
                pb, pr, _, features = predict_pair(
                    base_model, recovery_model, faultx, mask, quality, tb, tr)
                for i in np.flatnonzero(applied):
                    record = {
                        "fault_type": fault_type,
                        "base_correct": bool(pb[i].argmax() == gate_y[i]),
                        "recovery_correct": bool(pr[i].argmax() == gate_y[i]),
                    }
                    record.update(dict(zip(FEATURES, features[i])))
                    calibration_rows.append(record)

        selector, safe_threshold, balanced_threshold, threshold_audit = fit_selector(calibration_rows)
        threshold_audit.insert(0, "seed", seed)
        selector_rows.extend(threshold_audit.to_dict("records"))

        for fault_type in TEST_FAULTS:
            faultx, mask, quality, assigned = typed.fixed_typed_fault(
                testx, fault_type, seed=70001)
            applied = assigned & (mask[:, 0] > 0)
            assigned_unavailable = assigned & (mask[:, 0] == 0)
            pb, pr, consensus, features = predict_pair(
                base_model, recovery_model, faultx, mask, quality, tb, tr)
            selector_probability = selector.predict_proba(features)[:, 1]
            choose_safe = selector_probability >= safe_threshold
            choose_balanced = selector_probability >= balanced_threshold
            ps = np.where(choose_safe[:, None], pr, pb)
            pbal = np.where(choose_balanced[:, None], pr, pb)
            for subset, take in (
                ("fault_applied_available", applied),
                ("assigned", assigned),
                ("assigned_unavailable", assigned_unavailable),
                ("unaffected", ~assigned),
                ("all", np.ones(len(testy), bool)),
            ):
                if take.sum() < 2:
                    continue
                for method, p in (("PDRF", pb), ("RO-PDRF-Full", pr),
                                  ("SR-PDRF-Balanced", pbal),
                                  ("SR-PDRF-Safe", ps)):
                    append_metric_rows(metric_rows, seed, fault_type, method,
                                       subset, testy[take], p[take], int(take.sum()))

            # Prediction-level safety accounting uses the actual applied set.
            for i in np.flatnonzero(applied):
                prediction_rows.append({
                    "seed": seed, "fault_type": fault_type,
                    "sample": int(i), "batch": int(test_batches[i]),
                    "y": int(testy[i]),
                    "safe_threshold": safe_threshold,
                    "balanced_threshold": balanced_threshold,
                    "selector_probability": float(selector_probability[i]),
                    "choose_recovery_safe": bool(choose_safe[i]),
                    "choose_recovery_balanced": bool(choose_balanced[i]),
                    "base_prediction": int(pb[i].argmax()),
                    "recovery_prediction": int(pr[i].argmax()),
                    "safe_prediction": int(ps[i].argmax()),
                    "balanced_prediction": int(pbal[i].argmax()),
                    "base_correct": bool(pb[i].argmax() == testy[i]),
                    "recovery_correct": bool(pr[i].argmax() == testy[i]),
                    "safe_correct": bool(ps[i].argmax() == testy[i]),
                    "balanced_correct": bool(pbal[i].argmax() == testy[i]),
                    **dict(zip(FEATURES, features[i])),
                })

        cost_rows.append({
            "seed": seed, "base_temperature": tb,
            "recovery_temperature": tr,
            "safe_threshold": safe_threshold,
            "balanced_threshold": balanced_threshold,
            "elapsed_seconds": time.perf_counter()-seed_started,
            "base_train_seconds": base_cost["train_seconds"],
            "recovery_train_seconds": recovery_cost["train_seconds"],
            "base_parameters": base_cost["parameters"],
            "recovery_parameters": recovery_cost["parameters"],
        })

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(OUT/"major9_safe_recovery_metrics.csv", index=False)
    predictions.to_csv(OUT/"major9_safe_recovery_predictions.csv", index=False)
    pd.DataFrame(selector_rows).to_csv(
        OUT/"major9_safe_recovery_thresholds.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(
        OUT/"major9_safe_recovery_costs.csv", index=False)

    safety_rows = []
    for (seed, fault_type), group in predictions.groupby(["seed", "fault_type"]):
        recovery_opportunity = ~group.base_correct & group.recovery_correct
        transfer_opportunity = group.base_correct & ~group.recovery_correct
        for variant, correct_col, choose_col in (
            ("Balanced", "balanced_correct", "choose_recovery_balanced"),
            ("Safe", "safe_correct", "choose_recovery_safe"),
        ):
            safety_rows.append({
                "seed": seed, "fault_type": fault_type, "variant": variant,
                "n": len(group),
                "recovery_opportunities": int(recovery_opportunity.sum()),
                "recovery_preserved": int((recovery_opportunity & group[correct_col]).sum()),
                "recovery_preservation_rate": float(
                    (recovery_opportunity & group[correct_col]).sum()/max(1, recovery_opportunity.sum())),
                "transfer_opportunities": int(transfer_opportunity.sum()),
                "negative_transfer_remaining": int((transfer_opportunity & ~group[correct_col]).sum()),
                "negative_transfer_rate": float(
                    (transfer_opportunity & ~group[correct_col]).sum()/max(1, transfer_opportunity.sum())),
                "recovery_usage": float(group[choose_col].mean()),
            })
    pd.DataFrame(safety_rows).to_csv(
        OUT/"major9_safe_recovery_safety.csv", index=False)
    (OUT/"major9_safe_recovery_design.json").write_text(json.dumps({
        "mode": "fast development" if FAST else "formal frozen rerun",
        "seeds": list(SEEDS), "epochs": EPOCHS,
        "temperature_calibration": "first half of Batch 7 calibration partition",
        "selector_calibration": "second half of Batch 7; four controlled fault types; two realizations each",
        "selector": "standardized logistic preference model fitted only where PDRF and RO-PDRF correctness differ",
        "threshold_rules": {
            "Safe": "minimize calibration negative transfer, then maximize accuracy and recovery",
            "Balanced": "maximize calibration accuracy, then minimize negative transfer",
        },
        "inference": "use RO-PDRF-Full only above the corresponding frozen selector threshold; otherwise use PDRF",
        "test_used_for_selection": False,
        "elapsed_seconds": time.perf_counter()-started,
    }, indent=2), encoding="utf-8")

    show = metrics[(metrics.subset == "fault_applied_available") &
                   (metrics.metric == "macro_auroc")]
    print(show.groupby(["fault_type", "method"]).value.mean().unstack().round(4))
    print("\nSafety")
    print(pd.DataFrame(safety_rows).groupby(["fault_type", "variant"])[[
        "recovery_preservation_rate", "negative_transfer_rate", "recovery_usage"]].mean().round(4))


if __name__ == "__main__":
    main()
