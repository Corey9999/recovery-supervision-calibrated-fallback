"""Exploratory remedies for the two Section 2.3 aggregation/subset issues.

All model and ensemble choices are fitted without access to batches 8--10.
The script evaluates (i) calibration-set convex stacking and regularized
Dirichlet calibration, and (ii) degraded-view AUC/confidence objectives.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss

import run_major_revision_experiments as major
import run_q2_revision as q2
import run_uci_validation as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
SEEDS = tuple(range(101, 104)) if os.getenv("TARGET_QUICK") == "1" else tuple(range(101, 111))

SPECS = {
    "CAGF": major.SPECS["CAGF"],
    "PDRF": major.SPECS["PDRF"],
    "PDRF_AUC": replace(major.SPECS["PDRF"], degraded_auc=True),
    "PDRF_CML": replace(major.SPECS["PDRF"], confidence_mono=True),
    "PDRF_AUC_CML": replace(major.SPECS["PDRF"], degraded_auc=True, confidence_mono=True),
}
if os.getenv("TARGET_METHODS"):
    requested = [x.strip() for x in os.getenv("TARGET_METHODS", "").split(",") if x.strip()]
    SPECS = {name:SPECS[name] for name in requested}


def convex_stack(calibration_probs: np.ndarray, y: np.ndarray, ridge: float = 0.05):
    """Fit nonnegative ensemble weights on calibration data only."""
    members = calibration_probs.shape[0]

    def objective(theta):
        shifted = theta - theta.max()
        w = np.exp(shifted); w /= w.sum()
        p = np.tensordot(w, calibration_probs, axes=(0, 0))
        return log_loss(y, np.clip(p, 1e-12, 1), labels=np.arange(p.shape[1])) + ridge*np.square(w-1/members).sum()

    result = minimize(objective, np.zeros(members), method="L-BFGS-B")
    shifted = result.x-result.x.max()
    weights = np.exp(shifted); weights /= weights.sum()
    return weights, bool(result.success)


def dirichlet_fit(p: np.ndarray, y: np.ndarray):
    """Regularized multiclass linear map on log probabilities."""
    x = np.log(np.clip(p, 1e-8, 1.0))
    model = LogisticRegressionCV(
        Cs=np.array([0.01, 0.1, 1.0, 10.0]), cv=5, scoring="neg_log_loss",
        max_iter=3000, random_state=20260712,
    )
    model.fit(x, y)
    return model


def evaluate_subsets(y, p, affected):
    p = np.clip(p, 1e-12, None)
    p = p/p.sum(1, keepdims=True)
    rows = []
    for subset, take in (("all", np.ones(len(y), dtype=bool)), ("affected", affected), ("unaffected", ~affected)):
        for metric, value in major.metrics(y[take], p[take]).items():
            if metric in {"accuracy", "macro_f1", "macro_auprc", "macro_auroc", "nll"}:
                rows.append((subset, metric, value))
    return rows


def main():
    zip_path = base.ensure_zip(None)
    groups = [list(range(i, i+4)) for i in range(0, 16, 4)]
    split = q2.prepare_grouped(zip_path, groups)
    cx, cy, _ = split["calibration"]
    testx, testy, testb = split["test"]

    cal_fault = major.fixed_fault(cx, "silent", seed=76001)
    test_fault = major.fixed_fault(testx, "silent", seed=70001)
    cfx, cfm, cfq, _ = cal_fault
    tfx, tfm, tfq, affected = test_fault

    result_rows, prediction_rows = [], []
    cal_by_method, test_by_method = {}, {}
    for method, spec in SPECS.items():
        cal_members, test_members = [], []
        for seed in SEEDS:
            train, select, calibration, class_weights = q2.train_views(split, seed)
            model, _ = major.fit(spec, train, select, class_weights, seed,
                                 prior=(.45, .25, .18, .12), epochs=60)
            temperature = major.fit_temperature(model, calibration)
            pcal, _, _, _, _ = major.predict(model, cfx, cfm, cfq, temperature)
            ptest, _, _, _, _ = major.predict(model, tfx, tfm, tfq, temperature)
            cal_members.append(pcal); test_members.append(ptest)
            for subset, metric, value in evaluate_subsets(testy, ptest, affected):
                result_rows.append({"level":"single_model", "method":method, "seed":seed,
                                    "pool":"none", "subset":subset, "metric":metric, "value":value})
        cal_by_method[method] = np.stack(cal_members)
        test_by_method[method] = np.stack(test_members)

    ensemble_meta = {}
    for method in SPECS:
        cal_members = cal_by_method[method]; test_members = test_by_method[method]
        equal_cal = cal_members.mean(0); equal_test = test_members.mean(0)
        weights, success = convex_stack(cal_members, cy)
        stacked_cal = np.tensordot(weights, cal_members, axes=(0,0))
        stacked_test = np.tensordot(weights, test_members, axes=(0,0))
        calibrator = dirichlet_fit(stacked_cal, cy)
        dirichlet_test = calibrator.predict_proba(np.log(np.clip(stacked_test,1e-8,1.0)))
        pools = {"equal_probability":equal_test, "calibration_stacking":stacked_test,
                 "stacking_dirichlet":dirichlet_test}
        ensemble_meta[method] = {"stacking_success":success, "weights":weights.tolist(),
                                 "dirichlet_C":float(np.ravel(calibrator.C_)[0])}
        for pool, prob in pools.items():
            for subset, metric, value in evaluate_subsets(testy, prob, affected):
                result_rows.append({"level":"ensemble", "method":method, "seed":-1,
                                    "pool":pool, "subset":subset, "metric":metric, "value":value})
            for i in range(len(testy)):
                row={"sample":i,"batch":int(testb[i]),"y":int(testy[i]),"affected":bool(affected[i]),
                     "method":method,"pool":pool}
                row.update({f"p{j+1}":float(prob[i,j]) for j in range(prob.shape[1])})
                prediction_rows.append(row)

    pd.DataFrame(result_rows).to_csv(OUT/"targeted_improvement_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT/"targeted_ensemble_predictions.csv", index=False)
    (OUT/"targeted_improvement_design.json").write_text(json.dumps({
        "seeds":SEEDS, "calibration_fault_seed":76001, "test_fault_seed":70001,
        "model_variants":list(SPECS), "ensemble":ensemble_meta,
        "selection_boundary":"All weights, regularization choices and calibration maps use batch 7 only."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
