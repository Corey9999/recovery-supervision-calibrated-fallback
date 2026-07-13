"""Recovery-objective-matched 2x2 architecture experiment.

The experiment crosses bounded factorized versus unrestricted gated fusion with
the original versus recovery-oriented objective. Test corruptions are frozen
across methods and optimization seeds. Chemical and hydraulic evidence are
written separately so no dataset is treated as an optimization replicate.
"""
from dataclasses import replace
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd

import run_major_revision_experiments as major
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base
import run_hydraulic_validation as hyd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"source_data"
QUICK = os.getenv("M3_QUICK") == "1"
CHEM_SEEDS = (101,) if QUICK else tuple(range(101, 111))
HYD_SEEDS = (101,) if QUICK else tuple(range(101, 106))
CHEM_EPOCHS = 3 if QUICK else 60
HYD_EPOCHS = 3 if QUICK else 80

RO_PDRF = replace(
    major.SPECS["PDRF"], recovery_distillation=True, w_recovery=.30,
    brier_regularization=True, interior_barrier=True, w_interior=.005,
    degraded_auc=True, recovery_teacher="clean",
)
RO_CAGF = replace(
    major.SPECS["CAGF"], recovery_distillation=True, w_recovery=.30,
    brier_regularization=True, degraded_auc=True,
    gate_kl_regularization=True, w_gate_kl=.005, recovery_teacher="clean",
)
METHODS = {
    "CAGF": major.SPECS["CAGF"],
    "RO-CAGF": RO_CAGF,
    "PDRF": major.SPECS["PDRF"],
    "RO-PDRF": RO_PDRF,
}


def chemical_environments(x):
    rows = []
    # Broad fault-mechanism environments.
    for i, kind in enumerate(("silent_gaussian", "gain_loss", "offset", "linear_drift",
                              "stuck_at", "clipping", "correlated_dual")):
        # The principal silent-Gaussian condition reuses the frozen seed from
        # the original formal run, eliminating the former headline mismatch.
        fault_seed = 70001 if kind == "silent_gaussian" else 85000+i
        xx, mm, qq, aff = q2.make_fault(x, fault_seed, kind, modality=0)
        rows.append((f"type:{kind}", kind, 1, np.nan, .40, xx, mm, qq, aff))
    # Group location, severity and prevalence environments use the same fixed
    # generator as the formal primary experiment.
    for modality in range(4):
        xx, mm, qq, aff = major.fixed_fault(x, "silent", modality, 3., .40, 85100+modality)
        rows.append((f"group:{modality+1}", "silent_gaussian", modality+1, 3., .40, xx, mm, qq, aff))
    for scale in (1., 3., 5.):
        xx, mm, qq, aff = major.fixed_fault(x, "silent", 0, scale, .40, 85200)
        rows.append((f"scale:{scale:g}", "silent_gaussian", 1, scale, .40, xx, mm, qq, aff))
    for prevalence in (.10, .40, .70):
        xx, mm, qq, aff = major.fixed_fault(x, "silent", 0, 3., prevalence, 85300)
        rows.append((f"prevalence:{prevalence:g}", "silent_gaussian", 1, 3., prevalence, xx, mm, qq, aff))
    return rows


def run_chemical():
    split = q2.prepare_grouped(base.ensure_zip(None),
                               [list(range(i, i+4)) for i in range(0, 16, 4)])
    testx, testy, batches = split["test"]
    envs = chemical_environments(testx)
    metric_rows, cost_rows, pred_rows, mechanism_rows = [], [], [], []
    for seed in CHEM_SEEDS:
        train, select, calibration, cw = q2.train_views(split, seed)
        for method, spec in METHODS.items():
            model, cost = major.fit(spec, train, select, cw, seed,
                                    prior=(.45, .25, .18, .12), epochs=CHEM_EPOCHS)
            temp = major.fit_temperature(model, calibration)
            cost_rows.append({"dataset":"chemical", "task":"gas_class", "seed":seed,
                              "method":method, "training_forward_paths":3,
                              "inference_forward_paths":1, **cost})
            for env, kind, group, scale, prevalence, xx, mm, qq, aff in envs:
                p, scores, weights, _, infer = major.predict(model, xx, mm, qq, temp)
                cost_rows[-1].setdefault("inference_ms_per_observation",
                                         1000*infer/len(testy))
                for subset, take in (("all", np.ones(len(testy), bool)),
                                     ("affected", aff), ("unaffected", ~aff)):
                    if not take.any():
                        continue
                    for metric, value in risk.all_metrics(testy[take], p[take]).items():
                        metric_rows.append({
                            "dataset":"chemical", "task":"gas_class", "seed":seed,
                            "method":method, "environment":env, "fault_type":kind,
                            "group":group, "scale":scale, "prevalence":prevalence,
                            "subset":subset, "metric":metric, "value":value,
                        })
                if env == "type:silent_gaussian":
                    if method == "RO-PDRF":
                        clean_x = testx.copy() * mm[:, :, None]
                        clean_p, clean_scores, clean_weights, _, _ = major.predict(
                            model, clean_x, mm, qq, temp
                        )
                        for i in np.flatnonzero(aff):
                            mechanism_rows.append({
                                "seed": seed, "sample": int(i), "y": int(testy[i]),
                                "fault_score": float(scores[i, 0]),
                                "clean_score": float(clean_scores[i, 0]),
                                "fault_weight": float(weights[i, 0]),
                                "clean_weight": float(clean_weights[i, 0]),
                                "score_change": float(scores[i, 0] - clean_scores[i, 0]),
                                "weight_reduction": float(clean_weights[i, 0] - weights[i, 0]),
                                "clean_correct": bool(clean_p[i].argmax() == testy[i]),
                                "fault_correct": bool(p[i].argmax() == testy[i]),
                                "true_probability_change": float(
                                    p[i, testy[i]] - clean_p[i, testy[i]]
                                ),
                                "outer_5pct": bool(abs(scores[i, 0]) >= .95 * spec.bound),
                            })
                    for i in range(len(testy)):
                        pred_rows.append({
                            "dataset":"chemical", "seed":seed, "method":method,
                            "sample":i, "batch":int(batches[i]), "y":int(testy[i]),
                            "affected":bool(aff[i]),
                            **{f"p{j}":float(p[i,j]) for j in range(p.shape[1])},
                        })
    return metric_rows, cost_rows, pred_rows, mechanism_rows


def run_hydraulic():
    x, profile = hyd.load_features()
    metric_rows, cost_rows = [], []
    tasks = list(hyd.TASKS.items())
    if QUICK:
        tasks = tasks[:1]
    for task, column in tasks:
        train, select, calibration, test, values = hyd.split_task(x, profile, column)
        testx, _, _, testy = test
        ty = train[3]
        cw = len(ty)/(len(values)*np.bincount(ty, minlength=len(values)))
        for seed in HYD_SEEDS:
            for method, spec0 in METHODS.items():
                spec = replace(spec0, use_quality=False, train_quality=False)
                model, cost = major.fit(spec, train, select, cw, seed,
                                        prior=tuple([.2]*5), epochs=HYD_EPOCHS)
                temp = major.fit_temperature(model, calibration, neutral_q=True)
                cost_rows.append({"dataset":"hydraulic", "task":task, "seed":seed,
                                  "method":method, "training_forward_paths":3,
                                  "inference_forward_paths":1, **cost})
                for fault, modality in (("pressure", 0), ("vibration", 4)):
                    xx, mm, qq, aff = hyd.hydraulic_fault(testx, modality, 86000+modality)
                    p, _, _, _, infer = major.predict(model, xx, mm, qq, temp)
                    cost_rows[-1].setdefault("inference_ms_per_observation",
                                             1000*infer/len(testy))
                    for subset, take in (("all", np.ones(len(testy), bool)),
                                         ("affected", aff), ("unaffected", ~aff)):
                        for metric, value in risk.all_metrics(testy[take], p[take]).items():
                            metric_rows.append({
                                "dataset":"hydraulic", "task":task, "seed":seed,
                                "method":method, "environment":f"{task}:{fault}",
                                "fault_type":fault, "group":modality+1,
                                "scale":3., "prevalence":.40,
                                "subset":subset, "metric":metric, "value":value,
                            })
    return metric_rows, cost_rows


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    chem_metrics, chem_costs, preds, mechanism = run_chemical()
    hyd_metrics, hyd_costs = run_hydraulic()
    pd.DataFrame(chem_metrics+hyd_metrics).to_csv(
        OUT/"major3_objective_matched_metrics.csv", index=False)
    pd.DataFrame(chem_costs+hyd_costs).to_csv(
        OUT/"major3_objective_matched_costs.csv", index=False)
    pd.DataFrame(preds).to_csv(
        OUT/"major3_objective_matched_predictions.csv", index=False)
    pd.DataFrame(mechanism).to_csv(
        OUT/"major3_objective_matched_mechanism.csv", index=False)
    (OUT/"major3_objective_matched_design.json").write_text(json.dumps({
        "design":"2x2 architecture x recovery objective with architecture-specific regularization",
        "architectures":{"bounded":["PDRF","RO-PDRF"],
                         "unrestricted_gate":["CAGF","RO-CAGF"]},
        "recovery_signals":["clean-view recovery distillation", "clean/degraded Brier",
                            "degraded-view pairwise ranking"],
        "architecture_specific_regularizers":{
            "RO-PDRF":"bounded-score interior barrier, weight 0.005",
            "RO-CAGF":"gate KL-to-available-uniform regularizer, weight 0.005"},
        "chemical_seeds":list(CHEM_SEEDS), "hydraulic_seeds":list(HYD_SEEDS),
        "formal_teacher":"clean-only",
        "principal_test_fault_seed":70001,
        "test_missingness":0.20,
        "principal_fault":{"type":"silent_gaussian", "group":1,
                           "scale":3.0, "prevalence":0.40},
        "quality_setting":{"chemical":"q=1 at test", "hydraulic":"q disabled"},
        "checkpoint_criterion":"unweighted natural selection cross-entropy; patience 9",
        "training_forward_paths":3,
        "inference_forward_paths":1,
        "elapsed_seconds":time.perf_counter()-t0,
        "test_data_used_for_selection":False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
