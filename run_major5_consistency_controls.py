"""Consistency-training controls, repeated fault realizations and teacher audit.

The script separates clean-view distillation from the additional recovery
losses. It also evaluates independently generated fixed test corruptions while
keeping optimization seed as the lowest-level repeat. No test result is used
for model or hyperparameter selection.
"""
from dataclasses import replace
from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd

import run_major_revision_experiments as major
import run_major3_objective_matched as m3
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
QUICK = os.getenv("M5_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 111))
FAULT_MODEL_SEEDS = (101,) if QUICK else tuple(range(101, 106))
EPOCHS = 3 if QUICK else 60
FAULT_SEEDS = (70001, 70002) if QUICK else tuple(range(70001, 70009))


def clean_distillation(spec, threshold=0.0, modality_dropout=False):
    return replace(
        spec,
        recovery_distillation=True,
        recovery_teacher="clean",
        recovery_confidence_threshold=threshold,
        modality_dropout=modality_dropout,
        w_recovery=.30,
    )


CONTROL_METHODS = {
    "EF-PD+CD": clean_distillation(major.SPECS["EF_PD"]),
    "UF-PD+CD": clean_distillation(major.SPECS["UF_PD"]),
    "CAGF+CD": clean_distillation(major.SPECS["CAGF"]),
    "PDRF+CD": clean_distillation(major.SPECS["PDRF"]),
    "PDRF+CD-T07": clean_distillation(major.SPECS["PDRF"], threshold=.70),
    "ModDrop-SD": clean_distillation(
        major.SPECS["EF_PD"], modality_dropout=True),
}

FAULT_METHODS = {
    "CAGF": major.SPECS["CAGF"],
    "RO-CAGF": m3.RO_CAGF,
    "PDRF": major.SPECS["PDRF"],
    "RO-PDRF": m3.RO_PDRF,
}


def metric_rows(seed, method, realization, batches, y, p, affected):
    rows = []
    for batch in ("all", 8, 9, 10):
        in_batch = np.ones(len(y), bool) if batch == "all" else batches == batch
        for subset, take0 in (("all", np.ones(len(y), bool)),
                              ("affected", affected),
                              ("unaffected", ~affected)):
            take = in_batch & take0
            if take.sum() < 2 or len(np.unique(y[take])) < 2:
                continue
            for metric, value in risk.all_metrics(y[take], p[take]).items():
                rows.append({"seed": seed, "method": method,
                             "fault_realization": realization,
                             "batch": batch, "subset": subset,
                             "metric": metric, "value": value,
                             "n": int(take.sum())})
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i+4)) for i in range(0,16,4)])
    testx, testy, batches = split["test"]
    rows, preds, costs, traces = [], [], [], []
    t0 = time.perf_counter()

    # Six direct controls use all 10 optimization seeds and the unique
    # seed-70001 headline fault.
    principal = major.fixed_fault(testx, "silent", 0, 3., .40, 70001)
    for seed in SEEDS:
        train, select, calibration, cw = q2.train_views(split, seed)
        for method, spec in CONTROL_METHODS.items():
            trace = []
            model, cost = major.fit(
                spec, train, select, cw, seed,
                prior=(.45,.25,.18,.12), epochs=EPOCHS, trace=trace)
            temp = major.fit_temperature(model, calibration)
            xx, mm, qq, affected = principal
            p, _, _, _, infer = major.predict(model, xx, mm, qq, temp)
            rows.extend(metric_rows(seed, method, 70001, batches,
                                    testy, p, affected))
            costs.append({"seed":seed, "method":method, **cost,
                          "inference_ms_per_observation":1000*infer/len(testy)})
            for item in trace:
                traces.append({"method":method, **item})

    # Five matched fits per headline method are evaluated on eight independently
    # generated, method-shared test faults. This isolates fault-realization
    # variability from optimization variability.
    for seed in FAULT_MODEL_SEEDS:
        train, select, calibration, cw = q2.train_views(split, seed)
        for method, spec in FAULT_METHODS.items():
            model, cost = major.fit(
                spec, train, select, cw, seed,
                prior=(.45,.25,.18,.12), epochs=EPOCHS)
            temp = major.fit_temperature(model, calibration)
            for realization in FAULT_SEEDS:
                xx, mm, qq, affected = major.fixed_fault(
                    testx, "silent", 0, 3., .40, realization)
                # Teacher audit keeps availability/missingness identical and
                # removes only the injected Gaussian sensor corruption.
                clean_p, _, _, _, _ = major.predict(
                    model, testx*mm[:,:,None], mm, qq, temp)
                p, _, _, _, infer = major.predict(model, xx, mm, qq, temp)
                rows.extend(metric_rows(seed, method, realization, batches,
                                        testy, p, affected))
                for i in np.flatnonzero(affected):
                    preds.append({
                        "seed":seed, "method":method,
                        "fault_realization":realization,
                        "sample":int(i), "batch":int(batches[i]),
                        "y":int(testy[i]),
                        "clean_prediction":int(clean_p[i].argmax()),
                        "fault_prediction":int(p[i].argmax()),
                        "clean_correct":bool(clean_p[i].argmax()==testy[i]),
                        "fault_correct":bool(p[i].argmax()==testy[i]),
                        "clean_confidence":float(clean_p[i].max()),
                        "clean_true_probability":float(clean_p[i,testy[i]]),
                        "fault_true_probability":float(p[i,testy[i]]),
                    })

    pd.DataFrame(rows).to_csv(OUT/"major5_consistency_metrics.csv",index=False)
    pd.DataFrame(preds).to_csv(OUT/"major5_teacher_fault_predictions.csv",index=False)
    pd.DataFrame(costs).to_csv(OUT/"major5_consistency_costs.csv",index=False)
    pd.DataFrame(traces).to_csv(OUT/"major5_training_stability.csv",index=False)
    (OUT/"major5_consistency_design.json").write_text(json.dumps({
        "control_seeds":list(SEEDS),
        "fault_model_seeds":list(FAULT_MODEL_SEEDS),
        "fault_realizations":list(FAULT_SEEDS),
        "principal_fault":{"group":1,"scale":3.0,"prevalence":.40},
        "clean_distillation_weight":.30,
        "confidence_threshold":.70,
        "moddrop_baseline":"early fusion; one available group removed per training pair; clean-view self-distillation",
        "test_data_used_for_selection":False,
        "elapsed_seconds":time.perf_counter()-t0,
    },indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
