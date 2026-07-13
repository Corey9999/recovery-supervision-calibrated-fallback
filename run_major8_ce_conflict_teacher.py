"""Ground-truth CE-conflict distillation abstention for manuscript 16.

Training labels are already used by the clean and degraded task losses.  This
control suppresses the clean-to-fault KL term whenever the clean teacher has
higher cross-entropy than the current faulted prediction.  The gate is used
only during training; inference remains a single RO-PDRF forward pass.
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
QUICK = os.getenv("M8_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 111))
EPOCHS = 3 if QUICK else 60
METHOD = "RO-PDRF-CEG"
SPEC = replace(m3.RO_PDRF, recovery_teacher="ce_conflict_gate")


def main():
    OUT.mkdir(exist_ok=True)
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i + 4)) for i in range(0, 16, 4)])
    testx, testy, batches = split["test"]
    faultx, mask, quality, affected = major.fixed_fault(
        testx, "silent", 0, 3.0, .40, 70001)
    cleanx = (testx*mask[:, :, None]).astype(np.float32)
    rows, predictions, costs, traces = [], [], [], []
    started = time.perf_counter()

    for seed in SEEDS:
        train, select, calibration, class_weights = q2.train_views(split, seed)
        trace = []
        model, cost = major.fit(
            SPEC, train, select, class_weights, seed,
            prior=(.45, .25, .18, .12), epochs=EPOCHS, trace=trace)
        for record in trace:
            record["method"] = METHOD
            traces.append(record)
        temperature = major.fit_temperature(model, calibration)
        p_fault, _, _, _, inference_seconds = major.predict(
            model, faultx, mask, quality, temperature)
        p_clean, _, _, _, _ = major.predict(
            model, cleanx, mask, quality, temperature)
        costs.append({
            "method": METHOD, "seed": seed, "temperature": temperature,
            "inference_ms_per_observation":
                1000*inference_seconds/len(testy), **cost})
        for subset, take in (("all", np.ones(len(testy), bool)),
                             ("affected", affected),
                             ("unaffected", ~affected)):
            for metric, value in risk.all_metrics(testy[take], p_fault[take]).items():
                rows.append({"method": METHOD, "seed": seed,
                             "subset": subset, "metric": metric,
                             "value": value, "n": int(take.sum())})
        for i in np.flatnonzero(affected):
            y = int(testy[i])
            predictions.append({
                "method": METHOD, "seed": seed, "sample": int(i),
                "batch": int(batches[i]), "y": y,
                "clean_prediction": int(p_clean[i].argmax()),
                "fault_prediction": int(p_fault[i].argmax()),
                "clean_correct": bool(p_clean[i].argmax() == y),
                "fault_correct": bool(p_fault[i].argmax() == y),
                "clean_true_probability": float(p_clean[i, y]),
                "fault_true_probability": float(p_fault[i, y]),
                "ce_conflict_keep": bool(p_clean[i, y] >= p_fault[i, y]),
            })

    pd.DataFrame(rows).to_csv(
        OUT/"major8_ce_conflict_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        OUT/"major8_ce_conflict_predictions.csv", index=False)
    pd.DataFrame(costs).to_csv(
        OUT/"major8_ce_conflict_costs.csv", index=False)
    pd.DataFrame(traces).to_csv(
        OUT/"major8_ce_conflict_training_curves.csv", index=False)
    (OUT/"major8_ce_conflict_design.json").write_text(json.dumps({
        "method": METHOD, "seeds": list(SEEDS), "epochs": EPOCHS,
        "training_rule": "apply clean-to-fault KL only if clean CE <= degraded CE",
        "label_use": "training labels already used by clean and degraded task losses",
        "inference_rule": "ordinary single-pass RO-PDRF; no teacher or gate at inference",
        "principal_fault": {"type": "Gaussian", "group": 1, "scale": 3.0,
                            "prevalence": .40, "realization": 70001},
        "test_used_for_selection": False,
        "elapsed_seconds": time.perf_counter()-started,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
