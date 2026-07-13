"""Strong fusion, teacher, and calibration controls for the Q1 revision.

All controls use the unique chemical headline split, corruption realization,
optimization seeds, checkpoint rule, and per-seed temperature calibration.
The test set is not used to select a method or hyperparameter.
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
QUICK = os.getenv("M6_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 111))
EPOCHS = 3 if QUICK else 60


PDRF_CD = replace(
    major.SPECS["PDRF"], recovery_distillation=True,
    recovery_teacher="clean", w_recovery=.30)
PDRF_EMA_CD = replace(
    PDRF_CD, recovery_teacher="ema", teacher_ema_decay=.99)
PDRF_SOFT_CD = replace(
    PDRF_CD, recovery_confidence_weighted=True)
RO_AT_GATE = replace(
    m3.RO_CAGF, architecture="attention_gate")
RO_PDRF_CAL = replace(
    m3.RO_PDRF, w_brier=.20)

METHODS = {
    "PDRF+CD": PDRF_CD,
    "PDRF+EMA-CD": PDRF_EMA_CD,
    "PDRF+Soft-CD": PDRF_SOFT_CD,
    "RO-AT-GATE": RO_AT_GATE,
    "RO-PDRF-CAL": RO_PDRF_CAL,
}


def main():
    OUT.mkdir(exist_ok=True)
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i + 4)) for i in range(0, 16, 4)])
    testx, testy, batches = split["test"]
    xx, mm, qq, affected = major.fixed_fault(
        testx, "silent", 0, 3.0, .40, 70001)
    rows, predictions, costs = [], [], []
    started = time.perf_counter()

    for seed in SEEDS:
        train, select, calibration, class_weights = q2.train_views(split, seed)
        for method, spec in METHODS.items():
            model, cost = major.fit(
                spec, train, select, class_weights, seed,
                prior=(.45, .25, .18, .12), epochs=EPOCHS)
            temperature = major.fit_temperature(model, calibration)
            p, _, _, _, inference_seconds = major.predict(
                model, xx, mm, qq, temperature)
            costs.append({
                "method": method, "seed": seed, "temperature": temperature,
                "inference_ms_per_observation": 1000 * inference_seconds / len(testy),
                **cost,
            })
            for subset, take in (
                    ("all", np.ones(len(testy), dtype=bool)),
                    ("affected", affected), ("unaffected", ~affected)):
                for metric, value in risk.all_metrics(testy[take], p[take]).items():
                    rows.append({
                        "method": method, "seed": seed, "subset": subset,
                        "metric": metric, "value": value, "n": int(take.sum())})
            for i in range(len(testy)):
                record = {
                    "method": method, "seed": seed, "sample": i,
                    "batch": int(batches[i]), "y": int(testy[i]),
                    "affected": bool(affected[i])}
                record.update({f"p{c + 1}": float(p[i, c]) for c in range(p.shape[1])})
                predictions.append(record)

    pd.DataFrame(rows).to_csv(OUT / "major6_strong_control_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        OUT / "major6_strong_control_predictions.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "major6_strong_control_costs.csv", index=False)
    (OUT / "major6_strong_control_design.json").write_text(json.dumps({
        "seeds": list(SEEDS),
        "headline_fault": {"group": 1, "scale": 3.0, "prevalence": .40,
                           "realization": 70001},
        "attention_gate": {"group_token_width": 32, "heads": 4,
                           "feedforward_width": 64, "layers": 1},
        "ema_decay": .99,
        "soft_teacher_weight": "maximum clean-view class probability",
        "calibration_control_brier_weight": .20,
        "temperature_scaling": "per seed on calibration split",
        "test_used_for_selection": False,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
