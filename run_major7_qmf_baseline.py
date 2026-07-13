"""Published failure-aware fusion baseline for the manuscript-15 revision.

The baseline adapts the public Quality-aware Multimodal Fusion (QMF) rule
from Zhang et al. (ICML 2023) to the four sensor groups. It preserves QMF's
energy confidence, decisive decision-level fusion, group-specific
classification losses and training-trajectory confidence ranking. The sensor
encoders, split, paired degradation exposure, checkpoint rule and temperature
calibration are matched to the manuscript protocol. Test data are never used
for method or hyperparameter selection.
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


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
QUICK = os.getenv("M7_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 111))
EPOCHS = 3 if QUICK else 60

QMF_PD = replace(
    major.Spec(), architecture="qmf", paired=True, score_loss=False,
    consistency=False, norm_mono=False, raw_mono=False, score_rank=False,
    direct_rank=False, use_quality=False, train_quality=False,
    recovery_distillation=False, brier_regularization=False,
    gate_kl_regularization=False, qmf_rank_weight=.10)


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
        model, cost = major.fit(
            QMF_PD, train, select, class_weights, seed,
            prior=(.45, .25, .18, .12), epochs=EPOCHS)
        temperature = major.fit_temperature(model, calibration)
        p, _, weights, _, inference_seconds = major.predict(
            model, xx, mm, qq, temperature)
        costs.append({
            "method": "QMF-PD", "seed": seed, "temperature": temperature,
            "inference_ms_per_observation": 1000*inference_seconds/len(testy),
            **cost})
        for subset, take in (("all", np.ones(len(testy), bool)),
                             ("affected", affected),
                             ("unaffected", ~affected)):
            for metric, value in risk.all_metrics(testy[take], p[take]).items():
                rows.append({"method": "QMF-PD", "seed": seed,
                             "subset": subset, "metric": metric,
                             "value": value, "n": int(take.sum())})
        for i in range(len(testy)):
            row = {"method": "QMF-PD", "seed": seed, "sample": i,
                   "batch": int(batches[i]), "y": int(testy[i]),
                   "affected": bool(affected[i]),
                   "affected_group_weight": float(weights[i, 0])}
            row.update({f"p{c}": float(p[i, c]) for c in range(p.shape[1])})
            predictions.append(row)

    pd.DataFrame(rows).to_csv(OUT/"major7_qmf_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        OUT/"major7_qmf_predictions.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT/"major7_qmf_costs.csv", index=False)
    (OUT/"major7_qmf_design.json").write_text(json.dumps({
        "method": "sensor-group adaptation of QMF",
        "paper": "Provable Dynamic Fusion for Low-Quality Multimodal Data",
        "publication": "ICML 2023; PMLR 202:41753-41769",
        "official_code": "https://github.com/QingyangZhang/QMF",
        "energy_temperature": 1.0,
        "confidence": "-0.1 times negative logsumexp energy",
        "trajectory_rank_weight": .10,
        "seeds": list(SEEDS), "epochs": EPOCHS,
        "test_used_for_selection": False,
        "elapsed_seconds": time.perf_counter()-started,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
