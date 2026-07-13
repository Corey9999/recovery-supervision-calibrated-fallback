"""Full recovery objective with an exponential-moving-average clean teacher."""
from dataclasses import replace
from pathlib import Path
import json
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
SEEDS = tuple(range(101, 111))
SPEC = replace(
    m3.RO_PDRF, recovery_teacher="ema", teacher_ema_decay=.99)


def main():
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i + 4)) for i in range(0, 16, 4)])
    testx, testy, batches = split["test"]
    xx, mm, qq, affected = major.fixed_fault(testx, "silent", 0, 3.0, .40, 70001)
    rows, predictions, costs = [], [], []
    started = time.perf_counter()
    for seed in SEEDS:
        train, select, calibration, class_weights = q2.train_views(split, seed)
        model, cost = major.fit(
            SPEC, train, select, class_weights, seed,
            prior=(.45, .25, .18, .12), epochs=60)
        temperature = major.fit_temperature(model, calibration)
        p, _, _, _, inference_seconds = major.predict(model, xx, mm, qq, temperature)
        costs.append({
            "method": "RO-PDRF-EMA", "seed": seed, "temperature": temperature,
            "inference_ms_per_observation": 1000 * inference_seconds / len(testy),
            **cost})
        for subset, take in (("all", np.ones(len(testy), bool)),
                             ("affected", affected), ("unaffected", ~affected)):
            for metric, value in risk.all_metrics(testy[take], p[take]).items():
                rows.append({"method": "RO-PDRF-EMA", "seed": seed,
                             "subset": subset, "metric": metric,
                             "value": value, "n": int(take.sum())})
        for i in range(len(testy)):
            record = {"method": "RO-PDRF-EMA", "seed": seed, "sample": i,
                      "batch": int(batches[i]), "y": int(testy[i]),
                      "affected": bool(affected[i])}
            record.update({f"p{c + 1}": float(p[i, c]) for c in range(p.shape[1])})
            predictions.append(record)
    pd.DataFrame(rows).to_csv(OUT / "major6_ema_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "major6_ema_predictions.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "major6_ema_costs.csv", index=False)
    (OUT / "major6_ema_design.json").write_text(json.dumps({
        "seeds": list(SEEDS), "teacher": "EMA of model parameters on clean view",
        "ema_decay": .99, "test_used_for_selection": False,
        "elapsed_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
