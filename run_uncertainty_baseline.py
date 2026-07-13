"""Matched entropy-weighted uncertainty fusion on the chemical-array protocol."""
from pathlib import Path
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
SEEDS = tuple(range(101, 104)) if os.getenv("UNCERTAINTY_QUICK") == "1" else tuple(range(101, 111))
EPOCHS = 8 if os.getenv("UNCERTAINTY_QUICK") == "1" else 60


def main():
    zip_path = base.ensure_zip(None)
    groups = [list(range(i, i+4)) for i in range(0, 16, 4)]
    split = q2.prepare_grouped(zip_path, groups)
    tx, ty, batch = split["test"]
    _, testx, mask, quality, affected = risk.paired_fault(tx, 70001)
    rows, predictions, costs = [], [], []
    members = []
    for seed in SEEDS:
        train, select, calibration, class_weights = q2.train_views(split, seed)
        t0 = time.perf_counter()
        model, cost = major.fit(major.SPECS["ENTROPY_PD"], train, select, class_weights,
                                seed, prior=(.45, .25, .18, .12), epochs=EPOCHS)
        temperature = major.fit_temperature(model, calibration, neutral_q=True)
        p, scores, weights, _, infer_seconds = major.predict(model, testx, mask, quality, temperature)
        members.append(p)
        costs.append({"method": "ENT-PD", "seed": seed, "parameters": cost["parameters"],
                      "train_seconds": cost["train_seconds"], "wall_seconds": time.perf_counter()-t0,
                      "inference_seconds": infer_seconds, "epochs": cost["epochs"]})
        for subset, take in (("all", np.ones(len(ty), bool)), ("affected", affected),
                             ("unaffected", ~affected)):
            for metric, value in risk.all_metrics(ty[take], p[take]).items():
                rows.append({"method": "ENT-PD", "seed": seed, "subset": subset,
                             "metric": metric, "value": value})
        for i in range(len(ty)):
            row = {"method": "ENT-PD", "seed": seed, "sample": i, "batch": int(batch[i]),
                   "y": int(ty[i]), "affected": bool(affected[i]),
                   "affected_group_entropy": float(scores[i, 0]),
                   "affected_group_weight": float(weights[i, 0])}
            row.update({f"p{j+1}": float(p[i, j]) for j in range(p.shape[1])})
            predictions.append(row)
    ensemble = np.mean(members, axis=0)
    ensemble_rows = []
    for subset, take in (("all", np.ones(len(ty), bool)), ("affected", affected),
                         ("unaffected", ~affected)):
        for metric, value in risk.all_metrics(ty[take], ensemble[take]).items():
            ensemble_rows.append({"method": "ENT-PD", "pool": "probability_mean",
                                  "subset": subset, "metric": metric, "value": value})
    pd.DataFrame(rows).to_csv(OUT / "uncertainty_baseline_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "uncertainty_baseline_predictions.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT / "uncertainty_baseline_costs.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(OUT / "uncertainty_baseline_ensemble.csv", index=False)


if __name__ == "__main__":
    main()
