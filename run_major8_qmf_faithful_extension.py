"""Extend the closer-to-official QMF-PD adaptation to seeds 106--110."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

import run_major_revision_experiments as major
import run_major8_qmf_sensitivity as qmf
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
SEEDS = tuple(range(106, 111))
EPOCHS = 60
METHOD = "QMF-PD-faithful"
SPEC = qmf.METHODS[METHOD]


def main():
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
        p_clean, _, _, _, clean_seconds = major.predict(
            model, cleanx, mask, quality, temperature)
        p_fault, _, weights, _, fault_seconds = major.predict(
            model, faultx, mask, quality, temperature)
        costs.append({
            "method": METHOD, "seed": seed, "temperature": temperature,
            "clean_inference_ms_per_observation": 1000*clean_seconds/len(testy),
            "fault_inference_ms_per_observation": 1000*fault_seconds/len(testy),
            **cost})
        for subset, take, probabilities in (
                ("clean_matched_mask", np.ones(len(testy), bool), p_clean),
                ("fault_all", np.ones(len(testy), bool), p_fault),
                ("fault_affected", affected, p_fault),
                ("fault_unaffected", ~affected, p_fault)):
            for metric, value in risk.all_metrics(
                    testy[take], probabilities[take]).items():
                rows.append({"method": METHOD, "seed": seed, "subset": subset,
                             "metric": metric, "value": value,
                             "n": int(take.sum())})
        for i in np.flatnonzero(affected):
            row = {"method": METHOD, "seed": seed, "sample": int(i),
                   "batch": int(batches[i]), "y": int(testy[i]),
                   "affected_group_weight": float(weights[i, 0])}
            row.update({f"p{c}": float(p_fault[i, c])
                        for c in range(p_fault.shape[1])})
            predictions.append(row)
    pd.DataFrame(rows).to_csv(
        OUT/"major8_qmf_faithful_extension_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        OUT/"major8_qmf_faithful_extension_predictions.csv", index=False)
    pd.DataFrame(costs).to_csv(
        OUT/"major8_qmf_faithful_extension_costs.csv", index=False)
    pd.DataFrame(traces).to_csv(
        OUT/"major8_qmf_faithful_extension_curves.csv", index=False)
    (OUT/"major8_qmf_faithful_extension_design.json").write_text(json.dumps({
        "method": METHOD, "seeds": list(SEEDS), "epochs": EPOCHS,
        "purpose": "complete 10-seed evaluation when combined with sensitivity seeds 101--105",
        "test_used_for_selection": False,
        "elapsed_seconds": time.perf_counter()-started,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
