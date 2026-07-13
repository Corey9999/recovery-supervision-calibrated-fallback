"""Architecture-specific regularizer sensitivity under the unique headline protocol."""
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
QUICK = os.getenv("M4_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 111))
EPOCHS = 3 if QUICK else 60

RO_PDRF = replace(
    major.SPECS["PDRF"], recovery_distillation=True, recovery_teacher="clean",
    w_recovery=.30, brier_regularization=True, degraded_auc=True,
)
RO_CAGF = replace(
    major.SPECS["CAGF"], recovery_distillation=True, recovery_teacher="clean",
    w_recovery=.30, brier_regularization=True, degraded_auc=True,
)

SPECS = {
    "RO-CAGF KL=0": replace(RO_CAGF, gate_kl_regularization=False, w_gate_kl=0.0),
    "RO-CAGF KL=.001": replace(RO_CAGF, gate_kl_regularization=True, w_gate_kl=.001),
    "RO-CAGF KL=.020": replace(RO_CAGF, gate_kl_regularization=True, w_gate_kl=.020),
    "RO-PDRF interior=0": replace(RO_PDRF, interior_barrier=False, w_interior=0.0),
    "RO-PDRF interior=.001": replace(RO_PDRF, interior_barrier=True, w_interior=.001),
    "RO-PDRF interior=.020": replace(RO_PDRF, interior_barrier=True, w_interior=.020),
}


def main():
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i + 4)) for i in range(0, 16, 4)]
    )
    testx, testy, batches = split["test"]
    xx, mm, qq, affected = q2.make_fault(testx, 70001, "silent_gaussian", modality=0)
    metric_rows, cost_rows, prediction_rows = [], [], []
    t0 = time.perf_counter()
    for seed in SEEDS:
        train, select, calibration, class_weights = q2.train_views(split, seed)
        for method, spec in SPECS.items():
            model, cost = major.fit(
                spec, train, select, class_weights, seed,
                prior=(.45, .25, .18, .12), epochs=EPOCHS,
            )
            temperature = major.fit_temperature(model, calibration)
            probability, _, _, _, inference_seconds = major.predict(
                model, xx, mm, qq, temperature
            )
            cost_rows.append({
                "seed": seed, "method": method,
                "training_forward_paths": 3, "inference_forward_paths": 1,
                "inference_ms_per_observation": 1000 * inference_seconds / len(testy),
                **cost,
            })
            for subset, take in (
                ("all", np.ones(len(testy), bool)),
                ("affected", affected),
                ("unaffected", ~affected),
            ):
                for metric, value in risk.all_metrics(testy[take], probability[take]).items():
                    metric_rows.append({
                        "seed": seed, "method": method, "subset": subset,
                        "metric": metric, "value": value,
                    })
            for i in range(len(testy)):
                prediction_rows.append({
                    "seed": seed, "method": method, "sample": i,
                    "batch": int(batches[i]), "y": int(testy[i]),
                    "affected": bool(affected[i]),
                    **{f"p{j}": float(probability[i, j])
                       for j in range(probability.shape[1])},
                })

    pd.DataFrame(metric_rows).to_csv(OUT / "major4_regularizer_metrics.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / "major4_regularizer_costs.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        OUT / "major4_regularizer_predictions.csv", index=False
    )
    (OUT / "major4_regularizer_design.json").write_text(json.dumps({
        "purpose": "architecture-specific regularizer sensitivity",
        "formal_teacher": "clean-only",
        "principal_test_fault_seed": 70001,
        "settings": list(SPECS),
        "seeds": list(SEEDS),
        "checkpoint_criterion": "unweighted natural selection cross-entropy; patience 9",
        "elapsed_seconds": time.perf_counter() - t0,
        "test_data_used_for_selection": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
