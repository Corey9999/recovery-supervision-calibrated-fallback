"""Modern multi-expert routing baseline under the strict applied-fault estimand."""

from dataclasses import replace
from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd

import run_major_revision_experiments as major
import run_major3_objective_matched as m3
import run_major7_fault_types_ema as typed
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT/"source_data"
FAST = os.getenv("M9_FAST") == "1"
SEEDS = (101, 102, 103) if FAST else tuple(range(101, 111))
EPOCHS = 30 if FAST else 60
FAULT_TYPES = ("gaussian", "offset", "drift", "stuck_at")

METHODS = {
    "RO-CAGF": m3.RO_CAGF,
    "RO-PDRF-Full": m3.RO_PDRF,
    "RO-MER": replace(m3.RO_CAGF, architecture="expert_router"),
}


def main():
    OUT.mkdir(exist_ok=True)
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i+4)) for i in range(0, 16, 4)])
    testx, testy, batches = split["test"]
    rows, predictions, costs = [], [], []
    started = time.perf_counter()
    for seed in SEEDS:
        train, select, calibration, class_weights = q2.train_views(split, seed)
        for method, spec in METHODS.items():
            model, cost = major.fit(
                spec, train, select, class_weights, seed,
                prior=(.45, .25, .18, .12), epochs=EPOCHS)
            temperature = major.fit_temperature(model, calibration)
            costs.append({"seed": seed, "method": method,
                          "temperature": temperature, **cost})
            for fault_type in FAULT_TYPES:
                faultx, mask, quality, assigned = typed.fixed_typed_fault(
                    testx, fault_type, seed=70001)
                applied = assigned & (mask[:, 0] > 0)
                p, _, weights, _, inference_seconds = major.predict(
                    model, faultx, mask, quality, temperature)
                for subset, take in (
                    ("fault_applied_available", applied),
                    ("assigned", assigned),
                    ("unaffected", ~assigned),
                    ("all", np.ones(len(testy), bool)),
                ):
                    for metric, value in risk.all_metrics(testy[take], p[take]).items():
                        rows.append({
                            "seed": seed, "method": method,
                            "fault_type": fault_type, "subset": subset,
                            "metric": metric, "value": value,
                            "n": int(take.sum()),
                            "inference_ms_per_observation":
                                1000*inference_seconds/len(testy),
                        })
                for i in np.flatnonzero(applied):
                    predictions.append({
                        "seed": seed, "method": method,
                        "fault_type": fault_type, "sample": int(i),
                        "batch": int(batches[i]), "y": int(testy[i]),
                        "prediction": int(p[i].argmax()),
                        "correct": bool(p[i].argmax() == testy[i]),
                        "affected_group_weight": float(weights[i, 0]),
                        **{f"p{c}": float(p[i, c]) for c in range(p.shape[1])},
                    })
    pd.DataFrame(rows).to_csv(OUT/"major9_modern_baseline_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        OUT/"major9_modern_baseline_predictions.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT/"major9_modern_baseline_costs.csv", index=False)
    (OUT/"major9_modern_baseline_design.json").write_text(json.dumps({
        "mode": "fast development" if FAST else "formal frozen rerun",
        "seeds": list(SEEDS), "epochs": EPOCHS,
        "baseline": "RO-MER: one expert per sensor group plus a joint expert and a sample-dependent router",
        "objective": "same paired views, recovery distillation, Brier, degraded AUC, monotonicity and gate KL as RO-CAGF",
        "test_faults": list(FAULT_TYPES),
        "primary_subset": "fault actually applied while affected group is available",
        "test_used_for_selection": False,
        "elapsed_seconds": time.perf_counter()-started,
    }, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    show = frame[(frame.subset == "fault_applied_available") &
                 (frame.metric == "macro_auroc")]
    print(show.groupby(["fault_type", "method"]).value.mean().unstack().round(4))
    print("\nParameters and time")
    print(pd.DataFrame(costs).groupby("method")[["parameters", "train_seconds"]].mean().round(2))


if __name__ == "__main__":
    main()
