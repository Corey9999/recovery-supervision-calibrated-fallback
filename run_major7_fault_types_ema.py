"""Fault-type x realization x seed validation and EMA-decay sensitivity.

The evaluation crosses four controlled sensor-corruption mechanisms with eight
method-shared realizations and five independently fitted models. Gaussian,
offset, acquisition-ordered drift and stuck-at interventions use the same
affected-row and missingness draws within each realization. The experiment
does not turn repeated corruptions of one instrument into device replication.
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
QUICK = os.getenv("M7_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 106))
REALIZATIONS = (70001, 70002) if QUICK else tuple(range(70001, 70009))
FAULT_TYPES = ("gaussian", "offset", "drift", "stuck_at")
EPOCHS = 3 if QUICK else 60

METHODS = {
    "RO-CAGF": m3.RO_CAGF,
    "RO-PDRF": m3.RO_PDRF,
    "RO-PDRF-EMA95": replace(
        m3.RO_PDRF, recovery_teacher="ema", teacher_ema_decay=.95),
    "RO-PDRF-EMA99": replace(
        m3.RO_PDRF, recovery_teacher="ema", teacher_ema_decay=.99),
    "RO-PDRF-EMA995": replace(
        m3.RO_PDRF, recovery_teacher="ema", teacher_ema_decay=.995),
}


def fixed_typed_fault(x, fault_type, modality=0, scale=3.0,
                      prevalence=.40, seed=70001):
    rng = np.random.default_rng(seed)
    xx = x.copy()
    n, m, d = x.shape
    mask = base.random_mask(n, rng, .20)
    affected = rng.random(n) < prevalence
    direction = rng.choice(np.array([-1., 1.], np.float32), size=d)
    if fault_type == "gaussian":
        xx[affected, modality] += rng.normal(
            0, scale, (affected.sum(), d)).astype(np.float32)
    elif fault_type == "offset":
        xx[affected, modality] += scale*direction
    elif fault_type == "drift":
        # The monotone multiplier follows held-out acquisition order. A
        # non-zero starting magnitude avoids redefining early affected rows as
        # nearly clean while retaining a progressive drift trajectory.
        trajectory = np.linspace(.25, 1.0, n, dtype=np.float32)
        xx[affected, modality] += (
            scale*trajectory[affected, None]*direction[None, :])
    elif fault_type == "stuck_at":
        xx[affected, modality] = scale*direction
    else:
        raise ValueError(fault_type)
    q = np.ones((n, m), np.float32)
    xx *= mask[:, :, None]
    return xx.astype(np.float32), mask, q, affected


def main():
    OUT.mkdir(exist_ok=True)
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i + 4)) for i in range(0, 16, 4)])
    testx, testy, batches = split["test"]
    rows, costs = [], []
    started = time.perf_counter()

    for seed in SEEDS:
        train, select, calibration, class_weights = q2.train_views(split, seed)
        for method, spec in METHODS.items():
            model, cost = major.fit(
                spec, train, select, class_weights, seed,
                prior=(.45, .25, .18, .12), epochs=EPOCHS)
            temperature = major.fit_temperature(model, calibration)
            costs.append({"method": method, "seed": seed,
                          "temperature": temperature, **cost})
            for fault_type in FAULT_TYPES:
                for realization in REALIZATIONS:
                    xx, mm, qq, affected = fixed_typed_fault(
                        testx, fault_type, seed=realization)
                    p, _, _, _, inference_seconds = major.predict(
                        model, xx, mm, qq, temperature)
                    for subset, take in (
                            ("all", np.ones(len(testy), bool)),
                            ("affected", affected),
                            ("unaffected", ~affected)):
                        for metric, value in risk.all_metrics(
                                testy[take], p[take]).items():
                            rows.append({
                                "method": method, "seed": seed,
                                "fault_type": fault_type,
                                "fault_realization": realization,
                                "subset": subset, "metric": metric,
                                "value": value, "n": int(take.sum()),
                                "inference_ms_per_observation":
                                    1000*inference_seconds/len(testy)})

    pd.DataFrame(rows).to_csv(
        OUT/"major7_fault_type_ema_metrics.csv", index=False)
    pd.DataFrame(costs).to_csv(
        OUT/"major7_fault_type_ema_costs.csv", index=False)
    (OUT/"major7_fault_type_ema_design.json").write_text(json.dumps({
        "seeds": list(SEEDS), "fault_realizations": list(REALIZATIONS),
        "fault_types": list(FAULT_TYPES), "group": 1, "scale": 3.0,
        "prevalence": .40, "missingness": .20,
        "shared_within_realization":
            "affected rows and group-missingness masks; intervention-specific values are fixed across methods",
        "ema_decays": [.95, .99, .995],
        "test_used_for_selection": False,
        "scope": "repeated controlled faults on one chemical instrument, not independent devices",
        "elapsed_seconds": time.perf_counter()-started,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
