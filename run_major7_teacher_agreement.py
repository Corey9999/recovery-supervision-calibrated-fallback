"""Multi-view agreement teachers for the manuscript-15 revision.

Two remedies replace a single confidence threshold with agreement between
independently perturbed views: clean/removal agreement (CRA) and current/EMA
clean agreement (ECA). Both use the complete RO-PDRF objective and the fixed
principal chemical fault. The held-out test set is used only once after model
selection and temperature calibration.
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
SEEDS = (101,) if QUICK else tuple(range(101, 111))
EPOCHS = 3 if QUICK else 60

METHODS = {
    "RO-PDRF-CRA": replace(
        m3.RO_PDRF, recovery_teacher="clean_removal_agreement"),
    "RO-PDRF-ECA": replace(
        m3.RO_PDRF, recovery_teacher="ema_clean_agreement",
        teacher_ema_decay=.99),
}


def main():
    OUT.mkdir(exist_ok=True)
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i + 4)) for i in range(0, 16, 4)])
    testx, testy, batches = split["test"]
    xx, mm, qq, affected = major.fixed_fault(
        testx, "silent", 0, 3.0, .40, 70001)
    cleanx = (testx*mm[:, :, None]).astype(np.float32)
    removal_mask = mm.copy()
    valid_removal = affected & (mm[:, 0] > 0) & (mm.sum(1) > 1)
    removal_mask[valid_removal, 0] = 0
    removal_x = (testx*removal_mask[:, :, None]).astype(np.float32)
    rows, predictions, costs = [], [], []
    started = time.perf_counter()

    for seed in SEEDS:
        train, select, calibration, class_weights = q2.train_views(split, seed)
        for method, spec in METHODS.items():
            model, cost = major.fit(
                spec, train, select, class_weights, seed,
                prior=(.45, .25, .18, .12), epochs=EPOCHS)
            temperature = major.fit_temperature(model, calibration)
            p_fault, _, _, _, inference_seconds = major.predict(
                model, xx, mm, qq, temperature)
            p_clean, _, _, _, _ = major.predict(
                model, cleanx, mm, qq, temperature)
            if method == "RO-PDRF-CRA":
                p_second, _, _, _, _ = major.predict(
                    model, removal_x, removal_mask, qq*removal_mask, temperature)
                agreement_view = "affected-group removal"
            else:
                p_second, _, _, _, _ = major.predict(
                    model.ema_teacher, cleanx, mm, qq, temperature)
                agreement_view = "EMA clean"
            teacher = .5*(p_clean+p_second)
            agreement = p_clean.argmax(1) == p_second.argmax(1)
            costs.append({
                "method": method, "seed": seed, "temperature": temperature,
                "inference_ms_per_observation":
                    1000*inference_seconds/len(testy), **cost})
            for subset, take in (("all", np.ones(len(testy), bool)),
                                 ("affected", affected),
                                 ("unaffected", ~affected)):
                for metric, value in risk.all_metrics(testy[take], p_fault[take]).items():
                    rows.append({"method": method, "seed": seed,
                                 "subset": subset, "metric": metric,
                                 "value": value, "n": int(take.sum())})
            for i in np.flatnonzero(affected):
                predictions.append({
                    "method": method, "seed": seed, "sample": int(i),
                    "batch": int(batches[i]), "y": int(testy[i]),
                    "valid_removal": bool(valid_removal[i]),
                    "clean_prediction": int(p_clean[i].argmax()),
                    "second_view": agreement_view,
                    "second_prediction": int(p_second[i].argmax()),
                    "teacher_prediction": int(teacher[i].argmax()),
                    "fault_prediction": int(p_fault[i].argmax()),
                    "clean_correct": bool(p_clean[i].argmax() == testy[i]),
                    "teacher_correct": bool(teacher[i].argmax() == testy[i]),
                    "fault_correct": bool(p_fault[i].argmax() == testy[i]),
                    "teacher_agreement": bool(agreement[i]),
                    "clean_confidence": float(p_clean[i].max()),
                    "teacher_confidence": float(teacher[i].max()),
                    "fault_true_probability": float(p_fault[i, testy[i]]),
                })

    pd.DataFrame(rows).to_csv(
        OUT/"major7_teacher_agreement_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        OUT/"major7_teacher_agreement_predictions.csv", index=False)
    pd.DataFrame(costs).to_csv(
        OUT/"major7_teacher_agreement_costs.csv", index=False)
    (OUT/"major7_teacher_agreement_design.json").write_text(json.dumps({
        "seeds": list(SEEDS), "epochs": EPOCHS,
        "CRA": "distil only when clean and affected-group-removal argmax agree; target is their mean probability",
        "ECA": "distil only when student-clean and EMA-clean argmax agree; target is their mean probability",
        "ema_decay": .99,
        "principal_fault": {"type": "Gaussian", "group": 1,
                            "scale": 3.0, "prevalence": .40,
                            "realization": 70001},
        "test_used_for_selection": False,
        "elapsed_seconds": time.perf_counter()-started,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
