"""Generate selector-calibration and frozen-test predictions for shift audits.

The endpoint training protocol, split, temperatures, optimization seeds and
test realization match CEE-CF10-R2-LITE.  Only controlled-fault severity and
affected sensor group are varied.  Test labels are written for evaluation but
never used to fit endpoints, temperatures, selectors or thresholds.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
UPGRADE = ROOT.parent / "paper_package_q1_upgrade"
sys.path.insert(0, str(UPGRADE))

import run_major_revision_experiments as major
import run_major7_fault_types_ema as typed
import run_q2_revision as q2
import run_uci_validation as base

from run_cee_lite_routing_validation import features6, predict


OUT = ROOT / "source_data"
SEEDS = tuple(range(101, 111))
FAULTS = ("gaussian", "offset", "drift", "stuck_at")
SEVERITIES = (1.0, 2.0, 3.0)
TEST_GROUPS = (0, 1, 2, 3)
TEST_REALIZATION = 70001
TEST_PREVALENCE = 0.40
EPOCHS = 60
LITE_SPEC = replace(
    major.SPECS["PDRF"],
    recovery_distillation=True,
    w_recovery=0.30,
    recovery_teacher="clean",
)


def add_probability_columns(row: dict, pb: np.ndarray, pr: np.ndarray, index: int) -> None:
    for class_index in range(pb.shape[1]):
        row[f"base_p{class_index}"] = float(pb[index, class_index])
        row[f"lite_p{class_index}"] = float(pr[index, class_index])


def main() -> None:
    started = time.perf_counter()
    split = q2.prepare_grouped(
        base.ensure_zip(None),
        [list(range(index, index + 4)) for index in range(0, 16, 4)],
    )
    test_x, test_y, test_batch = split["test"]
    calibration_x, calibration_y, _ = split["calibration"]
    cut = len(calibration_x) // 2
    temperature_x, temperature_y = calibration_x[:cut], calibration_y[:cut]
    selector_x, selector_y = calibration_x[cut:], calibration_y[cut:]

    calibration_rows: list[dict] = []
    test_rows: list[dict] = []
    temperature_rows: list[dict] = []

    for seed in SEEDS:
        print(f"fitting endpoint pair {seed}", flush=True)
        train, select, _, class_weights = q2.train_views(split, seed)
        temperature_view = (
            *base.make_view(
                temperature_x, 25000 + seed, "missing_drift", train=True
            ),
            temperature_y,
        )
        base_model, _ = major.fit(
            major.SPECS["PDRF"],
            train,
            select,
            class_weights,
            seed,
            prior=(0.45, 0.25, 0.18, 0.12),
            epochs=EPOCHS,
        )
        lite_model, _ = major.fit(
            LITE_SPEC,
            train,
            select,
            class_weights,
            seed,
            prior=(0.45, 0.25, 0.18, 0.12),
            epochs=EPOCHS,
        )
        base_temperature = major.fit_temperature(base_model, temperature_view)
        lite_temperature = major.fit_temperature(lite_model, temperature_view)
        temperature_rows.append(
            {
                "seed": seed,
                "base_temperature": base_temperature,
                "lite_temperature": lite_temperature,
            }
        )

        # Selector-calibration predictions: affected group 1, three severities.
        for severity in SEVERITIES:
            for fault_index, fault in enumerate(FAULTS):
                for realization in range(2):
                    realization_id = 96000 + 100 * fault_index + 10 * realization + seed
                    fault_x, mask, quality, assigned = typed.fixed_typed_fault(
                        selector_x,
                        fault,
                        modality=0,
                        scale=severity,
                        prevalence=TEST_PREVALENCE,
                        seed=realization_id,
                    )
                    strict = assigned & (mask[:, 0] > 0)
                    pb = predict(
                        base_model,
                        fault_x,
                        mask,
                        quality,
                        base_temperature,
                    )
                    pr = predict(
                        lite_model,
                        fault_x,
                        mask,
                        quality,
                        lite_temperature,
                    )
                    feature = features6(pb, pr)
                    for sample in np.flatnonzero(strict):
                        row = {
                            "seed": seed,
                            "fault_type": fault,
                            "severity": severity,
                            "affected_group": 1,
                            "realization": realization,
                            "realization_id": realization_id,
                            "sample": int(sample),
                            "y": int(selector_y[sample]),
                            "base_correct": bool(pb[sample].argmax() == selector_y[sample]),
                            "recovery_correct": bool(pr[sample].argmax() == selector_y[sample]),
                        }
                        row.update(dict(zip(
                            (
                                "base_confidence",
                                "recovery_confidence",
                                "confidence_delta",
                                "base_entropy",
                                "recovery_entropy",
                                "entropy_delta",
                            ),
                            feature[sample],
                        )))
                        add_probability_columns(row, pb, pr, sample)
                        calibration_rows.append(row)

        # Test predictions: all severities for group 1, and all groups at the
        # formal scale 3.  This avoids duplicating scale-3/group-1 rows.
        conditions = [(severity, 0) for severity in SEVERITIES]
        conditions.extend((3.0, group) for group in TEST_GROUPS[1:])
        for severity, group in conditions:
            for fault in FAULTS:
                fault_x, mask, quality, assigned = typed.fixed_typed_fault(
                    test_x,
                    fault,
                    modality=group,
                    scale=severity,
                    prevalence=TEST_PREVALENCE,
                    seed=TEST_REALIZATION,
                )
                strict = assigned & (mask[:, group] > 0)
                pb = predict(
                    base_model,
                    fault_x[strict],
                    mask[strict],
                    quality[strict],
                    base_temperature,
                )
                pr = predict(
                    lite_model,
                    fault_x[strict],
                    mask[strict],
                    quality[strict],
                    lite_temperature,
                )
                samples = np.flatnonzero(strict)
                for local, sample in enumerate(samples):
                    row = {
                        "seed": seed,
                        "fault_type": fault,
                        "severity": severity,
                        "affected_group": group + 1,
                        "sample": int(sample),
                        "batch": int(test_batch[sample]),
                        "y": int(test_y[sample]),
                    }
                    add_probability_columns(row, pb, pr, local)
                    test_rows.append(row)

    calibration = pd.DataFrame(calibration_rows)
    test = pd.DataFrame(test_rows)
    temperatures = pd.DataFrame(temperature_rows)
    calibration.to_csv(
        OUT / "major_shift_selector_calibration.csv.gz",
        index=False,
        compression="gzip",
    )
    test.to_csv(
        OUT / "major_shift_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    temperatures.to_csv(OUT / "major_shift_temperatures.csv", index=False)
    design = {
        "run": "CEE-CF10-R2-LITE-SHIFT",
        "source_protocol": "CEE-CF10-R2-LITE",
        "optimization_seeds": list(SEEDS),
        "fault_mechanisms": list(FAULTS),
        "selector_calibration_severities": list(SEVERITIES),
        "test_severities": list(SEVERITIES),
        "formal_severity": 3.0,
        "selector_calibration_group": 1,
        "test_fault_groups_at_formal_severity": [group + 1 for group in TEST_GROUPS],
        "fault_prevalence": TEST_PREVALENCE,
        "test_realization": TEST_REALIZATION,
        "strict_subset": "fault assigned and affected group available",
        "test_labels_used_for_fitting": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "major_shift_design.json").write_text(
        json.dumps(design, indent=2), encoding="utf-8"
    )
    print(calibration.groupby("severity").size())
    print(test.groupby(["severity", "affected_group"]).size())


if __name__ == "__main__":
    main()
