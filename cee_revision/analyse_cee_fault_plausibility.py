"""Audit standardized-space fault magnitude in original units and percentiles."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
UPGRADE = ROOT.parent / "paper_package_q1_upgrade"
sys.path.insert(0, str(UPGRADE))

import run_major7_fault_types_ema as typed
import run_q2_revision as q2
import run_uci_validation as base


OUT = ROOT / "source_data"
FAULTS = ("gaussian", "offset", "drift", "stuck_at")
SCALES = (1.0, 3.0, 5.0)
TEST_REALIZATION = 70001


def percentile(sorted_training: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        result[:, column] = np.searchsorted(
            sorted_training[:, column], values[:, column], side="right"
        ) / len(sorted_training)
    return result


def main():
    raw, y, batches = base.read_dataset(base.ensure_zip(None))
    train = batches <= 6
    test = batches >= 8
    scaler = StandardScaler().fit(raw[train])
    standardized = np.clip(scaler.transform(raw), -8.0, 8.0).astype(np.float32)
    grouped = standardized.reshape(-1, 16, 8)
    group0_columns = np.arange(32)
    test_grouped = np.stack(
        [grouped[:, list(range(index, index + 4))].reshape(len(grouped), -1)
         for index in range(0, 16, 4)],
        axis=1,
    )[test]
    test_y = y[test]
    test_batches = batches[test]
    training_group0 = raw[train][:, group0_columns]
    sorted_training = np.sort(training_group0, axis=0)
    center = scaler.mean_[group0_columns]
    spread = scaler.scale_[group0_columns]

    audit_rows = []
    example_rows = []
    count_rows = []
    for batch in sorted(np.unique(test_batches)):
        take = test_batches == batch
        for class_index, count in pd.Series(test_y[take]).value_counts().sort_index().items():
            count_rows.append(
                {
                    "subset": "complete_test",
                    "fault_type": "none",
                    "batch": int(batch),
                    "class": int(class_index + 1),
                    "n": int(count),
                }
            )

    for scale in SCALES:
        for fault in FAULTS:
            fault_x, mask, _, assigned = typed.fixed_typed_fault(
                test_grouped,
                fault,
                modality=0,
                scale=scale,
                prevalence=0.40,
                seed=TEST_REALIZATION,
            )
            strict = assigned & (mask[:, 0] > 0)
            before_z = test_grouped[strict, 0]
            after_z = fault_x[strict, 0]
            before_raw = before_z * spread + center
            after_raw = after_z * spread + center
            before_percentile = percentile(sorted_training, before_raw)
            after_percentile = percentile(sorted_training, after_raw)
            standardized_shift = np.abs(after_z - before_z)
            percentile_shift = np.abs(after_percentile - before_percentile)
            lower = np.quantile(training_group0, 0.01, axis=0)
            upper = np.quantile(training_group0, 0.99, axis=0)
            minimum = training_group0.min(axis=0)
            maximum = training_group0.max(axis=0)
            audit_rows.append(
                {
                    "fault_type": fault,
                    "scale": scale,
                    "strict_observations": int(strict.sum()),
                    "values": int(after_raw.size),
                    "median_abs_standardized_shift": float(np.median(standardized_shift)),
                    "p95_abs_standardized_shift": float(np.quantile(standardized_shift, 0.95)),
                    "median_abs_percentile_shift": float(np.median(percentile_shift)),
                    "p95_abs_percentile_shift": float(np.quantile(percentile_shift, 0.95)),
                    "fraction_within_training_1_99_percentiles": float(
                        ((after_raw >= lower) & (after_raw <= upper)).mean()
                    ),
                    "fraction_within_training_min_max": float(
                        ((after_raw >= minimum) & (after_raw <= maximum)).mean()
                    ),
                    "fraction_outside_preprocessed_clip_range": float(
                        (np.abs(after_z) > 8.0).mean()
                    ),
                }
            )
            if scale == 3.0:
                strict_indices = np.flatnonzero(strict)
                for local, sample in enumerate(strict_indices[:5]):
                    row = {
                        "fault_type": fault,
                        "scale": scale,
                        "sample": int(sample),
                        "batch": int(test_batches[sample]),
                        "class": int(test_y[sample] + 1),
                        "median_before_percentile": float(np.median(before_percentile[local])),
                        "median_after_percentile": float(np.median(after_percentile[local])),
                        "maximum_abs_percentile_shift": float(np.max(percentile_shift[local])),
                    }
                    for feature in range(4):
                        row[f"feature_{feature + 1}_before_original"] = float(before_raw[local, feature])
                        row[f"feature_{feature + 1}_after_original"] = float(after_raw[local, feature])
                        row[f"feature_{feature + 1}_before_percentile"] = float(before_percentile[local, feature])
                        row[f"feature_{feature + 1}_after_percentile"] = float(after_percentile[local, feature])
                    example_rows.append(row)
                for batch in sorted(np.unique(test_batches)):
                    take = strict & (test_batches == batch)
                    for class_index, count in pd.Series(test_y[take]).value_counts().sort_index().items():
                        count_rows.append(
                            {
                                "subset": "strict_applied_available",
                                "fault_type": fault,
                                "batch": int(batch),
                                "class": int(class_index + 1),
                                "n": int(count),
                            }
                        )

    pd.DataFrame(audit_rows).to_csv(OUT / "q1_fault_plausibility.csv", index=False)
    pd.DataFrame(example_rows).to_csv(OUT / "q1_fault_examples_original_units.csv", index=False)
    pd.DataFrame(count_rows).to_csv(OUT / "q1_complete_and_strict_class_counts.csv", index=False)
    design = {
        "standardization_fit": "UCI gas-array batches 1-6 only",
        "standardization_transform": "training mean and standard deviation, followed by clipping to [-8,8] before grouping",
        "fault_space": "standardized grouped feature space after preprocessing",
        "group_1_original_columns": group0_columns.tolist(),
        "missingness": "20% independent group missingness generated before strict eligibility is determined",
        "corruption": "40% independent assignment; strict estimand requires assigned group 1 to remain available",
        "quality": "all ones in formal CEE-CF10-R2 tests; independently varied quality during paired training is a stress design",
        "independence_rationale": "factorial stress isolation rather than a claim that every missingness-corruption-quality combination is a natural hardware event",
        "test_realization": TEST_REALIZATION,
    }
    (OUT / "q1_fault_plausibility_design.json").write_text(
        json.dumps(design, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
