"""Export explicit split indices and the principal controlled-fault realization.

This script performs no fitting.  It reconstructs the deterministic index and
intervention objects used by the manuscript so an archive contains more than
split counts and prose descriptions.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

import run_uci_validation as uci
import run_q2_revision as q2
import run_major_revision_experiments as major
import run_hydraulic_validation as hydraulic
import run_major5_hydraulic_blocked as blocked


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"


def export_chemical() -> dict:
    path = uci.ensure_zip(None)
    _, labels, batches = uci.read_dataset(path)
    groups = [list(range(i, i + 4)) for i in range(0, 16, 4)]
    split = q2.prepare_grouped(path, groups)

    batch7 = np.flatnonzero(batches == 7)
    cut = int(round(0.60 * len(batch7)))
    assignments = {
        "train": np.flatnonzero(batches <= 6),
        "selection": batch7[:cut],
        "calibration": batch7[cut:],
        "test": np.flatnonzero(batches >= 8),
    }
    rows = []
    for name, indices in assignments.items():
        for within, index in enumerate(indices):
            rows.append({
                "dataset": "chemical",
                "task": "gas_class",
                "split": name,
                "within_split_index": within,
                "source_index": int(index),
                "class_value": int(labels[index]),
                "batch_or_cycle": int(batches[index]),
                "guard": False,
            })
    pd.DataFrame(rows).to_csv(
        OUT / "chemical_split_indices.csv", index=False)

    testx, testy, testb = split["test"]
    faulted, masks, _, affected = major.fixed_fault(
        testx, "silent", modality=0, scale=3.0, prevalence=0.40,
        seed=70001)
    delta = faulted[:, 0] - testx[:, 0] * masks[:, 0, None]
    realized = pd.DataFrame({
        "test_index": np.arange(len(testx), dtype=int),
        "source_index": assignments["test"],
        "batch": testb.astype(int),
        "class_value": testy.astype(int),
        "affected": affected.astype(bool),
        "mask_group_1": masks[:, 0].astype(int),
        "mask_group_2": masks[:, 1].astype(int),
        "mask_group_3": masks[:, 2].astype(int),
        "mask_group_4": masks[:, 3].astype(int),
        "group_1_delta_mean": delta.mean(axis=1),
        "group_1_delta_l2": np.linalg.norm(delta, axis=1),
    })
    realized.to_csv(OUT / "chemical_principal_fault_realization.csv",
                    index=False)
    return {name: int(len(indices)) for name, indices in assignments.items()}


def export_hydraulic() -> dict:
    x, profile = hydraulic.load_features()
    stable_ids = np.flatnonzero(profile[:, 4] == 0)
    rows = []
    counts = {}
    for task, column in hydraulic.TASKS.items():
        (_, _, _, _, values, pieces, guards, _, y) = blocked.blocked_split(
            x, profile, column)
        counts[task] = {}
        for split_name, local_indices in pieces.items():
            counts[task][split_name] = int(len(local_indices))
            for within, local_index in enumerate(local_indices):
                rows.append({
                    "dataset": "hydraulic",
                    "task": task,
                    "split": split_name,
                    "within_split_index": within,
                    "source_index": int(stable_ids[local_index]),
                    "stable_cycle_index": int(local_index),
                    "class_value": int(values[y[local_index]]),
                    "batch_or_cycle": int(stable_ids[local_index]),
                    "guard": False,
                })
        counts[task]["guard"] = int(len(guards))
        for local_index in guards:
            rows.append({
                "dataset": "hydraulic",
                "task": task,
                "split": "guard",
                "within_split_index": -1,
                "source_index": int(stable_ids[local_index]),
                "stable_cycle_index": int(local_index),
                "class_value": int(values[y[local_index]]),
                "batch_or_cycle": int(stable_ids[local_index]),
                "guard": True,
            })
    pd.DataFrame(rows).to_csv(
        OUT / "hydraulic_guarded_split_indices.csv", index=False)
    return counts


def main() -> None:
    chemical_counts = export_chemical()
    hydraulic_counts = export_hydraulic()
    protocol = {
        "chemical": {
            "groups_zero_based_sensor_indices": [
                [0, 1, 2, 3], [4, 5, 6, 7],
                [8, 9, 10, 11], [12, 13, 14, 15]],
            "split": "batches 1-6 train; first 60% of batch 7 selection; remainder calibration; batches 8-10 test",
            "split_counts": chemical_counts,
            "principal_fault": {
                "fault_seed": 70001,
                "missingness_and_fault_share_one_generator": True,
                "missing_group_probability": 0.20,
                "affected_probability_sampled_independently_of_missingness": 0.40,
                "affected_group_zero_based": 0,
                "standardized_gaussian_scale": 3.0,
                "quality_values": 1.0,
            },
        },
        "hydraulic": {
            "stable_cycle_filter": "profile column 4 equals zero",
            "split": "within-native-condition acquisition-order 60/15/10/15 blocks",
            "guard_cycles_per_boundary": 5,
            "counts": hydraulic_counts,
            "controlled_faults": {
                "affected_fraction": 0.40,
                "pressure_group_zero_based": 0,
                "vibration_group_zero_based": 4,
                "fault_seeds": {"pressure": 89000, "vibration": 89004},
            },
        },
        "selection_policy": "minimum unweighted selection cross-entropy; patience 9; threshold 1e-4",
        "test_used_for_model_or_checkpoint_selection": False,
    }
    (OUT / "reproducibility_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8")
    print("Explicit chemical and hydraulic split/intervention manifests written.")


if __name__ == "__main__":
    main()
