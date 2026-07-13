"""Audit the fault estimand for the principal chemical experiment.

The historical ``affected`` flag records assignment to the corruption arm.
Because group missingness is sampled independently and applied after the
corruption, a subset of assigned observations has the target group masked and
therefore receives no effective corruption.  This script separates assignment
from an actually applied, available-channel fault and recomputes all headline
metrics without refitting any model.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        take = (confidence >= left) & (confidence < right)
        if right == 1.0:
            take |= confidence == 1.0
        if take.any():
            value += take.sum() / total * abs(correct[take].mean() - confidence[take].mean())
    return float(value)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = p / p.sum(axis=1, keepdims=True)
    classes = np.arange(p.shape[1])
    onehot = label_binarize(y, classes=classes)
    return {
        "accuracy": float(accuracy_score(y, p.argmax(axis=1))),
        "macro_f1": float(f1_score(y, p.argmax(axis=1), average="macro")),
        "macro_auroc": float(roc_auc_score(onehot, p, average="macro", multi_class="ovr")),
        "macro_auprc": float(average_precision_score(onehot, p, average="macro")),
        "nll": float(log_loss(y, p, labels=classes)),
        "brier": float(np.square(p - onehot).sum(axis=1).mean()),
        "ece15": ece(y, p, 15),
    }


def exact_paired_p(left: pd.Series, right: pd.Series) -> float:
    common = left.index.intersection(right.index)
    delta = left.loc[common] - right.loc[common]
    if np.allclose(delta, 0):
        return 1.0
    return float(wilcoxon(delta, alternative="two-sided", method="exact").pvalue)


def fixed_ensemble_bootstrap(frame: pd.DataFrame, subset: str, left: str, right: str,
                             reps: int = 5000, seed: int = 94001) -> np.ndarray:
    """Batch-then-observation bootstrap for the fixed ten-model ensembles.

    This interval reflects the held-out observations and the three acquisition
    batches.  It does not represent device-level or model-refitting uncertainty.
    """
    pcols = [f"p{i}" for i in range(6)]
    key = ["sample", "batch", "y", "estimand_subset"]
    ens = (frame[frame.estimand_subset == subset]
           .groupby(key + ["method"], as_index=False)[pcols].mean())
    a = ens[ens.method == left].drop(columns="method")
    b = ens[ens.method == right].drop(columns="method")
    z = a.merge(b, on=key, suffixes=("_left", "_right"), validate="one_to_one")
    batches = sorted(z.batch.unique())
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pieces = []
        for batch in rng.choice(batches, size=len(batches), replace=True):
            block = z[z.batch == batch]
            pieces.append(block.iloc[rng.integers(0, len(block), len(block))])
        sample = pd.concat(pieces, ignore_index=True)
        y = sample.y.to_numpy(int)
        if np.unique(y).size < len(pcols):
            continue
        pl = sample[[f"p{i}_left" for i in range(6)]].to_numpy(float)
        pr = sample[[f"p{i}_right" for i in range(6)]].to_numpy(float)
        pl /= pl.sum(axis=1, keepdims=True)
        pr /= pr.sum(axis=1, keepdims=True)
        onehot = label_binarize(y, classes=np.arange(6))
        out.append(roc_auc_score(onehot, pl, average="macro", multi_class="ovr") -
                   roc_auc_score(onehot, pr, average="macro", multi_class="ovr"))
    return np.asarray(out, dtype=float)


def main() -> None:
    manifest = pd.read_csv(SRC / "chemical_principal_fault_realization.csv")
    pred = pd.read_csv(SRC / "major3_objective_matched_predictions.csv")
    pred = pred[pred.dataset == "chemical"].copy()
    pcols = [f"p{i}" for i in range(6)]

    required = {"test_index", "affected", "mask_group_1", "group_1_delta_l2"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Fault manifest lacks required columns: {sorted(missing)}")
    if not np.allclose(pred[pcols].sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Prediction rows do not sum to one")

    meta = manifest[["test_index", "affected", "mask_group_1", "group_1_delta_l2"]].rename(
        columns={"test_index": "sample"})
    frame = pred.merge(meta, on="sample", how="left", validate="many_to_one")
    if frame[["affected_y", "mask_group_1"]].isna().any().any():
        raise ValueError("Prediction-to-fault-manifest join is incomplete")
    if not (frame.affected_x.astype(bool) == frame.affected_y.astype(bool)).all():
        raise ValueError("Stored prediction and fault-manifest assignment flags disagree")

    frame["assigned"] = frame.affected_y.astype(bool)
    frame["fault_applied_available"] = (
        frame.assigned & (frame.mask_group_1 == 1) & (frame.group_1_delta_l2 > 0))
    frame["assigned_unavailable"] = frame.assigned & (frame.mask_group_1 == 0)
    frame["unaffected"] = ~frame.assigned
    frame["unaffected_available"] = (~frame.assigned) & (frame.mask_group_1 == 1)

    # Long form duplicates each row only across explicitly named estimands.
    subset_masks = {
        "assigned": frame.assigned,
        "fault_applied_available": frame.fault_applied_available,
        "assigned_unavailable": frame.assigned_unavailable,
        "unaffected": frame.unaffected,
        "unaffected_available": frame.unaffected_available,
    }
    expanded = []
    metric_rows = []
    for subset, take in subset_masks.items():
        block = frame[take].copy()
        block["estimand_subset"] = subset
        expanded.append(block)
        for (seed, method), group in block.groupby(["seed", "method"]):
            y = group.y.to_numpy(int)
            p = group[pcols].to_numpy(float)
            if np.unique(y).size < 2:
                continue
            for metric, value in metrics(y, p).items():
                metric_rows.append({
                    "seed": int(seed), "method": method,
                    "subset": subset, "metric": metric,
                    "value": value, "n": int(len(group)),
                })
    expanded = pd.concat(expanded, ignore_index=True)
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(SRC / "major9_estimand_metrics.csv", index=False)

    manifest_masks = {
        "assigned": manifest.affected.astype(bool),
        "fault_applied_available": manifest.affected.astype(bool) & (manifest.mask_group_1 == 1),
        "assigned_unavailable": manifest.affected.astype(bool) & (manifest.mask_group_1 == 0),
        "unaffected": ~manifest.affected.astype(bool),
        "unaffected_available": (~manifest.affected.astype(bool)) & (manifest.mask_group_1 == 1),
    }
    counts = pd.DataFrame([
        {"subset": subset, "n": int(mask.sum()), "fraction_of_test": float(mask.mean())}
        for subset, mask in manifest_masks.items()
    ])
    counts.to_csv(SRC / "major9_estimand_counts.csv", index=False)

    contrasts = []
    for subset in ("assigned", "fault_applied_available", "assigned_unavailable"):
        z = metric_frame[(metric_frame.subset == subset) &
                         (metric_frame.metric == "macro_auroc")]
        pivot = z.pivot(index="seed", columns="method", values="value")
        for label, left, right in (
            ("RO-PDRF - PDRF", "RO-PDRF", "PDRF"),
            ("RO-CAGF - CAGF", "RO-CAGF", "CAGF"),
            ("RO-PDRF - RO-CAGF", "RO-PDRF", "RO-CAGF"),
        ):
            delta = pivot[left] - pivot[right]
            contrasts.append({
                "subset": subset, "contrast": label,
                "n_seeds": int(delta.notna().sum()),
                "mean_difference": float(delta.mean()),
                "sd_difference": float(delta.std(ddof=1)),
                "wins": int((delta > 0).sum()),
                "wilcoxon_p": exact_paired_p(pivot[left], pivot[right]),
            })
    effect_frame = pd.DataFrame(contrasts)

    boot_rows = []
    for left, right in (("RO-PDRF", "PDRF"), ("RO-PDRF", "RO-CAGF")):
        samples = fixed_ensemble_bootstrap(
            expanded, "fault_applied_available", left, right)
        boot_rows.append({
            "subset": "fault_applied_available",
            "contrast": f"{left} - {right}",
            "resampling_unit": "acquisition batch then observation; fixed ten-model ensemble",
            "reps_retained": int(len(samples)),
            "mean_difference": float(samples.mean()),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
        })
    pd.DataFrame(boot_rows).to_csv(SRC / "major9_estimand_ensemble_bootstrap.csv", index=False)
    effect_frame.to_csv(SRC / "major9_estimand_effects.csv", index=False)

    audit = {
        "question": "Does the principal conclusion persist when affected means a fault actually applied to an available group?",
        "test_rows": int(len(manifest)),
        "assigned_rows": int(manifest.affected.sum()),
        "applied_available_rows": int((manifest.affected & (manifest.mask_group_1 == 1)).sum()),
        "assigned_unavailable_rows": int((manifest.affected & (manifest.mask_group_1 == 0)).sum()),
        "assignment_and_applied_partition": bool(
            manifest.affected.sum() ==
            (manifest.affected & (manifest.mask_group_1 == 1)).sum() +
            (manifest.affected & (manifest.mask_group_1 == 0)).sum()),
        "fixed_ensemble_bootstrap_scope": "test-sample and batch uncertainty only; not model refitting or device replication",
    }
    (SRC / "major9_estimand_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")

    print(counts.to_string(index=False))
    print("\nPaired seed contrasts")
    print(effect_frame.to_string(index=False))
    print("\nFixed-ensemble batch/observation bootstrap")
    print(pd.DataFrame(boot_rows).to_string(index=False))


if __name__ == "__main__":
    main()
