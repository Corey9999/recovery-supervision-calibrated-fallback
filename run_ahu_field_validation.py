"""External validation on expert-annotated, real-operational AHU sensor faults.

The held-out unit is an entire building (office, auditorium, or hospital).
Only normal and native return/supply-air temperature sensor-fault labels are
used. No artificial fault is added to any test record. Paired degradation is
confined to model training, as defined in run_major_revision_experiments.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import run_major_revision_experiments as major


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "external_data" / "ahu_field"
OUT = ROOT / "source_data"
QUICK = os.getenv("AHU_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 106))
EPOCHS = 8 if QUICK else 12
BUILDINGS = ("auditorium", "hospital", "office")

RC_NOQ = replace(
    major.SPECS["PDRF"],
    use_quality=False,
    train_quality=False,
    recovery_distillation=True,
    w_recovery=0.30,
    brier_regularization=True,
    interior_barrier=True,
    w_interior=0.005,
    degraded_auc=True,
)
METHODS = {
    "EF-PD": major.SPECS["EF_PD"],
    "UF-PD": major.SPECS["UF_PD"],
    "CAGF": major.SPECS["CAGF"],
    "ENT-PD": major.SPECS["ENTROPY_PD"],
    "PDRF-NOQ": major.SPECS["PDRF_NOQ"],
    "RC-PDRF-NOQ": RC_NOQ,
}


def _numeric(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    cols = [pd.to_numeric(frame[n], errors="coerce").to_numpy(float) for n in names if n in frame]
    if not cols:
        return np.full((len(frame), 1), np.nan)
    return np.column_stack(cols)


def _row_mean(frame: pd.DataFrame, contains: tuple[str, ...]) -> np.ndarray:
    names = [c for c in frame.columns if any(s.lower() in c.lower() for s in contains)]
    values = _numeric(frame, names)
    count = np.sum(np.isfinite(values), axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(total, count, out=np.full(len(frame), np.nan), where=count > 0)


def load_building(name: str) -> dict[str, np.ndarray]:
    path = DATA / f"{name}_scientific_data.csv"
    frame = pd.read_csv(path, low_memory=False)
    labels = frame["labeling"].astype(str).str.strip()
    keep = labels.isin(("Normal condition", "Return air temperature fault", "Supply air temperature fault"))
    frame = frame.loc[keep].reset_index(drop=True)
    labels = labels.loc[keep].reset_index(drop=True)
    if "DATE" in frame:
        record_key = (frame["AHU name"].astype(str) + "|" + frame["DATE"].astype(str) + "|" +
                      frame["Time"].astype(str))
    else:
        record_key = frame["AHU name"].astype(str) + "|" + frame["Time"].astype(str)
    # The hospital file contains a small number of repeated AHU-time keys,
    # including keys with incompatible labels. Exclude conflicting keys and
    # retain one copy of same-label repetitions before any split.
    label_counts = labels.groupby(record_key).nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    valid = ~record_key.isin(conflicting)
    frame, labels, record_key = frame.loc[valid].copy(), labels.loc[valid].copy(), record_key.loc[valid].copy()
    unique = ~record_key.duplicated(keep="first")
    frame, labels = frame.loc[unique].reset_index(drop=True), labels.loc[unique].reset_index(drop=True)

    # Five physical groups, each represented in a common four-position vector.
    # Padding is structural and never estimated from test labels.
    groups = [
        _numeric(frame, ["Set point temperature"]),
        _numeric(frame, ["Return temperature"]),
        _numeric(frame, ["Supply air temperature"]),
        _numeric(frame, ["Supply fan", "Valve position"]),
        np.column_stack([
            _row_mean(frame, ("heating supply temperature",)),
            _row_mean(frame, ("cooling supply temperature",)),
            _row_mean(frame, ("heating pump",)),
            _row_mean(frame, ("cooling pump",)),
        ]),
    ]
    x = np.full((len(frame), 5, 4), np.nan, np.float32)
    for j, values in enumerate(groups):
        width = min(values.shape[1], 4)
        x[:, j, :width] = values[:, :width]
    mask = np.isfinite(x).any(axis=2).astype(np.float32)
    y = (labels != "Normal condition").astype(np.int64).to_numpy()
    subtype = labels.to_numpy()
    ahu = frame["AHU name"].astype(str).to_numpy()
    if "DATE" in frame:
        timestamp = pd.to_datetime(frame["DATE"].astype(str) + " " + frame["Time"].astype(str), dayfirst=True,
                                   errors="coerce").astype(str).to_numpy()
    else:
        timestamp = pd.to_datetime(frame["Time"], errors="coerce").astype(str).to_numpy()
    return {"x": x, "mask": mask, "y": y, "subtype": subtype, "ahu": ahu, "time": timestamp}


def source_duplicate_profile(name: str) -> dict[str, int]:
    frame = pd.read_csv(DATA / f"{name}_scientific_data.csv", low_memory=False)
    labels = frame["labeling"].astype(str).str.strip()
    keep = labels.isin(("Normal condition", "Return air temperature fault", "Supply air temperature fault"))
    frame, labels = frame.loc[keep], labels.loc[keep]
    if "DATE" in frame:
        key = frame["AHU name"].astype(str) + "|" + frame["DATE"].astype(str) + "|" + frame["Time"].astype(str)
    else:
        key = frame["AHU name"].astype(str) + "|" + frame["Time"].astype(str)
    label_counts = labels.groupby(key).nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    conflict_rows = int(key.isin(conflicting).sum())
    same_label_repeats = int((key[~key.isin(conflicting)].duplicated()).sum())
    return {"conflicting_duplicate_rows_excluded": conflict_rows,
            "same_label_duplicate_rows_collapsed": same_label_repeats}


def subset(data: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[ids] for k, v in data.items()}


def capped_stratified(ids: np.ndarray, y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(y[ids]):
        take = ids[y[ids] == cls]
        if len(take) > cap:
            take = rng.choice(take, cap, replace=False)
        selected.append(take)
    return np.concatenate(selected)


def fit_scaler(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def view(data: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray):
    x = (data["x"] - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mask = data["mask"].astype(np.float32)
    x *= mask[:, :, None]
    q = np.ones_like(mask, np.float32)
    return x, mask, q, data["y"].astype(np.int64)


def split_sources(data: dict[str, np.ndarray], seed: int = 42001):
    ids = np.arange(len(data["y"]))
    train_ids, rest = train_test_split(ids, test_size=0.30, random_state=seed, stratify=data["y"])
    select_ids, cal_ids = train_test_split(rest, test_size=0.50, random_state=seed + 1,
                                           stratify=data["y"][rest])
    train_cap = 6000 if QUICK else 30000
    tune_cap = 1500 if QUICK else 6000
    train_ids = capped_stratified(train_ids, data["y"], train_cap, seed + 2)
    select_ids = capped_stratified(select_ids, data["y"], tune_cap, seed + 3)
    cal_ids = capped_stratified(cal_ids, data["y"], tune_cap, seed + 4)
    return subset(data, train_ids), subset(data, select_ids), subset(data, cal_ids)


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = major.metrics(y, p)
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, p.argmax(1)))
    return out


def paired_bootstrap(y, pa, pb, reps=None, seed=99001):
    reps = (100 if QUICK else 500) if reps is None else reps
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y, pa[:, 1]) - roc_auc_score(y, pb[:, 1])
    diffs = []
    for _ in range(reps):
        ids = rng.integers(0, len(y), len(y))
        if np.unique(y[ids]).size < 2:
            continue
        diffs.append(roc_auc_score(y[ids], pa[ids, 1]) -
                     roc_auc_score(y[ids], pb[ids, 1]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return observed, float(lo), float(hi), len(diffs)


def main():
    OUT.mkdir(exist_ok=True)
    raw = {name: load_building(name) for name in BUILDINGS}
    quality_rows = []
    for name, data in raw.items():
        quality_rows.append({
            "building": name,
            "rows_normal_or_sensor_fault": len(data["y"]),
            "normal": int((data["y"] == 0).sum()),
            "sensor_fault": int((data["y"] == 1).sum()),
            "return_sensor_fault": int((data["subtype"] == "Return air temperature fault").sum()),
            "supply_sensor_fault": int((data["subtype"] == "Supply air temperature fault").sum()),
            "ahu_units": int(pd.Series(data["ahu"]).nunique()),
            "exact_duplicate_rows": int(pd.DataFrame(data["x"].reshape(len(data["x"]), -1)).duplicated().sum()),
            "unavailable_group_rate": float(1 - data["mask"].mean()),
            **source_duplicate_profile(name),
        })
    pd.DataFrame(quality_rows).to_csv(OUT / "ahu_field_quality.csv", index=False)

    metric_rows, ensemble_rows, bootstrap_rows = [], [], []
    for fold_index, held_out in enumerate(BUILDINGS):
        source = concatenate([raw[n] for n in BUILDINGS if n != held_out])
        train_raw, select_raw, cal_raw = split_sources(source)
        mean, std = fit_scaler(train_raw["x"])
        train = view(train_raw, mean, std)
        select = view(select_raw, mean, std)
        calibration = view(cal_raw, mean, std)
        test = view(raw[held_out], mean, std)
        test_x, test_m, test_q, test_y = test
        class_weights = len(train[3]) / (2 * np.bincount(train[3], minlength=2))
        ensemble = {method: [] for method in METHODS}
        for seed in SEEDS:
            for method, spec in METHODS.items():
                model, cost = major.fit(spec, train, select, class_weights, seed,
                                        prior=tuple([0.2] * 5), epochs=EPOCHS)
                temperature = major.fit_temperature(model, calibration, neutral_q=(method != "CAGF"))
                p, scores, weights, _, infer_seconds = major.predict(model, test_x, test_m, test_q, temperature)
                ensemble[method].append(p)
                metrics = safe_metrics(test_y, p)
                for metric, value in metrics.items():
                    metric_rows.append({"held_out_building": held_out, "seed": seed, "method": method,
                                        "metric": metric, "value": value})
                metric_rows.append({"held_out_building": held_out, "seed": seed, "method": method,
                                    "metric": "native_sensor_fault_recall",
                                    "value": float((p.argmax(1)[test_y == 1] == 1).mean())})
                metric_rows.append({"held_out_building": held_out, "seed": seed, "method": method,
                                    "metric": "mean_temperature_group_weight_on_faults",
                                    "value": float(weights[test_y == 1, 1:3].sum(1).mean())})
                metric_rows.append({"held_out_building": held_out, "seed": seed, "method": method,
                                    "metric": "train_seconds", "value": cost["train_seconds"]})
                metric_rows.append({"held_out_building": held_out, "seed": seed, "method": method,
                                    "metric": "inference_seconds", "value": infer_seconds})
        ensemble_mean = {method: np.mean(ps, axis=0) for method, ps in ensemble.items()}
        for method, p in ensemble_mean.items():
            for metric, value in safe_metrics(test_y, p).items():
                ensemble_rows.append({"held_out_building": held_out, "method": method,
                                      "metric": metric, "value": value})
        for comparator in ("PDRF-NOQ", "CAGF", "EF-PD", "UF-PD"):
            effect, lo, hi, reps = paired_bootstrap(test_y, ensemble_mean["RC-PDRF-NOQ"],
                                                    ensemble_mean[comparator], seed=99001 + fold_index)
            bootstrap_rows.append({"held_out_building": held_out,
                                   "contrast": f"RC-PDRF-NOQ - {comparator}",
                                   "metric": "macro_auroc", "effect": effect,
                                   "ci_low": lo, "ci_high": hi, "reps": reps})

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "ahu_field_metrics.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(OUT / "ahu_field_ensemble.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "ahu_field_bootstrap.csv", index=False)
    summary = metrics.groupby(["method", "metric"], as_index=False).value.agg(["mean", "std", "count"])
    summary.to_csv(OUT / "ahu_field_summary.csv", index=False)

    tests = []
    primary = metrics[metrics.metric == "macro_auroc"]
    for comparator in ("PDRF-NOQ", "CAGF", "EF-PD", "UF-PD"):
        left = primary[primary.method == "RC-PDRF-NOQ"].sort_values(["held_out_building", "seed"]).value.to_numpy()
        right = primary[primary.method == comparator].sort_values(["held_out_building", "seed"]).value.to_numpy()
        stat, pvalue = wilcoxon(left, right, alternative="two-sided", method="exact")
        tests.append({"contrast": f"RC-PDRF-NOQ - {comparator}", "n_pairs": len(left),
                      "mean_difference": float(np.mean(left - right)), "wilcoxon_statistic": float(stat),
                      "p_two_sided_exact": float(pvalue), "wins": int(np.sum(left > right)),
                      "ties": int(np.sum(left == right)), "losses": int(np.sum(left < right))})
    pd.DataFrame(tests).to_csv(OUT / "ahu_field_paired_tests.csv", index=False)

    files = sorted(DATA.glob("*scientific_data.csv"))
    metadata = {
        "dataset_doi": "10.6084/m9.figshare.27147678.v3",
        "article_doi": "10.1038/s41597-025-05825-9",
        "license": "CC BY 4.0 (dataset repository)",
        "test_fault_source": "native expert annotations in real-operational BMS records; no test corruption",
        "ground_truth_boundary": "expert rule-based labels, not maintenance-adjudicated hardware replacement records",
        "held_out_design": "leave-one-building-out across auditorium, hospital, and office",
        "task": "binary normal condition versus return/supply-air temperature sensor fault",
        "seeds": list(SEEDS), "epochs": EPOCHS,
        "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    }
    (OUT / "ahu_field_design.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
