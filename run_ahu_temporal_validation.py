"""Within-building chronological validation on native AHU sensor-fault labels."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

import run_ahu_field_validation as ahu
import run_major_revision_experiments as major


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
QUICK = os.getenv("AHU_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 106))
EPOCHS = 8 if QUICK else 12


def add_lag_features(data):
    """Add leakage-safe current/1 h/6 h/24 h context keyed by AHU and time."""
    times = pd.to_datetime(data["time"], errors="coerce")
    keys = pd.MultiIndex.from_arrays([data["ahu"], times])
    blocks = []
    for lag in (0, 1, 6, 24):
        query = pd.MultiIndex.from_arrays([data["ahu"], times-pd.to_timedelta(lag, unit="h")])
        loc = keys.get_indexer(query)
        x = np.full((len(times), 20), np.nan, np.float32)
        m = np.zeros((len(times), 5), np.float32)
        valid = loc >= 0
        x[valid] = data["x"][loc[valid]].reshape(valid.sum(), -1)
        m[valid] = data["mask"][loc[valid]]
        blocks.extend([x, m])
    data["current_flat"] = np.concatenate(blocks[:2], axis=1)
    data["lag_flat"] = np.concatenate(blocks, axis=1)
    return data


def scale_probabilities(cal_p, cal_y, test_p):
    cal_log = np.log(np.clip(cal_p, 1e-8, 1.0))
    test_log = np.log(np.clip(test_p, 1e-8, 1.0))
    def objective(log_t):
        z = cal_log/np.exp(log_t); z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return log_loss(cal_y, p, labels=[0, 1])
    t = float(np.exp(minimize_scalar(objective, bounds=(-3, 3), method="bounded").x))
    z = test_log/t; z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p, t


def chronological_split(data):
    time = pd.to_datetime(data["time"], errors="coerce")
    order = np.argsort(time.to_numpy())
    n = len(order)
    cuts = (int(0.60*n), int(0.75*n), int(0.85*n))
    ids = (order[:cuts[0]], order[cuts[0]:cuts[1]], order[cuts[1]:cuts[2]], order[cuts[2]:])
    caps = (6000, 1500, 1500, 10**9)
    result = []
    for j, (part, cap) in enumerate(zip(ids, caps)):
        if j < 3:
            part = ahu.capped_stratified(part, data["y"], cap, 43001+j)
        result.append(ahu.subset(data, part))
    return result


def main():
    metric_rows = []
    split_rows = []
    prediction_rows = []
    calibration_rows = []
    confusion_rows = []
    for building in ahu.BUILDINGS:
        data = add_lag_features(ahu.load_building(building))
        test_prob_store, cal_prob_store = {}, {}
        train_raw, select_raw, cal_raw, test_raw = chronological_split(data)
        for split_name, part in zip(("train", "selection", "calibration", "test"),
                                    (train_raw, select_raw, cal_raw, test_raw)):
            split_rows.append({"building": building, "split": split_name, "rows": len(part["y"]),
                               "normal": int((part["y"] == 0).sum()),
                               "sensor_fault": int((part["y"] == 1).sum()),
                               "start": str(pd.to_datetime(part["time"]).min()),
                               "end": str(pd.to_datetime(part["time"]).max())})
        mean, std = ahu.fit_scaler(train_raw["x"])
        train, select, calibration, test = [ahu.view(p, mean, std)
                                            for p in (train_raw, select_raw, cal_raw, test_raw)]
        class_weights = len(train[3]) / (2*np.bincount(train[3], minlength=2))
        for seed in SEEDS:
            for method, spec in ahu.METHODS.items():
                model, cost = major.fit(spec, train, select, class_weights, seed,
                                        prior=tuple([0.2]*5), epochs=EPOCHS)
                temperature = major.fit_temperature(model, calibration, neutral_q=(method != "CAGF"))
                p, scores, weights, _, infer_seconds = major.predict(model, *test[:3], temperature)
                cal_p, _, _, _, _ = major.predict(model, *calibration[:3], temperature)
                test_prob_store.setdefault(method, []).append(p)
                cal_prob_store.setdefault(method, []).append(cal_p)
                for metric, value in ahu.safe_metrics(test[3], p).items():
                    metric_rows.append({"building": building, "seed": seed, "method": method,
                                        "metric": metric, "value": value})
                metric_rows.append({"building": building, "seed": seed, "method": method,
                                    "metric": "native_sensor_fault_recall",
                                    "value": float((p.argmax(1)[test[3] == 1] == 1).mean())})
                metric_rows.append({"building": building, "seed": seed, "method": method,
                                    "metric": "mean_temperature_group_weight_on_faults",
                                    "value": float(weights[test[3] == 1, 1:3].sum(1).mean())})
                metric_rows.append({"building": building, "seed": seed, "method": method,
                                    "metric": "train_seconds", "value": cost["train_seconds"]})
                metric_rows.append({"building": building, "seed": seed, "method": method,
                                    "metric": "inference_seconds", "value": infer_seconds})
                for subtype in ("Return air temperature fault", "Supply air temperature fault"):
                    take = test_raw["subtype"] == subtype
                    metric_rows.append({"building": building, "seed": seed, "method": method,
                                        "metric": "recall_"+subtype.lower().replace(" ", "_"),
                                        "value": float((p.argmax(1)[take] == 1).mean()) if take.any() else np.nan})
                pred = p.argmax(1)
                for true in (0, 1):
                    for guessed in (0, 1):
                        confusion_rows.append({"building": building, "seed": seed, "method": method,
                                               "true": true, "predicted": guessed,
                                               "count": int(np.sum((test[3] == true) & (pred == guessed)))})

            # Task-specific supervised diagnosis controls. HGB-current uses
            # only the present record; HGB-lag additionally uses 1/6/24-h history.
            for method, feature in (("HGB-current", "current_flat"), ("HGB-lag", "lag_flat")):
                import time
                t0 = time.perf_counter()
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                     max_leaf_nodes=31, l2_regularization=1e-3,
                                                     random_state=seed, early_stopping=True)
                weights = class_weights[train_raw["y"]]
                clf.fit(train_raw[feature], train_raw["y"], sample_weight=weights)
                train_seconds = time.perf_counter()-t0
                cal_raw_p = clf.predict_proba(cal_raw[feature])
                test_raw_p = clf.predict_proba(test_raw[feature])
                p, temperature = scale_probabilities(cal_raw_p, cal_raw["y"], test_raw_p)
                cal_p, _ = scale_probabilities(cal_raw_p, cal_raw["y"], cal_raw_p)
                test_prob_store.setdefault(method, []).append(p)
                cal_prob_store.setdefault(method, []).append(cal_p)
                for metric, value in ahu.safe_metrics(test_raw["y"], p).items():
                    metric_rows.append({"building": building, "seed": seed, "method": method,
                                        "metric": metric, "value": value})
                pred = p.argmax(1)
                metric_rows.extend([
                    {"building": building, "seed": seed, "method": method,
                     "metric": "native_sensor_fault_recall",
                     "value": float((pred[test_raw["y"] == 1] == 1).mean())},
                    {"building": building, "seed": seed, "method": method,
                     "metric": "train_seconds", "value": train_seconds},
                ])
                for subtype in ("Return air temperature fault", "Supply air temperature fault"):
                    take = test_raw["subtype"] == subtype
                    metric_rows.append({"building": building, "seed": seed, "method": method,
                                        "metric": "recall_"+subtype.lower().replace(" ", "_"),
                                        "value": float((pred[take] == 1).mean()) if take.any() else np.nan})
                for true in (0, 1):
                    for guessed in (0, 1):
                        confusion_rows.append({"building": building, "seed": seed, "method": method,
                                               "true": true, "predicted": guessed,
                                               "count": int(np.sum((test_raw["y"] == true) & (pred == guessed)))})

        # Store one fixed deep-ensemble prediction per building and method.
        for method, probs in test_prob_store.items():
            p = np.mean(probs, axis=0)
            for i in range(len(test_raw["y"])):
                prediction_rows.append({"building": building, "method": method, "sample": i,
                                        "time": test_raw["time"][i], "ahu": test_raw["ahu"][i],
                                        "subtype": test_raw["subtype"][i], "y": int(test_raw["y"][i]),
                                        "p0": float(p[i, 0]), "p1": float(p[i, 1])})
        for method, probs in cal_prob_store.items():
            p = np.mean(probs, axis=0)
            for i in range(len(cal_raw["y"])):
                calibration_rows.append({"building": building, "method": method, "sample": i,
                                         "y": int(cal_raw["y"][i]),
                                         "p0": float(p[i, 0]), "p1": float(p[i, 1])})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "ahu_temporal_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "ahu_temporal_splits.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "ahu_temporal_predictions.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "ahu_temporal_calibration_predictions.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUT / "ahu_temporal_confusion.csv", index=False)
    metrics.groupby(["method", "metric"], as_index=False).value.agg(["mean", "std", "count"]).to_csv(
        OUT / "ahu_temporal_summary.csv", index=False)
    primary = metrics[metrics.metric == "macro_auroc"]
    tests = []
    for comparator in ("PDRF-NOQ", "CAGF", "ENT-PD", "EF-PD", "UF-PD", "HGB-current", "HGB-lag"):
        left = primary[primary.method == "RC-PDRF-NOQ"].sort_values(["building", "seed"]).value.to_numpy()
        right = primary[primary.method == comparator].sort_values(["building", "seed"]).value.to_numpy()
        stat, pvalue = wilcoxon(left, right, method="exact")
        tests.append({"contrast": f"RC-PDRF-NOQ - {comparator}", "n_pairs": len(left),
                      "mean_difference": float(np.mean(left-right)), "wilcoxon_statistic": float(stat),
                      "p_two_sided_exact": float(pvalue), "wins": int((left > right).sum()),
                      "ties": int((left == right).sum()), "losses": int((left < right).sum())})
    pd.DataFrame(tests).to_csv(OUT / "ahu_temporal_paired_tests.csv", index=False)


if __name__ == "__main__":
    main()
