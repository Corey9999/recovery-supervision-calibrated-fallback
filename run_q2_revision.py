"""Q2-review revision experiments with prediction-level outputs.

The script evaluates bounded paired-degradation reliability fusion (PDRF),
matched corruption-aware baselines, sensor grouping sensitivity, unseen-sensor
generalization, target ablations, physical feature-level faults, disjoint model
selection/calibration subsets and complete multiclass metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler, label_binarize

import run_uci_validation as base
import run_revision_experiments as rev


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
SEEDS = (101,) if os.getenv("Q2_QUICK") == "1" else base.CFG.seeds
if os.getenv("Q2_QUICK") == "1":
    base.CFG.epochs = 3
METHODS = ("EF", "UF", "CAGF", "DWR", "BCRF")
FAULTS = ("natural", "silent_gaussian", "gain_loss", "offset", "linear_drift",
          "stuck_at", "clipping", "correlated_dual")


def prepare_grouped(path: Path, groups):
    raw, y, batches = base.read_dataset(path)
    train_sel = batches <= 6
    scaler = StandardScaler().fit(raw[train_sel])
    z = np.clip(scaler.transform(raw), -8.0, 8.0).astype(np.float32).reshape(-1, 16, 8)
    x = np.stack([z[:, g].reshape(len(z), -1) for g in groups], axis=1)
    b7 = np.flatnonzero(batches == 7)
    cut = int(round(0.60 * len(b7)))
    select_idx, calibration_idx = b7[:cut], b7[cut:]
    return {
        "train": (x[train_sel], y[train_sel], batches[train_sel]),
        "select": (x[select_idx], y[select_idx], batches[select_idx]),
        "calibration": (x[calibration_idx], y[calibration_idx], batches[calibration_idx]),
        "test": (x[batches >= 8], y[batches >= 8], batches[batches >= 8]),
    }


def make_fault(x, seed, kind, modality=0):
    rng = np.random.default_rng(seed)
    xx = x.copy()
    mask = base.random_mask(len(x), rng, base.CFG.test_missing if kind != "natural" else 0.0)
    q = np.ones((len(x), base.CFG.modalities), dtype=np.float32)
    affected = rng.random(len(x)) < base.CFG.corruption_fraction
    if kind == "natural":
        affected[:] = False
    elif kind == "silent_gaussian":
        xx[affected, modality] += rng.normal(0, 3.0, (affected.sum(), base.CFG.feature_dim)).astype(np.float32)
    elif kind == "gain_loss":
        xx[affected, modality] *= 0.30
    elif kind == "offset":
        xx[affected, modality] += 2.5
    elif kind == "linear_drift":
        idx = np.flatnonzero(affected)
        drift = np.linspace(0.0, 3.0, len(idx), dtype=np.float32)
        xx[idx, modality] += drift[:, None]
    elif kind == "stuck_at":
        xx[affected, modality] = 0.0
    elif kind == "clipping":
        xx[affected, modality] = np.clip(xx[affected, modality], -0.5, 0.5)
    elif kind == "correlated_dual":
        common = rng.normal(0, 2.5, (affected.sum(), base.CFG.feature_dim)).astype(np.float32)
        xx[affected, 0] += common
        xx[affected, 1] += common
    else:
        raise ValueError(kind)
    xx *= mask[:, :, None]
    return xx.astype(np.float32), mask, q, affected


def predict_prob(model, data, temperature=1.0):
    x, mask, q, _ = data
    with torch.no_grad():
        logits, scores, modal_logits, weights = model(torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(q))
        raw = torch.softmax(logits, 1).numpy()
        calibrated = torch.softmax(logits / temperature, 1).numpy()
    return raw, calibrated, scores.numpy(), weights.numpy()


def fixed_ece(y, p, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    correct = pred == y
    out = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        sel = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if sel.any():
            out += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(out)


def adaptive_ece(y, p, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    correct = pred == y
    order = np.argsort(conf)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(conf[c].mean() - correct[c].mean()) for c in chunks if len(c)))


def full_metrics(y, p):
    ybin = label_binarize(y, classes=np.arange(base.CFG.classes))
    pred = p.argmax(1)
    onehot = np.eye(base.CFG.classes)[y]
    out = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "macro_auroc": roc_auc_score(ybin, p, average="macro", multi_class="ovr"),
        "macro_auprc": average_precision_score(ybin, p, average="macro"),
        "micro_auprc": average_precision_score(ybin, p, average="micro"),
        "nll": log_loss(y, p, labels=np.arange(base.CFG.classes)),
        "brier": np.mean(np.sum((p - onehot) ** 2, axis=1)),
        "ece_10": fixed_ece(y, p, 10), "ece_15": fixed_ece(y, p, 15),
        "ece_20": fixed_ece(y, p, 20), "adaptive_ece_15": adaptive_ece(y, p, 15),
    }
    for c in range(base.CFG.classes):
        out[f"class_{c+1}_auroc"] = roc_auc_score(ybin[:, c], p[:, c])
        out[f"class_{c+1}_auprc"] = average_precision_score(ybin[:, c], p[:, c])
        out[f"class_{c+1}_recall"] = recall_score(y == c, pred == c)
    return {k: float(v) for k, v in out.items()}


def train_views(split, seed):
    tx, ty, _ = split["train"]
    sx, sy, _ = split["select"]
    cx, cy, _ = split["calibration"]
    train = (*base.make_view(tx, 10000 + seed, "missing_drift", train=True), ty)
    select = (*base.make_view(sx, 20000 + seed, "missing_drift", train=True), sy)
    calibration = (*base.make_view(cx, 25000 + seed, "missing_drift", train=True), cy)
    weights = len(ty) / (base.CFG.classes * np.bincount(ty, minlength=base.CFG.classes))
    return train, select, calibration, weights


def main(zip_path=None):
    path = base.ensure_zip(zip_path)
    consecutive = [list(range(i, i + 4)) for i in range(0, 16, 4)]
    split = prepare_grouped(path, consecutive)
    metric_rows, confusion_rows, prediction_store = [], [], {}
    threshold_rows = []
    for seed in SEEDS:
        train, select, calibration, class_weights = train_views(split, seed)
        testx, testy, testb = split["test"]
        for method in METHODS:
            model = rev.fit(method, train, select, class_weights, seed,
                            severity_margin=(method == "BCRF"), target_mode="interpolation" if method == "BCRF" else "removal")
            temperature = base.fit_temperature(model, calibration)
            for fault in FAULTS:
                xx, mm, qq, affected = make_fault(testx, 40000 + seed, fault)
                raw, cal, scores, weights = predict_prob(model, (xx, mm, qq, testy), temperature)
                for calibrated, probs in ((False, raw), (True, cal)):
                    for metric, value in full_metrics(testy, probs).items():
                        metric_rows.append({"seed": seed, "method": method, "fault": fault,
                                            "calibrated": calibrated, "metric": metric, "value": value})
                cm = confusion_matrix(testy, cal, labels=np.arange(base.CFG.classes)) if cal.ndim == 1 else confusion_matrix(testy, cal.argmax(1), labels=np.arange(base.CFG.classes))
                for true in range(base.CFG.classes):
                    for pred in range(base.CFG.classes):
                        confusion_rows.append({"seed": seed, "method": method, "fault": fault,
                                               "true_class": true + 1, "pred_class": pred + 1,
                                               "count": int(cm[true, pred])})
                prediction_store.setdefault((method, fault), {"raw": [], "cal": []})["raw"].append(raw)
                prediction_store[(method, fault)]["cal"].append(cal)
                if method == "BCRF" and fault == "silent_gaussian":
                    valid = affected & (mm[:, 0] > 0)
                    for threshold in (2.5, 2.7, 2.8, 2.9, 2.95):
                        threshold_rows.append({"seed": seed, "threshold": threshold,
                                               "fraction": float((scores[valid, 0] >= threshold).mean())})

    pd.DataFrame(metric_rows).to_csv(OUT / "q2_metrics_long.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUT / "q2_confusion_long.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT / "q2_boundary_thresholds.csv", index=False)
    pred_rows = []
    _, testy, testb = split["test"]
    for (method, fault), values in prediction_store.items():
        raw = np.mean(values["raw"], axis=0)
        cal = np.mean(values["cal"], axis=0)
        for i in range(len(testy)):
            row = {"sample": i, "batch": int(testb[i]), "y": int(testy[i]),
                   "method": method, "fault": fault}
            for c in range(base.CFG.classes):
                row[f"raw_p{c+1}"] = raw[i, c]
                row[f"cal_p{c+1}"] = cal[i, c]
            pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(OUT / "q2_ensemble_predictions.csv", index=False)

    # Sensor-grouping sensitivity: consecutive, interleaved and five fixed random partitions.
    groupings = {"consecutive": consecutive,
                 "interleaved": [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]}
    for k in range(5):
        perm = np.random.default_rng(8100 + k).permutation(16)
        groupings[f"random_{k+1}"] = [perm[i:i+4].tolist() for i in range(0, 16, 4)]
    grouping_rows = []
    for grouping_name, groups in groupings.items():
        gs = prepare_grouped(path, groups)
        for seed in SEEDS[:5]:
            train, select, calibration, cw = train_views(gs, seed)
            model = rev.fit("BCRF", train, select, cw, seed, severity_margin=True, target_mode="interpolation")
            temperature = base.fit_temperature(model, calibration)
            x, y, _ = gs["test"]
            for fault in ("natural", "silent_gaussian"):
                data = (*make_fault(x, 50000 + seed, fault)[:3], y)
                _, p, _, _ = predict_prob(model, data, temperature)
                grouping_rows.append({"grouping": grouping_name, "groups": json.dumps(groups),
                                      "seed": seed, "fault": fault,
                                      "macro_auroc": full_metrics(y, p)["macro_auroc"]})
    pd.DataFrame(grouping_rows).to_csv(OUT / "q2_grouping_sensitivity.csv", index=False)

    # Sampling and unseen-sensor tests. Exclude modality 4 during degradation training,
    # then fault only modality 4 at test time.
    sampling = {"dominant_prior": (0.45, 0.25, 0.18, 0.12),
                "uniform": (0.25, 0.25, 0.25, 0.25),
                "exclude_test_sensor": (0.40, 0.33, 0.27, 0.0)}
    sampling_rows = []
    for sampling_name, prior in sampling.items():
        for seed in SEEDS[:5]:
            train, select, calibration, cw = train_views(split, seed)
            model = rev.fit("BCRF", train, select, cw, seed, degradation_prior=prior,
                            severity_margin=True, target_mode="interpolation")
            temperature = base.fit_temperature(model, calibration)
            x, y, _ = split["test"]
            for modality in (0, 3):
                data = (*make_fault(x, 55000 + seed, "silent_gaussian", modality=modality)[:3], y)
                _, p, _, _ = predict_prob(model, data, temperature)
                sampling_rows.append({"sampling": sampling_name, "seed": seed,
                                      "test_fault_modality": modality + 1,
                                      "macro_auroc": full_metrics(y, p)["macro_auroc"]})
    pd.DataFrame(sampling_rows).to_csv(OUT / "q2_sampling_unseen_sensor.csv", index=False)

    # Consistency-target and severity-margin ablation.
    target_specs = (("removal_fixed", "removal", False), ("clean_fixed", "clean", False),
                    ("teacher_fixed", "teacher", False), ("interpolation_fixed", "interpolation", False),
                    ("interpolation_severity", "interpolation", True))
    target_rows = []
    for label, target, severity in target_specs:
        for seed in SEEDS[:5]:
            train, select, calibration, cw = train_views(split, seed)
            model = rev.fit("BCRF", train, select, cw, seed, target_mode=target, severity_margin=severity)
            temperature = base.fit_temperature(model, calibration)
            x, y, _ = split["test"]
            for scale in (0.5, 3.0, 5.0):
                xx, mm, qq = base.make_view(x, 58000 + seed, "silent_fault", corruption_scale=scale)
                _, p, _, _ = predict_prob(model, (xx, mm, qq, y), temperature)
                target_rows.append({"target": label, "seed": seed, "fault_scale": scale,
                                    "macro_auroc": full_metrics(y, p)["macro_auroc"]})
    pd.DataFrame(target_rows).to_csv(OUT / "q2_target_margin_ablation.csv", index=False)

    metadata = {"model_selection": "first 60% of batch 7 in acquisition order",
                "temperature_calibration": "remaining 40% of batch 7",
                "main_methods": METHODS, "faults": FAULTS, "seeds": list(SEEDS),
                "groupings": groupings, "sampling_priors": sampling,
                "main_method": "BCRF interpreted as bounded paired-degradation reliability fusion (PDRF)",
                "exploratory": ["grouping sensitivity", "sampling sensitivity", "target-margin ablation"]}
    (OUT / "q2_revision_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(metric_rows).query("calibrated == True and metric == 'macro_auroc'").groupby(["method", "fault"]).value.mean())


if __name__ == "__main__":
    main()
