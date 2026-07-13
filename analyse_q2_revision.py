"""Prediction-level bootstrap, batch, calibration and deployment-utility analyses."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)
CLASSES = np.arange(6)
METHOD_LABEL = {"EF": "Early fusion", "UF": "Uniform fusion", "CAGF": "Corruption-aware gate",
                "DWR": "Direct weight ranking", "BCRF": "PDRF"}


def probs(frame, prefix="cal"):
    p = frame[[f"{prefix}_p{i}" for i in range(1, 7)]].to_numpy(float)
    return p / p.sum(1, keepdims=True)


def metric(y, p, name):
    pred = p.argmax(1)
    ybin = label_binarize(y, classes=CLASSES)
    if name == "macro_auroc": return roc_auc_score(ybin, p, average="macro", multi_class="ovr")
    if name == "macro_auprc": return average_precision_score(ybin, p, average="macro")
    if name == "accuracy": return np.mean(pred == y)
    if name == "macro_f1": return f1_score(y, pred, average="macro")
    if name == "nll": return log_loss(y, p, labels=CLASSES)
    raise ValueError(name)


pred = pd.read_csv(SRC / "q2_ensemble_predictions.csv")
methods = ["EF", "UF", "CAGF", "DWR", "BCRF"]
faults = pred.fault.drop_duplicates().tolist()

# Paired sample and batch-stratified bootstrap on ensemble-mean predictions.
rng = np.random.default_rng(20260711)
bootstrap_rows = []
for fault in ("natural", "silent_gaussian", "correlated_dual"):
    data = {m: pred[(pred.method == m) & (pred.fault == fault)].sort_values("sample") for m in methods}
    y = data["BCRF"].y.to_numpy(int)
    batch = data["BCRF"].batch.to_numpy(int)
    cached_probs = {m: probs(data[m]) for m in methods}
    for comparator in ("EF", "UF", "CAGF", "DWR"):
        for metric_name in ("macro_auroc", "macro_auprc", "accuracy", "macro_f1", "nll"):
            values, stratified = [], []
            for _ in range(300):
                idx = rng.integers(0, len(y), len(y))
                pieces = []
                for b in np.unique(batch):
                    ib = np.flatnonzero(batch == b)
                    pieces.append(rng.choice(ib, len(ib), replace=True))
                sidx = np.concatenate(pieces)
                for target, store in ((idx, values), (sidx, stratified)):
                    a = metric(y[target], cached_probs["BCRF"][target], metric_name)
                    c = metric(y[target], cached_probs[comparator][target], metric_name)
                    store.append((c - a) if metric_name == "nll" else (a - c))
            point_a = metric(y, cached_probs["BCRF"], metric_name)
            point_c = metric(y, cached_probs[comparator], metric_name)
            point = (point_c - point_a) if metric_name == "nll" else (point_a - point_c)
            bootstrap_rows.append({"fault": fault, "contrast": f"PDRF-{comparator}", "metric": metric_name,
                                   "aligned_difference": point,
                                   "sample_ci_low": np.quantile(values, .025), "sample_ci_high": np.quantile(values, .975),
                                   "batch_stratified_ci_low": np.quantile(stratified, .025),
                                   "batch_stratified_ci_high": np.quantile(stratified, .975), "bootstrap_reps": 300})
pd.DataFrame(bootstrap_rows).to_csv(SRC / "q2_bootstrap_effects.csv", index=False)

# Per-batch performance.
batch_rows = []
for (method, fault, batch), d in pred.groupby(["method", "fault", "batch"]):
    y = d.y.to_numpy(int); p = probs(d)
    for name in ("macro_auroc", "macro_auprc", "accuracy", "macro_f1", "nll"):
        batch_rows.append({"method": method, "fault": fault, "batch": batch, "metric": name,
                           "value": metric(y, p, name), "n_samples": len(y)})
pd.DataFrame(batch_rows).to_csv(SRC / "q2_batch_performance.csv", index=False)

# Calibration before and after temperature scaling.
metrics = pd.read_csv(SRC / "q2_metrics_long.csv")
calibration = metrics[metrics.metric.isin(["nll", "brier", "ece_10", "ece_15", "ece_20", "adaptive_ece_15"])]
cal_summary = calibration.groupby(["method", "fault", "calibrated", "metric"], as_index=False).value.mean()
cal_summary.to_csv(SRC / "q2_calibration_summary.csv", index=False)

# Deployment utility. Costs are computed from ensemble hard predictions and rejection by confidence.
cost_schemes = {"uniform": np.ones((6, 6)) - np.eye(6),
                "class1_double": np.ones((6, 6)) - np.eye(6),
                "class6_double": np.ones((6, 6)) - np.eye(6)}
cost_schemes["class1_double"][0, 1:] = 2.0
cost_schemes["class6_double"][5, :5] = 2.0
utility_rows = []
for method in methods:
    natural = pred[(pred.method == method) & (pred.fault == "natural")].sort_values("sample")
    faulty = pred[(pred.method == method) & (pred.fault == "silent_gaussian")].sort_values("sample")
    y = natural.y.to_numpy(int)
    for scheme, matrix in cost_schemes.items():
        costs = {}
        for label, d in (("clean", natural), ("fault", faulty)):
            p = probs(d); hard = p.argmax(1); conf = p.max(1)
            base_cost = matrix[y, hard]
            for coverage in (0.8, 0.9, 1.0):
                keep = np.argsort(-conf)[:int(round(coverage * len(y)))]
                accepted = np.zeros(len(y), dtype=bool); accepted[keep] = True
                for reject_cost in (0.10, 0.25, 0.50):
                    costs[(label, coverage, reject_cost)] = np.mean(np.where(accepted, base_cost, reject_cost))
        for coverage in (0.8, 0.9, 1.0):
            for reject_cost in (0.10, 0.25, 0.50):
                for pi in np.linspace(0, 1, 101):
                    value = ((1 - pi) * costs[("clean", coverage, reject_cost)] +
                             pi * costs[("fault", coverage, reject_cost)])
                    utility_rows.append({"method": method, "cost_scheme": scheme, "coverage": coverage,
                                         "reject_cost": reject_cost, "fault_prevalence": pi,
                                         "expected_cost": value})
utility = pd.DataFrame(utility_rows)
utility.to_csv(SRC / "q2_deployment_utility.csv", index=False)

break_rows = []
for keys, d in utility.groupby(["cost_scheme", "coverage", "reject_cost"]):
    wide = d.pivot(index="fault_prevalence", columns="method", values="expected_cost")
    for comparator in ("EF", "UF", "CAGF", "DWR"):
        delta = wide["BCRF"] - wide[comparator]
        wins = delta <= 0
        threshold = float(wide.index[wins][0]) if wins.any() else np.nan
        break_rows.append({"cost_scheme": keys[0], "coverage": keys[1], "reject_cost": keys[2],
                           "comparator": comparator, "break_even_fault_prevalence": threshold,
                           "pdrf_cost_at_zero": wide["BCRF"].iloc[0],
                           "comparator_cost_at_zero": wide[comparator].iloc[0],
                           "pdrf_cost_at_one": wide["BCRF"].iloc[-1],
                           "comparator_cost_at_one": wide[comparator].iloc[-1]})
pd.DataFrame(break_rows).to_csv(SRC / "q2_break_even.csv", index=False)

# Compact supplementary tables.
headline = metrics[(metrics.calibrated == True) & metrics.metric.isin(
    ["accuracy", "macro_f1", "macro_auroc", "macro_auprc", "nll", "brier", "ece_15"])]
headline = headline.groupby(["method", "fault", "metric"], as_index=False).value.mean()
headline.to_csv(SRC / "q2_headline_metrics.csv", index=False)

def latex_table(frame, path, caption_order):
    lines = ["\\begin{tabular}{ll" + "c" * len(caption_order) + "}", "\\toprule",
             "Method & View & " + " & ".join(caption_order.values()) + " \\\\", "\\midrule"]
    for method in methods:
        for fault in ("natural", "silent_gaussian", "correlated_dual"):
            vals = []
            for metric_name in caption_order:
                v = frame[(frame.method == method) & (frame.fault == fault) & (frame.metric == metric_name)].value.iloc[0]
                vals.append(f"{v:.3f}")
            lines.append(f"{METHOD_LABEL[method]} & {fault.replace('_',' ')} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

latex_table(headline, TABLES / "q2_headline_results.tex",
            {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "macro_auroc": "Macro-AUROC",
             "macro_auprc": "Macro-AUPRC", "nll": "NLL", "brier": "Brier", "ece_15": "ECE"})

print(pd.DataFrame(bootstrap_rows).query("fault == 'silent_gaussian' and metric in ['macro_auroc','accuracy','nll']").to_string(index=False))
print(pd.DataFrame(break_rows).query("cost_scheme == 'uniform' and coverage == 1.0 and reject_cost == 0.25").to_string(index=False))
