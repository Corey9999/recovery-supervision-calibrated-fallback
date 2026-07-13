"""QMF-PD adaptation and fairness sensitivity for manuscript 16.

The experiment keeps the chemical split, group encoders, paired-degradation
exposure and checkpoint rule fixed while varying only QMF-specific choices:
confidence scale, normalized versus official unnormalized fusion, ranking-loss
weight, and a closer implementation of the official cumulative-history margin
with detached fusion confidences.  These are sensor-group adaptations, not
reproductions of the original text--image or RGB-D systems.
"""
from dataclasses import replace
from pathlib import Path
import json
import os
import time

import numpy as np
import pandas as pd

import run_major_revision_experiments as major
import run_q1_risk_sensitive as risk
import run_q2_revision as q2
import run_uci_validation as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "source_data"
QUICK = os.getenv("M8_QUICK") == "1"
SEEDS = (101,) if QUICK else tuple(range(101, 106))
EPOCHS = 3 if QUICK else 60

QMF_BASE = replace(
    major.Spec(), architecture="qmf", paired=True, score_loss=False,
    consistency=False, norm_mono=False, raw_mono=False, score_rank=False,
    direct_rank=False, use_quality=False, train_quality=False,
    recovery_distillation=False, brier_regularization=False,
    gate_kl_regularization=False, qmf_rank_weight=.10,
    qmf_confidence_scale=.10, qmf_normalized_fusion=False,
    qmf_faithful_rank=False, qmf_detach_fusion_confidence=False)

METHODS = {
    "QMF-PD-current": QMF_BASE,
    "QMF-PD-scale003": replace(QMF_BASE, qmf_confidence_scale=.03),
    "QMF-PD-scale030": replace(QMF_BASE, qmf_confidence_scale=.30),
    "QMF-PD-normalized": replace(QMF_BASE, qmf_normalized_fusion=True),
    "QMF-PD-rank0": replace(QMF_BASE, qmf_rank_weight=0.0),
    "QMF-PD-rank1": replace(QMF_BASE, qmf_rank_weight=1.0),
    "QMF-PD-faithful": replace(
        QMF_BASE, qmf_rank_weight=1.0, qmf_faithful_rank=True,
        qmf_detach_fusion_confidence=True),
}


def main():
    OUT.mkdir(exist_ok=True)
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(i, i + 4)) for i in range(0, 16, 4)])
    testx, testy, batches = split["test"]
    faultx, mask, quality, affected = major.fixed_fault(
        testx, "silent", 0, 3.0, .40, 70001)
    cleanx = (testx * mask[:, :, None]).astype(np.float32)
    rows, predictions, costs, traces = [], [], [], []
    started = time.perf_counter()

    for seed in SEEDS:
        train, select, calibration, class_weights = q2.train_views(split, seed)
        for method, spec in METHODS.items():
            trace = []
            model, cost = major.fit(
                spec, train, select, class_weights, seed,
                prior=(.45, .25, .18, .12), epochs=EPOCHS, trace=trace)
            for record in trace:
                record["method"] = method
                traces.append(record)
            temperature = major.fit_temperature(model, calibration)
            p_clean, _, _, _, clean_seconds = major.predict(
                model, cleanx, mask, quality, temperature)
            p_fault, _, weights, _, fault_seconds = major.predict(
                model, faultx, mask, quality, temperature)
            costs.append({
                "method": method, "seed": seed, "temperature": temperature,
                "clean_inference_ms_per_observation":
                    1000*clean_seconds/len(testy),
                "fault_inference_ms_per_observation":
                    1000*fault_seconds/len(testy), **cost})
            subsets = (
                ("clean_matched_mask", np.ones(len(testy), bool), p_clean),
                ("fault_all", np.ones(len(testy), bool), p_fault),
                ("fault_affected", affected, p_fault),
                ("fault_unaffected", ~affected, p_fault),
            )
            for subset, take, probabilities in subsets:
                for metric, value in risk.all_metrics(
                        testy[take], probabilities[take]).items():
                    rows.append({
                        "method": method, "seed": seed, "subset": subset,
                        "metric": metric, "value": value, "n": int(take.sum())})
            for i in np.flatnonzero(affected):
                row = {
                    "method": method, "seed": seed, "sample": int(i),
                    "batch": int(batches[i]), "y": int(testy[i]),
                    "affected_group_weight": float(weights[i, 0]),
                }
                row.update({f"p{c}": float(p_fault[i, c])
                            for c in range(p_fault.shape[1])})
                predictions.append(row)

    pd.DataFrame(rows).to_csv(OUT/"major8_qmf_sensitivity_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        OUT/"major8_qmf_sensitivity_predictions.csv", index=False)
    pd.DataFrame(costs).to_csv(OUT/"major8_qmf_sensitivity_costs.csv", index=False)
    pd.DataFrame(traces).to_csv(OUT/"major8_qmf_training_curves.csv", index=False)
    (OUT/"major8_qmf_sensitivity_design.json").write_text(json.dumps({
        "seeds": list(SEEDS), "epochs": EPOCHS,
        "official_QMF_tasks": ["text-image classification", "RGB-D scene recognition"],
        "official_confidence": "logsumexp(logits)/10",
        "official_fusion": "unnormalized sum of detached confidence times modality logits",
        "official_rank": "cumulative per-sample CE history, normalized pairwise margin, unit loss weight",
        "adaptation_constants": {
            "optimizer": "AdamW lr=0.002 weight_decay=0.0001",
            "batch_size": 256, "paired_fault_exposure": True,
            "checkpoint": "minimum clean selection CE with patience 9",
            "temperature": "one calibration-split temperature per fitted seed"},
        "test_used_for_selection": False,
        "scope": "QMF-specific sensitivity within a sensor-group adaptation, not an exact reproduction",
        "elapsed_seconds": time.perf_counter()-started,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
