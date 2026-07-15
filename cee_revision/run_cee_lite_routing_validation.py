"""Evaluate compute-matched selective routing from PDRF to RO-PDRF-Lite.

The protocol preserves the CEE-CF10-R2 split, masks, fault seeds, temperature
partition, optimization seeds and test realization. The selector uses only six
endpoint confidence/entropy features, so inference needs two forward passes and
no leave-one-group-out recovery evaluations.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
UPGRADE = ROOT.parent / "paper_package_q1_upgrade"
sys.path.insert(0, str(UPGRADE))

import run_major_revision_experiments as major
import run_major7_fault_types_ema as typed
import run_q2_revision as q2
import run_uci_validation as base

from analyse_cee_q1_scores import (
    FEATURES_6,
    grouped_binary_fit,
    normalized_entropy,
    probability_metrics,
    selection_metrics,
)


OUT = ROOT / "source_data"
SEEDS = tuple(range(101, 111))
FAULTS = ("gaussian", "offset", "drift", "stuck_at")
PREVALENCES = (0.10, 0.40, 0.70)
TEST_REALIZATION = 70001
EPOCHS = 60
LITE_SPEC = replace(
    major.SPECS["PDRF"],
    recovery_distillation=True,
    w_recovery=0.30,
    recovery_teacher="clean",
)


def features6(base_p: np.ndarray, recovery_p: np.ndarray) -> np.ndarray:
    base_confidence = base_p.max(axis=1)
    recovery_confidence = recovery_p.max(axis=1)
    base_entropy = normalized_entropy(base_p)
    recovery_entropy = normalized_entropy(recovery_p)
    return np.column_stack(
        [
            base_confidence,
            recovery_confidence,
            recovery_confidence - base_confidence,
            base_entropy,
            recovery_entropy,
            recovery_entropy - base_entropy,
        ]
    )


def predict(model, x, mask, quality, temperature):
    probability, _, _, _, _ = major.predict(
        model, x, mask, quality, temperature
    )
    return probability


def benchmark(callable_, repeats=7):
    """Match the earlier CEE timing protocol: 3 warm-ups and 7 timed calls."""
    for _ in range(3):
        callable_()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        times.append(time.perf_counter() - started)
    return (
        float(np.median(times)),
        float(np.quantile(times, 0.25)),
        float(np.quantile(times, 0.75)),
    )


def cpu_model_name():
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (OSError, ImportError):
            pass
    return platform.processor() or platform.machine() or "unavailable"


def route_probability(
    base_model,
    lite_model,
    x,
    mask,
    quality,
    base_temperature,
    lite_temperature,
    selector,
    threshold,
    return_identity=False,
):
    pb = predict(base_model, x, mask, quality, base_temperature)
    pl = predict(lite_model, x, mask, quality, lite_temperature)
    score = selector.predict_proba(
        pd.DataFrame(features6(pb, pl), columns=FEATURES_6)
    )[:, 1]
    choose_recovery = score >= threshold
    selected = np.where(choose_recovery[:, None], pl, pb)
    if return_identity:
        endpoint_identity = np.where(
            choose_recovery, "RO-PDRF-Lite", "PDRF"
        )
        return selected, endpoint_identity, score
    return selected


def calibration_frame(
    seed,
    selector_x,
    selector_y,
    base_model,
    lite_model,
    base_temperature,
    lite_temperature,
):
    rows = []
    for fault_index, fault in enumerate(FAULTS):
        for realization in range(2):
            realization_id = 96000 + 100 * fault_index + 10 * realization + seed
            fault_x, mask, quality, assigned = typed.fixed_typed_fault(
                selector_x, fault, seed=realization_id
            )
            applied = assigned & (mask[:, 0] > 0)
            pb = predict(base_model, fault_x, mask, quality, base_temperature)
            pl = predict(lite_model, fault_x, mask, quality, lite_temperature)
            features = features6(pb, pl)
            for sample in np.flatnonzero(applied):
                row = {
                    "seed": seed,
                    "fault_type": fault,
                    "realization": realization,
                    "realization_id": realization_id,
                    "sample": int(sample),
                    "base_correct": bool(pb[sample].argmax() == selector_y[sample]),
                    "recovery_correct": bool(pl[sample].argmax() == selector_y[sample]),
                }
                row.update(dict(zip(FEATURES_6, features[sample])))
                rows.append(row)
    return pd.DataFrame(rows)


def add_prediction_rows(
    rows,
    seed,
    fault,
    stream,
    y,
    batch,
    samples,
    pb,
    pl,
    choose,
):
    selected = np.where(choose[:, None], pl, pb)
    for local, sample in enumerate(samples):
        row = {
            "seed": seed,
            "fault_type": fault,
            "stream": stream,
            "sample": int(sample),
            "batch": int(batch[local]),
            "y": int(y[local]),
            "choose_lite": bool(choose[local]),
        }
        for class_index in range(pb.shape[1]):
            row[f"base_p{class_index}"] = float(pb[local, class_index])
            row[f"lite_p{class_index}"] = float(pl[local, class_index])
            row[f"selected_p{class_index}"] = float(selected[local, class_index])
        rows.append(row)


def main():
    started = time.perf_counter()
    split = q2.prepare_grouped(
        base.ensure_zip(None), [list(range(index, index + 4)) for index in range(0, 16, 4)]
    )
    test_x, test_y, test_batch = split["test"]
    calibration_x, calibration_y, _ = split["calibration"]
    cut = len(calibration_x) // 2
    temperature_x, temperature_y = calibration_x[:cut], calibration_y[:cut]
    selector_x, selector_y = calibration_x[cut:], calibration_y[cut:]

    metric_rows = []
    safety_rows = []
    prediction_rows = []
    diagnostic_rows = []
    cost_rows = []
    latency_rows = []
    calibration_rows = []

    for seed in SEEDS:
        train, select, _, class_weights = q2.train_views(split, seed)
        temperature_view = (
            *base.make_view(temperature_x, 25000 + seed, "missing_drift", train=True),
            temperature_y,
        )
        base_model, base_cost = major.fit(
            major.SPECS["PDRF"],
            train,
            select,
            class_weights,
            seed,
            prior=(0.45, 0.25, 0.18, 0.12),
            epochs=EPOCHS,
        )
        lite_model, lite_cost = major.fit(
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
        calibration = calibration_frame(
            seed,
            selector_x,
            selector_y,
            base_model,
            lite_model,
            base_temperature,
            lite_temperature,
        )
        calibration_rows.extend(calibration.to_dict("records"))
        fit = grouped_binary_fit(
            calibration, FEATURES_6, seed + 5000, c_value=0.05
        )
        informative = calibration.base_correct != calibration.recovery_correct
        diagnostic_rows.append(
            {
                "seed": seed,
                "calibration_rows": len(calibration),
                "informative_rows": int(informative.sum()),
                "base_better_rows": int(
                    (calibration.base_correct & ~calibration.recovery_correct).sum()
                ),
                "lite_better_rows": int(
                    (~calibration.base_correct & calibration.recovery_correct).sum()
                ),
                "threshold": fit["threshold"],
                "threshold_tie_count": fit["tie_count"],
                "base_temperature": base_temperature,
                "lite_temperature": lite_temperature,
            }
        )
        cost_rows.extend(
            [
                {"seed": seed, "method": "PDRF", **base_cost},
                {"seed": seed, "method": "RO-PDRF-Lite", **lite_cost},
            ]
        )

        for fault in FAULTS:
            for prevalence in PREVALENCES:
                fault_x, mask, quality, assigned = typed.fixed_typed_fault(
                    test_x,
                    fault,
                    prevalence=prevalence,
                    seed=TEST_REALIZATION,
                )
                strict = assigned & (mask[:, 0] > 0)
                pb_all = predict(base_model, fault_x, mask, quality, base_temperature)
                pl_all = predict(lite_model, fault_x, mask, quality, lite_temperature)
                score_all = fit["model"].predict_proba(
                    pd.DataFrame(features6(pb_all, pl_all), columns=FEATURES_6)
                )[:, 1]
                choose_all = score_all >= fit["threshold"]
                streams = [
                    (
                        "full_mixed_stream"
                        if np.isclose(prevalence, 0.40)
                        else f"full_mixed_{int(100 * prevalence)}pct",
                        np.ones(len(test_y), bool),
                    )
                ]
                if np.isclose(prevalence, 0.40):
                    streams.insert(0, ("strict_fault_applied_available", strict))
                for stream, take in streams:
                    pb = pb_all[take]
                    pl = pl_all[take]
                    choose = choose_all[take]
                    y = test_y[take]
                    selected = np.where(choose[:, None], pl, pb)
                    base_correct = pb.argmax(axis=1) == y
                    lite_correct = pl.argmax(axis=1) == y
                    for method, probability in (
                        ("PDRF", pb),
                        ("RO-PDRF-Lite", pl),
                        ("Lite-CF-6", selected),
                    ):
                        metric_rows.append(
                            {
                                "seed": seed,
                                "fault_type": fault,
                                "fault_prevalence": prevalence,
                                "stream": stream,
                                "method": method,
                                "n": len(y),
                                **probability_metrics(y, probability),
                            }
                        )
                    safety_rows.append(
                        {
                            "seed": seed,
                            "fault_type": fault,
                            "fault_prevalence": prevalence,
                            "stream": stream,
                            "method": "Lite-CF-6",
                            "mean_forward_passes": 2.0,
                            **selection_metrics(base_correct, lite_correct, choose),
                        }
                    )
                    if stream == "strict_fault_applied_available":
                        add_prediction_rows(
                            prediction_rows,
                            seed,
                            fault,
                            stream,
                            y,
                            test_batch[take],
                            np.flatnonzero(take),
                            pb,
                            pl,
                            choose,
                        )

        clean_rng = np.random.default_rng(TEST_REALIZATION)
        clean_mask = base.random_mask(len(test_x), clean_rng, 0.20)
        clean_x = test_x.copy() * clean_mask[:, :, None]
        clean_quality = np.ones((len(test_x), clean_mask.shape[1]), np.float32)
        base_timing = benchmark(
            lambda: predict(
                base_model, clean_x, clean_mask, clean_quality, base_temperature
            )
        )
        lite_timing = benchmark(
            lambda: predict(
                lite_model, clean_x, clean_mask, clean_quality, lite_temperature
            )
        )
        routed_timing = benchmark(
            lambda: route_probability(
                base_model,
                lite_model,
                clean_x,
                clean_mask,
                clean_quality,
                base_temperature,
                lite_temperature,
                fit["model"],
                fit["threshold"],
            )
        )
        base_timing_one = benchmark(
            lambda: predict(
                base_model,
                clean_x[:1],
                clean_mask[:1],
                clean_quality[:1],
                base_temperature,
            )
        )
        lite_timing_one = benchmark(
            lambda: predict(
                lite_model,
                clean_x[:1],
                clean_mask[:1],
                clean_quality[:1],
                lite_temperature,
            )
        )
        routed_timing_one = benchmark(
            lambda: route_probability(
                base_model,
                lite_model,
                clean_x[:1],
                clean_mask[:1],
                clean_quality[:1],
                base_temperature,
                lite_temperature,
                fit["model"],
                fit["threshold"],
            )
        )
        for method, timing, timing_one, parameters, passes in (
            ("PDRF", base_timing, base_timing_one, base_cost["parameters"], 1),
            (
                "RO-PDRF-Lite",
                lite_timing,
                lite_timing_one,
                lite_cost["parameters"],
                1,
            ),
            (
                "Lite-CF-6",
                routed_timing,
                routed_timing_one,
                base_cost["parameters"]
                + lite_cost["parameters"]
                + len(FEATURES_6)
                + 1,
                2,
            ),
        ):
            median, q1, q3 = timing
            median_one, q1_one, q3_one = timing_one
            latency_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "batch_size": len(clean_x),
                    "cpu_model": cpu_model_name(),
                    "cpu_threads": torch.get_num_threads(),
                    "cpu_interop_threads": torch.get_num_interop_threads(),
                    "torch_version": torch.__version__,
                    "latency_batch_ms_median": 1000.0 * median,
                    "latency_batch_ms_q1": 1000.0 * q1,
                    "latency_batch_ms_q3": 1000.0 * q3,
                    "throughput_observations_per_second": len(clean_x) / median,
                    "latency_batch1_ms_median": 1000.0 * median_one,
                    "latency_batch1_ms_q1": 1000.0 * q1_one,
                    "latency_batch1_ms_q3": 1000.0 * q3_one,
                    "forward_passes_per_observation": passes,
                    "stored_parameters": parameters,
                    "model_state_kib_fp32": 4.0 * parameters / 1024.0,
                }
            )
        clean_pb = predict(
            base_model, clean_x, clean_mask, clean_quality, base_temperature
        )
        clean_pl = predict(
            lite_model, clean_x, clean_mask, clean_quality, lite_temperature
        )
        clean_score = fit["model"].predict_proba(
            pd.DataFrame(features6(clean_pb, clean_pl), columns=FEATURES_6)
        )[:, 1]
        clean_choose = clean_score >= fit["threshold"]
        clean_selected = np.where(clean_choose[:, None], clean_pl, clean_pb)
        for method, probability in (
            ("PDRF", clean_pb),
            ("RO-PDRF-Lite", clean_pl),
            ("Lite-CF-6", clean_selected),
        ):
            metric_rows.append(
                {
                    "seed": seed,
                    "fault_type": "none",
                    "fault_prevalence": 0.0,
                    "stream": "clean_no_imposed_fault",
                    "method": method,
                    "n": len(test_y),
                    **probability_metrics(test_y, probability),
                }
            )
        safety_rows.append(
            {
                "seed": seed,
                "fault_type": "none",
                "fault_prevalence": 0.0,
                "stream": "clean_no_imposed_fault",
                "method": "Lite-CF-6",
                "mean_forward_passes": 2.0,
                **selection_metrics(
                    clean_pb.argmax(axis=1) == test_y,
                    clean_pl.argmax(axis=1) == test_y,
                    clean_choose,
                ),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    safety = pd.DataFrame(safety_rows)
    metrics.to_csv(OUT / "q1_lite_routing_metrics.csv", index=False)
    safety.to_csv(OUT / "q1_lite_routing_safety.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        OUT / "q1_lite_routing_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(calibration_rows).to_csv(
        OUT / "q1_lite_selector_calibration.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        OUT / "q1_lite_selector_diagnostics.csv", index=False
    )
    pd.DataFrame(cost_rows).to_csv(OUT / "q1_lite_endpoint_costs.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(OUT / "q1_lite_cpu_latency.csv", index=False)
    summary = (
        metrics[metrics.stream == "strict_fault_applied_available"]
        .groupby("method")[[
            "accuracy",
            "macro_auroc",
            "macro_auprc",
            "nll",
            "brier",
            "ece15_equal_width",
        ]]
        .agg(["mean", "std"])
    )
    summary.to_csv(OUT / "q1_lite_routing_summary.csv")
    (OUT / "q1_lite_routing_design.json").write_text(
        json.dumps(
            {
                "run_id": "CEE-CF10-R2-LITE",
                "source_run": "CEE-CF10-R2",
                "seeds": list(SEEDS),
                "faults": list(FAULTS),
                "fault_prevalences": list(PREVALENCES),
                "test_realization": TEST_REALIZATION,
                "selector_features": FEATURES_6,
                "selector_C": 0.05,
                "selector_training": "correctness disagreements in Batch-7 selector half; grouped OOF threshold",
                "temperature": "one scalar per endpoint and seed fitted on the disjoint first half of Batch 7",
                "post_routing_recalibration": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary)


if __name__ == "__main__":
    main()
