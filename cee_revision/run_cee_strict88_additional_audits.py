"""Additional fixed-protocol audits requested in the Strict-88 review.

The script reuses the frozen CEE-CF10-R2 split, endpoint specifications,
temperature partitions, selector calibration and test realization.  It adds
only (i) a scale-1 strict-subset evaluation and (ii) fixed intra-op CPU-thread
timing.  Neither audit is used to select a model, threshold or headline run.
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
    probability_metrics,
    selection_metrics,
)
from run_cee_lite_routing_validation import (
    benchmark,
    calibration_frame,
    cpu_model_name,
    features6,
    predict,
    route_probability,
)


OUT = ROOT / "source_data"
SEEDS = tuple(range(101, 111))
FAULTS = ("gaussian", "offset", "drift", "stuck_at")
TEST_REALIZATION = 70001
TEST_SCALE = 1.0
TEST_PREVALENCE = 0.40
EPOCHS = 60
THREAD_SETTINGS = (1, 12)
LITE_SPEC = replace(
    major.SPECS["PDRF"],
    recovery_distillation=True,
    w_recovery=0.30,
    recovery_teacher="clean",
)


def processor_name() -> str:
    """Return the same CPU label as the primary timing script."""
    return cpu_model_name() or platform.processor() or "unavailable"


def main() -> None:
    started = time.perf_counter()
    split = q2.prepare_grouped(
        base.ensure_zip(None),
        [list(range(index, index + 4)) for index in range(0, 16, 4)],
    )
    test_x, test_y, _ = split["test"]
    calibration_x, calibration_y, _ = split["calibration"]
    cut = len(calibration_x) // 2
    temperature_x, temperature_y = calibration_x[:cut], calibration_y[:cut]
    selector_x, selector_y = calibration_x[cut:], calibration_y[cut:]

    metric_rows: list[dict] = []
    safety_rows: list[dict] = []
    latency_rows: list[dict] = []
    original_threads = torch.get_num_threads()

    for seed in SEEDS:
        train, select, _, class_weights = q2.train_views(split, seed)
        temperature_view = (
            *base.make_view(
                temperature_x, 25000 + seed, "missing_drift", train=True
            ),
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
        fit = grouped_binary_fit(
            calibration, FEATURES_6, seed + 5000, c_value=0.05
        )

        for fault in FAULTS:
            fault_x, mask, quality, assigned = typed.fixed_typed_fault(
                test_x,
                fault,
                scale=TEST_SCALE,
                prevalence=TEST_PREVALENCE,
                seed=TEST_REALIZATION,
            )
            strict = assigned & (mask[:, 0] > 0)
            pb = predict(
                base_model,
                fault_x[strict],
                mask[strict],
                quality[strict],
                base_temperature,
            )
            pl = predict(
                lite_model,
                fault_x[strict],
                mask[strict],
                quality[strict],
                lite_temperature,
            )
            score = fit["model"].predict_proba(
                pd.DataFrame(features6(pb, pl), columns=FEATURES_6)
            )[:, 1]
            choose = score >= fit["threshold"]
            selected = np.where(choose[:, None], pl, pb)
            y = test_y[strict]
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
                        "fault_scale": TEST_SCALE,
                        "fault_prevalence": TEST_PREVALENCE,
                        "stream": "strict_fault_applied_available",
                        "method": method,
                        "n": len(y),
                        **probability_metrics(y, probability),
                    }
                )
            safety_rows.append(
                {
                    "seed": seed,
                    "fault_type": fault,
                    "fault_scale": TEST_SCALE,
                    "fault_prevalence": TEST_PREVALENCE,
                    "stream": "strict_fault_applied_available",
                    "method": "Lite-CF-6",
                    "mean_forward_passes": 2.0,
                    **selection_metrics(base_correct, lite_correct, choose),
                }
            )

        clean_rng = np.random.default_rng(TEST_REALIZATION)
        clean_mask = base.random_mask(len(test_x), clean_rng, 0.20)
        clean_x = test_x.copy() * clean_mask[:, :, None]
        clean_quality = np.ones(
            (len(test_x), clean_mask.shape[1]), dtype=np.float32
        )
        for requested_threads in THREAD_SETTINGS:
            torch.set_num_threads(requested_threads)
            base_timing = benchmark(
                lambda: predict(
                    base_model,
                    clean_x,
                    clean_mask,
                    clean_quality,
                    base_temperature,
                )
            )
            lite_timing = benchmark(
                lambda: predict(
                    lite_model,
                    clean_x,
                    clean_mask,
                    clean_quality,
                    lite_temperature,
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
                (
                    "PDRF",
                    base_timing,
                    base_timing_one,
                    base_cost["parameters"],
                    1,
                ),
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
                        "requested_intraop_threads": requested_threads,
                        "actual_intraop_threads": torch.get_num_threads(),
                        "interop_threads": torch.get_num_interop_threads(),
                        "cpu_model": processor_name(),
                        "torch_version": torch.__version__,
                        "batch_size": len(clean_x),
                        "latency_batch_ms_median": 1000.0 * median,
                        "latency_batch_ms_q1": 1000.0 * q1,
                        "latency_batch_ms_q3": 1000.0 * q3,
                        "throughput_observations_per_second": len(clean_x)
                        / median,
                        "latency_batch1_ms_median": 1000.0 * median_one,
                        "latency_batch1_ms_q1": 1000.0 * q1_one,
                        "latency_batch1_ms_q3": 1000.0 * q3_one,
                        "forward_passes_per_observation": passes,
                        "stored_parameters": parameters,
                    }
                )

    torch.set_num_threads(original_threads)
    metrics = pd.DataFrame(metric_rows)
    safety = pd.DataFrame(safety_rows)
    latency = pd.DataFrame(latency_rows)
    metrics.to_csv(OUT / "strict88_mild_scale1_metrics.csv", index=False)
    safety.to_csv(OUT / "strict88_mild_scale1_safety.csv", index=False)
    latency.to_csv(OUT / "strict88_cpu_thread_sensitivity.csv", index=False)
    (OUT / "strict88_additional_audits_design.json").write_text(
        json.dumps(
            {
                "source_run": "CEE-CF10-R2-LITE",
                "optimization_seeds": list(SEEDS),
                "fault_mechanisms": list(FAULTS),
                "mild_severity": TEST_SCALE,
                "fault_prevalence": TEST_PREVALENCE,
                "test_realization": TEST_REALIZATION,
                "strict_subset": "fault assigned and affected group available",
                "selector_calibration_severity": 3.0,
                "selector_refit_or_retuned_on_scale1": False,
                "thread_settings": list(THREAD_SETTINGS),
                "timing_warmups": 3,
                "timing_repetitions": 7,
                "data_loading_included": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        metrics.groupby("method")[["accuracy", "macro_auroc"]]
        .mean()
        .round(6)
    )
    print(
        safety[
            [
                "negative_transfer_prevention",
                "recovery_retention",
                "recovery_selection_rate",
                "net_correct_change_per_10000",
            ]
        ]
        .mean()
        .round(6)
    )


if __name__ == "__main__":
    main()
