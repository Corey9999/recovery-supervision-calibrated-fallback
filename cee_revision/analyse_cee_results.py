"""Create manuscript-ready tables and a single audited fact ledger."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)


def pm(mean, std, digits=4):
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def ci(mean, lo, hi, digits=3):
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def save_latex(frame, name, column_format=None):
    text = frame.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        na_rep="--",
    )
    text = text.replace("%", "\\%")
    (TABLES / name).write_text(text, encoding="utf-8")


def main():
    metrics = pd.read_csv(DATA / "cee_cf10_stream_metrics.csv")
    safety = pd.read_csv(DATA / "cee_cf10_stream_safety.csv")
    diagnostics = pd.read_csv(DATA / "cee_cf10_selector_diagnostics.csv")
    paired = pd.read_csv(DATA / "cee_cf10_paired_effects.csv")
    simple = pd.read_csv(DATA / "cee_cf10_simple_rule_test.csv")
    lofo = pd.read_csv(DATA / "cee_cf10_lofo.csv")
    unseen = pd.read_csv(DATA / "cee_cf10_unseen_faults.csv")
    cost = pd.read_csv(DATA / "cee_cf10_cpu_cost.csv")
    scaling = pd.read_csv(DATA / "cee_cf10_group_scaling.csv")
    selector_parameters = pd.read_csv(DATA / "cee_cf10_selector_parameters.csv")
    reliability = pd.read_csv(DATA / "cee_cf10_selector_reliability.csv")
    intervals = pd.read_csv(DATA / "cee_cf10_hierarchical_intervals.csv")
    design = json.loads((DATA / "cee_cf10_design.json").read_text(encoding="utf-8"))

    strict = metrics[
        (metrics.prevalence == 0.40)
        & (metrics.stream == "strict_fault_applied_available")
        & (metrics.metric == "macro_auroc")
    ]
    strict_summary = strict.groupby(["fault_type", "method"]).value.agg(["mean", "std"]).reset_index()
    strict_pivot = strict_summary.pivot(index="fault_type", columns="method", values=["mean", "std"])
    strict_rows = []
    labels = {
        "gaussian": "Gaussian noise",
        "offset": "Offset",
        "drift": "Drift",
        "stuck_at": "Stuck-at",
    }
    for fault in ("gaussian", "offset", "drift", "stuck_at"):
        base_mean = strict_pivot.loc[fault, ("mean", "PDRF")]
        full_mean = strict_pivot.loc[fault, ("mean", "RO-PDRF-Full")]
        safe_mean = strict_pivot.loc[fault, ("mean", "SR-PDRF-Safe-CF")]
        strict_rows.append(
            {
                "Fault mechanism": labels[fault],
                "PDRF": pm(base_mean, strict_pivot.loc[fault, ("std", "PDRF")]),
                "RO-PDRF-Full": pm(full_mean, strict_pivot.loc[fault, ("std", "RO-PDRF-Full")]),
                "Safe-CF": pm(safe_mean, strict_pivot.loc[fault, ("std", "SR-PDRF-Safe-CF")]),
                "Full--PDRF": f"{full_mean-base_mean:+.4f}",
                "Safe--PDRF": f"{safe_mean-base_mean:+.4f}",
            }
        )
    save_latex(pd.DataFrame(strict_rows), "cee_strict_primary.tex", "lccccc")

    stream_specs = [
        ("Clean / no imposed fault", "clean_no_imposed_fault", 0.0),
        ("Mixed stream, 10% prevalence", "full_mixed_stream", 0.10),
        ("Mixed stream, 40% prevalence", "full_mixed_stream", 0.40),
        ("Mixed stream, 70% prevalence", "full_mixed_stream", 0.70),
        ("Assigned but masked, 40%", "assigned_but_masked", 0.40),
        ("Unaffected rows, 40%", "unaffected", 0.40),
    ]
    stream_rows = []
    for label, stream_name, prevalence in stream_specs:
        part = metrics[
            (metrics.stream == stream_name)
            & (metrics.prevalence == prevalence)
            & (metrics.metric == "macro_auroc")
        ]
        row = {"Evaluation stream": label}
        for method, short in (
            ("PDRF", "PDRF"),
            ("RO-PDRF-Full", "Full"),
            ("SR-PDRF-Safe-CF", "Safe-CF"),
        ):
            values = part.loc[part.method == method, "value"]
            row[short] = pm(values.mean(), values.std())
        stream_rows.append(row)
    save_latex(pd.DataFrame(stream_rows), "cee_deployment_streams.tex", "lccc")

    safe_strict = safety[
        (safety.prevalence == 0.40)
        & (safety.stream == "full_mixed_stream")
        & (safety.variant == "Safe-CF")
    ]
    clean_safe = safety[
        (safety.stream == "clean_no_imposed_fault")
        & (safety.variant == "Safe-CF")
    ]
    interval_map = intervals.set_index("quantity").to_dict("index")
    diag_rows = [
        {
            "Quantity": "OOF selector AUROC",
            "Estimate": f"{diagnostics.selector_auroc.mean():.3f} $\\pm$ {diagnostics.selector_auroc.std():.3f}",
        },
        {
            "Quantity": "OOF selector AUPRC",
            "Estimate": f"{diagnostics.selector_auprc.mean():.3f} $\\pm$ {diagnostics.selector_auprc.std():.3f}",
        },
        {
            "Quantity": "OOF selector Brier score",
            "Estimate": f"{diagnostics.selector_brier.mean():.3f} $\\pm$ {diagnostics.selector_brier.std():.3f}",
        },
        {
            "Quantity": "Safe threshold, median (range)",
            "Estimate": f"{diagnostics.safe_threshold.median():.3f} ({diagnostics.safe_threshold.min():.3f}--{diagnostics.safe_threshold.max():.3f})",
        },
        {
            "Quantity": "Negative-transfer prevention",
            "Estimate": ci(
                interval_map["negative_transfer_prevention"]["bootstrap_mean"],
                interval_map["negative_transfer_prevention"]["ci_2.5"],
                interval_map["negative_transfer_prevention"]["ci_97.5"],
            ),
        },
        {
            "Quantity": "Recovery retention",
            "Estimate": ci(
                interval_map["recovery_retention"]["bootstrap_mean"],
                interval_map["recovery_retention"]["ci_2.5"],
                interval_map["recovery_retention"]["ci_97.5"],
            ),
        },
        {
            "Quantity": "Clean false-switch rate",
            "Estimate": f"{clean_safe.false_switch_rate.mean():.3f} $\\pm$ {clean_safe.false_switch_rate.std():.3f}",
        },
        {
            "Quantity": "Clean harmful-switch rate",
            "Estimate": f"{clean_safe.harmful_switch_rate.mean():.3f} $\\pm$ {clean_safe.harmful_switch_rate.std():.3f}",
        },
    ]
    save_latex(pd.DataFrame(diag_rows), "cee_selector_diagnostics.tex", "lp{0.34\\textwidth}")

    transfer_rows = []
    for label, frame, fault_col in (
        ("Leave-one-family-out", lofo, "held_out_fault"),
        ("Unseen mechanism", unseen, "unseen_fault"),
    ):
        for fault, group in frame.groupby(fault_col):
            transfer_rows.append(
                {
                    "Audit": label,
                    "Held-out mechanism": labels.get(fault, fault.replace("_", " ").title()),
                    "Safe--PDRF": f"{(group.safe_macro_auroc-group.pdrf_macro_auroc).mean():+.4f}",
                    "Safe--Full": f"{(group.safe_macro_auroc-group.full_macro_auroc).mean():+.4f}",
                    "Prevention": f"{group.negative_transfer_prevention.mean():.3f}",
                    "Retention": f"{group.recovery_retention.mean():.3f}",
                }
            )
    save_latex(pd.DataFrame(transfer_rows), "cee_unseen_transfer.tex", "llcccc")

    simple_order = [
        ("Cross-fitted-logistic-Safe", "proposed"),
        ("Higher-confidence", "prevention_matched"),
        ("Lower-entropy", "prevention_matched"),
        ("Lower-LOO-disagreement", "prevention_matched"),
        ("Higher-JS-disagreement", "prevention_matched"),
        ("Random-matched-selection", "selection_rate_matched"),
        ("Always-PDRF", "endpoint"),
        ("Always-RO-PDRF-Full", "endpoint"),
        ("Oracle", "upper_bound"),
    ]
    simple_rows = []
    for rule, matching in simple_order:
        group = simple[(simple.rule == rule) & (simple.matching == matching)]
        if group.empty:
            continue
        simple_rows.append(
            {
                "Selection rule": ("Decision oracle" if rule == "Oracle" else rule.replace("-", " ")),
                "Matching": ("label informed" if matching == "upper_bound" else matching.replace("_", " ")),
                "Prevention": f"{group.negative_transfer_prevention.mean():.3f}",
                "Retention": f"{group.recovery_retention.mean():.3f}",
                "Full use": f"{group.full_selection_rate.mean():.3f}",
                "Macro-AUROC": pm(group.macro_auroc.mean(), group.macro_auroc.std()),
            }
        )
    save_latex(pd.DataFrame(simple_rows), "cee_simple_baselines.tex", "llcccc")

    group4 = scaling.loc[scaling.groups == 4].iloc[0]
    flops = {
        "PDRF": int(group4.pdrf_flops),
        "RO-PDRF-Full": int(group4.pdrf_flops),
        "SR-PDRF-Safe-CF": int(group4.inference_flops),
    }
    passes = {"PDRF": 1, "RO-PDRF-Full": 1, "SR-PDRF-Safe-CF": int(group4.forward_passes)}
    cost_rows = []
    for method in ("PDRF", "RO-PDRF-Full", "SR-PDRF-Safe-CF"):
        group = cost[cost.method == method]
        cost_rows.append(
            {
                "Method": method,
                "Parameters": f"{int(round(group.stored_parameters.mean())):,}",
                "State (KiB)": f"{group.model_state_kib_fp32.mean():.1f}",
                "Passes": passes[method],
                "FLOPs / sample": f"{flops[method]:,}",
                "CPU ms / sample": f"{group.latency_ms_per_observation.median():.4f}",
            }
        )
    save_latex(pd.DataFrame(cost_rows), "cee_compute_cost.tex", "lrrrrr")

    parameter_labels = {
        "base_confidence": "Base confidence",
        "recovery_confidence": "Recovery confidence",
        "confidence_delta": "Confidence difference",
        "base_entropy": "Base normalized entropy",
        "recovery_entropy": "Recovery normalized entropy",
        "entropy_delta": "Entropy difference",
        "base_recovery_js": "Base--recovery JS divergence",
        "base_consensus_js": "Base--LOO-consensus JS",
        "recovery_consensus_js": "Recovery--LOO-consensus JS",
        "base_consensus_agreement": "Base--LOO class agreement",
        "recovery_consensus_agreement": "Recovery--LOO class agreement",
        "removal_fraction": "Legal-removal fraction",
    }
    parameter_rows = []
    for feature, group in selector_parameters.groupby("feature", sort=False):
        parameter_rows.append(
            {
                "Feature": parameter_labels.get(feature, feature),
                "Center": pm(group.standardization_mean.mean(), group.standardization_mean.std(), 3),
                "Scale": pm(group.standardization_scale.mean(), group.standardization_scale.std(), 3),
                "Coefficient": pm(group.logistic_coefficient.mean(), group.logistic_coefficient.std(), 3),
            }
        )
    save_latex(pd.DataFrame(parameter_rows), "cee_selector_parameters.tex", "lccc")

    seed_rows = diagnostics[
        [
            "seed",
            "calibration_rows",
            "correctness_disagreements",
            "safe_threshold",
            "balanced_threshold",
            "selector_auroc",
            "selector_auprc",
            "selector_brier",
        ]
    ].copy()
    seed_rows.columns = ["Seed", "Calibration n", "Disagreement n", "Safe threshold", "Balanced threshold", "AUROC", "AUPRC", "Brier"]
    for column in ("Safe threshold", "Balanced threshold", "AUROC", "AUPRC", "Brier"):
        seed_rows[column] = seed_rows[column].map(lambda value: f"{value:.3f}")
    save_latex(seed_rows, "cee_selector_seed_diagnostics.tex", "rrrrrrrr")

    reliability_rows = []
    for bin_id, group in reliability.groupby("bin"):
        group = group[(group.n > 0) & group.mean_probability.notna() & group.observed_recovery_preference.notna()]
        total = int(group.n.sum())
        reliability_rows.append(
            {
                "Probability bin": f"{group.lower.min():.1f}--{group.upper.max():.1f}",
                "n": total,
                "Mean probability": f"{np.average(group.mean_probability, weights=group.n):.3f}",
                "Observed recovery preference": f"{np.average(group.observed_recovery_preference, weights=group.n):.3f}",
            }
        )
    save_latex(pd.DataFrame(reliability_rows), "cee_selector_reliability.tex", "lrrr")

    scaling_rows = []
    for row in scaling.itertuples(index=False):
        scaling_rows.append(
            {
                "Groups": int(row.groups),
                "PDRF FLOPs": f"{int(row.pdrf_flops):,}",
                "Safe-CF passes": int(row.forward_passes),
                "Safe-CF FLOPs": f"{int(row.inference_flops):,}",
            }
        )
    save_latex(pd.DataFrame(scaling_rows), "cee_group_scaling.tex", "rrrr")

    manifest = pd.DataFrame(
        [
            ["Run identifier", design["run_id"]],
            ["Optimization seeds", "101--110"],
            ["Training rerun", design["training_rerun_id"]],
            ["Test realization", str(design["test_realization"])],
            ["Temperature protocol", "First chronological half of Batch 7"],
            ["Selector protocol", "Second half; five-fold sample-group cross-fitting"],
            ["Checkpoint reuse", "One PDRF/Full pair reused across all selector audits per seed"],
        ],
        columns=["Item", "Frozen setting"],
    )
    save_latex(manifest, "cee_run_manifest.tex", "lp{0.62\\textwidth}")

    gaussian_effects = paired[paired.fault_type == "gaussian"].set_index("comparison")
    all_fault = strict.pivot_table(
        index=["fault_type", "seed"], columns="method", values="value"
    )
    all_full_minus_base = all_fault["RO-PDRF-Full"] - all_fault["PDRF"]
    all_safe_minus_base = all_fault["SR-PDRF-Safe-CF"] - all_fault["PDRF"]
    facts = {
        "run_id": design["run_id"],
        "strict_n_per_fault": int(
            metrics[
                (metrics.prevalence == 0.40)
                & (metrics.stream == "strict_fault_applied_available")
            ].n.mode().iloc[0]
        ),
        "gaussian_full_minus_pdrf": gaussian_effects.loc["Full_minus_PDRF"].to_dict(),
        "gaussian_safe_minus_pdrf": gaussian_effects.loc["Safe_minus_PDRF"].to_dict(),
        "four_fault_full_minus_pdrf_mean": float(all_full_minus_base.mean()),
        "four_fault_safe_minus_pdrf_mean": float(all_safe_minus_base.mean()),
        "hierarchical_intervals": interval_map,
        "selector": {
            "calibration_rows_mean": float(diagnostics.calibration_rows.mean()),
            "disagreement_rows_mean": float(diagnostics.correctness_disagreements.mean()),
            "safe_threshold_median": float(diagnostics.safe_threshold.median()),
            "safe_threshold_range": [float(diagnostics.safe_threshold.min()), float(diagnostics.safe_threshold.max())],
            "oof_auroc_mean": float(diagnostics.selector_auroc.mean()),
            "oof_auprc_mean": float(diagnostics.selector_auprc.mean()),
            "oof_brier_mean": float(diagnostics.selector_brier.mean()),
        },
        "clean": {
            "false_switch_rate_mean": float(clean_safe.false_switch_rate.mean()),
            "harmful_switch_rate_mean": float(clean_safe.harmful_switch_rate.mean()),
        },
        "lofo": {
            "safe_minus_pdrf_mean": float((lofo.safe_macro_auroc-lofo.pdrf_macro_auroc).mean()),
            "safe_minus_full_mean": float((lofo.safe_macro_auroc-lofo.full_macro_auroc).mean()),
            "prevention_mean": float(lofo.negative_transfer_prevention.mean()),
            "retention_mean": float(lofo.recovery_retention.mean()),
        },
        "unseen": {
            "safe_minus_pdrf_mean": float((unseen.safe_macro_auroc-unseen.pdrf_macro_auroc).mean()),
            "safe_minus_full_mean": float((unseen.safe_macro_auroc-unseen.full_macro_auroc).mean()),
            "prevention_mean": float(unseen.negative_transfer_prevention.mean()),
            "retention_mean": float(unseen.recovery_retention.mean()),
        },
        "cost": {
            method: {
                "latency_ms_per_observation_median": float(cost.loc[cost.method == method, "latency_ms_per_observation"].median()),
                "stored_parameters": int(round(cost.loc[cost.method == method, "stored_parameters"].mean())),
                "flops_per_sample": flops[method],
                "forward_passes": passes[method],
            }
            for method in ("PDRF", "RO-PDRF-Full", "SR-PDRF-Safe-CF")
        },
    }
    (DATA / "cee_manuscript_facts.json").write_text(
        json.dumps(facts, indent=2), encoding="utf-8"
    )
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
