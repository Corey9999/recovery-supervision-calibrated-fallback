"""Statistical synthesis for the strict-estimand, safety and modern-baseline upgrade."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
SRC = ROOT/"source_data"
TAB = ROOT/"tables"
TAB.mkdir(exist_ok=True)


def paired_p(delta):
    delta = np.asarray(delta, float)
    return 1.0 if np.allclose(delta, 0) else float(
        wilcoxon(delta, alternative="two-sided", method="exact").pvalue)


def crossed_bootstrap(pivot, reps=5000, seed=99009):
    """Resample fault types, then paired optimization-seed columns."""
    rng = np.random.default_rng(seed)
    matrix = pivot.to_numpy(float)
    out = np.empty(reps)
    for i in range(reps):
        faults = rng.integers(0, matrix.shape[0], matrix.shape[0])
        values = []
        for fault in faults:
            seeds = rng.integers(0, matrix.shape[1], matrix.shape[1])
            values.extend(matrix[fault, seeds])
        out[i] = np.mean(values)
    return out


def effects(frame, contrasts, output):
    z = frame[(frame.subset == "fault_applied_available") &
              (frame.metric == "macro_auroc")]
    rows = []
    for label, left, right in contrasts:
        pivot = z[z.method.isin([left, right])].pivot_table(
            index="fault_type", columns=["method", "seed"], values="value")
        faults = sorted(pivot.index)
        cube = []
        for fault in faults:
            lv = z[(z.fault_type == fault) & (z.method == left)].set_index("seed").value
            rv = z[(z.fault_type == fault) & (z.method == right)].set_index("seed").value
            common = lv.index.intersection(rv.index)
            delta = lv.loc[common]-rv.loc[common]
            cube.append(delta.to_numpy())
            rows.append({
                "scope": fault, "contrast": label, "left": left, "right": right,
                "mean_difference": float(delta.mean()),
                "sd_difference": float(delta.std(ddof=1)),
                "wins": int((delta > 0).sum()), "n": int(len(delta)),
                "wilcoxon_p": paired_p(delta),
                "ci_low": np.nan, "ci_high": np.nan,
                "inference": "optimization-seed variability only",
            })
        cube = pd.DataFrame(np.vstack(cube), index=faults)
        boot = crossed_bootstrap(cube)
        rows.append({
            "scope": "four_fault_types", "contrast": label,
            "left": left, "right": right,
            "mean_difference": float(cube.to_numpy().mean()),
            "sd_difference": float(cube.to_numpy().std(ddof=1)),
            "wins": int((cube.mean(axis=1) > 0).sum()), "n": len(faults),
            "wilcoxon_p": np.nan,
            "ci_low": float(np.quantile(boot, .025)),
            "ci_high": float(np.quantile(boot, .975)),
            "inference": "descriptive crossed fault-type and seed bootstrap; four top-level fault types",
        })
    result = pd.DataFrame(rows)
    result.to_csv(SRC/output, index=False)
    return result


def safety_summary():
    p = pd.read_csv(SRC/"major9_safe_recovery_predictions.csv")
    rows = []
    for fault_type, group in p.groupby("fault_type"):
        recovery_opportunity = ~group.base_correct & group.recovery_correct
        transfer_opportunity = group.base_correct & ~group.recovery_correct
        for variant, correct, choose in (
            ("Balanced", "balanced_correct", "choose_recovery_balanced"),
            ("Safe", "safe_correct", "choose_recovery_safe"),
        ):
            rows.append({
                "scope": fault_type, "variant": variant,
                "n_predictions": len(group),
                "recovery_opportunities": int(recovery_opportunity.sum()),
                "recovery_preserved": int((recovery_opportunity & group[correct]).sum()),
                "recovery_preservation_rate": float(
                    (recovery_opportunity & group[correct]).sum()/max(1, recovery_opportunity.sum())),
                "negative_transfer_opportunities": int(transfer_opportunity.sum()),
                "negative_transfer_remaining": int((transfer_opportunity & ~group[correct]).sum()),
                "negative_transfer_remaining_rate": float(
                    (transfer_opportunity & ~group[correct]).sum()/max(1, transfer_opportunity.sum())),
                "negative_transfer_prevented_rate": float(
                    (transfer_opportunity & group[correct]).sum()/max(1, transfer_opportunity.sum())),
                "recovery_usage": float(group[choose].mean()),
            })
    frame = pd.DataFrame(rows)
    totals = []
    for variant, group in frame.groupby("variant"):
        totals.append({
            "scope": "four_fault_types", "variant": variant,
            "n_predictions": int(group.n_predictions.sum()),
            "recovery_opportunities": int(group.recovery_opportunities.sum()),
            "recovery_preserved": int(group.recovery_preserved.sum()),
            "recovery_preservation_rate": float(
                group.recovery_preserved.sum()/group.recovery_opportunities.sum()),
            "negative_transfer_opportunities": int(group.negative_transfer_opportunities.sum()),
            "negative_transfer_remaining": int(group.negative_transfer_remaining.sum()),
            "negative_transfer_remaining_rate": float(
                group.negative_transfer_remaining.sum()/group.negative_transfer_opportunities.sum()),
            "negative_transfer_prevented_rate": float(
                1-group.negative_transfer_remaining.sum()/group.negative_transfer_opportunities.sum()),
            "recovery_usage": float(np.average(group.recovery_usage, weights=group.n_predictions)),
        })
    frame = pd.concat([frame, pd.DataFrame(totals)], ignore_index=True)
    frame.to_csv(SRC/"major9_safe_recovery_summary.csv", index=False)
    return frame


def write_tables(strict, safe, modern, safety):
    strict_show = strict[(strict.subset == "fault_applied_available") &
                         (strict.contrast.isin(["RO-PDRF - PDRF", "RO-PDRF - RO-CAGF"]))]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Contrast & Mean $\Delta$ AUROC & SD & Wins/10 & $P$ \\", r"\midrule"]
    for _, x in strict_show.iterrows():
        lines.append(f"{x.contrast} & {x.mean_difference:+.4f} & {x.sd_difference:.4f} & "
                     f"{int(x.wins)}/10 & {x.wilcoxon_p:.4f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB/"major9_strict_estimand.tex").write_text("\n".join(lines), encoding="utf-8")

    z = pd.read_csv(SRC/"major9_safe_recovery_metrics.csv")
    z = z[(z.subset == "fault_applied_available") & (z.metric == "macro_auroc")]
    mean = z.groupby(["fault_type", "method"]).value.agg(["mean", "std"]).reset_index()
    order = ["PDRF", "RO-PDRF-Full", "SR-PDRF-Balanced", "SR-PDRF-Safe"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             "Fault & " + " & ".join(order) + r" \\", r"\midrule"]
    for fault in ["gaussian", "offset", "drift", "stuck_at"]:
        values = []
        for method in order:
            x = mean[(mean.fault_type == fault) & (mean.method == method)].iloc[0]
            values.append(f"{x['mean']:.3f} $\\pm$ {x['std']:.3f}")
        lines.append(fault.replace("_", "-") + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB/"major9_safe_recovery.tex").write_text("\n".join(lines), encoding="utf-8")

    z = pd.read_csv(SRC/"major9_modern_baseline_metrics.csv")
    z = z[(z.subset == "fault_applied_available") & (z.metric == "macro_auroc")]
    mean = z.groupby(["fault_type", "method"]).value.agg(["mean", "std"]).reset_index()
    order = ["RO-CAGF", "RO-MER", "RO-PDRF-Full"]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             "Fault & " + " & ".join(order) + r" \\", r"\midrule"]
    for fault in ["gaussian", "offset", "drift", "stuck_at"]:
        values = []
        for method in order:
            x = mean[(mean.fault_type == fault) & (mean.method == method)].iloc[0]
            values.append(f"{x['mean']:.3f} $\\pm$ {x['std']:.3f}")
        lines.append(fault.replace("_", "-") + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB/"major9_modern_baseline.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    strict = pd.read_csv(SRC/"major9_estimand_effects.csv")
    safe_metrics = pd.read_csv(SRC/"major9_safe_recovery_metrics.csv")
    modern_metrics = pd.read_csv(SRC/"major9_modern_baseline_metrics.csv")
    safe = effects(safe_metrics, [
        ("Full - PDRF", "RO-PDRF-Full", "PDRF"),
        ("Balanced - Full", "SR-PDRF-Balanced", "RO-PDRF-Full"),
        ("Safe - Full", "SR-PDRF-Safe", "RO-PDRF-Full"),
        ("Safe - PDRF", "SR-PDRF-Safe", "PDRF"),
    ], "major9_safe_recovery_effects.csv")
    modern = effects(modern_metrics, [
        ("Full - MER", "RO-PDRF-Full", "RO-MER"),
        ("MER - CAGF", "RO-MER", "RO-CAGF"),
        ("Full - CAGF", "RO-PDRF-Full", "RO-CAGF"),
    ], "major9_modern_baseline_effects.csv")
    safety = safety_summary()
    write_tables(strict, safe, modern, safety)

    aggregate_safe = safe[safe.scope == "four_fault_types"]
    aggregate_modern = modern[modern.scope == "four_fault_types"]
    aggregate_safety = safety[safety.scope == "four_fault_types"]
    summary = {
        "strict_estimand": strict[strict.subset == "fault_applied_available"].to_dict("records"),
        "safe_recovery_aggregate": aggregate_safe.to_dict("records"),
        "modern_baseline_aggregate": aggregate_modern.to_dict("records"),
        "safety_aggregate": aggregate_safety.to_dict("records"),
        "interpretation": {
            "strict_subset": "Recovery supervision remains beneficial relative to PDRF, but the bounded-versus-gate architecture contrast is unresolved.",
            "safe_layer": "The calibration-only selector prevents most base-correct/recovery-wrong losses but sacrifices recovery opportunities and is not uniformly AUROC-improving.",
            "modern_baseline": "RO-MER is stronger than RO-CAGF for three fault types, yet RO-PDRF-Full retains the highest mean over all four controlled mechanisms."
        }
    }
    (SRC/"major9_upgrade_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("Safe aggregate\n", aggregate_safe.to_string(index=False))
    print("\nModern aggregate\n", aggregate_modern.to_string(index=False))
    print("\nSafety aggregate\n", aggregate_safety.to_string(index=False))


if __name__ == "__main__":
    main()
