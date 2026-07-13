"""Summarize revision experiments for tables, macros and paired effect intervals."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def fmt(x, ci):
    return f"{x:.3f} $\\pm$ {ci:.3f}"


faults = pd.read_csv(SRC / "revision_faults_long.csv")
rolling = pd.read_csv(SRC / "revision_rolling_long.csv")
audit = pd.read_csv(SRC / "revision_boundary_audit.csv")

fault_order = ["silent_gaussian", "bias", "clipping", "burst", "correlated",
               "misleading_quality", "delayed_quality"]
fault_labels = {
    "silent_gaussian": "Silent Gaussian", "bias": "Fixed bias", "clipping": "Clipping",
    "burst": "Contiguous burst", "correlated": "Correlated dual-sensor",
    "misleading_quality": "Misleading diagnostic", "delayed_quality": "Delayed diagnostic",
}
method_order = ["UF", "MOME_A", "CRF", "BCRF"]
method_labels = {"UF": "Uniform fusion", "MOME_A": "Routed experts", "CRF": "CRF", "BCRF": "Bounded CRF"}

auc = faults[faults.metric.eq("macro_auroc")]
rows = []
for fault in fault_order:
    row = [fault_labels[fault]]
    for method in method_order:
        values = auc[(auc.fault == fault) & (auc.method == method)].value
        row.append(fmt(*mean_ci(values)))
    rows.append(row)
table = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Fault & Uniform fusion & Routed experts & CRF & Bounded CRF \\\\", "\\midrule"]
for row in rows:
    table.append(" & ".join(row) + " \\\\")
table += ["\\bottomrule", "\\end{tabular}"]
(TABLES / "revision_fault_results.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

rolling_auc = rolling[rolling.metric.eq("macro_auroc")]
rows = []
origin_words = {1: "One", 2: "Two", 3: "Three"}
for origin in (1, 2, 3):
    for regime in ("natural", "silent"):
        row = [str(origin), regime.capitalize()]
        for method in method_order:
            values = rolling_auc[(rolling_auc.origin == origin) & (rolling_auc.regime == regime) &
                                 (rolling_auc.method == method)].value
            row.append(fmt(*mean_ci(values)))
        rows.append(row)
table = ["\\begin{tabular}{llcccc}", "\\toprule",
         "Origin & View & Uniform fusion & Routed experts & CRF & Bounded CRF \\\\", "\\midrule"]
for row in rows:
    table.append(" & ".join(row) + " \\\\")
table += ["\\bottomrule", "\\end{tabular}"]
(TABLES / "revision_rolling_results.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

paired_rows = []
for fault in fault_order:
    wide = auc[auc.fault.eq(fault)].pivot(index="seed", columns="method", values="value")
    for contrast, left, right in (("BCRF-CRF", "BCRF", "CRF"),
                                  ("CRF-routed", "CRF", "MOME_A"),
                                  ("CRF-uniform", "CRF", "UF")):
        values = wide[left] - wide[right]
        mean, ci = mean_ci(values)
        paired_rows.append({"fault": fault, "contrast": contrast, "mean_difference": mean,
                            "ci95": ci, "n_seeds": len(values)})
pd.DataFrame(paired_rows).to_csv(SRC / "revision_paired_effects.csv", index=False)

audit_summary = audit.groupby("method", as_index=False).agg(
    delta_logvar=("delta_logvar", "mean"),
    delta_logvar_sd=("delta_logvar", "std"),
    near_upper_bound=("near_upper_bound", "mean"),
    near_upper_bound_sd=("near_upper_bound", "std"),
    delta_weight=("delta_weight", "mean"),
    delta_weight_sd=("delta_weight", "std"),
)
audit_summary.to_csv(SRC / "revision_boundary_summary.csv", index=False)

def macro(name, value, pct=False):
    if pct:
        value = f"{100 * value:.1f}\\%"
    else:
        value = f"{value:.3f}"
    return f"\\newcommand{{\\{name}}}{{{value}}}"


macros = []
for fault, stem in (("silent_gaussian", "RevSilent"), ("correlated", "RevCorrelated"),
                    ("misleading_quality", "RevMisleading")):
    for method, suffix in (("MOME_A", "Routed"), ("CRF", "CRF"), ("BCRF", "BCRF")):
        value = auc[(auc.fault == fault) & (auc.method == method)].value.mean()
        macros.append(macro(stem + suffix + "AUC", value))
for method, suffix in (("CRF", "CRF"), ("BCRF", "BCRF")):
    row = audit_summary[audit_summary.method.eq(method)].iloc[0]
    macros.append(macro("Rev" + suffix + "DeltaLogVar", row.delta_logvar))
    macros.append(macro("Rev" + suffix + "Boundary", row.near_upper_bound, pct=True))
    macros.append(macro("Rev" + suffix + "DeltaWeight", row.delta_weight))
for origin in (1, 2, 3):
    for method, suffix in (("UF", "UF"), ("CRF", "CRF"), ("BCRF", "BCRF")):
        value = rolling_auc[(rolling_auc.origin == origin) & (rolling_auc.regime == "silent") &
                            (rolling_auc.method == method)].value.mean()
        macros.append(macro(f"RevOrigin{origin_words[origin]}{suffix}SilentAUC", value))
(TABLES / "revision_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

print(audit_summary.to_string(index=False))
print(pd.DataFrame(paired_rows).to_string(index=False))
