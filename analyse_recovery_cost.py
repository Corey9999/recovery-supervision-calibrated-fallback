"""Summarize the matched three-seed parameter and runtime audit."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT/"source_data"/"recovery_cost_audit.csv")
summary = data.groupby("method").agg(
    parameters=("parameters", "first"),
    train_seconds_mean=("train_seconds", "mean"),
    train_seconds_std=("train_seconds", "std"),
    epochs_mean=("epochs", "mean"),
    inference_ms_mean=("inference_ms_per_observation", "mean"),
    inference_ms_std=("inference_ms_per_observation", "std"),
).reset_index()
summary.to_csv(ROOT/"source_data"/"recovery_cost_summary.csv", index=False)

order = ["CAGF", "ENT-PD", "PDRF", "RO-PDRF"]
summary = summary.set_index("method").reindex(order)
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Method & Parameters & Training time (s) & Epochs & Inference (ms/observation) \\",
         r"\midrule"]
for method, z in summary.iterrows():
    lines.append(
        f"{method} & {int(z.parameters):,} & "
        f"{z.train_seconds_mean:.1f} $\\pm$ {z.train_seconds_std:.1f} & "
        f"{z.epochs_mean:.1f} & "
        f"{z.inference_ms_mean:.4f} $\\pm$ {z.inference_ms_std:.4f}" + r" \\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
(ROOT/"tables"/"recovery_cost.tex").write_text("\n".join(lines), encoding="utf-8")
