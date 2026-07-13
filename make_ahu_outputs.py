"""Generate compact LaTeX tables for the real-operational AHU validation."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)

LABELS = {"EF-PD": "EF-PD", "HGB-current": "HGB-current",
          "HGB-lag": "HGB-lag", "ENT-PD": "ENT-PD", "CAGF": "CAGF",
          "PDRF-NOQ": "PDRF-NOQ", "RC-PDRF-NOQ": "RO-PDRF-NOQ"}
BUILDINGS = ["auditorium", "hospital", "office"]


def pm(mean, sd):
    return f"{mean:.3f} $\\pm$ {sd:.3f}"


metrics = pd.read_csv(SRC / "ahu_temporal_metrics.csv")
focus = metrics[metrics.metric.isin(["macro_auroc", "macro_auprc"])]
agg = focus.groupby(["building", "method", "metric"]).value.agg(["mean", "std"])
lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"& \multicolumn{2}{c}{Auditorium} & \multicolumn{2}{c}{Hospital} & \multicolumn{2}{c}{Office} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Method & AUROC & AUPRC & AUROC & AUPRC & AUROC & AUPRC \\", r"\midrule"]
for method in LABELS:
    cells = []
    for building in BUILDINGS:
        for metric in ("macro_auroc", "macro_auprc"):
            row = agg.loc[(building, method, metric)]
            cells.append(pm(row["mean"], row["std"]))
    lines.append(LABELS[method] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "ahu_temporal_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

splits = pd.read_csv(SRC / "ahu_temporal_splits.csv")
lines = [r"\begin{tabular}{llrrr}", r"\toprule",
         r"Building & Split & Records & Normal & Sensor fault \\", r"\midrule"]
for _, row in splits.iterrows():
    lines.append(f"{row.building.capitalize()} & {row.split.capitalize()} & {row.rows:,} & "
                 f"{row.normal:,} & {row.sensor_fault:,} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "ahu_temporal_splits.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

tests = pd.read_csv(SRC / "ahu_temporal_paired_tests.csv")
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Contrast & Mean $\Delta$ AUROC & Wins & Losses & Exact $P$ \\", r"\midrule"]
for _, row in tests.iterrows():
    lines.append(f"{row.contrast.replace('RC-PDRF-NOQ - ', 'RO-PDRF-NOQ vs. ')} & {row.mean_difference:+.4f} & "
                 f"{int(row.wins)} & {int(row.losses)} & {row.p_two_sided_exact:.4f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "ahu_temporal_tests.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

cross_path = SRC / "ahu_field_metrics.csv"
if cross_path.exists():
    cross = pd.read_csv(cross_path)
    cross = cross[cross.metric == "macro_auroc"]
    cross_agg = cross.groupby(["held_out_building", "method"]).value.agg(["mean", "std"])
    cross_methods = [m for m in LABELS if m in cross_agg.index.get_level_values("method")]
    lines = [r"\begin{tabular}{l" + "c"*len(cross_methods) + "}", r"\toprule",
             "Held-out building & " + " & ".join(LABELS[m] for m in cross_methods) + r" \\", r"\midrule"]
    for building in BUILDINGS:
        lines.append(building.capitalize() + " & " + " & ".join(
            pm(cross_agg.loc[(building, method), "mean"], cross_agg.loc[(building, method), "std"])
            for method in cross_methods) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "ahu_cross_building_exploratory.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
