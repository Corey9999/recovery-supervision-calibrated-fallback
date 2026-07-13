"""Merge the frozen baseline formal run with the frozen final-method run."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "source_data"
FINAL = "PDRF_RCIA_R30"
KEYS = {
    "q1_risk_metrics": ["method", "seed", "subset", "metric"],
    "q1_risk_predictions": ["method", "seed", "sample"],
    "q1_mechanism_samples": ["method", "seed", "sample"],
    "q1_ensemble_metrics": ["method", "pool", "subset", "metric"],
    "q1_ensemble_predictions": ["method", "pool", "sample"],
}

for stem, keys in KEYS.items():
    baseline = pd.read_csv(SRC / f"{stem}_formal10.csv")
    final = pd.read_csv(SRC / f"{stem}.csv")
    final = final[final["method"] == FINAL]
    merged = pd.concat([baseline, final], ignore_index=True)
    merged = merged.drop_duplicates(keys, keep="last").sort_values(keys)
    merged.to_csv(SRC / f"{stem}.csv", index=False)
    print(stem, len(merged), sorted(merged.method.unique()))

design_path = SRC / "q1_risk_design.json"
design = json.loads(design_path.read_text(encoding="utf-8"))
design["formal_methods"] = sorted(pd.read_csv(SRC / "q1_risk_metrics.csv").method.unique())
design["full_method"] = FINAL
design["formal_seed_count"] = 10
design_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
