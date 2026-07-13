"""Normalize reader-facing model labels after generated-table analyses.

Raw CSV method identifiers are intentionally retained for backward-compatible
reproduction.  This post-processing step expands the legacy ``RO-PDRF`` ID in
LaTeX tables so the practical Lite, mechanistic Full and calibration-oriented
EMA roles remain explicit in every public-facing table.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"


EXACT = {
    "RO-PDRF-CEG": "CE-conflict training audit",
    "RO-PDRF-CRA": "RO-PDRF-Full-CRA",
    "RO-PDRF-ECA": "RO-PDRF-Full-ECA",
    "RO-PDRF-CAL": "RO-PDRF-Full-CAL",
    "RO-PDRF-NOQ": "RO-PDRF-Full-NOQ",
    "CE-conflict training gate": "CE-conflict training audit",
    "cross-entropy-conflict gate": "CE-conflict training audit",
}


def normalize(text: str) -> str:
    for old, new in EXACT.items():
        text = text.replace(old, new)
    return re.sub(
        r"RO-PDRF(?!-(?:Lite|Full|EMA)(?:\b|\d))",
        "RO-PDRF-Full",
        text,
    )


def main() -> None:
    changed = 0
    for path in sorted(TABLES.glob("*.tex")):
        old = path.read_text(encoding="utf-8")
        new = normalize(old)
        if new != old:
            path.write_text(new, encoding="utf-8")
            changed += 1
    print(f"Normalized reader-facing labels in {changed} generated tables.")


if __name__ == "__main__":
    main()
