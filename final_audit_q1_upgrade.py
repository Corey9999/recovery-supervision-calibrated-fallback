"""Fail-fast audit for the isolated Q1-upgrade manuscript package."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "paper_package"
OUT = ROOT / "source_data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, tol: float = 5e-7) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"Expected {expected}, obtained {actual}")


def main() -> None:
    original_present = ORIGINAL.exists()
    if original_present:
        for name in ("manuscript.tex", "supplementary.tex"):
            assert sha256(ORIGINAL / name) == sha256(ROOT / name), name

    manuscript = (ROOT / "manuscript_q1_upgrade.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "supplementary_q1_upgrade.tex").read_text(encoding="utf-8")
    combined = manuscript + "\n" + supplement
    for token in ("待添加", "TBD", "TODO", "placeholder", "??"):
        assert token not in combined, token

    bib = (ROOT / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    cited = set()
    for content in re.findall(r"\\cite[pt]?\{([^}]+)\}", manuscript):
        cited.update(k.strip() for k in content.split(","))
    assert cited <= bib_keys, sorted(cited - bib_keys)

    labels = set(re.findall(r"\\label\{([^}]+)\}", manuscript))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", manuscript))
    assert refs <= labels, sorted(refs - labels)

    metrics = pd.read_csv(OUT / "major9_estimand_metrics.csv")
    strict = metrics[(metrics.subset == "fault_applied_available") &
                     (metrics.metric == "macro_auroc")]
    assert set(strict.seed) == set(range(101, 111))
    assert set(strict.n) == {1436}
    means = strict.groupby("method").value.mean()
    close(means["PDRF"], 0.780351279143952)
    close(means["RO-PDRF"], 0.795982225705321)

    safety = pd.read_csv(OUT / "major9_safe_recovery_summary.csv")
    safety = safety[safety.scope == "four_fault_types"].set_index("variant")
    assert int(safety.loc["Safe", "recovery_preserved"]) == 303
    assert int(safety.loc["Safe", "negative_transfer_remaining"]) == 137
    close(float(safety.loc["Safe", "negative_transfer_prevented_rate"]),
          0.9003636363636364)

    modern = pd.read_csv(OUT / "major9_modern_baseline_metrics.csv")
    modern = modern[(modern.subset == "fault_applied_available") &
                    (modern.metric == "macro_auroc")]
    assert set(modern.method) == {"RO-CAGF", "RO-MER", "RO-PDRF-Full"}
    assert set(modern.seed) == set(range(101, 111))
    assert modern.value.between(0, 1).all()

    required = [
        ROOT / "manuscript_q1_upgrade.pdf",
        ROOT / "supplementary_q1_upgrade.pdf",
        ROOT / "figures" / "figure8_strict_safe_upgrade.pdf",
        ROOT / "figures" / "figure8_strict_safe_upgrade.svg",
        OUT / "major9_frozen_protocol.json",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in required)

    for log_name in ("build_manuscript_pass3.log", "build_supplement_pass2.log"):
        log_path = ROOT / log_name
        if not log_path.exists():
            continue
        log = log_path.read_text(encoding="utf-8")
        for failure in ("Undefined control sequence", "LaTeX Error",
                        "There were undefined references", "Overfull \\hbox"):
            assert failure not in log, (log_name, failure)

    report = {
        "status": "pass",
        "original_tex_byte_identical": True if original_present else "not_applicable_in_standalone_repository",
        "strict_subset_n": 1436,
        "optimization_seeds": list(range(101, 111)),
        "safe_negative_transfer_prevented": 0.9003636363636364,
        "missing_citations": [],
        "missing_cross_references": [],
        "placeholders": [],
        "compiled_pdfs": ["manuscript_q1_upgrade.pdf",
                           "supplementary_q1_upgrade.pdf"],
    }
    (OUT / "major9_final_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
