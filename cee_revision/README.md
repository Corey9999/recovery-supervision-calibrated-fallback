# Strict-88 revision: supported two-pass Lite-CF with severity and cost audits

This directory contains the frozen computational snapshot for the *Computers & Electrical Engineering* manuscript **Cross-fitted selective recovery under controlled multi-sensor corruption**.

## Primary design

- Frozen endpoint run: `CEE-CF10-R2`, optimization seeds 101--110.
- Primary subset: a controlled fault is applied to an available sensor group (`n = 1,436` per mechanism and seed).
- Router: Lite-CF logistic selector with six temperature-scaled endpoint features and `C = 0.05`.
- Validation: five-fold cross-fitting grouped by original observation; coefficients and threshold selection are separated.
- Deployment: one bounded-base pass plus one recovery pass; no leave-one-group-out passes are required.
- Primary objective: hard-decision safety (negative-transfer prevention, retained corrections and cost-weighted utility).
- Secondary outcomes: macro-AUROC, macro-AUPRC, NLL, Brier score and ECE of the emitted probability stream.
- Fault mechanisms: Gaussian noise, offset, drift and stuck-at corruption, applied after training-only standardization and clipping.
- Complete-stream prevalence sensitivity: 0%, 10%, 40% and 70% imposed-fault assignment.
- Router-feature standardization is refitted within every cross-fitting training fold and on the full informative calibration set for the final model.
- Prospective deployment support requires sufficient events in both preference classes, repeated grouped discrimination and threshold stability; unsupported fits revert to PDRF.
- A mild scale-1 audit reuses the scale-3 endpoints, temperatures, selector coefficients and thresholds without refitting or retuning.

The test label and fault identity are not router inputs. Fault scale 3 is treated as a severe controlled stress test, not as a natural-failure prevalence model.

## Reproduction

Create the environment using `environment.yml` or install the exact versions in `requirements-lock.txt`, then run from the repository root:

```bash
python cee_revision/run_cee_selector_validation.py
python cee_revision/run_cee_lite_routing_validation.py
python cee_revision/analyse_cee_q1_scores.py
python cee_revision/analyse_cee_lite_results.py
python cee_revision/analyse_cee_fault_plausibility.py
python cee_revision/run_cee_strict88_additional_audits.py
python cee_revision/analyse_cee_strict88_revision.py
python cee_revision/make_cee_q1_figures.py
```

`cee_cf10_r2_lite_config.json` records the frozen selector, split, threshold-grid and utility settings. Exact derived outputs are included under `source_data/`; no rerun is needed to inspect the reported values.

## Primary audited results

- PDRF macro-AUROC: 0.8115; unconditional recovery: 0.8175; Lite-CF: 0.8151.
- Lite-CF prevented 94.7% of recovery-induced harmful decision changes and retained 9.7% of available corrections.
- Equal-cost net utility: +15.8 correct decisions per 10,000 strict observations.
- The utility changes sign when one harmful change is valued at approximately 2.9 corrections.
- Mechanism-fixed Lite-CF AUROC effects were positive for Gaussian and stuck-at corruption; offset and drift intervals included zero.
- Repeated grouped validation (100 fits) gave mean selector AUROC 0.720 with SD 0.138, documenting sparse-event instability.
- Only 89.7 endpoint-correctness disagreements informed each fitted router on average; repeated thresholds ranged from 0.525 to 0.90.
- Complete-stream equal-cost utility was negative at 0% and 10% controlled-fault prevalence and positive at 40% and 70%; pass penalties shift each value downward.
- Only 5/10 fitted endpoint pairs passed the prospective event-count, discrimination and threshold-stability gate; the other pairs use PDRF under the support-gated policy.
- At mild scale 1, Lite-CF macro-AUROC was 0.8357, prevention was 96.0%, retention was 5.3% and equal-cost utility was +4.00 decisions per 10,000 observations.
- Direct CPU timing fixes intra-op threads at one and 12 in separate runs, with three warm-ups and seven timed calls for batch sizes 1 and 4,364. Lite-CF requires two endpoint passes. Energy consumption was not measured.

These results establish risk control under controlled available-group corruption. They do not establish maintenance-confirmed natural sensor failure, population-level failure prevalence or device-independent field recovery.

## Contents

- `manuscript_cee.*` and `supplementary_cee.*`: manuscript and supplementary sources, editable Word exports and compiled PDFs.
- `run_cee_lite_routing_validation.py`: frozen two-pass Lite-CF evaluation.
- `analyse_cee_q1_scores.py`, `analyse_cee_lite_results.py`: stability, comparator, interval, probability-quality and utility analyses.
- `analyse_cee_fault_plausibility.py`: standardized- and original-unit intervention audit.
- `run_cee_strict88_additional_audits.py`: mild-severity and fixed-thread reruns without selector retuning.
- `analyse_cee_strict88_revision.py`: deployment-support, event-count, fitting-dispersion and severity summaries.
- `make_cee_q1_figures.py`: Figures 3 and 4 in PDF, SVG, PNG and TIFF formats.
- `source_data/`: machine-readable frozen inputs and derived outputs.
- `tables/`: exact LaTeX table bodies.
- `STRICT88_REVISION_AUDIT.md`: reviewer-comment-to-evidence completeness audit.

The repository is released under the MIT License. The Strict-88 release tag is `v1.3.0-strict88`. A software DOI has not been assigned; a DOI must not be inferred from the GitHub URL.
