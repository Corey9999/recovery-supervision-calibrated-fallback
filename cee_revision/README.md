# CEE Q1-scores revision: two-pass Lite-CF

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

The test label and fault identity are not router inputs. Fault scale 3 is treated as a severe controlled stress test, not as a natural-failure prevalence model.

## Reproduction

Create the environment using `environment.yml` or install the exact versions in `requirements-lock.txt`, then run from the repository root:

```bash
python cee_revision/run_cee_selector_validation.py
python cee_revision/run_cee_lite_routing_validation.py
python cee_revision/analyse_cee_q1_scores.py
python cee_revision/analyse_cee_lite_results.py
python cee_revision/analyse_cee_fault_plausibility.py
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
- Lite-CF requires two endpoint passes. Energy consumption was not measured.

These results establish risk control under controlled available-group corruption. They do not establish maintenance-confirmed natural sensor failure, population-level failure prevalence or device-independent field recovery.

## Contents

- `manuscript_cee.*` and `supplementary_cee.*`: manuscript and supplementary sources, editable Word exports and compiled PDFs.
- `run_cee_lite_routing_validation.py`: frozen two-pass Lite-CF evaluation.
- `analyse_cee_q1_scores.py`, `analyse_cee_lite_results.py`: stability, comparator, interval, probability-quality and utility analyses.
- `analyse_cee_fault_plausibility.py`: standardized- and original-unit intervention audit.
- `make_cee_q1_figures.py`: Figures 3 and 4 in PDF, SVG, PNG and TIFF formats.
- `source_data/`: machine-readable frozen inputs and derived outputs.
- `tables/`: exact LaTeX table bodies.

The repository is released under the MIT License. The versioned Git tag is `v1.1.0-cee-q1`. A software DOI has not been assigned.
