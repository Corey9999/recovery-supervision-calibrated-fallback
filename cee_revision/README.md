# Support-gated selective recovery under controlled multi-sensor corruption

This directory is the frozen reproducibility package for the *Computers & Electrical Engineering* manuscript **Support-gated selective recovery under controlled multi-sensor corruption**.

## Decision problem

The study compares a stable PDRF base endpoint with an RO-PDRF-Lite recovery endpoint under controlled corruption of an available sensor group. A router must retain useful corrections while preventing recovery-induced harmful decision changes. Test labels and fault identity are not router inputs.

The submission reports two distinct operating regimes:

- **Safety policy:** a prespecified support gate permits routing only when calibration event counts, repeated grouped discrimination and threshold stability pass. Unsupported fitted pairs revert to PDRF.
- **Utility policy:** thresholds are selected only from out-of-fold calibration predictions for harm-to-correction ratios 1, 2, 5 and 10.

Always-base and always-recovery policies are reported beside every selective policy. Chemical faults are controlled interventions, not natural field-failure annotations.

## Major-review additions

- All-row multinomial and two-stage outcome models use base-better, equivalent and recovery-better labels on every calibration row, addressing the selected-disagreement limitation.
- Complete prevention-retention frontiers and calibration-matched random, entropy, confidence, uncertainty, rank, tree and logit controls are included.
- Severity transport covers all 3 x 3 calibration/test scales; affected-group transport covers groups 1--4.
- Leave-one-mechanism-out, Gaussian-only and calibration-mixture sensitivity analyses test mechanism transport.
- Batch results report the complete mechanism x fitted-pair x acquisition-batch x policy matrix, plus observation-weighted, equal-batch, worst-batch, Batches 8--9 and Batch 10 views.
- Engineering outputs report extra endpoint passes, selected observations per retained/net correction, model memory and fixed-thread residual latency.
- Figures 3 and 4 are supplied as editable PDF/SVG, PNG and 600-dpi TIFF.

## Headline audited results

- Always PDRF: macro-AUROC 0.8115, 100% prevention, 0% retention.
- Always RO-PDRF-Lite: macro-AUROC 0.8175, 0% prevention, 100% retention.
- Frozen conditional router: macro-AUROC 0.8151, 94.7% prevention, 9.7% retention and +15.8 equal-cost utility per 10,000 strict observations.
- Prospective support gate: only 5/10 fitted pairs pass; the gate yields 96.2% prevention, 6.6% retention and +7.5 equal-cost utility.
- Calibration-selected all-row and two-stage policies yield +107.9 and +110.6 utility at a 1:1 harm ratio and +30.8 and +37.8 at 2:1. At ratios 5 and 10 all learned policies are negative and always PDRF is preferred.
- Batch 10 contains 1,180/1,436 strict observations per cell, so observation-weighted summaries are dominated by Batch 10; equal-batch and worst-batch views are reported separately.

## Reproduction

Create the environment using `environment.yml` or install the versions in `requirements-lock.txt`. From the repository root, the major-review entry points are:

```bash
python cee_revision/analyse_cee_major_router_audits.py
python cee_revision/run_cee_major_severity_transport.py
python cee_revision/analyse_cee_major_shift_audits.py
python cee_revision/make_cee_major_revision_figures.py
```

The severity/group generation script retrains the same endpoint architecture only to create prediction files absent from the earlier frozen run. Its formal scale-3/group-1 output is checked against the frozen endpoint predictions before transport results are used. Derived machine-readable outputs are already included under `source_data/`, so readers can audit the reported numbers without rerunning training.

Earlier frozen-run entry points and all negative/sensitivity analyses remain in this directory for traceability.

## Scope and version

These results support controlled-corruption risk control on fixed future batches. They do not establish maintenance-confirmed natural sensor failure, device-independent downstream recovery or population-level failure prevalence. Fitted-seed intervals quantify optimization variation on one fixed test realization.

- Computational snapshot: `f8ef1a17bdfc98377334dd9aa6903702cdc789c1`
- Release tag: `v1.4.0-major-router`
- License: MIT
- Software DOI: not assigned
