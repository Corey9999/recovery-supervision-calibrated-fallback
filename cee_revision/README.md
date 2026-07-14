# CEE cross-fitted deployment revision

This directory reproduces frozen run `CEE-CF10-R2` for the *Computers & Electrical Engineering* submission.

## Design

- Optimization seeds: 101--110
- Test realization: 70001
- Temperature fitting: first chronological half of the Batch 7 calibration partition
- Selector development: second half, five-fold stratified group cross-fitting by original observation
- Calibration faults: Gaussian noise, offset, drift and stuck-at
- Unseen faults: gain loss, clipping and correlated dual-group corruption
- Primary estimand: the controlled fault is applied while the affected group remains available
- Uncertainty: 1,000 crossed-cluster replicates that independently resample fault mechanisms and fitted-pair seeds, retain the seed draw across faults, and then resample observations within cells

The test label and fault type are not selector inputs. Out-of-fold conditional preference scores tune the thresholds; final coefficients are then refitted while thresholds remain frozen. The score is not claimed as a generally calibrated reliability probability.

## Reproduction

From the repository root:

```bash
python cee_revision/run_cee_selector_validation.py
python cee_revision/analyse_cee_results.py
python cee_revision/make_cee_validation_figure.py
python cee_revision/make_cee_graphical_abstract.py
```

For a short smoke test:

```powershell
$env:CEE_FAST='1'
python cee_revision/run_cee_selector_validation.py
```

The formal rerun takes several minutes on a CPU-only environment. It writes only inside `cee_revision/source_data/`.

## Principal audited results

- Gaussian RO-PDRF-Full minus PDRF macro-AUROC: +0.0143, positive in 10/10 fits.
- Safe-CF minus PDRF macro-AUROC: +0.00483, crossed-cluster 95% interval +0.00166 to +0.00780.
- Negative-transfer prevention: 90.3%, interval 82.3% to 95.2%.
- Recovery retention: 14.4%, interval 5.6% to 26.2%.
- At 40% prevalence: 185.6 harmful decisions prevented, 20.9 corrections retained and 3.4 net additional correct decisions per 10,000 observations.
- Leave-one-family-out prevention: 86.2% mean.
- Unseen-mechanism prevention: 86.8% mean.
- Four-group Safe-CF deployment: six forward passes and 231,168 counted FLOPs per observation.

These results concern controlled corruptions on one chemical-sensor array. They do not establish maintenance-confirmed natural-failure or device-independent transfer.

## Files

- `run_cee_selector_validation.py`: frozen model, cross-fitting, stream, transfer, baseline, uncertainty and cost analysis.
- `analyse_cee_results.py`: manuscript fact ledger and LaTeX tables.
- `make_cee_validation_figure.py`: main deployment/transfer/cost figure.
- `make_cee_graphical_abstract.py`: optional CEE graphical abstract.
- `source_data/`: exact CSV/JSON outputs, including crossed-bootstrap replicates, selector calibration rows, feature ablations, coefficients, fold diagnostics, recovery eligibility and batch/batch-1 timing.
- `tables/`: generated CEE table bodies.
- `figures/`: editable SVG/PDF and PNG previews.
- `manuscript_cee.*`, `supplementary_cee.*`: submitted CEE-specific manuscript sources and compiled PDFs.

The repository is released under the MIT License. No software DOI is claimed in this revision; cite the repository URL and exact Git commit.
