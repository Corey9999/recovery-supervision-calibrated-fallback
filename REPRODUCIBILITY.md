# Reproducibility workflow

## Environment

The recorded environment uses Python 3.10 with CPU-compatible PyTorch. Create the environment from `environment.yml` or install `requirements.txt`. All random seeds, split definitions, and controlled-fault parameters used by the final analyses are retained in code and in `source_data/reproducibility_protocol.json`.

## Data preparation

```bash
python download_public_data.py
python export_reproducibility_indices.py
```

The first command downloads external raw data from the official records. The second exports deterministic observation-level split and principal-fault manifests. Existing manifests are included for auditability.

## Full experiment order

The commands below reflect the dependency order used for the final release. The complete run is computationally substantial; the supplied prediction-level outputs allow the analysis and figure stages to be rerun independently.

```bash
python run_major3_objective_matched.py
python run_major4_regularizer_sensitivity.py
python run_major3_ahu_shift_adaptation.py
python run_major3_teacher_ablation.py
python run_major3_loss_ablation.py
python run_major5_consistency_controls.py
python run_major5_basic_distillation_rerun.py
python run_major5_hydraulic_blocked.py
python run_major6_strong_controls.py
python run_major6_ema_extension.py
python run_major7_qmf_baseline.py
python run_major7_teacher_agreement.py
python run_major7_fault_types_ema.py
python run_major8_qmf_sensitivity.py
python run_major8_qmf_faithful_extension.py
python run_major8_ce_conflict_teacher.py

python analyse_major3_revision.py
python analyse_major5_revision.py
python analyse_major6_revision.py
python analyse_major7_revision.py
python analyse_major8_revision.py
python make_five_manuscript_figures.py
python normalize_reader_labels.py
```

Earlier `run_*.py`, `analyse_*.py`, and `make_*.py` files are retained because they reproduce baseline, uncertainty, ensemble, risk-coverage, conformal, no-interpolation, and quality-metadata analyses reported in the study or its supplementary information.

## Main audit artifacts

- `source_data/chemical_split_indices.csv`: all 13,910 chemical-array observations and their split assignments.
- `source_data/chemical_principal_fault_realization.csv`: principal held-out chemical-fault realizations.
- `source_data/hydraulic_guarded_split_indices.csv`: guarded hydraulic split assignments for all component-condition tasks.
- `source_data/reproducibility_protocol.json`: protocol metadata and deterministic seeds.
- `source_data/major8_principal_method_summary.csv`: principal method summary.
- `source_data/major8_fault_type_architecture_effects.csv`: per-fault architecture effects.
- `source_data/major8_qmf_sensitivity_summary.csv`: QMF sensitivity summary.
- `source_data/major8_teacher_conditional_rates.csv`: recovery and error-transfer audit rates.

## Interpretation constraints

- Controlled sensor interventions are not maintenance-confirmed natural failures.
- The hydraulic data contain native component-condition labels, while sensor corruptions are controlled.
- The AHU data contain native expert-labelled sensor-fault classes but do not provide an independent downstream recovery target.
- The CE-conflict training audit is a diagnostic used during training, not a deployable test-time health signal.
- QMF-PD is an in-protocol sensor-group comparator rather than a claim of general superiority over QMF methods.
