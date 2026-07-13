# External datasets

Raw third-party files are excluded from the Git repository. Run:

```bash
python download_public_data.py
```

The downloader creates the paths expected by the validation scripts under `external_data/`.

## Gas Sensor Array Drift Dataset

- Official record: https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
- DOI: https://doi.org/10.24432/C5RP6W
- Repository licence: CC BY 4.0, as displayed by UCI.
- Local file: `external_data/gas_sensor_array_drift.zip`
- Used by: `run_uci_validation.py`

The dataset contains 13,910 measurements from 16 chemical sensors collected over 10 temporal batches. The study retains the native gas labels and temporal ordering; test-time sensor faults are controlled interventions.

## Condition Monitoring of Hydraulic Systems

- Official record: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
- DOI: https://doi.org/10.24432/C5CW21
- Repository licence: CC BY 4.0, as displayed by UCI.
- Local file: `external_data/hydraulic_condition_monitoring.zip`
- Used by: `run_hydraulic_validation.py` and related guarded-split scripts.

The component-condition labels are native to the public test-rig dataset. Pressure and vibration sensor corruptions used in the study remain controlled interventions.

## Labelled AHU data from three building types

- Official record: https://doi.org/10.6084/m9.figshare.27147678.v3
- DOI: https://doi.org/10.6084/m9.figshare.27147678.v3
- Repository licence: CC BY 4.0, as displayed by Figshare version 3.
- Local directory: `external_data/ahu_field/`
- Used by: `run_ahu_field_validation.py`, `run_ahu_temporal_validation.py`, and AHU adaptation analyses.

Figshare's source filename `hosptial_scientific_data.csv` contains a spelling error. The downloader saves it as `hospital_scientific_data.csv`, which is the filename expected by the analysis code.

## Attribution

Users must cite the original dataset records and comply with their licences. The repository's MIT License does not replace the licences of these third-party datasets.
