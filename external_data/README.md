# Local external-data cache

This directory intentionally contains no third-party raw data in Git.

Run:

```bash
python download_public_data.py
```

Expected local paths after downloading:

```text
external_data/
  gas_sensor_array_drift.zip
  hydraulic_condition_monitoring.zip
  ahu_field/
    auditorium_scientific_data.csv
    hospital_scientific_data.csv
    office_scientific_data.csv
    FDD_processing.ipynb
```

See `../DATASETS.md` for attribution, DOI, licence, and scope details.
