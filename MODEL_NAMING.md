# Public model naming and default

- **RO-PDRF-Lite** is the primary practical model and software default. It is
  base PDRF plus clean-view recovery distillation at weight 0.30.
- **RO-PDRF-Full** is the mechanistic complete-objective model. It adds Brier,
  pairwise-ranking and response-interior terms to Lite.
- **RO-PDRF-EMA** is the calibration-oriented sensitivity based on Full.
- **CE-conflict training audit** is a label-assisted, training-only diagnostic.
  It is not a deployable safeguard.

Legacy prediction and metric CSV files retain the internal identifier
`RO-PDRF` for RO-PDRF-Full so existing hashes and analysis joins remain
reproducible. Reader-facing manuscript text, tables and figures use the
expanded public names. `default_model_config.json` specifies Lite explicitly.
