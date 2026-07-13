# AHU field-data quality audit

## Intended use and grain

The external validation treats one row as one hourly observation from one air-handling unit (AHU). The analysis task is binary recognition of normal operation versus an expert-annotated return- or supply-air temperature sensor fault. Building identity, AHU identity, timestamps and labels are excluded from model inputs.

## Source and licensing

- Article: Wang, *Scientific Data* 12, 1481 (2025), DOI `10.1038/s41597-025-05825-9`.
- Dataset: Figshare version 3, DOI `10.6084/m9.figshare.27147678.v3`.
- Repository license: CC BY 4.0.
- Files used: office, auditorium and hospital CSV records; SHA-256 hashes are stored in `source_data/ahu_field_design.json`.

## High-signal checks

| Building | Retained rows | Normal | Sensor fault | Return fault | Supply fault | AHUs |
|---|---:|---:|---:|---:|---:|---:|
| Auditorium | 62,357 | 60,219 | 2,138 | 1,734 | 404 | 13 |
| Hospital | 62,407 | 53,619 | 8,788 | 8,779 | 9 | 8 |
| Office | 127,432 | 119,402 | 8,030 | 6,999 | 1,031 | 20 |

- The hospital source contained repeated AHU-time keys. Sixty-six rows belonging to conflicting-label keys were excluded, and 128 same-label repeated rows were collapsed before splitting.
- Structural plant-loop missingness is present in the hospital data. Group availability is represented explicitly; missing values are zero-filled only after training-set standardization.
- Supply-air sensor faults are extremely rare in the hospital (9 records), so the primary field task combines return- and supply-air temperature sensor faults. Subtype-specific hospital claims are not made.
- Long normal runs create substantial class imbalance. Training, checkpoint-selection and calibration subsets are deterministically capped by class, while the chronological test periods are retained in full.
- Chronological splitting prevents future observations from entering training or calibration. A separate leave-one-building-out audit measures the additional building-domain shift.

## Evidence boundary

The records come from operating buildings, and the fault labels were assigned by experienced HVAC engineers using stated engineering criteria with an independent expert spot check. They are stronger field evidence than synthetic or laboratory fault injection. They are not maintenance-adjudicated hardware replacement records, and the public dataset does not provide inter-annotator agreement for all rows. The manuscript therefore describes them as “real-operational, expert-annotated sensor faults,” not confirmed physical sensor failures.
