# Public-package audit

Audit date: 2026-07-13

## Inventory

- Total files: 490, including this audit report
- Total size: approximately 474.26 MB before Git compression
- Python scripts: 62
- Generated source-data files: 266, including `source_data/README.md`
- Figure files: 54 editable PDF/SVG or preview PNG files
- Generated table files: 93
- Largest file: `source_data/major3_ahu_adaptation_predictions.csv`, approximately 60.88 MB

## Checks passed

- All 62 Python files pass `python -m py_compile`.
- The public-data downloader exposes a valid command-line interface.
- No password, access token, API key, private-key header, personal Windows path, unresolved `TODO`, or publication placeholder was found in the public text and code files.
- No individual file exceeds GitHub's 100 MiB hard limit.
- TIFF figures, LaTeX build caches, Word-layout QA files, old manuscript versions, reviewer files, and third-party raw datasets are excluded.
- UCI and Figshare sources, DOIs, licences, and expected local paths are documented in `DATASETS.md`.

## Licence

The repository's original software and documentation use the MIT License, with copyright assigned to Riyang Luo. Third-party raw data are excluded and remain under their source licences.
