# Recovery supervision with calibrated fallback: isolated Q1 upgrade

Journal-neutral submission package for **Recovery supervision with calibrated fallback for multi-sensor fusion under available-but-corrupted channels**. Journal-specific formatting remains deferred until the target is selected. This directory is an independent upgrade: the original `paper_package` article and supplement are not modified.

## Q1 upgrade additions

- Strict primary estimand restricted to 1,436 observations for which the controlled fault was nonzero and the affected group remained available; 329 assigned-but-masked rows are isolated as sensitivity evidence.
- Calibration-only selective recovery with Balanced and Safe thresholds fitted without test labels.
- Safe fallback prevents 90.0% of observed negative-transfer opportunities while retaining 17.9% of recovery opportunities across four controlled fault mechanisms.
- Recovery-oriented multi-expert router with one expert per physical group, a joint expert and sample-dependent routing.
- Formal 10-seed reruns, fixed-protocol metadata, batch-to-observation ensemble bootstrap and prediction-level safety accounting.
- New four-panel publication figure in PDF, SVG, PNG and 600-dpi TIFF formats.
- Public-data feasibility audit that preserves the natural-failure and multi-device evidence boundary.

## Revision completed

- Unique clean-teacher recovery-objective-matched comparison for CAGF, RO-CAGF, PDRF and RO-PDRF-Full.
- Effect-size table separating paired degradation, clean distillation and complete recovery for early, uniform, gated and bounded fusion.
- Published QMF confidence, detached unnormalized fusion and cumulative trajectory-ranking rules aligned to the same sensor groups (`QMF-PD`), with confidence-scale, normalization and ranking-weight sensitivity.
- Parameter-matched group-token attention gate with full implementation details.
- Eight shared Gaussian fault realizations plus an expanded Gaussian/offset/drift/stuck-at design with eight realizations and five fitted seeds.
- Crossed-factor bootstrap preserving the reuse of each fitted seed across controlled fault cells.
- EMA decay 0.95/0.99/0.995 sensitivity and four-fault probability-quality analysis.
- Clean/removal and student/EMA teacher-agreement experiments with prediction-level recovery/error-transfer audits.
- CE-conflict training audit, retained as a negative mechanistic result because conditional error transfer remains and the rule is unavailable at test time.
- Explicit RO-PDRF-Lite versus RO-PDRF-Full comparison, including measured training-time standard deviations and a practical recommendation.
- Per-fault RO-PDRF-Full--RO-CAGF effects and leave-one-fault-type-out analysis; the four-type aggregate is labelled descriptive.
- Guarded within-condition hydraulic split promoted to the main table and figure; random stratification retained as supplementary sensitivity evidence.
- Ranking--calibration Pareto figure, classwise calibration, aggregation and selective-risk boundaries.
- Concise AHU task/transfer boundary in the main paper; full shift, reversal, adaptation, conformal and subtype analyses in the supplement.
- Redrawn Figure 1 with larger text, separated connectors and no in-canvas explanatory sentence.
- Explicit chemical and hydraulic split-index manifests, principal fault-realization manifest, environment specification and Lite default configuration.
- 54 cited references verified through Crossref, DataCite or official publication records.

## Principal new results

- Official-rule-aligned QMF-PD affected macro-AUROC: 0.775; RO-PDRF-Full: 0.804 under the same principal chemical fault protocol. QMF-PD clean matched-mask AUROC was 0.845.
- RO-PDRF-Lite reached 0.802 affected AUROC versus 0.804 for Full and trained in 9.2 ± 2.7 s versus 22.7 ± 5.6 s.
- Per-fault RO-PDRF-Full--RO-CAGF effects were +0.011, +0.030, +0.035 and +0.034 for Gaussian, offset, drift and stuck-at interventions; the four-type descriptive aggregate was +0.027 [0.002, 0.051].
- Across four controlled fault mechanisms, RO-PDRF-EMA minus Full was +0.0038 AUROC (95% interval -0.0007 to +0.0091), -0.283 NLL (-0.535 to -0.094) and -0.0228 ECE (-0.0396 to -0.0085).
- Clean/removal agreement retained 65.3% active coverage with 24.6% conditional recovery and 32.9% error transfer; student/EMA agreement retained 72.4% coverage with 28.1% recovery and 40.3% transfer. Agreement did not solve the teacher-error route.
- The CE-conflict training audit reached 24.7% conditional recovery and 35.7% error transfer; it is not a label-free test-time selector.
- Guarded-block hydraulic RO-PDRF-Full minus RO-CAGF mean: +0.015; hierarchical 95% interval -0.006 to +0.036.

## Reproduction order

```bash
python analyse_major9_estimand.py
python run_major9_safe_recovery.py
python run_major9_modern_baseline.py
python analyse_major9_upgrade.py
python make_major9_figure.py
pdflatex manuscript_q1_upgrade.tex
bibtex manuscript_q1_upgrade
pdflatex manuscript_q1_upgrade.tex
pdflatex manuscript_q1_upgrade.tex
pdflatex supplementary_q1_upgrade.tex
pdflatex supplementary_q1_upgrade.tex
python final_audit_q1_upgrade.py

# Earlier analyses retained in the package
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
python export_reproducibility_indices.py
python normalize_reader_labels.py
python verify_references.py
latexmk -pdf manuscript.tex
latexmk -pdf supplementary.tex
make4ht -u -f html5 manuscript.tex "html5,charset=utf-8"
python export_word_pandoc.py
python final_audit.py
python build_final_archive.py
```

Earlier scripts remain because they reproduce baseline, uncertainty, ensemble, risk--coverage, conformal, no-interpolation and quality-metadata analyses cited by the article.

## Main files

- `manuscript_q1_upgrade.tex`, `manuscript_q1_upgrade.pdf`: independently upgraded article.
- `supplementary_q1_upgrade.tex`, `supplementary_q1_upgrade.pdf`: independently upgraded supplementary information.
- `manuscript.tex`, `supplementary.tex`: untouched byte-identical copies used to verify isolation from the original package.
- `final_audit_q1_upgrade.py`, `source_data/major9_final_audit.json`: fail-fast consistency and isolation audit.
- `REVIEW_COMMENT_TRACKER.md`: concern-by-concern implementation audit; not a response letter.
- `REFERENCE_VERIFICATION_54.md`: record-level reference verification.
- `source_data/`: seed-, fault-, environment- and prediction-level outputs.
- `default_model_config.json`, `MODEL_NAMING.md`: explicit Lite software default and public naming map.
- `environment.yml`, `requirements.txt`: reproducible software environment.
- `figures/`: editable SVG/PDF, 300-dpi PNG and 600-dpi TIFF figures.
- `tables/`: generated LaTeX tables.

## Interpretation boundary

The evidence supports a controlled recovery formulation and a bounded, parameter-efficient implementation. It does not establish a new distillation principle, calibrated health score, universal ranking/calibration superiority, natural hardware-failure recovery or device-independent field validity. The chemical fault mechanisms remain controlled interventions on one instrument; the hydraulic sensor faults are controlled interventions on one rig; the AHU target is direct sensor-fault diagnosis rather than an independent downstream recovery outcome.

## Author and declarations

Riyang Luo is the sole author. Affiliation: School of Mechanical and Power Engineering, Nanjing Tech University, Nanjing, China. Email: lry@njtech.edu.cn. No specific funding was received and no competing interests are declared. The AI disclosure remains omitted until the target journal is selected. A permanent public repository URL and software DOI remain author-controlled publication actions.
