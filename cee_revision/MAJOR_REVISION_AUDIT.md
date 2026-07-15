# Major-revision completion audit

Target: *Computers & Electrical Engineering*

Revision basis: `Q1_Reviewer_Report_Cross_fitted_Selective_Recovery.docx`

## Reviewer-critical items

| Requested item | Revision status | Evidence location |
|---|---|---|
| Put always-base and always-recovery policies beside the selective router | Complete | Main Table 3 and Fig. 3b–c |
| Add simple matched controls and the complete prevention–retention frontier | Complete | Main Section 4.2, Fig. 3a; machine-readable full curves and matched summaries in `source_data/major_*` |
| Correct the selected-sample analysis | Complete | All-row multinomial and two-stage comparators use base-better, equivalent and recovery-better labels on every calibration row; Main Sections 3.5 and 4.2; Fig. 3d |
| Make the prespecified support gate the primary deployment policy | Complete | Unsupported fitted pairs revert to PDRF; only 5/10 pairs pass; Main Table 3 and Supplementary Tables S6–S7 |
| Report calibration support and fitted-pair variation without pseudoreplication | Complete | Fitted-pair/mechanism-fixed summaries and intervals; Supplementary source-data manifest |
| Test utility-aware thresholds | Complete | Harm-to-correction ratios 1, 2, 5 and 10 selected only from out-of-fold calibration predictions; Main Table 7 and Fig. 3c |
| Test severity, affected-group and mechanism transport | Complete | Full 3x3 severity matrix, four affected groups, leave-one-mechanism-out, Gaussian-only and mixture calibrations; Main Section 4.4, Fig. 4a–b,d and supplementary machine-readable tables |
| Resolve Batch-10 dominance | Complete | Batch counts are stated explicitly (91/165/1,180); overall, equal-batch, worst-batch, Batches 8–9 and Batch 10 results are reported; Main Table 8 and Fig. 4c |
| Add mechanism-by-batch method results | Complete | Full fitted-pair x mechanism x batch x policy file in `source_data/major_mechanism_batch_method_matrix.csv` |
| Compare calibration variants | Complete | Endpoint temperature, uncalibrated, Platt, isotonic, rank-normalized, class-conditional and logit-margin controls are reported as sensitivity analyses |
| Quantify engineering cost | Complete | Extra endpoint passes, selected observations per retained correction/net correction, model memory and one-/twelve-thread residual latency are reported |
| Avoid claiming natural sensor-failure recovery | Complete | Title, Abstract, Discussion and Conclusion consistently describe controlled-corruption risk control; hydraulic and AHU boundaries remain explicit |
| Ensure figure legibility | Complete | Figures 3–4 rebuilt in Python and exported as PDF/SVG/PNG/600-dpi TIFF; all main and supplementary key pages visually inspected with no overlap, clipping or text outside panels |

## Conclusions supported by the added experiments

- The frozen conditional router is not a universal replacement for either endpoint. Its 94.7% prevention and 9.7% retention are conditional safety metrics, not evidence of broad recovery benefit.
- The support gate is the primary prospective policy because unsupported selector fits are forced to the base model. It improves prevention to 96.2% but reduces retention to 6.6%.
- All-row and two-stage outcome models remove the correctness-disagreement selection restriction. They recover substantially more corrections at equal cost, while exposing the trade-off in harmful transfers.
- Utility-selected routing can be beneficial at harm ratios 1 and 2 but fails at ratios 5 and 10, where always using the base endpoint is preferred.
- Severity, affected-group, mechanism and batch results demonstrate material transport sensitivity; therefore the manuscript does not claim a universal selector.

## Residual limitations retained transparently

- Chemical sensor faults are controlled interventions rather than naturally annotated field failures.
- The hydraulic experiment is a single-rig sensitivity analysis and the AHU data provide direct diagnosis without an independent downstream recovery target.
- Fitted-seed intervals quantify optimization variation on one fixed test realization; they are not population-level device intervals.
- The repository has a public GitHub URL but no archived software DOI; no DOI is invented.
