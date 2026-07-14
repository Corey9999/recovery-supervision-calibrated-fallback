# Q1 review-completion audit

This is an internal revision checklist, not a response letter.

## Major comments

1. **Response-head specification — completed.** Main Methods now gives every affine layer, bias and activation. The final output is linear before `3*tanh(u/3)`. Empirical response ranges are in the main text, Supplementary Information and `source_data/q1_response_range_audit.csv`.
2. **Sparse selector events — completed.** Added 100 repeated grouped fits, coefficient/sign stability, a six-value regularization path, calibration-size sensitivity and the six-feature strongly shrunk Lite-CF selector. The deployment fallback when grouped folds lack both preference classes is stated.
3. **Selected-sample routing — completed.** Added an all-row three-outcome model, a shallow tree and endpoint-agreement/disagreement probability audits.
4. **Objective alignment — completed.** Negative-transfer prevention, recovery retention and decision utility are primary. AUROC/AUPRC/NLL/Brier/ECE are secondary and are evaluated on the final routed vector.
5. **Highest-level uncertainty — completed.** The seed P value was removed from the abstract. New claims use mechanism-fixed and mechanism--batch-fixed paired intervals. The four-mechanism bootstrap is labelled legacy sensitivity only.
6. **Breadth of external validation — completed by scope reduction.** Title and contributions are limited to controlled corruption. Hydraulic results are single-rig sensitivity; AHU results are a direct-diagnosis boundary.
7. **Practical benefit and six-pass cost — completed.** Lite-CF requires two passes. Added cost-sensitive utility over harm value, correction value and pass penalty; staged Cascade-25; matched hard-decision/compute comparisons. Energy is explicitly unmeasured.
8. **Learned selector baselines — completed.** Added all-row multinomial routing, a depth-two tree, a six-feature logistic selector and a staged cascade under grouped cross-fitting.
9. **Endpoint hierarchy — completed.** PDRF versus RO-PDRF-Lite is the operational pair. Full is retained only for mechanistic and legacy sensitivity analyses.
10. **Final routed probability quality — completed.** Added raw source data and main/SI summaries for strict, mixed and no-imposed-fault streams, agreement/disagreement subsets, two ECE definitions, classwise ECE, AUPRC and classwise metrics. The temperature and no-post-routing-calibration protocol are explicit.
11. **Fault realism and preprocessing — completed.** Added batches-1--6-only standardization, clipping order, standardized-space fault placement, percentile displacement, scale 1/3/5 sensitivity, representative inverse-transformed examples and the factorial-stress rationale.
12. **Supplementary and reproducibility package — content completed.** Full Supplementary Information and a result-to-run manifest are included. Tagged-release/archive creation is completed during final packaging and recorded in the package manifest.

## Minor comments

1. Narrow title — completed.
2. Shorter abstract without seed P value — completed.
3. Compact event-notation table — completed.
4. Unambiguous response-head activation — completed.
5. Temperature-scaled selector features — completed.
6. Exact lexicographic tie rule and tie frequency — completed.
7. Paired 95% intervals — completed with mechanism-fixed tables.
8. Mathematical harm/correction definitions — completed.
9. Opportunity-boundary/AUROC distinction — completed in text and caption.
10. Figure readability — completed with two redesigned full-width figures and enlarged labels.
11. Complete and strict class counts by batch — completed.
12. Hydraulic guarded test counts by condition — completed.
13. AHU prevalence and subtype counts by building — completed.
14. ECE binning definitions — completed.
15. Tagged release, lockfile and machine-readable configuration — lockfile/configuration present; release archive and checksum are created in final packaging.
16. AI declaration — intentionally absent at the author's request; it will be supplied separately according to the selected journal workflow.
17. Dense language — revised with shorter Methods, Results and Discussion paragraphs.
