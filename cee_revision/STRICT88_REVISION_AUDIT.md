# Strict-88 reviewer-action audit

This internal audit maps the review in `Q1_Reviewer_Report_manuscript_cee_Strict_88.docx` to the independent Strict-88 revision. It is not a response letter.

## Major comments

1. **Operational support beyond class presence — completed.** Methods now define a prospective all-conditions support gate: at least 25 events in each preference class, both classes in every fitting fold, repeated grouped median out-of-fold AUROC at least 0.65, no repeat below 0.60 and threshold range at most 0.10. Results and Supplementary Tables report that 5/10 fitted pairs pass. Unsupported pairs revert to PDRF. The manuscript states that the gate was introduced after review and does not replace the frozen headline estimand.

2. **Pair prevention with retention and absolute counts — completed.** Table 1 fixes denominators and distinguishes opportunity counts (`N_C`, `N_H`) from realized events (`n_C`, `n_H`). The Results report absolute counts and ranges by mechanism for every 1,436-observation fitted pair and explain why a 52.2% recovery endpoint selection rate can coexist with 9.7% recovery retention.

3. **Make utility and prevalence dependence central — completed.** The abstract, Results, Discussion and Conclusion now state that complete-stream utility is negative at 0% and 10% controlled-fault prevalence and that the strict-set advantage reverses above a descriptive harm-to-correction ratio of about 2.9. A deployment rule requires prevalence, event values and compute penalty to be fixed externally before enabling Lite-CF.

4. **Add mild-severity evidence — completed.** A new scale-1 audit reuses the same scale-3-fitted endpoints, temperatures, selector coefficients and thresholds. No refitting or retuning occurs. New CSV evidence and main/Supplementary tables report AUROC, prevention, retention, endpoint selection and equal-cost utility.

5. **Clarify fixed-test inference and show fitting dispersion — completed.** Main endpoint values are mean plus standard deviation over 10 fitted pairs after mechanism averaging. Mechanism-fixed accuracy, macro-AUROC and equal-cost-utility differences have paired 95% Student-t intervals. Captions and text state that these are fitting-stability summaries on one fixed test set, not device- or mechanism-population intervals.

6. **Expose endpoint identity and calibrate interpretation — completed within current evidence.** The routing API can optionally return endpoint identity and the conditional preference score together with the selected vector. Figure 4 and the probability subsection explicitly report overconfidence and state that confidence is endpoint-specific rather than globally calibrated. Post-routing recalibration is not claimed.

7. **Strengthen external-validation boundary and prospective protocol — completed.** The Discussion specifies a future multi-device protocol with independently operated devices, maintenance-adjudicated sensor failures, an independent downstream target, endpoint fitting and site-specific calibration before outcome analysis, and a frozen support gate, fallback endpoint and utility values. Hydraulic and AHU evidence remain explicitly bounded.

8. **Archive DOI and conditional artificial-intelligence declaration — partly open.** Repository URL, exact snapshot and release tag are retained, and the manuscript truthfully states that no permanent software DOI has been assigned. The author requested that no artificial-intelligence declaration be inserted at this stage; journal-specific disclosure remains an author-side submission action.

## Minor comments

1. **Low-prevalence utility in abstract — completed.**
2. **Consistent routing terminology — completed.** The principal term is “recovery endpoint selection rate”; “recovery retention” is reserved for `n_C/N_C`.
3. **Notation and denominator definitions — completed.** Opportunity and realized counts are separated in equations, captions and tables.
4. **Seed-level dispersion — completed.** The primary endpoint table includes mean plus standard deviation across fitted pairs.
5. **Numerical offset/rounding language — completed.** The exact float32 saturation boundary and rounding interpretation are stated.
6. **Units for lambda — completed.** Lambda is a pass-cost proxy in correction-equivalents per 10,000 observations; energy was not measured.
7. **Figure 3 local zoom — completed.** Panel c retains the full range and adds a labelled PDRF--Lite-CF inset.
8. **Figure 4 overconfidence — completed.** Caption and text identify the reliability curves as substantially overconfident.
9. **Fixed-thread latency — completed.** New one-thread and 12-thread controlled measurements use identical warm-up/timing protocols and report batch-one and complete-batch values.
10. **Legacy analyses retained — completed.** Supplementary Information keeps legacy selectors, severity grids, calibration-size analyses, coefficients and the result manifest.
11. **Verify 2026 references — completed.** The TMLR survey and Building and Environment article were checked against official publication records; acronym capitalization is protected in BibTeX.
12. **Software DOI — open and disclosed.** No DOI is fabricated; the text states that it has not been assigned.
13. **Shorten dense Methods/Discussion — completed.** Comparator/recovery Methods and Discussion were compressed while retaining estimands, equations, boundaries and reproducibility details.

## Reproducibility additions

- `run_cee_strict88_additional_audits.py`
- `analyse_cee_strict88_revision.py`
- `source_data/strict88_mild_scale1_metrics.csv`
- `source_data/strict88_mild_scale1_safety.csv`
- `source_data/strict88_cpu_thread_sensitivity.csv`
- `source_data/strict88_deployment_support.csv`
- `source_data/strict88_opportunity_counts_by_mechanism.csv`
- `source_data/strict88_endpoint_seed_dispersion.csv`
- `source_data/strict88_support_gated_policy.csv`
- `source_data/strict88_severity_comparison.csv`
- `source_data/strict88_mechanism_paired_effects.csv`

## Remaining author-side actions

- Obtain a permanent archive DOI only after the final public release has been pushed and archived; then replace the truthful “not assigned” statement.
- Complete the journal-specific artificial-intelligence disclosure outside this revision, as requested by the author.
