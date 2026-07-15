# CEE6 strict-review revision audit

This file is an internal completeness audit, not a response letter. The source
review was `Q1_Reviewer_Report_CEE6_Strict_85.docx` (Minor Revision, 85/100).

## Major comments

1. **Sparse, model-specific selector calibration — completed.** The Abstract,
   Methods, Table 3 caption, Discussion and Conclusion now report 89.7 mean
   informative disagreements, the repeated threshold range 0.525--0.90,
   deployment-specific recalibration and mandatory fallback to PDRF when both
   preference classes are unsupported.
2. **Pair prevention with retention — completed.** Every occurrence of the
   94.7% headline prevention result is paired with 9.7% recovery retention.
   Lite-CF is described as a conservative safety layer that sacrifices most
   recovery opportunities.
3. **Cost-model dependence — completed.** A new complete-stream sensitivity
   crosses 0%, 10%, 40% and 70% fault prevalence with pass penalties of 0,
   0.5 and 2 correction-equivalents per 10,000. The 2.9 break-even ratio is
   labelled descriptive and dataset-specific. Source:
   `source_data/q1_lite_prevalence_pass_utility.csv`.
4. **Mechanism dependence — completed.** Main-text interpretation identifies
   Gaussian and stuck-at as the clearest supported mechanisms and states that
   the offset interval is effectively centred near the null and crosses zero.
   Scale-1, scale-3 and scale-5 severity results remain in Supplementary
   Table S and `source_data/q1_fault_plausibility.csv`.
5. **Hard decisions versus probability quality — completed.** The main text,
   Figure 4 caption and Conclusion state that routing emits one
   temperature-scaled endpoint vector, uses no probability mixture or
   post-routing recalibration, and is not uniformly better than both endpoints.
   Seed-level probability dispersion is added to the Supplementary Information.
6. **External validity — completed.** Hydraulic and AHU sections are labelled
   single-rig sensitivity and direct-diagnosis boundary analyses. The Discussion
   specifies a future study with multiple devices, adjudicated failures,
   independent downstream labels, prospective site calibration, predeclared
   fallback and predeclared utility values.
7. **Compute — completed.** Direct CPU timing was added without retraining:
   three warm-ups and seven timed calls per seed at batch sizes 1 and 4,364.
   Lite-CF is described as two-pass/lower-compute than Safe-CF-12, not
   energy-efficient or real-time ready. Source:
   `source_data/q1_lite_cpu_latency.csv`.
8. **Main-text density — completed.** Legacy Safe-CF-12 transport tables,
   scalar controls, staged-cascade detail and architecture/teacher detail were
   removed from the main evidence chain and retained in the Supplementary
   Information.

## Minor comments

1. Abstract includes sparse/model-specific calibration — completed.
2. Canonical event terminology and denominator reminder — completed.
3. ``Lite'' defined as routing to RO-PDRF-Lite — completed.
4. Scaler refitting inside each cross-fitting fold and on the full calibration
   set stated — completed.
5. Figure 4 caption states temperature scaling, no mixture and no post-routing
   calibration — completed.
6. Seed-level routed-output dispersion added to Supplementary Information —
   completed.
7. Offset interval described as near-null and crossing zero — completed.
8. Figure 3 fonts enlarged; panel c gives exact `lambda=0.5` — completed.
9. Figure 4a replaced by separate seed-wise AUROC and threshold distributions —
   completed.
10. Empirical training range after training-only standardization/clipping stated —
    completed.
11. Hydraulic/AHU headings and captions use sensitivity/boundary wording —
    completed.
12. Environment lock, exact run configuration and script-to-result manifest are
    included. A new immutable GitHub release is created during packaging. Zenodo
    DOI remains pending user authorization of the external Zenodo account link;
    no DOI is fabricated.
13. Methods 3.4--3.6 and Discussion were shortened and sentence structure was
    simplified — completed.

## QA

- Main PDF: 28 pages; Supplementary PDF: 45 pages.
- No undefined references, overfull boxes, float-too-large errors or placeholders.
- Abstract: 239 words.
- Figure 1/2 arrows and Figure 3/4 labels/legends were inspected at original
  raster resolution; no text, arrows or legends overlap.
- Editable Word versions were exported and rendered to PDF for visual QA.
- The AI-use declaration remains absent at the author's explicit request.
