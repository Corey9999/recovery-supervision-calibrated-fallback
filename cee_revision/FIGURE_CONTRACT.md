# CEE main validation figure contract

Core conclusion: Safe-CF trades retained RO-PDRF-Full corrections for fewer RO-PDRF-Full negative transfers; inference must respect the crossed fault-by-model-seed design and the ten unique seed units in no-fault or unaffected streams.

Figure archetype: quantitative grid with a dominant risk--recovery panel.

Target journal/output: Computers & Electrical Engineering; double-column width (183 mm), 2x2 panels; editable PDF/SVG plus 600-dpi TIFF and 300-dpi PNG preview.

Backend: Python (matplotlib only).

Panel map:

- a: full mixed-stream macro-AUROC at 0%, 10%, 40%, and 70% fault prevalence.
- b: Safe-CF macro-AUROC differences for leave-one-family-out and unseen mechanisms.
- c: simple-rule versus cross-fitted selector trade-off between negative-transfer prevention and recovery retention; crossed-cluster 95% interval for Safe-CF and a correctness opportunity-boundary marker.
- d: inference FLOPs and forward-pass scaling at 4, 8, and 16 sensor groups.

Evidence hierarchy:

- Hero evidence: panel c, because it tests whether the learned selector adds value beyond simple fallback rules.
- Validation evidence: panels a and b, covering deployment mixture and mechanism transfer.
- Cost boundary: panel d, showing the G+2 pass requirement.

Statistics: 10 matched optimization seeds; strict model--fault--observation predictions; Safe-CF interval from 1,000 crossed-cluster replicates that independently resample fault mechanisms and model-pair seed identities, retain the same seed draw across faults, and then resample observations within cells. Clean 0% points use exactly one value per seed. Other error bars are seed-level two-sided 95% Student-t intervals.

Reviewer risks: selector threshold leakage, use of labels at test time, unequal stream definitions, pseudo-replication at 0% or in unaffected rows, crossed-factor misspecification, treating four mechanisms as a fault population, mislabelling the correctness oracle as an AUROC bound, ambiguous reference frames for prevented harm, unreadable annotations, and hidden six-pass deployment cost.

Image integrity: no generated or altered experimental images; all panels are direct plots of source CSV outputs produced by the frozen CEE run.

## Optional CEE graphical abstract

- Conclusion: cross-fitted selection converts unconditional recovery into an explicit safety--recovery trade-off.
- Evidence chain: paired endpoint models -> out-of-fold selector -> frozen deployment decision -> quantified protection, retention, transfer and cost.
- Reviewer risks: avoid implying natural-fault or device generalization; distinguish prevention from recovery-model usage; state the six-pass cost at four groups.
- Export: 1328 x 531 pixels or proportionally larger, plus editable PDF and 500-dpi TIFF.

## Q1 major-revision figures

Core conclusion: A six-feature, strongly regularized Lite-CF router controls most recovery-induced harmful switches with two endpoint passes. Repeated calibration splits expose residual threshold and coefficient instability, and final-output calibration remains a deployment limitation.

Figure archetype: quantitative grid with a practical-trade-off hero panel.

Panel map:

- Figure 3a: prevention versus retention; marker area encodes mean forward passes.
- Figure 3b: mechanism-fixed Lite-CF minus PDRF macro-AUROC intervals across 10 fitted pairs.
- Figure 3c: decision utility across harmful-transfer cost ratios; forward-pass cost is a transparent proxy, not measured energy.
- Figure 3d: performance versus calibration-set size under repeated grouped subsampling.
- Figure 4a: repeated grouped cross-validation discrimination and threshold spread.
- Figure 4b: standardized coefficient sign stability for six prespecified features.
- Figure 4c: strict-subset reliability curves for PDRF, RO-PDRF-Lite and Lite-CF.
- Figure 4d: final-output NLL and equal-width ECE separated by endpoint class agreement.

Evidence hierarchy:

- Hero evidence: two-pass prevention--retention--compute comparison.
- Validation evidence: mechanism-fixed intervals and emitted-probability metrics.
- Robustness evidence: repeated grouped splits, regularization, coefficient signs and calibration-set-size curves.

Statistics and export contract: Python only; 183-mm double-column width; body text at least 7 pt; editable SVG/PDF, 600-dpi TIFF and PNG. Mechanism-fixed intervals are Student-t intervals across 10 fitted model pairs. Source data are the `q1_lite_*`, `q1_learned_routing_*`, `q1_decision_utility*` and `q1_fault_plausibility*` files.

Reviewer risks: routing thresholds remain variable; forward passes and FLOPs are compute proxies rather than energy measurements; AUROC is secondary to hard-decision prevention, retention and utility; no interval is interpreted as a fault-population or device-population interval.
