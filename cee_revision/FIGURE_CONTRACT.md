# CEE main validation figure contract

Core conclusion: Cross-fitted selective recovery limits negative transfer beyond simple rules and remains measurable on mixed and unseen-fault streams, but its benefit is conservative and its inference cost grows with the number of sensor groups.

Figure archetype: quantitative grid with a dominant risk--recovery panel.

Target journal/output: Computers & Electrical Engineering; double-column width (183 mm), 2x2 panels; editable PDF/SVG plus 600-dpi TIFF and 300-dpi PNG preview.

Backend: Python (matplotlib only).

Panel map:

- a: full mixed-stream macro-AUROC at 0%, 10%, 40%, and 70% fault prevalence.
- b: Safe-CF macro-AUROC differences for leave-one-family-out and unseen mechanisms.
- c: simple-rule versus cross-fitted selector trade-off between negative-transfer prevention and recovery retention; hierarchical 95% interval for Safe-CF and an ideal-corner marker.
- d: inference FLOPs and forward-pass scaling at 4, 8, and 16 sensor groups.

Evidence hierarchy:

- Hero evidence: panel c, because it tests whether the learned selector adds value beyond simple fallback rules.
- Validation evidence: panels a and b, covering deployment mixture and mechanism transfer.
- Cost boundary: panel d, showing the G+2 pass requirement.

Statistics: 10 matched optimization seeds; strict model--fault--observation predictions; Safe-CF interval from 1,000 hierarchical fault-type -> seed/model-pair -> observation bootstrap replicates; other error bars are seed/cell standard errors and are labelled as such.

Reviewer risks: selector threshold leakage, use of labels at test time, unequal stream definitions, inflated independent-sample interpretation, missing uncertainty, unreadable dense annotations, and hidden six-pass deployment cost.

Image integrity: no generated or altered experimental images; all panels are direct plots of source CSV outputs produced by the frozen CEE run.

## Optional CEE graphical abstract

- Conclusion: cross-fitted selection converts unconditional recovery into an explicit safety--recovery trade-off.
- Evidence chain: paired endpoint models -> out-of-fold selector -> frozen deployment decision -> quantified protection, retention, transfer and cost.
- Reviewer risks: avoid implying natural-fault or device generalization; distinguish prevention from recovery-model usage; state the six-pass cost at four groups.
- Export: 1328 x 531 pixels or proportionally larger, plus editable PDF and 500-dpi TIFF.
