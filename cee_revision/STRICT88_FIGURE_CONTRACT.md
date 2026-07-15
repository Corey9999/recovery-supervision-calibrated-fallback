# Strict-88 figure contract

## Figure 3: safety, mechanism and utility evidence

- Intended conclusion: Lite-CF controls most recovery-induced hard-decision harm, but retains few correction opportunities and its value depends on mechanism, prevalence and cost.
- Panel a: negative-transfer prevention versus recovery retention; marker area represents forward-pass count.
- Panel b: Lite-CF minus PDRF macro-AUROC by fixed controlled mechanism; paired 95% Student-t intervals use 10 fitted endpoint pairs on the same test observations.
- Panel c: utility per 10,000 strict observations across harm-to-correction ratios. The main axis retains the full unconditional-recovery range; a single inset enlarges only the PDRF--Lite-CF region. The inset must not use a second y-axis or conceal sign changes.
- Panel d: repeated calibration-group subsampling with uncertainty over seed--repeat summaries.
- Review risks: no population-level fault-mechanism claim; no inference from marker area; inset bounds and methods must be explicit in the caption; labels, arrows and legends must remain inside their panels.
- Required exports: editable SVG and PDF, 600-dpi TIFF, and PNG preview generated from the same Python source.

## Figure 4: selector stability and probability quality

- Intended conclusion: sparse calibration creates unstable routing fits, and all final outputs remain overconfident despite temperature scaling.
- Panel a: repeated grouped out-of-fold AUROC and threshold variation by fitted endpoint pair.
- Panel b: standardized coefficient median, interquartile range and dominant-sign frequency.
- Panel c: reliability curves with identity line; caption must state the observed overconfidence.
- Panel d: NLL and ECE separated by endpoint-class agreement and disagreement.
- Review risks: the conditional preference score is not a globally calibrated reliability probability; final probability vectors are endpoint-specific; no probability mixture or post-routing recalibration was used.
- Required exports: editable SVG and PDF, 600-dpi TIFF, and PNG preview generated from the same Python source.

## Visual QA acceptance criteria

- No label, legend or panel letter crosses a panel boundary.
- No arrow intersects node text, and no two arrows overlap in method diagrams.
- Insets do not cover the primary trend or legend.
- Minimum text remains readable at manuscript page width.
- PDF, SVG, TIFF and PNG contain the same panel content and ordering.
