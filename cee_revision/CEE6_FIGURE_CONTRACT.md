# CEE6 figure contract

## Figure 3

- Core conclusion: Lite-CF trades recovery retention for negative-transfer prevention, and its deployment value depends on corruption mechanism, calibration support and the declared utility costs.
- Archetype: quantitative grid.
- Target/output: Computers & Electrical Engineering, full-width vector PDF/SVG plus 600-dpi TIFF and PNG preview.
- Backend: Python/matplotlib only.
- Final size: 183 mm wide, approximately 135 mm high.
- Panel map: (a) prevention versus retention with pass-count encoding; (b) mechanism-fixed AUROC contrasts; (c) utility versus harm cost with the exact pass penalty printed in the panel; (d) calibration-size sensitivity.
- Hero evidence: the prevention-retention plane.
- Controls/robustness: mechanism intervals, utility boundary and calibration-size sensitivity.
- Statistics: fixed-mechanism paired 95% intervals; 10 fitted endpoint pairs; calibration-size means with 95% normal-approximation intervals.
- Reviewer risks: headline prevention must remain paired with retention; utility must show the pass-cost proxy; labels and legends must remain readable at journal scale.

## Figure 4

- Core conclusion: Lite-CF calibration is sparse and model-specific, while its routed probability vector is not uniformly better than both endpoints.
- Archetype: quantitative grid with a nested distribution panel.
- Target/output: Computers & Electrical Engineering, full-width vector PDF/SVG plus 600-dpi TIFF and PNG preview.
- Backend: Python/matplotlib only.
- Final size: 183 mm wide, approximately 142 mm high.
- Panel map: (a) separate seed-wise distributions for OOF AUROC and selected threshold, replacing the overlapping scatter; (b) coefficient/sign stability; (c) strict-subset reliability; (d) probability quality by endpoint-class agreement.
- Hero evidence: seed-wise OOF-AUROC and threshold distributions.
- Controls/robustness: coefficient signs and final emitted-probability audits.
- Statistics: 100 repeated grouped fits, 10 repeats per endpoint-pair seed; reliability-bin summaries over fixed mechanism-seed cells.
- Reviewer risks: no point or text overlap; seed identity remains visible; temperature scaling and absence of mixture/post-routing recalibration are stated in the caption.

## Visual QA

- Use at least 8-pt labels/legends and 9-pt panel letters at final export size.
- Keep legends outside dense data regions.
- Check PDF, SVG, PNG and TIFF exports.
- Render final manuscript and Word/PDF figure pages, then inspect for clipping, overlap and unreadable text.
