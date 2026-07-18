# Implementation Plan and Dataset Observations

## Goal

Turn batches of parcel photographs into conservative, auditable shipment records. A wrong AWB is treated as more harmful than a review flag. The application therefore keeps raw evidence, field sources, confidence, warnings, and detailed status flags.

## Supplied Dataset Inspection

- 17 valid JPEG files were inspected programmatically and visually.
- 15 are landscape and 2 are portrait.
- Dimensions range from approximately 2748×1504 to 4096×4604.
- All frames use a dark conveyor/scanner background and contain detectable yellow machine-overlay text.
- Mean brightness is approximately 26–35 and underexposed pixels range from approximately 44%–76%.
- The 2748×~1520 group forms a repeated capture layout with a smaller overlay and mostly compact cartons.
- Five larger images form another capture family with larger overlay text and more varied parcel placement.
- Overlay locations are consistently near the upper-left in supplied data, but scale and footprint vary. Production code searches the complete image with HSV/component-density logic and does not use fixed coordinates.
- Weight appears in both `gm` and `kg`; AWBs appear with 11 and 14 digits.
- Timestamp text appears with slash or hyphen dates and sometimes lacks spaces before the time or AM/PM.
- Some images have visible courier labels/barcodes while others have valid overlay fields but no visible label.
- The supplied set does not adequately cover empty frames, multiple parcels, strong occlusion, corruption, or duplicates; derived fixtures cover those cases.

The reproducible per-file metrics are stored in `sample_data/dataset_observations.csv`.

## Extraction Strategy

1. Validate signatures, image limits, ZIP safety, and decodeability.
2. Hash the original bytes and reuse results for within-batch duplicates.
3. Correct EXIF orientation and calculate quality metrics.
4. Detect yellow overlay text by HSV mask and normalized connected-component density.
5. OCR multiple overlay variants with Tesseract and select the result using anchor coverage, parsed-field count, and OCR confidence.
6. Decode barcodes from the full frame and rotated candidate-label crops.
7. Propose label regions using rectangularity, brightness, text/edge density, and barcode-like vertical gradients.
8. OCR label candidates independently from the overlay.
9. Estimate parcel count and boundary visibility using conservative foreground heuristics.
10. Reconcile candidates by normalized-value agreement; unresolved AWB disagreement always requires review.
11. Optionally ask xAI vision only for ambiguous classifications and never let it override strong local evidence.
12. Persist one typed result per image and generate CSV plus audit JSON.

## Confidence and Status Rules

- OCR confidence comes from contributing OCR tokens.
- Explicit anchors add evidence; repeatability and independent-source agreement increase confidence.
- Character correction, low quality, and source conflict reduce confidence.
- AWB `>=0.90` may be clean, `0.75–0.899` is review-only, and lower candidates remain only in audit evidence.
- `NO_PARCEL` needs strong combined evidence; segmentation failure alone becomes `PARCEL_DETECTION_UNCERTAIN`.
- A reliable overlay with no visible label becomes `SUCCESS_OVERLAY_ONLY` plus `LABEL_NOT_VISIBLE`.
- Multiple parcels, partial visibility, unresolved barcodes, or AWB conflicts take precedence and become `REVIEW_REQUIRED`.

## Implementation Phases

1. Dataset inspection and parser experiments.
2. Overlay extraction and golden tests.
3. Barcode, label proposals, and label OCR.
4. Parcel, label-status, confidence, and reconciliation rules.
5. SQLite jobs, threaded processing, CSV/JSON, and annotations.
6. Streamlit bulk upload, progress, filtering, review, and downloads.
7. Optional xAI provider behind explicit consent.
8. Derived edge fixtures, full supplied batch, and 100-image bulk run.
9. Docker, Streamlit Cloud configuration, documentation, sample output, and demo.

## Limitations

- Parcel and label detection use lightweight OpenCV heuristics, not a trained detector.
- Very different conveyor backgrounds may reduce segmentation accuracy.
- Tiny, rotated, blurred, reflective, or occluded labels may require review.
- Barcode presence does not automatically prove that a value is the AWB.
- Streamlit Community Cloud storage and background threads are process-local and not durable across restarts.

