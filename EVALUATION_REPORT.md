# Labelled Evaluation Report

Run date: 2026-07-19

Policy under test: an AWB read only from the machine overlay is evidence, not a clean shipment result. It is routed to review until matching courier-label OCR or barcode evidence is available.

Readiness decision: **not yet ready for unattended parcel-condition decisions**. AWB safety was strong on this limited cohort, but the detector result requires a trained, independently validated five-class checkpoint.

## Headline results

The available human labels cover shipment fields on 17 private target photographs. Review and parcel-condition behavior is measured separately on 11 labelled adverse-condition fixtures derived from a private parcel photograph. Keeping these cohorts separate avoids presenting synthetic transformations as independent warehouse photographs.

| Requested measure | Cohort | Result |
|---|---:|---:|
| AWB exact-match precision | 17 target photos | 100% (17/17 extracted AWBs exact) |
| False clean AWB count | Both cohorts | 0 |
| Review recall | 11 adverse fixtures | 100% (10/10 review-positive fixtures flagged) |
| Parcel-condition accuracy | 11 adverse fixtures | 9.1% (1/11 exact count-and-visibility matches) |
| Field coverage | 17 target photos | 100% overall (85/85 labelled fields present) |

The target-photo field coverage was 100% for AWB, weight, length, width, and height (17/17 for each field). The target-photo review rate was 82.4% (14/17) after applying the overlay-only review policy. Three target AWBs had independent label or barcode support and were clean; all three were exact matches.

## Adverse-condition detail

The adverse cohort contains blur, darkness, overexposure, rotation, cropping, label obstruction, unreadable labels, multiple parcels, an empty frame, conflicting overlay text, and a renamed duplicate.

| Measure | Result |
|---|---:|
| AWB exact-match precision | 77.8% (7/9 extracted AWBs exact) |
| AWB exact-match recall | 70.0% (7/10 labelled AWBs exact) |
| Clean AWB records | 0 |
| False clean AWBs | 0 |
| Review recall | 100% (10/10) |
| Parcel-condition accuracy | 9.1% (1/11) |
| AWB coverage | 90.0% (9/10) |
| Weight coverage | 80.0% (8/10) |
| Length/width/height coverage | 100% each (10/10) |
| Overall field coverage | 94.0% (47/50) |
| Processing failures | 0% (0/11) |

Every adverse fixture was routed away from a clean result. This prevented the two incorrect extracted AWBs from becoming clean shipment data, but it also means the cohort does not establish clean-AWB precision because it contains no clean AWB records.

## Interpretation

The AWB safety policy behaved as intended on this small run: no incorrect AWB was marked clean. Extraction remained complete on the 17 target photographs, but the conservative policy produced a high review rate because 13 AWBs were supported only by overlay OCR.

Parcel-condition accuracy is not acceptable for a production claim. The installed `models/parcel_detector.onnx` is a broad logistics baseline, not a trained five-class parcel checkpoint for this warehouse domain. The result supports using its outputs only as conservative evidence and continuing to route ambiguous parcel states to review. A custom detector must be trained and frozen against independent parcel-condition labels before claiming operational detector accuracy.

## Method

- The production `src.pipeline.process_image` path processed every image with the installed ONNX baseline, local OCR, barcode extraction, reconciliation, and status assignment.
- The 17 target photographs used manually curated private manifests for AWB, weight, and dimensions.
- The 11 adverse fixtures used explicit condition labels for parcel count, parcel visibility, and review expectation. The corrupted fixture was excluded because parcel-condition accuracy applies to decoded images.
- AWB exact-match precision is `exact extracted AWBs / extracted AWBs` among rows with a labelled AWB.
- A false clean AWB is a non-review result whose extracted AWB differs from the labelled AWB.
- Review recall is `flagged review-positive images / all review-positive images`. Rejected images that need no human decision may be labelled review-negative.
- Parcel-condition accuracy requires both parcel count and parcel visibility to match on the same image.
- Field coverage is `labelled fields with a non-empty extraction / labelled fields` and does not imply field correctness.

The reusable evaluator can reproduce the aggregate calculations from a completed private ground-truth CSV and prediction CSV:

```powershell
python -m scripts.evaluate_accuracy `
  --truth evaluation\ground_truth.csv `
  --predictions path\to\shipment_results.csv `
  --split test `
  --output-dir evaluation\results
```

It writes `metrics.json`, false-clean/processing failures, threshold sweeps, and condition confusion tables. Private filenames, images, AWBs, and derived row-level reports are intentionally excluded from Git.

## Limitations

- Seventeen target photographs are too few to support a general accuracy claim.
- The 11 adverse fixtures are transformations of one source photograph, so they are not independent unseen warehouse samples.
- The target photographs are overlay-heavy and do not measure broad courier-label layout coverage.
- No validated five-class ONNX detector checkpoint is installed; blocked-label accuracy is not claimed.
- These results are a transparent readiness baseline, not proof of performance on the evaluator's unseen inputs.

The next evidence milestone is a frozen, human-labelled, independent test set spanning empty frames, multiple parcels, partial parcels, obstructions, unreadable labels, courier variants, and warehouse backgrounds.
