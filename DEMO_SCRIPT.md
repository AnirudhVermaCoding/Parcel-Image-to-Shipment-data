# Five-Minute Demo Script

## 0:00–0:30 — Problem

Warehouse operators currently read parcel photographs and type shipment information manually. This application converts a batch into traceable shipment records and prefers an honest review flag over a guessed AWB.

## 0:30–1:00 — Data Observation

Show one supplied image. Point out the yellow machine overlay and the separate courier label. Explain that overlay extraction is not called label extraction, and that some images have overlay data even when the courier label is absent.

## 1:00–1:35 — Bulk Upload

Open the Streamlit app, choose multiple images, upload several supplied examples, keep local-only mode selected, enable annotated previews, and start the batch.

## 1:35–2:00 — Live Progress

Show the job ID, processing stage, processed/total counts, success/review/error counts, and progress bar. Mention that work runs outside the Streamlit rendering path and that one corrupt image cannot fail the batch.

## 2:00–2:40 — Readable Result

Open a result with a visible label. Show the AWB, its source, confidence, barcode candidates, overlay OCR, label OCR, and annotated regions. Explain source agreement when available.

## 2:40–3:20 — Overlay-Only Result

Open an image without a visible courier label. Show that overlay fields are still returned while the status says `SUCCESS_OVERLAY_ONLY` and includes `LABEL_NOT_VISIBLE`; no label verification is claimed.

## 3:20–4:00 — Honest Failure

Upload or open the blurred/occluded fixture. Show `REVIEW_REQUIRED`, low-confidence warnings, and the absence of a guessed clean AWB.

## 4:00–4:30 — CSV

Download the CSV and show one row per uploaded image, exact AWB strings, normalized grams/centimetres, source columns, confidence columns, statuses, warnings, and processing duration.

## 4:30–5:00 — Architecture and Limitation

Mention that local OCR/barcode processing needs no paid key. Optional xAI vision can be enabled only for ambiguous cases with explicit consent. Close with the honest limitation that parcel and label detection use lightweight CV heuristics rather than a custom-trained detector.

