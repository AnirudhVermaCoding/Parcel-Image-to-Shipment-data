# Prozo Case Study 1 — Parcel Images to Shipment Data

A local-first Streamlit application that processes parcel photographs in bulk and produces one auditable shipment record per image. It extracts AWB/tracking numbers, weight, dimensions, recorded volume, location, and capture time when the evidence is reliable, and otherwise returns explicit review or rejection flags.

> Design principle: a wrong AWB is worse than an honest failure flag.

## How This Meets the Brief

| Evaluation requirement | Implementation |
|---|---|
| Bulk warehouse input | Upload multiple JPG/JPEG/PNG images or one ZIP archive. The default configurable limit is 500 images. |
| One result per image | Every accepted upload receives its own typed record and exception boundary, so a corrupt or difficult image cannot terminate the batch. |
| Shipment fields | Extract AWB/tracking number, weight, and length/width/height when present. Every technical output records the evidence source and confidence. |
| Honest failure handling | Return explicit review or rejection states for uncertain parcel detection, multiple parcels, partial parcels, missing/blocked/unreadable labels, AWB conflicts, low confidence, unsupported files, and processing failures. |
| Downloadable output | Download a warehouse-friendly CSV at any point after results exist; full audit CSV and JSON are also available. |

The application is an end-to-end conservative prototype, not a claim of production accuracy. The upload, processing, review, recovery, and CSV paths work. The current detector is a generic logistics baseline, and the published labelled evaluation shows that parcel-condition classification still needs a trained five-class model. See [Evaluation evidence](#evaluation-evidence).

## What Goes In and What Comes Out

Input:

- Multiple JPG/JPEG/PNG files, or one ZIP archive.
- Dark, rotated, partial, blurred, occluded, empty, or multi-parcel frames are accepted and evaluated conservatively.

Output:

- One typed result per image.
- Required UTF-8 CSV download.
- Audit JSON with candidates, raw OCR, confidence, bounding boxes, and warnings.
- Optional annotated previews.

The warehouse CSV contains filename, tracking number, weight, dimensions, location, a readable result, a review marker, and plain-English notes. The audit output additionally exposes parcel/label state, extraction source, confidence, candidates, raw OCR evidence, detector metadata, warnings, and safe error information.

## Important Supplied-Data Observation

The supplied images contain yellow machine-generated overlays with fields such as `Location`, `AWB No`, `Length`, `Width`, `Height`, `R.Vol.`, `Weight`, and `Time`.

That information is processed as `OVERLAY_OCR`. It is not presented as courier-label extraction. Courier labels and barcodes are independently processed as `LABEL_OCR` and `BARCODE`. An image may therefore return reliable overlay fields while honestly reporting `LABEL_NOT_VISIBLE`. An AWB supported only by the machine overlay is retained as evidence but is always routed to review with `AWB_NOT_LABEL_VERIFIED`; it becomes clean only after matching courier-label OCR or barcode evidence is found. The full audit output makes this source distinction explicit.

## Architecture

```text
Streamlit / CLI / Tests
        |
        v
Typed processing pipeline
        |
        +-- validation, hashing, orientation, quality
        +-- machine overlay OCR
        +-- barcode decoding
        +-- ONNX detector, with explicit OpenCV fallback
        +-- multi-pass Tesseract + conditional RapidOCR
        +-- cached, rate-limited optional xAI observation
        +-- candidate reconciliation and confidence
        +-- status engine and annotations
        |
        v
SQLite jobs + CSV/JSON reports
```

Computer-vision and extraction modules do not import Streamlit. The same `process_image` function is used by the UI, CLI, tests, and threaded job workers.

## Processing Pipeline

1. Validate file signature, size, dimensions, pixel count, and decodeability.
2. Calculate SHA-256 and identify duplicates.
3. Correct EXIF orientation and record dimensions.
4. Calculate blur, brightness, contrast, exposure, edge density, and skew.
5. Detect yellow machine-overlay components across the full image.
6. OCR overlay variants and parse explicitly labelled fields.
7. Decode barcodes from the full image.
8. Detect candidate label regions.
9. Enhance/rotate label crops, decode their barcodes, and OCR their text.
10. Estimate parcel count and boundary visibility conservatively.
11. Optionally invoke xAI only when local evidence is ambiguous.
12. Reconcile candidates by normalized-value agreement.
13. Assign field confidence, primary status, flags, warnings, and review state.
14. Persist the result and update batch progress.
15. Generate CSV, JSON, and optional annotations.

Each image is processed inside its own exception boundary.

## Accuracy Upgrade Status

The runtime supports CPU ONNX detection, bounded multi-pass Tesseract, conditional RapidOCR, configurable courier rules, calibrated thresholds, and cached/rate-limited xAI structured output. The repository contains CVAT validation, evaluation, Google Colab training, YOLO11n, and ONNX export tooling. Private images, ground truth, and derived shipment reports are intentionally excluded from Git.

`models/parcel_detector.onnx` is installed as a documented logistics baseline. It maps only supported upstream classes such as cardboard boxes and barcode regions, and it does **not** claim the full five-class custom detector capability or blocked-label detection. See `models/parcel_detector.model-card.md` for its checksum, provenance, licence, metrics, and limitations. If the artifact is removed or cannot load, the application continues with conservative OpenCV heuristics and records `DETECTOR_UNAVAILABLE`.

## Technology Choices

- Python 3.11
- Streamlit
- OpenCV headless, NumPy, Pillow
- Tesseract through pytesseract
- zxing-cpp
- Pandas and Pydantic v2
- SQLite and `ThreadPoolExecutor`
- ONNX Runtime and RapidOCR ONNX
- OpenAI-compatible client configured for the xAI Responses API
- Pytest
- httpx for the optional xAI provider

No GPU, CUDA, paid OCR service, authentication system, application server, Redis, or external database is required.

## Why Local OCR Is the Default

Shipment images may contain operational or customer information. Local OCR avoids mandatory cost, quota, latency, and data-transfer concerns. Tesseract remains the primary engine. RapidOCR is a conditional second opinion only when stronger local evidence has not resolved the label.

## Why Barcode Decoding Matters

OCR can confuse similar characters. A decoded barcode is strong independent evidence, especially when it agrees with the overlay or label text. A barcode is not automatically treated as an AWB: format, context, repetition, and source agreement are evaluated.

## Status Taxonomy

Primary statuses:

- `SUCCESS_LABEL_VERIFIED`
- `SUCCESS_OVERLAY_AND_LABEL`
- `SUCCESS_OVERLAY_ONLY`
- `SUCCESS_PARTIAL_FIELDS`
- `REVIEW_REQUIRED`
- `REJECTED`
- `PROCESSING_ERROR`

Detailed flags include no/multiple/partial parcel, uncertain detection, label visibility/readability, multiple labels/barcodes, missing/conflicting fields, low quality/OCR, unavailable providers, unsupported images, and duplicates.

`NO_PARCEL` requires strong combined evidence. Failed segmentation alone produces `PARCEL_DETECTION_UNCERTAIN`.

`SUCCESS_OVERLAY_ONLY` remains in the schema so older saved jobs can still be read. New overlay-only AWBs are `REVIEW_REQUIRED` and carry `AWB_NOT_LABEL_VERIFIED`.

## Confidence and Reconciliation

Every extracted field includes its source and confidence.

- OCR token confidence provides the base.
- Explicit field context, stable OCR, and independent-source agreement increase confidence.
- OCR corrections, poor image quality, and conflicts reduce confidence.
- AWB confidence `>=0.90` may be clean.
- AWB confidence `0.75–0.899` remains review-only.
- Lower AWB candidates stay in audit JSON rather than becoming a canonical value.

Agreement rules:

1. Barcode + overlay or barcode + label agreement is strongest.
2. Overlay + label agreement is high confidence.
3. A single strong barcode may be accepted when its format and context are credible.
4. An explicitly anchored overlay AWB remains review-only until label OCR or barcode evidence agrees.
5. Label OCR alone needs strong context and confidence.
6. Unresolved disagreement becomes `AWB_CONFLICT` and `REVIEW_REQUIRED`.

## Batch Processing

- UUID job IDs and internal filenames.
- One job directory for uploads, reports, and annotations.
- SQLite metadata in WAL mode.
- Synchronous verification and an optional background coordinator using the same pipeline.
- Per-image isolation and duplicate-result reuse.
- Active background jobs refresh once per second inside a Streamlit fragment; polling is disabled as soon as the job becomes idle or terminal.
- Up to 500 images per job, submitted to workers in bounded chunks.
- Two workers by default, with a shared 1-8 selector for both ZIP and multi-image uploads on Streamlit Community Cloud.
- Atomic partial CSV/JSON checkpoints after every completed image.
- Last-progress timestamps, stall warnings, and hash-based resume of unfinished inputs.

Jobs survive Streamlit reruns while the application process remains alive. A process restart marks an in-flight job interrupted without deleting its completed records. If the instance-local uploads still exist, **Resume unfinished images** keeps the existing results and processes only missing hashes.

## Background Jobs (Optional)

Synchronous verification is the authoritative path. Background mode runs the same job on a worker with SQLite state and continuously persisted reports, so it can be reopened by job ID while the app instance is running.

Unlike the previous all-or-nothing result screen, a running, cancelled, failed, or interrupted job now exposes every completed record, live review metrics, image details, and partial CSV/JSON downloads. If a job stops at—for example—252/259, those 252 results remain inspectable and downloadable. Resume processes only the seven unfinished files.

Background threads themselves are not durable. Streamlit Community Cloud may restart the process and may eventually discard local files; local execution is the reliable mode for long batches. Production durability still requires external object storage, a database, and a worker queue.

## Local Setup

The evaluator can run the complete local-only workflow without API credentials. Python 3.11 and Tesseract 5 are the only system prerequisites.

### 1. Python environment

Use Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Install Tesseract

Install Tesseract 5 and ensure `tesseract` is on `PATH`.

If installed in a nonstandard location:

```env
TESSERACT_CMD=C:\path\to\Tesseract-OCR\tesseract.exe
```

Linux:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

### 3. Run

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

No API key is required.

Evaluator smoke check:

1. Select **Simple** view.
2. Upload several JPG/PNG files, or one ZIP containing supported images.
3. Select **Process images**.
4. Inspect ready, review, and rejected results.
5. Download `parcel_results.csv` and confirm that every processed image has one row.

## CLI

```powershell
python -m src.cli --input path\to\parcel-images --output sample_outputs\generated
```

The input can be a directory, one image, or a ZIP file.

## Docker

```bash
docker build -t prozo-parcel-extraction .
docker run --rm -p 8501:8501 prozo-parcel-extraction
```

The Docker image installs Tesseract and runs without API credentials.

## Environment Variables

Required: none.

Useful local configuration:

```env
ENABLE_XAI_FALLBACK=false
STREAMLIT_CLOUD=false
RUNTIME_DIR=runtime
MAX_BATCH_IMAGES=500
MAX_WORKERS=2
MAX_WORKER_LIMIT=8
GENERATE_ANNOTATIONS=false
JOB_RETENTION_HOURS=24
JOB_STUCK_SECONDS=300
OCR_TIMEOUT_SECONDS=30
```

Optional xAI configuration:

```env
ENABLE_XAI_FALLBACK=true
XAI_API_KEY=replace-through-secrets
XAI_MODEL=grok-4.5
XAI_TIMEOUT_SECONDS=45
XAI_MAX_RETRIES=2
XAI_MAX_IMAGES_PER_JOB=50
XAI_MAX_CONCURRENCY=2
VISION_CACHE_ENABLED=true
VISION_PROMPT_VERSION=parcel-observation-v2
DETECTOR_MODEL_PATH=models/parcel_detector.onnx
DETECTOR_CONFIDENCE_THRESHOLD=0.35
DETECTOR_IOU_THRESHOLD=0.45
OCR_ENGINES=tesseract
OCR_MAX_PASSES_PER_LABEL=6
OCR_MAX_LABEL_CANDIDATES=1
CALIBRATION_PATH=config/calibration.json
COURIER_RULES_PATH=config/courier_rules.json
```

Never commit `.env` or `.streamlit/secrets.toml`.

Streamlit Community Cloud uses Tesseract plus `opencv-python-headless` only.
RapidOCR remains available for local experiments, but its package brings the
desktop `opencv-python` distribution and is intentionally excluded from the
production requirements. To enable it locally, run
`pip install -r requirements-optional-ocr.txt` and set
`OCR_ENGINES=tesseract,rapidocr`.

## Optional Vision Fallback

xAI is disabled by default. The UI displays a consent checkbox before any image can leave the local machine.

It is used only when parcel count, visibility, label condition, or local extraction is ambiguous. Field-only ambiguity sends the best label crop; parcel/visibility ambiguity sends the downscaled frame and crop. Calls are limited to two in flight, structured responses are cached by content/model/prompt/schema, and provider failures retain the local result.

The diagnostics panel provides an explicit configuration test using a synthetic blank image. Safe error categories include authentication, invalid request, rate limit, timeout, network, malformed response, and server error. API keys, raw images, and sensitive response bodies are not logged.

Vision evidence cannot override a strong barcode, silently resolve a conflict, or turn an unreliable AWB into a clean result.

## CSV Schema

The CSV includes identifiers, processing/status fields, parcel and label state, AWB value/source/confidence, original and normalized weight, dimensions, volume, location, timestamp, barcodes, warnings, confidence, duration, and safe error fields.

`awb_number` contains the exact raw identifier. `awb_number_excel` is an additional validated spreadsheet-safe display column. All fields are quoted and the file is UTF-8 with BOM.

## Privacy and Security

- Local-only by default.
- Signature validation instead of trusting extensions.
- ZIP traversal, nested-archive, compression-ratio, entry-count, and expanded-size controls.
- UUID internal paths and sanitized display filenames.
- Configurable file/image/batch limits.
- No arbitrary local-file serving.
- No raw tracebacks or internal paths in the UI.
- No secrets or full image data in normal logs.
- Runtime uploads, SQLite, reports, and annotations are ignored by Git.

## Testing

```powershell
python -m pytest tests\unit
```

Focused suites:

```powershell
python -m pytest tests\golden
python -m pytest tests\integration
```

Utilities:

```powershell
python -m scripts.inspect_dataset --input path\to\parcel-images
python -m scripts.create_test_fixtures --source path\to\local-parcel.jpg
```

The public unit suite covers validation, ZIP safety, parsing, normalization, AWB validation, reconciliation, confidence, CSV preservation, provider-disabled behavior, corrupt images, provider recovery, and job recovery. Private golden and integration suites remain local because they depend on shipment images and expected AWBs.

The GitHub `Tests` workflow builds a fresh Python 3.11 environment, installs Tesseract and production dependencies, runs the public unit suite, and performs a real Streamlit startup smoke test on every push to `main`.

## Evaluation Evidence

The locally generated `evaluation/ground_truth.csv` contains a human-labelling template with SHA-256 values, near-duplicate groups, and deterministic 70/15/15 splits. It is ignored by Git because it describes private shipment images. System predictions must never be copied into ground truth.

The locally generated `evaluation/baseline_summary.json` covers the larger unlabelled inventory and remains ignored by Git. Its accuracy fields are intentionally unset because unlabelled images cannot establish accuracy. The separate, manually labelled evaluation is reported in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

Build a template from a local job:

```powershell
python -m scripts.build_ground_truth --job-id YOUR_JOB_ID --output evaluation\ground_truth.csv
```

After completing the expected fields and setting `annotated=true`, evaluate the validation split:

```powershell
python -m scripts.evaluate_accuracy `
  --truth evaluation\ground_truth.csv `
  --predictions path\to\shipment_results.csv `
  --split val
```

The command writes `metrics.json`, `failures.csv`, and `threshold_sweep.csv`. Metrics include AWB exact-match precision and recall, false clean AWBs, review recall, parcel-condition accuracy, per-field coverage, review rate, and failure rate. Aggregate results from the private labelled run are published in [EVALUATION_REPORT.md](EVALUATION_REPORT.md); private filenames and shipment values remain excluded. The current result is not a production detector claim: parcel-condition accuracy on the small adverse fixture cohort was 9.1%.

Current measured snapshot:

| Measure | Result | Meaning |
|---|---:|---|
| Target-photo AWB exact-match precision | 100% (17/17) | Every extracted AWB matched the human label in this small target cohort. |
| False clean AWBs | 0 | No known incorrect AWB bypassed review. |
| Adverse-fixture review recall | 100% (10/10) | Every fixture labelled as needing review was routed to review. |
| Parcel-condition accuracy | 9.1% (1/11) | The current generic detector is not adequate for a production parcel-condition claim. |
| Target-photo field coverage | 100% (85/85) | All labelled fields were populated; coverage alone does not prove numeric weight or dimension correctness. |

These figures establish a transparent baseline, not expected performance on unseen evaluator images. Seventeen target photographs are a small sample, and the adverse fixtures are transformations of one source photograph. Weight and dimension accuracy, blocked-label accuracy, and broad courier/background generalisation remain unproven.

Freeze the lowest-review threshold meeting 99% precision:

```powershell
python -m scripts.calibrate_threshold `
  --sweep evaluation\results\threshold_sweep.csv `
  --minimum-precision 0.99 `
  --output config\calibration.json
```

Do not tune after inspecting the test split. The gates are clean-AWB precision ≥99%, detector F1 ≥0.85, review rate ≤35%, and valid-image failure rate <1%.

## Detector Annotation and Google Colab Training

1. Import the source images into CVAT.
2. Annotate boxes using exactly `parcel_full`, `parcel_partial`, `shipping_label_visible`, `shipping_label_blocked`, and `hand_or_obstruction`.
3. Export YOLO format while preserving the manifest splits. Multiple parcels are derived from multiple parcel boxes, not a separate class.
4. Validate the export:

```powershell
python -m scripts.validate_yolo_annotations `
  --dataset path\to\parcel-dataset `
  --manifest evaluation\ground_truth.csv
```

5. Open `training/parcel_detector_colab.ipynb` in a Colab GPU runtime, replace the repository URL placeholder, upload the CVAT export, and run all cells.
6. The notebook uses pinned training dependencies, trains YOLO11n at 640×640, exports fixed-input opset-17 ONNX, and writes a model card.
7. Replace the documented logistics baseline at `models/parcel_detector.onnx` only after the five-class custom artifact passes frozen validation; Streamlit loads the configured artifact once for CPU inference.

Training dependencies are isolated in `training/requirements.txt`. Raw training data, runs, and generated evaluation results are ignored by Git. Review the Ultralytics license before commercial deployment.

## Private Evaluation Data

Supplied parcel images, golden manifests, expected AWBs, generated annotations, and shipment reports remain local and are ignored by Git. The public repository contains the application and evaluation tooling but no shipment records. Generated submission examples are written under `sample_outputs/generated/` locally.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud and create an app.
3. Select `AnirudhVermaCoding/Parcel-Image-to-Shipment-data`, branch `main`, and entrypoint `app.py`.
4. Deploy with Python 3.11. The committed `.python-version` requests that runtime and `packages.txt` installs Tesseract.
5. Open **App settings > Secrets** and paste the non-secret hosted defaults from `.streamlit/secrets.toml.example`. At minimum set:

   ```toml
   STREAMLIT_CLOUD = true
   MAX_BATCH_IMAGES = 500
   MAX_WORKERS = 2
   MAX_WORKER_LIMIT = 8
   ENABLE_XAI_FALLBACK = false
   ```

6. If optional xAI vision is required, add `XAI_API_KEY` in the Secrets editor and set `ENABLE_XAI_FALLBACK = true`. Do not commit `.env` or `.streamlit/secrets.toml`.
7. The committed ONNX artifact is a documented logistics baseline. Replace it only after a custom five-class checkpoint passes frozen validation; without a loadable artifact, diagnostics report the OpenCV fallback.
8. Test ZIP upload, progress recovery, partial CSV/JSON downloads, and consent behavior in a private browser window.
9. Watch the GitHub `Tests` workflow and the Streamlit deployment logs for dependency or resource failures.
10. Record a public URL only if a hosted demonstration is intentionally provided. Local Streamlit and Docker are the supported evaluation paths in this repository.

Use only synthetic or explicitly approved images in a public demonstration.

Community Cloud uses instance-local, ephemeral storage. Completed checkpoints and partial reports remain available across Streamlit reruns on the same process, but they may disappear when the host restarts. A stuck or interrupted job should be resumed by job ID while its uploaded files remain present; production durability requires external storage and a worker queue.

## Known Limitations

- The installed ONNX artifact is a broad logistics baseline, not a validated five-class target-warehouse detector; unsupported parcel and label conditions still rely on conservative OpenCV heuristics.
- Different conveyor backgrounds may reduce segmentation accuracy.
- Unseen courier-label layouts may reduce OCR quality.
- Tiny, reflective, blurred, rotated, or occluded labels may require review.
- Barcode presence does not prove which barcode is the AWB.
- Free hosted infrastructure is not suitable for unlimited batches.
- Streamlit Cloud jobs are not durable across application restarts.
- No authentication or user-specific retention policy is included.
- The published private-set metrics describe only the labelled evaluation cohort and must not be presented as general unseen-warehouse accuracy.

## Production Scaling Path

A production version could add durable object storage, a dedicated API, PostgreSQL, Redis/SQS, worker containers, a trained parcel/label detector, a human-review workflow, access controls, monitoring, and explicit retention policies.

## Improvements With More Time

- Evaluate and train parcel/label detectors on diverse warehouse backgrounds.
- Calibrate confidence on a larger labelled validation set.
- Add courier-specific label templates.
- Add a durable review queue and reviewer feedback loop.
- Add operational monitoring and retention cleanup jobs.
