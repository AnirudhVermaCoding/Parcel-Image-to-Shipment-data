# Parcel Images to Shipment Data

A local-first Streamlit case study that converts batches of warehouse parcel photos into structured shipment records and a downloadable CSV.

The design principle is simple:

> A wrong AWB is worse than an honest review flag.

The app extracts tracking/AWB number, weight, and dimensions when evidence is reliable. Difficult images receive readable review or rejection reasons such as multiple parcels, partial parcel, unreadable label, conflicting AWBs, or no parcel.

## Run Locally in Under 15 Minutes

### Prerequisites

- Python 3.11
- Tesseract OCR 5 available on `PATH`
- Git

No API key, GPU, CUDA, Redis, PostgreSQL, or external OCR service is required.

### Windows PowerShell

```powershell
git clone https://github.com/AnirudhVermaCoding/Parcel-Image-to-Shipment-data.git
cd Parcel-Image-to-Shipment-data
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

If Tesseract is installed outside `PATH`, set its executable before starting Streamlit:

```powershell
$env:TESSERACT_CMD="C:\path\to\Tesseract-OCR\tesseract.exe"
```

### Linux

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

### Quick Evaluation Flow

1. Open **Simple** view.
2. Upload multiple JPG/JPEG/PNG files, or one ZIP archive.
3. Select **Process images**.
4. Review the ready, review, and rejected results.
5. Download `parcel_results.csv`.

The default batch limit is 500 images and is configurable through `MAX_BATCH_IMAGES`.

### CLI and Tests

```powershell
python -m src.cli --input path\to\parcel-images --output sample_outputs\generated
python -m pip install -r requirements-dev.txt
python -m pytest tests\unit -q
```

Docker is also supported:

```bash
docker build -t parcel-extraction .
docker run --rm -p 8501:8501 parcel-extraction
```

## Output

Every processed image receives one structured result. One image failing does not stop the rest of the batch.

The warehouse CSV contains:

- Filename
- Tracking/AWB number
- Weight
- Length, width, and height
- Location when present
- Readable result
- Needs-review marker
- Plain-English reason

The detailed CSV and JSON additionally contain evidence source, confidence, OCR text, barcode values, parcel and label state, detector metadata, warnings, and safe error information.

AWBs are preserved as text for spreadsheet compatibility.

## Approach and Why

### 1. Validate and isolate every image

File signature, size, dimensions, pixel count, and decoding are checked before extraction. Images are hashed for duplicate detection and corrected for EXIF orientation. Each image has its own exception boundary so a corrupt file cannot terminate a large batch.

### 2. Keep evidence sources separate

The pipeline distinguishes:

- `OVERLAY_OCR` — machine-generated yellow overlay
- `LABEL_OCR` — courier-label text
- `BARCODE` — decoded barcode
- `VISION_FALLBACK` — optional external assistance
- `NOT_EXTRACTED` — no reliable value

This matters because machine-overlay text is not courier-label evidence. An overlay-only AWB is retained but always sent to review until matching label OCR or barcode evidence is found.

### 3. Use several local extraction methods

OpenCV and Pillow handle image preparation and quality checks. Tesseract performs multi-pass OCR. `zxing-cpp` decodes barcodes. Label crops are enhanced and rotated before OCR. Explicit field labels such as `AWB`, `Weight`, `Length`, `Width`, and `Height` are parsed instead of accepting arbitrary numbers.

### 4. Treat barcodes as evidence, not automatic truth

A parcel image can contain product, routing, or inventory barcodes. A barcode is promoted as an AWB only when its format and context are credible or it agrees with OCR evidence. Conflicting values become `AWB_CONFLICT` and require review.

### 5. Reconcile evidence conservatively

Independent agreement is stronger than a single reading:

1. Barcode plus label or overlay agreement
2. Overlay plus label agreement
3. A strong contextual barcode or label reading
4. Overlay-only or medium-confidence evidence, which remains review-only

AWBs below the review threshold are not promoted to the canonical AWB field.

### 6. Flag uncertainty explicitly

The status engine covers no parcel, multiple parcels, partial visibility, uncertain detection, missing/blocked/unreadable labels, multiple labels or barcodes, missing fields, low image quality, OCR failure, AWB conflict, unsupported input, duplicates, and processing errors.

Background jobs checkpoint every completed image and support partial downloads, cancellation, resume, and process-local recovery.

## Technology Stack

- Python 3.11 and Streamlit
- OpenCV headless, Pillow, and NumPy
- Tesseract through `pytesseract`
- `zxing-cpp` for barcodes
- ONNX Runtime for CPU detection
- Pandas and Pydantic
- SQLite and `ThreadPoolExecutor`
- Pytest and GitHub Actions

Optional xAI vision assistance is disabled by default and requires explicit UI consent. Strong local evidence cannot be silently overridden by the optional provider.

## Current Evaluation Evidence

The detailed methodology is in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

| Measure | Result |
|---|---:|
| AWB exact-match precision on 17 target photos | 100% (17/17) |
| False clean AWBs across evaluated cohorts | 0 |
| Review recall on adverse fixtures | 100% (10/10) |
| Parcel-condition accuracy on adverse fixtures | 9.1% (1/11) |
| Target-photo field coverage | 100% (85/85) |

These numbers are a small, transparent baseline—not a general accuracy claim. Field coverage means a value was extracted; it does not prove weight or dimension correctness. The adverse fixtures are transformations of one source image and are not independent unseen warehouse photographs.

## Known Limitations

- The bundled ONNX model is a broad logistics baseline, not a validated five-class detector for this warehouse domain.
- Parcel-condition accuracy is currently poor; multiple, partial, empty, and visibility decisions need a trained detector and larger labelled test set.
- Blocked-label accuracy has not been established.
- Weight and dimension numeric accuracy has not yet been reported, only coverage.
- Tiny, reflective, blurred, rotated, dark, overexposed, or occluded labels may require review.
- A barcode cannot by itself prove that its value is the AWB.
- Background workers and files are process-local and are not durable across host restarts.
- There is no authentication or user-specific retention policy.
- The bundled model metadata references AGPL-3.0; licensing obligations must be reviewed before commercial use.

## What I Would Improve With More Time

1. Label a larger, independent dataset covering warehouse backgrounds, courier layouts, empty frames, multiple parcels, partial parcels, and obstructions.
2. Train and freeze the intended five-class detector: `parcel_full`, `parcel_partial`, `shipping_label_visible`, `shipping_label_blocked`, and `hand_or_obstruction`.
3. Report weight and dimension exact accuracy or tolerance-based error, not only field coverage.
4. Calibrate confidence thresholds on a frozen validation set and evaluate once on an untouched test set.
5. Add a durable human-review queue with reviewer corrections feeding future evaluation.
6. Add object storage, a durable worker queue, authentication, monitoring, and explicit retention controls for production use.

## Main Files

- `app.py` — Streamlit interface
- `src/pipeline.py` — shared image-processing pipeline
- `src/classification/status_engine.py` — status and review decisions
- `src/reconciliation/` — candidate agreement and confidence
- `src/reporting/` — CSV, JSON, presentation, and annotations
- `scripts/evaluate_accuracy.py` — labelled evaluation
- `models/parcel_detector.model-card.md` — detector provenance and limitations
