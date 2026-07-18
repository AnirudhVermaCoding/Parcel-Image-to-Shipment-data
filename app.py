from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.classification.onnx_detector import detector_diagnostics
from src.extraction.vision_fallback import XaiVisionProvider
from src.jobs.manager import JobManager
from src.reporting.csv_report import results_dataframe
from src.schemas import BatchOptions, JobStage
from src.validation.files import (
    InputFile,
    InputValidationError,
    extract_zip_bytes,
    safe_display_name,
    validate_image_bytes,
)

st.set_page_config(
    page_title="Parcel Images to Shipment Data",
    page_icon="📦",
    layout="wide",
)


@st.cache_resource
def get_manager(cache_api_version: int) -> JobManager:
    # The argument is deliberately part of Streamlit's cache key.
    del cache_api_version
    return JobManager(AppConfig.from_env())


config = AppConfig.from_env()
manager = get_manager(JobManager.CACHE_API_VERSION)
required_manager_methods = ("get_job", "get_results", "is_running", "resume")
if not all(callable(getattr(manager, name, None)) for name in required_manager_methods):
    # A development hot reload can otherwise leave an object created from an
    # older class definition in st.cache_resource.
    get_manager.clear()
    manager = get_manager(JobManager.CACHE_API_VERSION)

st.title("Parcel Images → Shipment Data")
st.caption(
    "Local-first OCR, barcode decoding, conservative status flags, and one CSV row per image."
)

with st.expander("Deployment diagnostics", expanded=False):
    detector_info = detector_diagnostics(config)
    st.json(
        {
            "hosted_mode": config.hosted_mode,
            "batch_limit": config.max_batch_images,
            "worker_default": config.default_workers,
            "worker_limit": config.max_workers,
            "detector": detector_info,
            "ocr_engines_configured": list(config.ocr_engines),
            "calibration_version": config.calibration_version,
            "clean_awb_threshold": config.awb_clean_threshold,
            "vision_configured": bool(config.enable_xai_fallback and config.xai_api_key),
            "vision_model": config.xai_model if config.enable_xai_fallback else None,
            "vision_job_cap": config.xai_max_images_per_job,
            "vision_concurrency": config.xai_max_concurrency,
        }
    )
    if config.enable_xai_fallback and config.xai_api_key:
        st.caption("The test sends one synthetic blank image to xAI and may incur a small API charge.")
        if st.button("Test vision configuration"):
            with st.spinner("Testing xAI configuration..."):
                diagnostic = XaiVisionProvider(config).diagnose()
            if diagnostic.success:
                st.success(
                    f"Vision configuration succeeded in {diagnostic.duration_ms} ms "
                    f"after {diagnostic.attempt_count} attempt(s)."
                )
            else:
                st.error(
                    f"Vision test failed: {diagnostic.category}"
                    + (f" (HTTP {diagnostic.http_status})" if diagnostic.http_status else "")
                )

with st.expander("Open an existing job", expanded=False):
    existing_job_id = st.text_input("Job ID", placeholder="Paste a job ID from this app instance")
    if st.button("Open job"):
        existing_job = manager.get_job(existing_job_id.strip())
        if existing_job:
            st.session_state["job_id"] = existing_job.job_id
            st.success(f"Opened job {existing_job.job_id}")
        else:
            st.error("Job not found in the current application instance.")

with st.expander("Background jobs and recovery", expanded=False):
    st.markdown(
        """
        **Synchronous verification** runs the processing path directly and is authoritative.

        **Background job** runs the same pipeline on a worker and checkpoints every completed
        image to SQLite plus partial CSV/JSON reports. You can reopen it by job ID, inspect and
        download intermediate results, and resume only unfinished images after interruption.

        Background threads are process-local. Streamlit reruns are safe, but a hosted platform
        restart stops in-flight work. Saved uploads and checkpoints remain resumable only while
        the platform retains this app instance's local files; local execution is the reliable mode.
        """
    )

with st.expander("How processing works", expanded=False):
    st.markdown(
        """
        1. Validate and hash each image.
        2. Read the yellow machine overlay separately from courier labels.
        3. Decode barcodes and OCR candidate label regions.
        4. Reconcile evidence and flag conflicts instead of guessing.
        5. Save an auditable CSV and JSON report.
        """
    )
    if config.enable_xai_fallback and config.xai_api_key:
        st.warning(
            "Optional vision-assisted mode may send ambiguous images to an external API. "
            "Local-only processing remains available."
        )
    else:
        st.info("External vision is disabled. The complete local-only workflow is available.")

upload_mode = st.radio(
    "Upload mode",
    ("Multiple images", "One ZIP archive"),
    horizontal=True,
)
uploaded = []
if upload_mode == "Multiple images":
    uploaded = st.file_uploader(
        "Upload JPG, JPEG, or PNG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
else:
    zip_upload = st.file_uploader("Upload a ZIP archive", type=["zip"])
    uploaded = [zip_upload] if zip_upload else []

if uploaded:
    total_upload_bytes = sum(len(item.getvalue()) for item in uploaded if item)
    st.info(
        f"Selected {len(uploaded)} upload object(s), "
        f"{total_upload_bytes / (1024 * 1024):.2f} MB total. "
        f"Mode: {'local-only' if not config.enable_xai_fallback else 'local-first'}."
    )

with st.form("job_configuration"):
    left, middle, right = st.columns(3)
    with left:
        generate_annotations = st.checkbox(
            "Generate annotated previews",
            value=config.generate_annotations,
        )
    with middle:
        max_workers = st.number_input(
            "Maximum workers",
            min_value=1,
            max_value=max(1, config.max_workers),
            value=max(1, min(config.default_workers, config.max_workers)),
            help=(
                "Controls how many images are processed concurrently for both upload modes. "
                "Higher values use more CPU and memory; xAI calls remain separately limited."
            ),
        )
        if config.hosted_mode and max_workers > 4:
            st.warning(
                "More than 4 workers can exhaust Streamlit Community Cloud memory on large images."
            )
    with right:
        threshold = st.number_input(
            "Clean AWB confidence",
            min_value=0.75,
            max_value=0.99,
            value=float(config.awb_clean_threshold),
            step=0.01,
            disabled=True,
            help="Loaded from the frozen calibration file or conservative default.",
        )
    enable_vision = False
    if config.enable_xai_fallback and config.xai_api_key:
        enable_vision = st.checkbox(
            "Enable vision assistance for ambiguous images",
            value=False,
        )
        st.checkbox(
            "I understand ambiguous images may leave this machine",
            key="vision_consent",
            value=False,
            help="Required only when vision assistance is enabled.",
        )
    execution_mode = st.radio(
        "Execution mode",
        ("Background job (reopenable)", "Synchronous verification"),
        horizontal=True,
        help=(
            "Synchronous verification is the authoritative direct path. Background mode uses "
            "the same pipeline, checkpoints every result, and can be reopened by job ID while "
            "this application instance is available."
        ),
    )
    submitted = st.form_submit_button("Start batch", type="primary")

if submitted:
    if not uploaded:
        st.error("Upload at least one image or ZIP archive.")
    elif enable_vision and not st.session_state.get("vision_consent"):
        st.error("External-processing consent is required for vision-assisted mode.")
    else:
        input_files: list[InputFile] = []
        validation_messages: list[str] = []
        if upload_mode == "One ZIP archive":
            try:
                input_files, validation_messages = extract_zip_bytes(
                    uploaded[0].name,
                    uploaded[0].getvalue(),
                    config,
                )
            except InputValidationError as exc:
                st.error(str(exc))
        else:
            for item in uploaded:
                data = item.getvalue()
                name = safe_display_name(item.name)
                try:
                    validate_image_bytes(name, data, config)
                except InputValidationError as exc:
                    validation_messages.append(str(exc))
                input_files.append(InputFile(name, data))
        if input_files:
            if len(input_files) > config.max_batch_images:
                st.error(f"Batch limit is {config.max_batch_images} images.")
            else:
                background = execution_mode == "Background job (reopenable)"
                job_id = manager.submit(
                    input_files,
                    BatchOptions(
                        local_only=not enable_vision,
                        enable_vision=enable_vision,
                        generate_annotations=generate_annotations,
                        max_workers=int(max_workers),
                        awb_acceptance_threshold=threshold,
                    ),
                    background=background,
                )
                st.session_state["job_id"] = job_id
                st.session_state["validation_messages"] = validation_messages
                st.success(
                    f"{'Batch queued' if background else 'Verification completed'}. "
                    f"Job ID: {job_id}"
                )

for message in st.session_state.get("validation_messages", []):
    st.warning(message)


TERMINAL_STAGES = {
    JobStage.COMPLETED,
    JobStage.COMPLETED_WITH_WARNINGS,
    JobStage.FAILED,
    JobStage.CANCELLED,
}

# Only enable the 1-second auto-refresh while a job is actually live. Leaving
# ``run_every`` active when idle or finished keeps scheduling fragment reruns; a
# later full-app rerun then removes the fragment before the pending timer fires,
# which is what logs the "fragment ... does not exist anymore" warning.
_monitor_job_id = st.session_state.get("job_id")
_monitor_job = manager.get_job(_monitor_job_id) if _monitor_job_id else None
_monitor_should_poll = _monitor_job is not None and _monitor_job.stage not in TERMINAL_STAGES


@st.fragment(run_every=1 if _monitor_should_poll else None)
def job_monitor() -> None:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return
    job = manager.get_job(job_id)
    if job is None:
        st.error("The selected job could not be found.")
        return
    st.subheader("Job progress")
    st.code(job.job_id)
    metric_columns = st.columns(5)
    metric_columns[0].metric("Stage", job.stage.value)
    metric_columns[1].metric("Processed", f"{job.processed_images}/{job.total_images}")
    metric_columns[2].metric("Success", job.successful_images)
    metric_columns[3].metric("Review", job.review_required_images)
    metric_columns[4].metric("Failed", job.failed_images)
    st.progress(min(1.0, job.progress_percentage / 100))
    if job.current_image:
        st.caption(f"Current image: {job.current_image}")
    terminal = job.stage in TERMINAL_STAGES
    if _monitor_should_poll and terminal:
        # The job finished during a polling tick. Trigger one full-app rerun so
        # the fragment is re-mounted with run_every disabled, cleanly stopping
        # the auto-refresh timer instead of leaving it pending.
        st.rerun()
    stalled = False
    if not terminal and job.last_progress_at:
        stalled_seconds = max(0, int((datetime.now() - job.last_progress_at).total_seconds()))
        stalled = stalled_seconds >= config.job_stuck_seconds
        st.caption(f"Last completed-image progress: {stalled_seconds} seconds ago")
        if stalled:
            st.warning(
                "This job has not completed another image within the configured stall window. "
                "Its existing results and checkpoint downloads remain available."
            )
    if not terminal:
        if st.button("Cancel job"):
            manager.cancel(job_id)
            st.info("Cancellation requested. A currently executing OCR/API call must return first.")
    if job.stage in {JobStage.FAILED, JobStage.CANCELLED}:
        st.error(job.error_message or f"Job ended with status {job.stage.value}")

    can_resume = (
        job.processed_images < job.total_images
        and not manager.is_running(job_id)
        and (terminal or stalled)
    )
    if can_resume:
        if st.button("Resume unfinished images", type="primary"):
            try:
                pending = manager.resume(job_id, background=True)
                st.success(
                    f"Resume started for {pending} unfinished image(s). "
                    f"The existing {job.processed_images} result(s) were preserved."
                )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

    results = manager.get_results(job_id)
    if not results:
        st.info("No image results have been checkpointed yet.")
        return
    dataframe = results_dataframe(results)
    st.subheader("Current results" if not terminal else "Batch summary")
    if job.checkpoint_at:
        st.caption(
            f"Checkpoint: {job.checkpoint_at.isoformat(timespec='seconds')} — "
            f"{len(results)} result(s) persisted."
        )
    summary_columns = st.columns(7)
    summary_columns[0].metric("Images", len(results))
    summary_columns[1].metric("Clean", int((~dataframe["requires_review"]).sum()))
    summary_columns[2].metric("Review", int(dataframe["requires_review"].sum()))
    summary_columns[3].metric("AWBs", int((dataframe["awb_number"] != "").sum()))
    summary_columns[4].metric("Barcodes", int((dataframe["barcode_values"] != "").sum()))
    summary_columns[5].metric(
        "Overlay only",
        int((dataframe["primary_status"] == "SUCCESS_OVERLAY_ONLY").sum()),
    )
    summary_columns[6].metric("Warnings", job.warning_count)

    review_reason_counts: dict[str, int] = {}
    for result in results:
        for reason in result.review_reasons:
            review_reason_counts[reason] = review_reason_counts.get(reason, 0) + 1
    vision_attempts = sum(result.vision_attempted for result in results)
    vision_successes = sum(result.vision_succeeded for result in results)
    vision_cache_hits = sum(result.vision_cache_hit for result in results)
    vision_failures = vision_attempts - vision_successes
    if review_reason_counts:
        st.caption(
            "Review reasons: "
            + ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(
                    review_reason_counts.items(), key=lambda item: item[1], reverse=True
                )
            )
        )
    if vision_attempts:
        st.caption(
            f"Vision usage: attempts={vision_attempts}, successes={vision_successes}, "
            f"cache hits={vision_cache_hits}, failures={vision_failures}, "
            f"configured cap={config.xai_max_images_per_job}."
        )

    filter_row_one = st.columns(3)
    status_values = sorted(dataframe["primary_status"].dropna().unique().tolist())
    selected_statuses = filter_row_one[0].multiselect("Primary status", status_values)
    source_values = sorted(dataframe["awb_source"].dropna().unique().tolist())
    selected_sources = filter_row_one[1].multiselect("AWB source", source_values)
    label_values = sorted(dataframe["label_status"].dropna().unique().tolist())
    selected_labels = filter_row_one[2].multiselect("Label status", label_values)

    filter_row_two = st.columns(4)
    flag_values = sorted(
        {
            flag
            for value in dataframe["status_flags"]
            for flag in str(value).split(";")
            if flag
        }
    )
    selected_flags = filter_row_two[0].multiselect("Status flag", flag_values)
    review_filter = filter_row_two[1].selectbox(
        "Review filter",
        ("All", "Requires review", "Clean only"),
    )
    filename_query = filter_row_two[2].text_input("Filename contains")
    confidence_range = filter_row_two[3].slider(
        "Overall confidence",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
    )
    filtered = dataframe
    if selected_statuses:
        filtered = filtered[filtered["primary_status"].isin(selected_statuses)]
    if selected_sources:
        filtered = filtered[filtered["awb_source"].isin(selected_sources)]
    if selected_labels:
        filtered = filtered[filtered["label_status"].isin(selected_labels)]
    if selected_flags:
        filtered = filtered[
            filtered["status_flags"].apply(
                lambda value: any(flag in str(value).split(";") for flag in selected_flags)
            )
        ]
    if review_filter == "Requires review":
        filtered = filtered[filtered["requires_review"]]
    elif review_filter == "Clean only":
        filtered = filtered[~filtered["requires_review"]]
    if filename_query:
        filtered = filtered[
            filtered["filename"].str.contains(filename_query, case=False, na=False)
        ]
    numeric_confidence = pd.to_numeric(filtered["overall_confidence"], errors="coerce").fillna(0)
    filtered = filtered[
        numeric_confidence.between(confidence_range[0], confidence_range[1])
    ]
    st.dataframe(filtered, width="stretch", hide_index=True)

    review_queue = dataframe[dataframe["requires_review"]]
    with st.expander(f"Review queue ({len(review_queue)})", expanded=False):
        st.dataframe(review_queue, width="stretch", hide_index=True)

    filenames = [result.original_filename for result in results]
    selected_filename = st.selectbox("Image detail", filenames)
    selected = next(result for result in results if result.original_filename == selected_filename)
    detail_columns = st.columns([1.2, 1])
    input_path = manager.database.result_input_path(job_id, selected.image_id)
    with detail_columns[0]:
        if input_path and Path(input_path).exists():
            st.image(input_path, caption=selected.original_filename, width="stretch")
        if selected.annotated_image_path and Path(selected.annotated_image_path).exists():
            st.image(
                selected.annotated_image_path,
                caption="Annotated evidence preview",
                width="stretch",
            )
    with detail_columns[1]:
        st.json(
            {
                "primary_status": selected.primary_status.value,
                "status_flags": [flag.value for flag in selected.status_flags],
                "requires_review": selected.requires_review,
                "review_reasons": selected.review_reasons,
                "awb": {
                    "value": selected.awb_number,
                    "source": selected.awb_source.value,
                    "confidence": selected.awb_confidence,
                },
                "weight_grams": selected.weight_grams,
                "weight_source": selected.weight_source.value,
                "weight_confidence": selected.weight_confidence,
                "dimensions_cm": [
                    selected.length_cm,
                    selected.width_cm,
                    selected.height_cm,
                ],
                "dimensions_source": selected.dimensions_source.value,
                "dimensions_confidence": selected.dimensions_confidence,
                "recorded_volume": selected.recorded_volume,
                "recorded_volume_source": selected.recorded_volume_source.value,
                "recorded_volume_confidence": selected.recorded_volume_confidence,
                "location": selected.location,
                "location_source": selected.location_source.value,
                "location_confidence": selected.location_confidence,
                "timestamp": (
                    selected.capture_timestamp.isoformat()
                    if selected.capture_timestamp
                    else None
                ),
                "timestamp_source": selected.capture_timestamp_source.value,
                "timestamp_confidence": selected.capture_timestamp_confidence,
                "parcel_count": selected.parcel_count,
                "parcel_visibility": selected.parcel_visibility.value,
                "label_status": selected.label_status.value,
                "barcode_values": selected.barcode_values,
                "warnings": selected.warnings,
                "detector": {
                    "used": selected.detector_used,
                    "version": selected.detector_version,
                    "confidence": selected.detector_confidence,
                },
                "ocr_engines_used": selected.ocr_engines_used,
                "vision": {
                    "attempted": selected.vision_attempted,
                    "succeeded": selected.vision_succeeded,
                    "cache_hit": selected.vision_cache_hit,
                    "error_category": selected.vision_error_category,
                    "attempt_count": selected.vision_attempt_count,
                    "duration_ms": selected.vision_duration_ms,
                },
            }
        )
        with st.expander("Raw OCR evidence"):
            st.text_area("Machine overlay OCR", selected.overlay_ocr_text, height=180)
            for index, text in enumerate(selected.label_ocr_texts):
                st.text_area(f"Label OCR {index + 1}", text, height=140)

    if job.report_path and Path(job.report_path).exists():
        st.download_button(
            f"Download {'batch' if job.stage in {JobStage.COMPLETED, JobStage.COMPLETED_WITH_WARNINGS} else 'current checkpoint'} CSV",
            data=Path(job.report_path).read_bytes(),
            file_name="shipment_results.csv",
            mime="text/csv",
        )
    if job.json_report_path and Path(job.json_report_path).exists():
        st.download_button(
            f"Download {'audit' if job.stage in {JobStage.COMPLETED, JobStage.COMPLETED_WITH_WARNINGS} else 'current checkpoint'} JSON",
            data=Path(job.json_report_path).read_bytes(),
            file_name="shipment_results.json",
            mime="application/json",
        )
    annotation_files = [
        Path(result.annotated_image_path)
        for result in results
        if result.annotated_image_path and Path(result.annotated_image_path).exists()
    ]
    if annotation_files:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in annotation_files:
                archive.write(path, path.name)
        st.download_button(
            "Download annotated previews",
            data=buffer.getvalue(),
            file_name="annotated_previews.zip",
            mime="application/zip",
        )


job_monitor()

st.divider()
with st.expander("Technical information", expanded=False):
    st.json(
        {
            "processing_mode": "local-first",
            "ocr": "Tesseract; development RapidOCR fallback when installed",
            "barcode_decoder": "zxing-cpp",
            "max_batch_images": config.max_batch_images,
            "default_workers": config.default_workers,
            "worker_limit": config.max_workers,
            "vision_configured": bool(config.enable_xai_fallback and config.xai_api_key),
            "detector": detector_diagnostics(config),
            "calibration_version": config.calibration_version,
            "runtime_persistence": "SQLite and per-job folders for this process instance",
        }
    )
st.caption(
    "Privacy: images are processed locally unless vision assistance is explicitly enabled. "
    "Runtime files are temporary and must not be committed."
)
