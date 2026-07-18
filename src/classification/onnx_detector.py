from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig
from src.schemas import BoundingBox, LabelStatus, ParcelVisibility


CLASS_NAMES = (
    "parcel_full",
    "parcel_partial",
    "shipping_label_visible",
    "shipping_label_blocked",
    "hand_or_obstruction",
)


@dataclass(slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(slots=True)
class DetectorEvidence:
    available: bool = False
    used: bool = False
    version: str | None = None
    confidence: float = 0.0
    detections: list[Detection] = field(default_factory=list)
    parcel_count: int | None = None
    parcel_visibility: ParcelVisibility = ParcelVisibility.UNCERTAIN
    parcel_bboxes: list[BoundingBox] = field(default_factory=list)
    label_status: LabelStatus = LabelStatus.LABEL_STATUS_UNCERTAIN
    label_bboxes: list[BoundingBox] = field(default_factory=list)
    error: str | None = None


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    x1, y1 = max(first.x, second.x), max(first.y, second.y)
    x2 = min(first.x + first.width, second.x + second.width)
    y2 = min(first.y + first.height, second.y + second.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


class OnnxParcelDetector:
    def __init__(self, model_path: str, input_size: int = 640):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self.input_size = input_size
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:12]
        self.version = f"{self.model_path.stem}:{digest}"
        self.lock = threading.Lock()

    def _preprocess(self, bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = bgr.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        left = (self.input_size - resized_width) // 2
        top = (self.input_size - resized_height) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[top : top + resized_height, left : left + resized_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None]
        return tensor, scale, left, top

    @staticmethod
    def _rows(output: np.ndarray) -> np.ndarray:
        values = np.asarray(output)
        if values.ndim == 3:
            values = values[0]
        if values.ndim != 2:
            raise ValueError(f"Unsupported detector output shape: {values.shape}")
        expected = 4 + len(CLASS_NAMES)
        if values.shape[0] == expected and values.shape[1] != expected:
            values = values.T
        if values.shape[1] < expected:
            raise ValueError(f"Detector output has {values.shape[1]} values per box; expected {expected}")
        return values

    def detect(self, bgr: np.ndarray, config: AppConfig) -> DetectorEvidence:
        tensor, scale, left, top = self._preprocess(bgr)
        with self.lock:
            output = self.session.run(None, {self.input_name: tensor})[0]
        rows = self._rows(output)
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        original_height, original_width = bgr.shape[:2]
        for row in rows:
            class_scores = row[4 : 4 + len(CLASS_NAMES)]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < config.detector_confidence_threshold:
                continue
            center_x, center_y, width, height = map(float, row[:4])
            x = int(round((center_x - width / 2 - left) / scale))
            y = int(round((center_y - height / 2 - top) / scale))
            width = int(round(width / scale))
            height = int(round(height / scale))
            x, y = max(0, x), max(0, y)
            width = min(width, original_width - x)
            height = min(height, original_height - y)
            if width <= 1 or height <= 1:
                continue
            boxes.append([x, y, width, height])
            scores.append(score)
            class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            config.detector_confidence_threshold,
            config.detector_iou_threshold,
        )
        detections = [
            Detection(
                class_name=CLASS_NAMES[class_ids[int(index)]],
                confidence=round(scores[int(index)], 4),
                bbox=BoundingBox(
                    x=boxes[int(index)][0],
                    y=boxes[int(index)][1],
                    width=boxes[int(index)][2],
                    height=boxes[int(index)][3],
                ),
            )
            for index in np.asarray(indices).reshape(-1)
        ] if len(indices) else []
        parcel_detections = [d for d in detections if d.class_name.startswith("parcel_")]
        label_detections = [d for d in detections if d.class_name.startswith("shipping_label_")]
        blocked = [d for d in label_detections if d.class_name == "shipping_label_blocked"]
        hands = [d for d in detections if d.class_name == "hand_or_obstruction"]
        if not blocked and any(
            _iou(label.bbox, hand.bbox) > 0.08 for label in label_detections for hand in hands
        ):
            blocked = label_detections
        visibility = ParcelVisibility.UNCERTAIN
        if parcel_detections:
            visibility = (
                ParcelVisibility.PARTIAL
                if any(d.class_name == "parcel_partial" for d in parcel_detections)
                else ParcelVisibility.FULL
            )
        if blocked:
            label_status = LabelStatus.LABEL_BLOCKED_OR_OCCLUDED
        elif label_detections:
            label_status = LabelStatus.LABEL_VISIBLE_LOW_CONFIDENCE
        else:
            label_status = LabelStatus.LABEL_STATUS_UNCERTAIN
        return DetectorEvidence(
            available=True,
            used=True,
            version=self.version,
            confidence=round(max((d.confidence for d in detections), default=0.0), 4),
            detections=detections,
            parcel_count=len(parcel_detections) or None,
            parcel_visibility=visibility,
            parcel_bboxes=[d.bbox for d in parcel_detections],
            label_status=label_status,
            label_bboxes=[d.bbox for d in label_detections],
        )


@lru_cache(maxsize=2)
def _load_detector(model_path: str, input_size: int) -> OnnxParcelDetector:
    return OnnxParcelDetector(model_path, input_size)


def detect_with_onnx(bgr: np.ndarray, config: AppConfig) -> DetectorEvidence:
    model_path = Path(config.detector_model_path)
    if not model_path.is_file():
        return DetectorEvidence(error="model_not_found")
    try:
        return _load_detector(str(model_path.resolve()), config.detector_input_size).detect(bgr, config)
    except Exception as exc:
        return DetectorEvidence(error=f"{type(exc).__name__}")


def detector_diagnostics(config: AppConfig) -> dict[str, object]:
    path = Path(config.detector_model_path)
    if not path.is_file():
        return {"available": False, "path": str(path), "version": None, "error": "model_not_found"}
    try:
        detector = _load_detector(str(path.resolve()), config.detector_input_size)
        return {"available": True, "path": str(path), "version": detector.version, "error": None}
    except Exception as exc:
        return {"available": False, "path": str(path), "version": None, "error": type(exc).__name__}
