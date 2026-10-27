"""Modality-specific adapters for the GhostVision application layer."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from .rgb_inference import RGBSegmenter


@dataclass(frozen=True)
class ApplicationSegmentationResult:
    """Application-facing RGB segmentation result."""

    mask: np.ndarray
    overlay: np.ndarray
    modality: str
    model_name: str
    metadata: Mapping[str, object]


@runtime_checkable
class SegmentationAdapter(Protocol):
    """Contract for segmentation backends such as the RGB model."""

    def predict(
        self, image_path: str | Path
    ) -> ApplicationSegmentationResult:
        """Return a mask, overlay, identity, and serializable metadata."""


class RGBSegmentationAdapter:
    """Bind one configured RGB model to the neutral adapter contract."""

    def __init__(
        self, segmenter: RGBSegmenter, *, model_name: str = "augmented"
    ) -> None:
        self._segmenter = segmenter
        self._model_name = model_name

    def predict(
        self, image_path: str | Path
    ) -> ApplicationSegmentationResult:
        prediction = self._segmenter.predict(
            image_path, model_name=self._model_name
        )
        return ApplicationSegmentationResult(
            mask=prediction.mask,
            overlay=prediction.overlay,
            modality="rgb",
            model_name=prediction.model_name,
            metadata={
                "checkpoint_path": str(prediction.checkpoint.path),
                "checkpoint_epoch": prediction.checkpoint.epoch,
                "architecture": prediction.checkpoint.architecture,
                "image_size": prediction.checkpoint.image_size,
                "validation_dice": prediction.checkpoint.validation_dice,
                "threshold": prediction.threshold,
            },
        )


@dataclass(frozen=True)
class ThermalDetection:
    """One thermal animal detection in XYXY pixel coordinates."""

    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ThermalApplicationResult:
    """Thermal detector output without fabricated segmentation fields."""

    image_path: Path
    animal_count: int
    detections: tuple[ThermalDetection, ...]
    model_path: Path
    confidence_threshold: float
    modality: str = "thermal"


ApplicationResult = ApplicationSegmentationResult | ThermalApplicationResult
ThermalDetectorFactory = Callable[..., Any]


class ThermalResultError(RuntimeError):
    """Raised when the thermal detector returns an invalid payload."""


class ThermalDependencyError(RuntimeError):
    """Raised when the lazy thermal inference dependency is unavailable."""


class ThermalDetectionAdapter:
    """Lazy adapter around the real ``ThermalAnimalDetector`` interface.

    The Ultralytics-backed module is imported only when the first thermal
    prediction is requested. Tests can inject a detector factory without
    importing Ultralytics.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.25,
        detector_factory: ThermalDetectorFactory | None = None,
    ) -> None:
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Thermal confidence must be between 0 and 1.")
        self._model_path = Path(model_path).expanduser()
        self._confidence = float(confidence)
        self._detector_factory = detector_factory
        self._detector: Any | None = None

    def _get_detector(self) -> Any:
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"Thermal checkpoint does not exist: {self._model_path}"
            )
        if self._detector is None:
            factory = self._detector_factory
            if factory is None:
                try:
                    module = import_module("thermal_detector")
                except ModuleNotFoundError as exc:
                    raise ThermalDependencyError(
                        "Thermal inference requires the ultralytics package."
                    ) from exc
                factory = module.ThermalAnimalDetector
            self._detector = factory(
                self._model_path, confidence=self._confidence
            )
        return self._detector

    @staticmethod
    def _validate_payload(
        payload: object,
        *,
        model_path: Path,
        confidence: float,
    ) -> ThermalApplicationResult:
        if not isinstance(payload, Mapping):
            raise ThermalResultError(
                "Thermal detector result must be a mapping."
            )
        required = {"image", "animal_count", "detections"}
        missing = required.difference(payload)
        if missing:
            raise ThermalResultError(
                f"Thermal result is missing keys: {sorted(missing)}"
            )

        raw_detections = payload["detections"]
        if not isinstance(raw_detections, (list, tuple)):
            raise ThermalResultError(
                "Thermal detections must be a list or tuple."
            )
        detections = []
        for index, detection in enumerate(raw_detections):
            if not isinstance(detection, Mapping):
                raise ThermalResultError(
                    f"Thermal detection {index} must be a mapping."
                )
            try:
                class_name = str(detection["class"])
                detection_confidence = float(detection["confidence"])
                raw_bbox = detection["bbox"]
                if len(raw_bbox) != 4:
                    raise ValueError("bbox must contain four coordinates")
                bbox = tuple(float(value) for value in raw_bbox)
            except (KeyError, TypeError, ValueError) as exc:
                raise ThermalResultError(
                    f"Thermal detection {index} is invalid: {exc}"
                ) from exc
            if (
                not isfinite(detection_confidence)
                or not 0.0 <= detection_confidence <= 1.0
            ):
                raise ThermalResultError(
                    f"Thermal detection {index} confidence is outside 0..1."
                )
            if not all(isfinite(value) for value in bbox):
                raise ThermalResultError(
                    f"Thermal detection {index} bbox is not finite."
                )
            detections.append(
                ThermalDetection(
                    class_name=class_name,
                    confidence=detection_confidence,
                    bbox=bbox,
                )
            )

        try:
            animal_count = int(payload["animal_count"])
        except (TypeError, ValueError) as exc:
            raise ThermalResultError(
                "Thermal animal_count must be an integer."
            ) from exc
        if animal_count != len(detections):
            raise ThermalResultError(
                "Thermal animal_count does not match detections."
            )

        return ThermalApplicationResult(
            image_path=Path(str(payload["image"])),
            animal_count=animal_count,
            detections=tuple(detections),
            model_path=model_path,
            confidence_threshold=confidence,
        )

    def predict(self, image_path: str | Path) -> ThermalApplicationResult:
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Thermal image does not exist: {path}")
        payload = self._get_detector().detect(path)
        return self._validate_payload(
            payload,
            model_path=self._model_path,
            confidence=self._confidence,
        )
