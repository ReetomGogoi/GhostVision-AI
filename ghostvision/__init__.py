"""Reusable inference interfaces for GhostVision-AI."""

from .adapters import (
    ApplicationResult,
    ApplicationSegmentationResult,
    RGBSegmentationAdapter,
    SegmentationAdapter,
    ThermalApplicationResult,
    ThermalDependencyError,
    ThermalDetection,
    ThermalDetectionAdapter,
    ThermalResultError,
)
from .application import (
    GhostVisionApplication,
    ModalityNotConfiguredError,
    UnsupportedModalityError,
)
from .rgb_inference import (
    CheckpointInfo,
    IncompatibleCheckpointError,
    InferenceDependencyError,
    InvalidImageError,
    MissingCheckpointError,
    RGBPredictionResult,
    RGBSegmenter,
    UnknownModelError,
)

__all__ = [
    "ApplicationResult",
    "ApplicationSegmentationResult",
    "CheckpointInfo",
    "GhostVisionApplication",
    "IncompatibleCheckpointError",
    "InferenceDependencyError",
    "InvalidImageError",
    "MissingCheckpointError",
    "ModalityNotConfiguredError",
    "RGBPredictionResult",
    "RGBSegmentationAdapter",
    "RGBSegmenter",
    "SegmentationAdapter",
    "ThermalApplicationResult",
    "ThermalDependencyError",
    "ThermalDetection",
    "ThermalDetectionAdapter",
    "ThermalResultError",
    "UnknownModelError",
    "UnsupportedModalityError",
]
