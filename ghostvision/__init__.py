"""Reusable inference interfaces for GhostVision-AI."""

from .adapters import (
    ApplicationSegmentationResult,
    RGBSegmentationAdapter,
    SegmentationAdapter,
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
    "ApplicationSegmentationResult",
    "CheckpointInfo",
    "IncompatibleCheckpointError",
    "InferenceDependencyError",
    "InvalidImageError",
    "MissingCheckpointError",
    "RGBPredictionResult",
    "RGBSegmentationAdapter",
    "RGBSegmenter",
    "SegmentationAdapter",
    "UnknownModelError",
]
