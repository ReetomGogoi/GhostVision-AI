"""Application-facing segmentation adapter contract.

No thermal inference interface is present on ``feature/rgb``. Thermal support
can implement ``SegmentationAdapter`` once its real return types and image
requirements are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from .rgb_inference import RGBSegmenter


@dataclass(frozen=True)
class ApplicationSegmentationResult:
    """Modality-neutral result consumed by a future application layer."""

    mask: np.ndarray
    overlay: np.ndarray
    modality: str
    model_name: str
    metadata: Mapping[str, object]


@runtime_checkable
class SegmentationAdapter(Protocol):
    """Minimal contract shared by independent RGB and thermal adapters."""

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
