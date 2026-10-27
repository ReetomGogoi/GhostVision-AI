"""Unified application router for RGB segmentation and thermal detection."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from .adapters import (
    ApplicationResult,
    RGBSegmentationAdapter,
    ThermalDetectionAdapter,
)
from .rgb_inference import RGBSegmenter


class ApplicationAdapter(Protocol):
    """Common call shape; each modality keeps its own result type."""

    def predict(self, image_path: str | Path) -> ApplicationResult:
        """Run one inference request."""


class UnsupportedModalityError(ValueError):
    """Raised when a modality other than RGB or thermal is requested."""


class ModalityNotConfiguredError(RuntimeError):
    """Raised when a supported modality has no configured backend."""


class GhostVisionApplication:
    """Route application requests to independently configured modalities.

    Passing checkpoint paths creates lazy RGB and thermal backends. Tests and
    host applications may inject adapters directly.
    """

    SUPPORTED_MODALITIES = frozenset({"rgb", "thermal"})

    def __init__(
        self,
        *,
        rgb_checkpoint_paths: Mapping[str, str | Path] | None = None,
        rgb_model_name: str = "augmented",
        thermal_model_path: str | Path | None = None,
        thermal_confidence: float = 0.25,
        rgb_adapter: ApplicationAdapter | None = None,
        thermal_adapter: ApplicationAdapter | None = None,
    ) -> None:
        if rgb_adapter is not None and rgb_checkpoint_paths is not None:
            raise ValueError(
                "Provide either rgb_adapter or rgb_checkpoint_paths, not both."
            )
        if thermal_adapter is not None and thermal_model_path is not None:
            raise ValueError(
                "Provide either thermal_adapter or thermal_model_path, "
                "not both."
            )

        adapters: dict[str, ApplicationAdapter] = {}
        if rgb_adapter is not None:
            adapters["rgb"] = rgb_adapter
        elif rgb_checkpoint_paths is not None:
            adapters["rgb"] = RGBSegmentationAdapter(
                RGBSegmenter(rgb_checkpoint_paths),
                model_name=rgb_model_name,
            )

        if thermal_adapter is not None:
            adapters["thermal"] = thermal_adapter
        elif thermal_model_path is not None:
            adapters["thermal"] = ThermalDetectionAdapter(
                thermal_model_path,
                confidence=thermal_confidence,
            )
        self._adapters = adapters

    @property
    def available_modalities(self) -> tuple[str, ...]:
        """Return configured modalities in stable order."""

        return tuple(
            modality
            for modality in ("rgb", "thermal")
            if modality in self._adapters
        )

    def predict(
        self, modality: str, image_path: str | Path
    ) -> ApplicationResult:
        """Run the selected modality without translating its result shape."""

        if not isinstance(modality, str):
            raise UnsupportedModalityError(
                "Modality must be 'rgb' or 'thermal'."
            )
        normalized = modality.strip().lower()
        if normalized not in self.SUPPORTED_MODALITIES:
            raise UnsupportedModalityError(
                f"Unsupported modality {modality!r}; choose rgb or thermal."
            )
        if normalized not in self._adapters:
            raise ModalityNotConfiguredError(
                f"Modality {normalized!r} is not configured."
            )
        return self._adapters[normalized].predict(image_path)
