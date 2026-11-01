"""FastAPI transport for the existing GhostVision inference application.

This module validates uploads and serializes the two modality-specific result
types. Model construction, preprocessing, and inference remain owned by
``GhostVisionApplication`` and its existing adapters.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .adapters import (
    ApplicationSegmentationResult,
    ThermalApplicationResult,
    ThermalDependencyError,
    ThermalResultError,
)
from .application import GhostVisionApplication, ModalityNotConfiguredError
from .rgb_inference import (
    IncompatibleCheckpointError,
    InferenceDependencyError,
    InvalidImageError,
    MissingCheckpointError,
    UnknownModelError,
)

LOGGER = logging.getLogger(__name__)

ENV_RGB_BASELINE = "GHOSTVISION_RGB_BASELINE"
ENV_RGB_AUGMENTED = "GHOSTVISION_RGB_AUGMENTED"
ENV_THERMAL_CHECKPOINT = "GHOSTVISION_THERMAL_CHECKPOINT"
ENV_THERMAL_CONFIDENCE = "GHOSTVISION_THERMAL_CONFIDENCE"
ENV_ALLOWED_ORIGINS = "GHOSTVISION_ALLOWED_ORIGINS"

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
RGB_MODELS = frozenset({"baseline", "augmented"})


@dataclass(frozen=True)
class APISettings:
    """Checkpoint paths supplied by the host environment."""

    rgb_checkpoints: Mapping[str, Path]
    thermal_checkpoint: Path | None
    thermal_confidence: float = 0.25
    allowed_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    @classmethod
    def from_environment(cls) -> "APISettings":
        rgb_checkpoints: dict[str, Path] = {}
        for model_name, variable_name in (
            ("baseline", ENV_RGB_BASELINE),
            ("augmented", ENV_RGB_AUGMENTED),
        ):
            value = os.environ.get(variable_name, "").strip()
            if value:
                rgb_checkpoints[model_name] = Path(value).expanduser()

        thermal_value = os.environ.get(ENV_THERMAL_CHECKPOINT, "").strip()
        confidence_text = os.environ.get(
            ENV_THERMAL_CONFIDENCE, "0.25"
        ).strip()
        try:
            confidence = float(confidence_text)
        except ValueError as exc:
            raise ValueError(
                f"{ENV_THERMAL_CONFIDENCE} must be a number."
            ) from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"{ENV_THERMAL_CONFIDENCE} must be between 0 and 1."
            )
        allowed_origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                ENV_ALLOWED_ORIGINS,
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        )

        return cls(
            rgb_checkpoints=rgb_checkpoints,
            thermal_checkpoint=(
                Path(thermal_value).expanduser() if thermal_value else None
            ),
            thermal_confidence=confidence,
            allowed_origins=allowed_origins,
        )


class ApplicationLike(Protocol):
    """Narrow protocol used by the HTTP layer and test doubles."""

    def predict(self, modality: str, image_path: str | Path) -> object:
        """Run one modality-specific prediction."""


class ApplicationProvider(Protocol):
    def __call__(self, modality: str, rgb_model: str) -> ApplicationLike:
        """Return the configured application for a request."""


class APIConfigurationError(RuntimeError):
    """Raised when a requested checkpoint has not been configured."""


class ConfiguredApplicationProvider:
    """Lazily build and cache existing ``GhostVisionApplication`` instances."""

    def __init__(self, settings: APISettings) -> None:
        self.settings = settings
        self._applications: dict[tuple[str, str], GhostVisionApplication] = {}

    def __call__(
        self, modality: str, rgb_model: str
    ) -> GhostVisionApplication:
        cache_key = (modality, rgb_model if modality == "rgb" else "thermal")
        cached = self._applications.get(cache_key)
        if cached is not None:
            return cached

        if modality == "rgb":
            checkpoint = self.settings.rgb_checkpoints.get(rgb_model)
            environment_name = (
                ENV_RGB_BASELINE
                if rgb_model == "baseline"
                else ENV_RGB_AUGMENTED
            )
            if checkpoint is None:
                raise APIConfigurationError(
                    f"RGB {rgb_model} checkpoint is not configured. "
                    f"Set {environment_name}."
                )
            if not checkpoint.is_file():
                raise APIConfigurationError(
                    f"RGB {rgb_model} checkpoint is unavailable. "
                    f"Check {environment_name}."
                )
            application = GhostVisionApplication(
                rgb_checkpoint_paths={rgb_model: checkpoint},
                rgb_model_name=rgb_model,
            )
        elif modality == "thermal":
            checkpoint = self.settings.thermal_checkpoint
            if checkpoint is None:
                raise APIConfigurationError(
                    "Thermal checkpoint is not configured. "
                    f"Set {ENV_THERMAL_CHECKPOINT}."
                )
            if not checkpoint.is_file():
                raise APIConfigurationError(
                    "Thermal checkpoint is unavailable. "
                    f"Check {ENV_THERMAL_CHECKPOINT}."
                )
            application = GhostVisionApplication(
                thermal_model_path=checkpoint,
                thermal_confidence=self.settings.thermal_confidence,
            )
        else:
            raise APIConfigurationError(f"Unsupported modality: {modality}")

        self._applications[cache_key] = application
        return application


def _checkpoint_status(path: Path | None) -> dict[str, bool]:
    return {
        "configured": path is not None,
        "checkpoint_available": bool(path and path.is_file()),
    }


async def _read_valid_image(upload: UploadFile) -> Image.Image:
    if upload.content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded image exceeds the 15 MB limit.",
        )

    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="File content is not JPEG, PNG, or WebP.",
                )
            width, height = source.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Image dimensions are invalid or exceed "
                        "40 megapixels."
                    ),
                )
            return source.convert("RGB").copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Uploaded file could not be decoded as an image.",
        ) from exc


def _png_payload(image: Image.Image) -> dict[str, object]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return {
        "media_type": "image/png",
        "encoding": "base64",
        "width": image.width,
        "height": image.height,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _rgb_payload(result: object) -> dict[str, object]:
    if not isinstance(result, ApplicationSegmentationResult):
        raise RuntimeError("RGB backend returned an unexpected result type.")
    mask_array = np.asarray(result.mask)
    overlay_array = np.asarray(result.overlay)
    if mask_array.ndim != 2:
        raise RuntimeError("RGB mask must be a two-dimensional array.")
    if overlay_array.shape != (*mask_array.shape, 3):
        raise RuntimeError("RGB overlay dimensions do not match the mask.")

    mask_image = Image.fromarray(
        (mask_array.astype(bool) * 255).astype(np.uint8), mode="L"
    )
    overlay_image = Image.fromarray(
        overlay_array.astype(np.uint8), mode="RGB"
    )
    metadata = dict(result.metadata)
    checkpoint_path = metadata.pop("checkpoint_path", None)
    if checkpoint_path is not None:
        metadata["checkpoint"] = Path(str(checkpoint_path)).name

    return {
        "modality": "rgb",
        "model_name": result.model_name,
        "mask": _png_payload(mask_image),
        "overlay": _png_payload(overlay_image),
        "metadata": metadata,
    }


def _thermal_payload(result: object) -> dict[str, object]:
    if not isinstance(result, ThermalApplicationResult):
        raise RuntimeError("Thermal backend returned an unexpected result type.")
    return {
        "modality": "thermal",
        "animal_count": result.animal_count,
        "detections": [
            {
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "bbox": list(detection.bbox),
            }
            for detection in result.detections
        ],
        "metadata": {
            "checkpoint": result.model_path.name,
            "confidence_threshold": result.confidence_threshold,
        },
    }


async def _run_prediction(
    *,
    upload: UploadFile,
    modality: str,
    rgb_model: str,
    provider: ApplicationProvider,
) -> dict[str, object]:
    image = await _read_valid_image(upload)
    try:
        application = provider(modality, rgb_model)
        with tempfile.TemporaryDirectory(prefix="ghostvision-api-") as temp_dir:
            image_path = Path(temp_dir) / "upload.png"
            image.save(image_path, format="PNG")
            result = await run_in_threadpool(
                application.predict, modality, image_path
            )
        return (
            _rgb_payload(result)
            if modality == "rgb"
            else _thermal_payload(result)
        )
    except APIConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (MissingCheckpointError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The selected checkpoint is unavailable.",
        ) from exc
    except (InferenceDependencyError, ThermalDependencyError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Inference dependency is unavailable: {exc}",
        ) from exc
    except (InvalidImageError, UnknownModelError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (IncompatibleCheckpointError, ThermalResultError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"The configured checkpoint or model output is invalid: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("%s inference request failed", modality)
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Inspect the backend log for details.",
        ) from exc


def create_app(
    settings: APISettings | None = None,
    *,
    application_provider: ApplicationProvider | None = None,
) -> FastAPI:
    """Create the API without loading model weights at startup."""

    resolved_settings = settings or APISettings.from_environment()
    provider = application_provider or ConfiguredApplicationProvider(
        resolved_settings
    )
    api = FastAPI(
        title="GhostVision-AI API",
        version="1.0.0",
        description=(
            "Server-side RGB camouflage segmentation and thermal wildlife "
            "detection using the existing GhostVisionApplication."
        ),
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @api.get("/health")
    def health() -> dict[str, object]:
        rgb_models = {
            model_name: _checkpoint_status(
                resolved_settings.rgb_checkpoints.get(model_name)
            )
            for model_name in ("baseline", "augmented")
        }
        thermal = _checkpoint_status(resolved_settings.thermal_checkpoint)
        return {
            "status": "ok",
            "modalities": {
                "rgb": {
                    "configured": any(
                        item["configured"] for item in rgb_models.values()
                    ),
                    "models": rgb_models,
                },
                "thermal": thermal,
            },
        }

    @api.post("/predict/rgb")
    async def predict_rgb(
        image: UploadFile = File(...),
        model_name: str = Form("augmented"),
    ) -> dict[str, object]:
        normalized_model = model_name.strip().lower()
        if normalized_model not in RGB_MODELS:
            raise HTTPException(
                status_code=422,
                detail="model_name must be baseline or augmented.",
            )
        return await _run_prediction(
            upload=image,
            modality="rgb",
            rgb_model=normalized_model,
            provider=provider,
        )

    @api.post("/predict/thermal")
    async def predict_thermal(
        image: UploadFile = File(...),
    ) -> dict[str, object]:
        return await _run_prediction(
            upload=image,
            modality="thermal",
            rgb_model="augmented",
            provider=provider,
        )

    return api


app = create_app()
