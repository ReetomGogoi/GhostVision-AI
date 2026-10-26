"""Inference-only RGB camouflage segmentation.

The implementation mirrors the completed RGB experiments: U-Net with a
ResNet34 encoder, 256 x 256 RGB input, ImageNet normalization, and a 0.5
probability threshold. Checkpoint locations are supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_SIZE = 256
THRESHOLD = 0.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ARCHITECTURE = "Unet-ResNet34"


class RGBInferenceError(RuntimeError):
    """Base error for RGB inference failures."""


class MissingCheckpointError(RGBInferenceError):
    """Raised when the configured checkpoint does not exist."""


class IncompatibleCheckpointError(RGBInferenceError):
    """Raised when checkpoint metadata or weights do not match the model."""


class InvalidImageError(RGBInferenceError):
    """Raised when an input image is missing or cannot be decoded."""


class UnknownModelError(RGBInferenceError):
    """Raised when a model other than baseline or augmented is requested."""


class InferenceDependencyError(RGBInferenceError):
    """Raised when the production PyTorch dependencies are unavailable."""


@dataclass(frozen=True)
class CheckpointSpec:
    """Expected identity of one completed RGB experiment checkpoint."""

    model_name: str
    expected_epoch: int
    expected_validation_dice: float


CHECKPOINT_SPECS = {
    "baseline": CheckpointSpec(
        "baseline",
        expected_epoch=19,
        expected_validation_dice=0.5686530063,
    ),
    "augmented": CheckpointSpec(
        "augmented",
        expected_epoch=7,
        expected_validation_dice=0.5737559270,
    ),
}


@dataclass(frozen=True)
class CheckpointInfo:
    """Validated metadata returned with a prediction."""

    path: Path
    epoch: int
    architecture: str
    image_size: int
    validation_dice: float


@dataclass(frozen=True)
class RGBPredictionResult:
    """Structured output from :class:`RGBSegmenter`.

    ``mask`` is a uint8 array containing only 0 and 1. ``overlay`` is an RGB
    uint8 image with predicted foreground highlighted in red. Both arrays use
    the original image dimensions.
    """

    mask: np.ndarray
    overlay: np.ndarray
    model_name: str
    checkpoint: CheckpointInfo
    image_path: Path
    threshold: float


ModelLoader = Callable[
    [CheckpointSpec, Path, str], tuple[Any, CheckpointInfo]
]
InferenceRunner = Callable[[Any, np.ndarray, str], np.ndarray]


def _load_production_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError as exc:
        raise InferenceDependencyError(
            "RGB inference requires PyTorch and "
            "segmentation-models-pytorch==0.5.0."
        ) from exc
    return torch, smp


def _validate_checkpoint(
    checkpoint: object,
    specification: CheckpointSpec,
    checkpoint_path: Path,
) -> CheckpointInfo:
    if not isinstance(checkpoint, Mapping):
        raise IncompatibleCheckpointError(
            f"Checkpoint is not a mapping: {checkpoint_path}"
        )

    required = {
        "epoch",
        "model_state_dict",
        "val_metrics",
        "architecture",
        "image_size",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise IncompatibleCheckpointError(
            f"Checkpoint {checkpoint_path} is missing keys: "
            f"{sorted(missing)}"
        )

    epoch = checkpoint["epoch"]
    if epoch != specification.expected_epoch:
        raise IncompatibleCheckpointError(
            f"{specification.model_name} checkpoint epoch is {epoch}; "
            f"expected {specification.expected_epoch}."
        )
    architecture = checkpoint["architecture"]
    if architecture != ARCHITECTURE:
        raise IncompatibleCheckpointError(
            f"Checkpoint architecture is {architecture!r}; "
            f"expected {ARCHITECTURE!r}."
        )
    image_size = checkpoint["image_size"]
    if image_size != IMAGE_SIZE:
        raise IncompatibleCheckpointError(
            f"Checkpoint image size is {image_size}; expected {IMAGE_SIZE}."
        )

    validation_metrics = checkpoint["val_metrics"]
    if not isinstance(validation_metrics, Mapping):
        raise IncompatibleCheckpointError(
            "Checkpoint val_metrics must be a mapping."
        )
    try:
        validation_dice = float(validation_metrics["dice"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IncompatibleCheckpointError(
            "Checkpoint val_metrics.dice is missing or invalid."
        ) from exc
    if not np.isfinite(validation_dice):
        raise IncompatibleCheckpointError(
            "Checkpoint validation Dice is not finite."
        )
    if not np.isclose(
        validation_dice,
        specification.expected_validation_dice,
        rtol=0.0,
        atol=1e-9,
    ):
        raise IncompatibleCheckpointError(
            f"{specification.model_name} checkpoint validation Dice is "
            f"{validation_dice:.10f}; expected "
            f"{specification.expected_validation_dice:.10f}."
        )

    return CheckpointInfo(
        path=checkpoint_path,
        epoch=int(epoch),
        architecture=str(architecture),
        image_size=int(image_size),
        validation_dice=validation_dice,
    )


def _default_model_loader(
    specification: CheckpointSpec,
    checkpoint_path: Path,
    device: str,
) -> tuple[Any, CheckpointInfo]:
    torch, smp = _load_production_dependencies()
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise IncompatibleCheckpointError(
            f"Could not safely load checkpoint {checkpoint_path}: {exc}"
        ) from exc

    information = _validate_checkpoint(
        checkpoint, specification, checkpoint_path
    )
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    try:
        model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
    except Exception as exc:
        raise IncompatibleCheckpointError(
            f"Checkpoint weights are incompatible with {ARCHITECTURE}: {exc}"
        ) from exc
    model = model.to(device)
    model.eval()
    return model, information


def _default_inference_runner(
    model: Any,
    normalized_batch: np.ndarray,
    device: str,
) -> np.ndarray:
    torch, _ = _load_production_dependencies()
    input_tensor = torch.from_numpy(normalized_batch).float().to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)

    if tuple(probabilities.shape) != (1, 1, IMAGE_SIZE, IMAGE_SIZE):
        raise RGBInferenceError(
            "Model returned shape "
            f"{tuple(probabilities.shape)}; expected "
            f"(1, 1, {IMAGE_SIZE}, {IMAGE_SIZE})."
        )
    if not torch.isfinite(probabilities).all().item():
        raise RGBInferenceError("Model prediction contains non-finite values.")
    return probabilities[0, 0].detach().cpu().numpy()


class RGBSegmenter:
    """Reusable predictor for the completed baseline and augmented models.

    Parameters
    ----------
    checkpoint_paths:
        Mapping containing externally configured ``baseline`` and/or
        ``augmented`` checkpoint paths. Google Drive is not assumed.
    device:
        Optional PyTorch device string. When omitted in production, CUDA is
        selected when available. Tests and alternative runtimes can inject a
        model loader and inference runner.
    """

    def __init__(
        self,
        checkpoint_paths: Mapping[str, str | Path],
        *,
        device: str | None = None,
        model_loader: ModelLoader | None = None,
        inference_runner: InferenceRunner | None = None,
    ) -> None:
        normalized_paths: dict[str, Path] = {}
        for name, path in checkpoint_paths.items():
            normalized_name = name.strip().lower()
            if normalized_name not in CHECKPOINT_SPECS:
                raise UnknownModelError(
                    f"Unsupported checkpoint name {name!r}; use baseline "
                    "or augmented."
                )
            normalized_paths[normalized_name] = Path(path).expanduser()

        self._checkpoint_paths = normalized_paths
        self._device = device
        self._model_loader = model_loader or _default_model_loader
        self._inference_runner = (
            inference_runner or _default_inference_runner
        )
        self._model_cache: dict[
            str, tuple[Any, CheckpointInfo]
        ] = {}

    def _resolve_device(self) -> str:
        if self._device is not None:
            return self._device
        if (
            self._model_loader is not _default_model_loader
            or self._inference_runner is not _default_inference_runner
        ):
            return "cpu"
        torch, _ = _load_production_dependencies()
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_model(
        self, model_name: str
    ) -> tuple[Any, CheckpointInfo, str]:
        normalized_name = model_name.strip().lower()
        if normalized_name not in CHECKPOINT_SPECS:
            raise UnknownModelError(
                f"Unknown RGB model {model_name!r}; choose baseline "
                "or augmented."
            )
        if normalized_name not in self._checkpoint_paths:
            raise MissingCheckpointError(
                f"No checkpoint path configured for {normalized_name}."
            )

        checkpoint_path = self._checkpoint_paths[normalized_name]
        if not checkpoint_path.is_file():
            raise MissingCheckpointError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )
        device = self._resolve_device()
        if normalized_name not in self._model_cache:
            model, information = self._model_loader(
                CHECKPOINT_SPECS[normalized_name],
                checkpoint_path,
                device,
            )
            if not hasattr(model, "eval"):
                raise IncompatibleCheckpointError(
                    "Loaded model does not provide eval()."
                )
            model.eval()
            self._model_cache[normalized_name] = (model, information)
        model, information = self._model_cache[normalized_name]
        return model, information, device

    @staticmethod
    def _load_image(image_path: str | Path) -> tuple[Path, Image.Image]:
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise InvalidImageError(f"Image does not exist: {path}")
        try:
            with Image.open(path) as source:
                image = source.convert("RGB").copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(
                f"Could not decode RGB image {path}: {exc}"
            ) from exc
        return path, image

    @staticmethod
    def _preprocess(image: Image.Image) -> np.ndarray:
        resized = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        normalized = (image_array - IMAGENET_MEAN) / IMAGENET_STD
        batch = normalized.transpose(2, 0, 1)[None, ...].copy()
        if batch.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise InvalidImageError(
                f"Unexpected preprocessed shape: {batch.shape}"
            )
        if not np.isfinite(batch).all():
            raise InvalidImageError(
                "Preprocessed image contains non-finite values."
            )
        return batch

    def predict(
        self,
        image_path: str | Path,
        *,
        model_name: str = "augmented",
    ) -> RGBPredictionResult:
        """Segment one user-provided RGB image without modifying files."""

        path, original = self._load_image(image_path)
        model, information, device = self._get_model(model_name)
        batch = self._preprocess(original)
        probabilities = np.asarray(
            self._inference_runner(model, batch, device),
            dtype=np.float32,
        )
        if probabilities.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise RGBInferenceError(
                f"Inference runner returned shape {probabilities.shape}; "
                f"expected ({IMAGE_SIZE}, {IMAGE_SIZE})."
            )
        if not np.isfinite(probabilities).all():
            raise RGBInferenceError(
                "Inference runner returned non-finite probabilities."
            )

        mask_256 = (probabilities >= THRESHOLD).astype(np.uint8)
        mask = np.asarray(
            Image.fromarray(mask_256 * 255).resize(
                original.size, Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        mask = (mask > 0).astype(np.uint8)

        original_array = np.asarray(original, dtype=np.uint8)
        overlay = original_array.astype(np.float32) / 255.0
        foreground = mask.astype(bool)
        highlight = np.array([1.0, 0.15, 0.05], dtype=np.float32)
        overlay[foreground] = (
            0.55 * overlay[foreground] + 0.45 * highlight
        )
        overlay = np.rint(np.clip(overlay, 0.0, 1.0) * 255).astype(
            np.uint8
        )

        return RGBPredictionResult(
            mask=mask,
            overlay=overlay,
            model_name=model_name.strip().lower(),
            checkpoint=information,
            image_path=path,
            threshold=THRESHOLD,
        )

    def clear_cache(self) -> None:
        """Release cached model references without touching checkpoints."""

        self._model_cache.clear()
