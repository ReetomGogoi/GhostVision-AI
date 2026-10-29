"""Helpers, metrics definitions, and visualization utilities for the GhostVision UI."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ghostvision.application import GhostVisionApplication


# Environment variable names for configuration
ENV_RGB_BASELINE = "GHOSTVISION_RGB_BASELINE"
ENV_RGB_AUGMENTED = "GHOSTVISION_RGB_AUGMENTED"
ENV_THERMAL_CHECKPOINT = "GHOSTVISION_THERMAL_CHECKPOINT"


# Verified Evaluation Metrics (from recorded experiment files and held-out test evaluations)
RGB_TEST_METRICS = {
    "dataset": "COD10K Test Split (Held-out)",
    "images_evaluated": 2026,
    "input_resolution": "256 x 256",
    "threshold": 0.5,
    "models": [
        {
            "name": "Baseline (No Augmentation)",
            "checkpoint_epoch": 19,
            "dice": 0.5549,
            "iou": 0.4420,
            "precision": 0.6606,
            "recall": 0.5703,
            "selected_val_dice": 0.5687,
        },
        {
            "name": "Augmented (Augmentation Pipeline)",
            "checkpoint_epoch": 7,
            "dice": 0.5644,
            "iou": 0.4446,
            "precision": 0.6252,
            "recall": 0.6063,
            "selected_val_dice": 0.5738,
        },
    ],
    "notes": (
        "Evaluated across 2,026 held-out COD10K test images. "
        "Augmented checkpoint achieved higher Dice (+0.0095) and IoU (+0.0026) over baseline."
    ),
}

THERMAL_TEST_METRICS = {
    "dataset": "BAMBI Thermal Wildlife Dataset (Held-out Test Split)",
    "training_augmented_images": 10000,
    "test_images": 1826,
    "validation_images": 1696,
    "architecture": "YOLO11n (Ultralytics / PyTorch)",
    "input_resolution": "640 x 640",
    "classes": "1 (animal)",
    "precision": 0.9086,
    "recall": 0.8925,
    "mAP50": 0.9576,
    "mAP50_95": 0.7227,
    "notes": (
        "Evaluated on 1,826 held-out test images from BAMBI thermal wildlife dataset. "
        "Pretrained on COCO and fine-tuned for 15 epochs."
    ),
}


def get_default_checkpoint_paths() -> dict[str, str]:
    """Retrieve default checkpoint paths from environment variables if present."""
    return {
        "rgb_baseline": os.environ.get(ENV_RGB_BASELINE, "").strip(),
        "rgb_augmented": os.environ.get(ENV_RGB_AUGMENTED, "").strip(),
        "thermal": os.environ.get(ENV_THERMAL_CHECKPOINT, "").strip(),
    }


def draw_thermal_detections(
    image: Image.Image,
    detections: list[dict[str, Any]] | tuple[Any, ...],
    color: str = "#FF3333",
    width: int = 3,
) -> Image.Image:
    """Draw bounding boxes and confidence labels onto a copy of the thermal image."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for item in detections:
        if hasattr(item, "bbox") and hasattr(item, "confidence"):
            bbox = item.bbox
            conf = item.confidence
            class_name = getattr(item, "class_name", "animal")
        elif isinstance(item, Mapping):
            bbox = item.get("bbox", [0, 0, 0, 0])
            conf = item.get("confidence", 0.0)
            class_name = item.get("class", "animal")
        else:
            continue

        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            label = f"{class_name}: {conf:.2f}"
            
            # Draw label background for legibility
            text_bbox = draw.textbbox((x1, max(0, y1 - 16)), label, font=font) if hasattr(draw, "textbbox") else (x1, max(0, y1 - 16), x1 + 80, max(0, y1))
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1 + 2, max(0, y1 - 15)), label, fill="#FFFFFF", font=font)

    return annotated


def create_application_backend(
    rgb_baseline_path: str | None = None,
    rgb_augmented_path: str | None = None,
    rgb_model_name: str = "augmented",
    thermal_path: str | None = None,
    thermal_confidence: float = 0.25,
) -> GhostVisionApplication:
    """Build a GhostVisionApplication instance configured with verified or provided paths."""
    rgb_paths: dict[str, str] = {}
    if rgb_baseline_path and Path(rgb_baseline_path).is_file():
        rgb_paths["baseline"] = rgb_baseline_path
    if rgb_augmented_path and Path(rgb_augmented_path).is_file():
        rgb_paths["augmented"] = rgb_augmented_path

    thermal_model = thermal_path if (thermal_path and Path(thermal_path).is_file()) else None

    return GhostVisionApplication(
        rgb_checkpoint_paths=rgb_paths if rgb_paths else None,
        rgb_model_name=rgb_model_name,
        thermal_model_path=thermal_model,
        thermal_confidence=thermal_confidence,
    )

