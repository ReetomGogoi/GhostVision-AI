"""API tests use application doubles and never load model checkpoints."""

import base64
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from ghostvision.adapters import (
    ApplicationSegmentationResult,
    ThermalApplicationResult,
    ThermalDetection,
)
from ghostvision.api import APISettings, create_app


class FakeApplication:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, modality, image_path):
        path = Path(image_path)
        self.calls.append((modality, path, path.is_file()))
        return self.result


class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.rgb_checkpoint = self.root / "augmented.pth"
        self.thermal_checkpoint = self.root / "best.pt"
        self.rgb_checkpoint.touch()
        self.thermal_checkpoint.touch()
        self.settings = APISettings(
            rgb_checkpoints={"augmented": self.rgb_checkpoint},
            thermal_checkpoint=self.thermal_checkpoint,
        )
        image = Image.new("RGB", (10, 6), color=(60, 85, 72))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        self.image_bytes = stream.getvalue()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_client(self, result):
        application = FakeApplication(result)
        api = create_app(
            self.settings,
            application_provider=lambda _modality, _model: application,
        )
        return TestClient(api), application

    def upload(self, client, endpoint, *, data=None, content=None):
        return client.post(
            endpoint,
            data=data or {},
            files={
                "image": (
                    "wildlife.png",
                    self.image_bytes if content is None else content,
                    "image/png",
                )
            },
        )

    def test_health_reports_configured_modalities_without_loading_models(self):
        response = TestClient(create_app(self.settings)).get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(
            payload["modalities"]["rgb"]["models"]["augmented"][
                "checkpoint_available"
            ]
        )
        self.assertFalse(
            payload["modalities"]["rgb"]["models"]["baseline"][
                "configured"
            ]
        )
        self.assertTrue(
            payload["modalities"]["thermal"]["checkpoint_available"]
        )

    def test_local_react_origin_is_allowed(self):
        client = TestClient(create_app(self.settings))
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://localhost:5173",
        )

    def test_rgb_endpoint_returns_mask_overlay_and_metadata(self):
        mask = np.zeros((6, 10), dtype=np.uint8)
        mask[:, :4] = 1
        overlay = np.full((6, 10, 3), 125, dtype=np.uint8)
        result = ApplicationSegmentationResult(
            mask=mask,
            overlay=overlay,
            modality="rgb",
            model_name="augmented",
            metadata={
                "checkpoint_path": str(self.rgb_checkpoint),
                "checkpoint_epoch": 7,
                "threshold": 0.5,
            },
        )
        client, application = self.make_client(result)
        response = self.upload(
            client,
            "/predict/rgb",
            data={"model_name": "augmented"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["modality"], "rgb")
        self.assertEqual(payload["model_name"], "augmented")
        self.assertEqual(payload["metadata"]["checkpoint"], "augmented.pth")
        self.assertNotIn("checkpoint_path", payload["metadata"])
        for output_name in ("mask", "overlay"):
            encoded = payload[output_name]
            self.assertEqual(encoded["media_type"], "image/png")
            with Image.open(io.BytesIO(base64.b64decode(encoded["data"]))) as image:
                image.verify()
        self.assertEqual(application.calls[0][0], "rgb")
        self.assertTrue(application.calls[0][2])

    def test_thermal_endpoint_preserves_boxes_and_confidence_scores(self):
        result = ThermalApplicationResult(
            image_path=Path("thermal.png"),
            animal_count=2,
            detections=(
                ThermalDetection("animal", 0.91, (1.0, 2.0, 8.0, 5.0)),
                ThermalDetection("animal", 0.77, (2.0, 1.0, 9.0, 6.0)),
            ),
            model_path=self.thermal_checkpoint,
            confidence_threshold=0.25,
        )
        client, application = self.make_client(result)
        response = self.upload(client, "/predict/thermal")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["animal_count"], 2)
        self.assertEqual(payload["detections"][0]["confidence"], 0.91)
        self.assertEqual(payload["detections"][0]["bbox"], [1.0, 2.0, 8.0, 5.0])
        self.assertEqual(payload["metadata"]["checkpoint"], "best.pt")
        self.assertEqual(application.calls[0][0], "thermal")

    def test_invalid_image_content_is_rejected_before_inference(self):
        client, application = self.make_client(object())
        response = self.upload(client, "/predict/rgb", content=b"not an image")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(application.calls, [])

    def test_unsupported_media_type_is_rejected(self):
        client, application = self.make_client(object())
        response = client.post(
            "/predict/thermal",
            files={"image": ("notes.txt", b"text", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(application.calls, [])

    def test_unknown_rgb_model_is_rejected(self):
        client, application = self.make_client(object())
        response = self.upload(
            client,
            "/predict/rgb",
            data={"model_name": "experimental"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(application.calls, [])

    def test_missing_checkpoint_returns_clear_service_error(self):
        settings = APISettings(
            rgb_checkpoints={},
            thermal_checkpoint=None,
        )
        client = TestClient(create_app(settings))
        response = self.upload(client, "/predict/thermal")
        self.assertEqual(response.status_code, 503)
        self.assertIn("GHOSTVISION_THERMAL_CHECKPOINT", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
