import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from ghostvision.adapters import (
    ThermalApplicationResult,
    ThermalDependencyError,
    ThermalDetectionAdapter,
    ThermalResultError,
)
from ghostvision.application import (
    GhostVisionApplication,
    ModalityNotConfiguredError,
    UnsupportedModalityError,
)


class FakeAdapter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, image_path):
        self.calls.append(image_path)
        return self.result


class FakeThermalDetector:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def detect(self, image_path):
        self.calls.append(image_path)
        return self.payload


class ApplicationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image = self.root / "thermal.png"
        self.image.touch()
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.touch()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_application_routes_rgb_and_thermal_without_conversion(self):
        rgb_result = object()
        thermal_result = object()
        rgb = FakeAdapter(rgb_result)
        thermal = FakeAdapter(thermal_result)
        application = GhostVisionApplication(
            rgb_adapter=rgb,
            thermal_adapter=thermal,
        )

        self.assertIs(application.predict("RGB", self.image), rgb_result)
        self.assertIs(
            application.predict("thermal", self.image), thermal_result
        )
        self.assertEqual(application.available_modalities, ("rgb", "thermal"))
        self.assertEqual(rgb.calls, [self.image])
        self.assertEqual(thermal.calls, [self.image])

    def test_rgb_only_use_does_not_import_thermal_detector(self):
        sys.modules.pop("thermal_detector", None)
        application = GhostVisionApplication(
            rgb_adapter=FakeAdapter(object())
        )
        application.predict("rgb", self.image)
        self.assertNotIn("thermal_detector", sys.modules)

    def test_unsupported_modality_has_clear_error(self):
        application = GhostVisionApplication()
        with self.assertRaisesRegex(
            UnsupportedModalityError, "choose rgb or thermal"
        ):
            application.predict("lidar", self.image)

    def test_non_string_modality_has_clear_error(self):
        application = GhostVisionApplication()
        with self.assertRaisesRegex(
            UnsupportedModalityError, "Modality must be"
        ):
            application.predict(None, self.image)

    def test_supported_but_unconfigured_modality_has_clear_error(self):
        application = GhostVisionApplication(
            rgb_adapter=FakeAdapter(object())
        )
        with self.assertRaisesRegex(
            ModalityNotConfiguredError, "thermal.*not configured"
        ):
            application.predict("thermal", self.image)

    def test_thermal_adapter_preserves_detector_result_fields(self):
        payload = {
            "image": str(self.image),
            "animal_count": 2,
            "detections": [
                {
                    "class": "animal",
                    "confidence": 0.9123,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                },
                {
                    "class": "animal",
                    "confidence": 0.7654,
                    "bbox": [1.5, 2.5, 8.5, 9.5],
                },
            ],
        }
        detector = FakeThermalDetector(payload)
        factory_calls = []

        def factory(model_path, confidence=0.25):
            factory_calls.append((model_path, confidence))
            return detector

        adapter = ThermalDetectionAdapter(
            self.checkpoint,
            confidence=0.3,
            detector_factory=factory,
        )
        self.assertEqual(factory_calls, [])
        result = adapter.predict(self.image)

        self.assertIsInstance(result, ThermalApplicationResult)
        self.assertEqual(result.modality, "thermal")
        self.assertEqual(result.image_path, self.image)
        self.assertEqual(result.animal_count, 2)
        self.assertEqual(result.detections[0].class_name, "animal")
        self.assertEqual(result.detections[0].confidence, 0.9123)
        self.assertEqual(
            result.detections[0].bbox, (10.0, 20.0, 30.0, 40.0)
        )
        self.assertEqual(result.model_path, self.checkpoint)
        self.assertEqual(result.confidence_threshold, 0.3)
        self.assertEqual(factory_calls, [(self.checkpoint, 0.3)])
        self.assertEqual(detector.calls, [self.image])

    def test_thermal_missing_image_fails_before_detector_loading(self):
        factory_calls = []

        def factory(*args, **kwargs):
            factory_calls.append((args, kwargs))
            return FakeThermalDetector({})

        adapter = ThermalDetectionAdapter(
            self.checkpoint, detector_factory=factory
        )
        with self.assertRaisesRegex(FileNotFoundError, "Thermal image"):
            adapter.predict(self.root / "missing.png")
        self.assertEqual(factory_calls, [])

    def test_thermal_missing_checkpoint_has_clear_error(self):
        adapter = ThermalDetectionAdapter(self.root / "missing.pt")
        with self.assertRaisesRegex(FileNotFoundError, "Thermal checkpoint"):
            adapter.predict(self.image)

    def test_missing_thermal_dependency_has_clear_error(self):
        adapter = ThermalDetectionAdapter(self.checkpoint)
        with patch(
            "ghostvision.adapters.import_module",
            side_effect=ModuleNotFoundError("ultralytics"),
        ):
            with self.assertRaisesRegex(
                ThermalDependencyError, "requires the ultralytics"
            ):
                adapter.predict(self.image)

    def test_thermal_malformed_result_is_rejected(self):
        detector = FakeThermalDetector({
            "image": str(self.image),
            "animal_count": 2,
            "detections": [{
                "class": "animal",
                "confidence": 0.9,
                "bbox": [1, 2, 3, 4],
            }],
        })
        adapter = ThermalDetectionAdapter(
            self.checkpoint,
            detector_factory=lambda *args, **kwargs: detector,
        )
        with self.assertRaisesRegex(
            ThermalResultError, "does not match detections"
        ):
            adapter.predict(self.image)


if __name__ == "__main__":
    unittest.main()
