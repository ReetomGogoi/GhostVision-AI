"""Unit and AppTest tests for the GhostVision Streamlit UI."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from ghostvision.adapters import (
    ApplicationSegmentationResult,
    ThermalApplicationResult,
    ThermalDetection,
)
from ghostvision.ui_utils import (
    ENV_RGB_AUGMENTED,
    ENV_RGB_BASELINE,
    ENV_THERMAL_CHECKPOINT,
    draw_thermal_detections,
    get_default_checkpoint_paths,
)


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestUIUtils(unittest.TestCase):
    def test_default_checkpoint_paths_from_env(self):
        with patch.dict(
            os.environ,
            {
                ENV_RGB_BASELINE: "/path/to/baseline.pth",
                ENV_RGB_AUGMENTED: "/path/to/augmented.pth",
                ENV_THERMAL_CHECKPOINT: "/path/to/best.pt",
            },
        ):
            paths = get_default_checkpoint_paths()
            self.assertEqual(paths["rgb_baseline"], "/path/to/baseline.pth")
            self.assertEqual(paths["rgb_augmented"], "/path/to/augmented.pth")
            self.assertEqual(paths["thermal"], "/path/to/best.pt")

    def test_draw_thermal_detections(self):
        test_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        detections = [
            ThermalDetection(
                class_name="animal",
                confidence=0.88,
                bbox=(10.0, 10.0, 50.0, 50.0),
            )
        ]
        annotated = draw_thermal_detections(test_img, detections)
        self.assertEqual(annotated.size, (100, 100))
        self.assertIsInstance(annotated, Image.Image)


class TestStreamlitAppTest(unittest.TestCase):
    def test_app_renders_title_and_tabs(self):
        """Verify the Streamlit app loads and displays core UI headers and tabs."""
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        at.run()
        
        # Verify page title and subheader
        self.assertFalse(at.exception, f"App raised exception: {at.exception}")
        self.assertTrue(any("GhostVision-AI" in title.value for title in at.title))
        self.assertTrue(any("Dual-Modality" in sub.value for sub in at.subheader))

        # Verify tabs exist
        tab_labels = [tab.label for tab in at.tabs]
        self.assertTrue(any("RGB" in label for label in tab_labels))
        self.assertTrue(any("Thermal" in label for label in tab_labels))
        self.assertTrue(any("Metrics" in label for label in tab_labels))

    def test_metrics_tab_displays_verified_data(self):
        """Verify that the verified metrics dataframes are rendered."""
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        at.run()
        
        # Dataframes should contain RGB and Thermal evaluation results
        self.assertGreaterEqual(len(at.dataframe), 2)

    def test_sidebar_inputs_and_env_integration(self):
        """Verify sidebar inputs populate and reflect environment configuration."""
        with patch.dict(
            os.environ,
            {
                ENV_RGB_BASELINE: "C:/models/baseline.pth",
                ENV_RGB_AUGMENTED: "C:/models/augmented.pth",
                ENV_THERMAL_CHECKPOINT: "C:/models/thermal.pt",
            },
        ):
            at = AppTest.from_file(APP_PATH, default_timeout=10)
            at.run()

            # Check that text inputs reflect the environment values
            text_values = [t.value for t in at.sidebar.text_input]
            self.assertIn("C:/models/baseline.pth", text_values)
            self.assertIn("C:/models/augmented.pth", text_values)
            self.assertIn("C:/models/thermal.pt", text_values)

    def test_rgb_missing_checkpoint_handles_gracefully(self):
        """Verify error display when attempting RGB inference without checkpoint."""
        from app import run_rgb_inference_pipeline
        import streamlit as st

        # Mock an uploaded file
        mock_file = MagicMock()
        mock_file.name = "test.png"
        mock_file.getbuffer.return_value = b"fake bytes"

        with patch("streamlit.error") as mock_error:
            run_rgb_inference_pipeline(
                uploaded_file=mock_file,
                model_name="augmented",
                baseline_path="",
                augmented_path="/nonexistent/checkpoint.pth",
            )
            mock_error.assert_called()
            self.assertTrue(
                any("not configured or does not exist" in str(arg) for call in mock_error.call_args_list for arg in call[0])
            )

    def test_thermal_missing_checkpoint_handles_gracefully(self):
        """Verify error display when attempting Thermal inference without checkpoint."""
        from app import run_thermal_inference_pipeline
        import streamlit as st

        mock_file = MagicMock()
        mock_file.name = "thermal.jpg"
        mock_file.getbuffer.return_value = b"fake bytes"

        with patch("streamlit.error") as mock_error:
            run_thermal_inference_pipeline(
                uploaded_file=mock_file,
                thermal_path="/nonexistent/best.pt",
                confidence=0.25,
            )
            mock_error.assert_called()
            self.assertTrue(
                any("not configured or does not exist" in str(arg) for call in mock_error.call_args_list for arg in call[0])
            )

    def test_invalid_image_extension_handles_gracefully(self):
        """Verify error display when attempting upload of unsupported file types."""
        from app import run_rgb_inference_pipeline

        mock_file = MagicMock()
        mock_file.name = "data.txt"

        with tempfile.NamedTemporaryFile() as fake_ckpt:
            with patch("streamlit.error") as mock_error:
                run_rgb_inference_pipeline(
                    uploaded_file=mock_file,
                    model_name="augmented",
                    baseline_path=fake_ckpt.name,
                    augmented_path=fake_ckpt.name,
                )
                mock_error.assert_called()
                self.assertTrue(
                    any("Invalid file format" in str(arg) for call in mock_error.call_args_list for arg in call[0])
                )


if __name__ == "__main__":
    unittest.main()
