import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ghostvision.adapters import RGBSegmentationAdapter
from ghostvision.rgb_inference import (
    ARCHITECTURE,
    CHECKPOINT_SPECS,
    IMAGE_SIZE,
    CheckpointInfo,
    IncompatibleCheckpointError,
    InvalidImageError,
    MissingCheckpointError,
    RGBInferenceError,
    RGBSegmenter,
    UnknownModelError,
    _validate_checkpoint,
)


class FakeModel:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


class RGBSegmenterTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.baseline_checkpoint = self.root / "baseline.pth"
        self.augmented_checkpoint = self.root / "augmented.pth"
        self.baseline_checkpoint.touch()
        self.augmented_checkpoint.touch()
        self.image_path = self.root / "camouflage.png"
        pixels = np.zeros((2, 4, 3), dtype=np.uint8)
        pixels[..., 0] = 255
        Image.fromarray(pixels, mode="RGB").save(self.image_path)
        self.loader_calls = []
        self.runner_batches = []

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_segmenter(self, probabilities=None):
        if probabilities is None:
            probabilities = np.zeros(
                (IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
            )
            probabilities[:, : IMAGE_SIZE // 2] = 0.75

        def loader(specification, checkpoint_path, device):
            self.loader_calls.append(
                (specification, checkpoint_path, device)
            )
            model = FakeModel()
            information = CheckpointInfo(
                path=checkpoint_path,
                epoch=specification.expected_epoch,
                architecture=ARCHITECTURE,
                image_size=IMAGE_SIZE,
                validation_dice=specification.expected_validation_dice,
            )
            return model, information

        def runner(model, batch, device):
            self.runner_batches.append((model, batch.copy(), device))
            return probabilities

        return RGBSegmenter(
            {
                "baseline": self.baseline_checkpoint,
                "augmented": self.augmented_checkpoint,
            },
            device="cpu",
            model_loader=loader,
            inference_runner=runner,
        )

    def test_prediction_is_structured_and_uses_expected_preprocessing(self):
        segmenter = self.make_segmenter()
        result = segmenter.predict(
            self.image_path, model_name="Baseline"
        )

        self.assertEqual(result.model_name, "baseline")
        self.assertEqual(result.checkpoint.epoch, 19)
        self.assertEqual(result.mask.shape, (2, 4))
        self.assertEqual(result.overlay.shape, (2, 4, 3))
        self.assertEqual(result.mask.dtype, np.uint8)
        self.assertEqual(result.overlay.dtype, np.uint8)
        np.testing.assert_array_equal(
            result.mask, np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
        )
        self.assertFalse(
            np.array_equal(
                result.overlay[:, :2],
                np.asarray(Image.open(self.image_path))[:, :2],
            )
        )
        batch = self.runner_batches[0][1]
        self.assertEqual(batch.shape, (1, 3, 256, 256))
        self.assertAlmostEqual(
            float(batch[0, 0, 0, 0]), (1.0 - 0.485) / 0.229,
            places=5,
        )
        self.assertAlmostEqual(
            float(batch[0, 1, 0, 0]), (0.0 - 0.456) / 0.224,
            places=5,
        )

    def test_augmented_selection_uses_epoch_7_and_models_are_cached(self):
        segmenter = self.make_segmenter()
        first = segmenter.predict(
            self.image_path, model_name="augmented"
        )
        second = segmenter.predict(
            self.image_path, model_name="augmented"
        )
        self.assertEqual(first.checkpoint.epoch, 7)
        self.assertEqual(second.checkpoint.epoch, 7)
        self.assertEqual(len(self.loader_calls), 1)
        self.assertGreaterEqual(self.runner_batches[0][0].eval_calls, 1)

    def test_missing_checkpoint_has_clear_error(self):
        missing = self.root / "missing.pth"
        segmenter = RGBSegmenter(
            {"baseline": missing},
            model_loader=lambda *_: None,
            inference_runner=lambda *_: None,
        )
        with self.assertRaisesRegex(
            MissingCheckpointError, "Checkpoint does not exist"
        ):
            segmenter.predict(self.image_path, model_name="baseline")

    def test_invalid_image_has_clear_error(self):
        invalid = self.root / "invalid.png"
        invalid.write_text("not an image", encoding="utf-8")
        segmenter = self.make_segmenter()
        with self.assertRaisesRegex(InvalidImageError, "Could not decode"):
            segmenter.predict(invalid, model_name="baseline")
        self.assertEqual(self.loader_calls, [])

    def test_unknown_model_is_rejected(self):
        segmenter = self.make_segmenter()
        with self.assertRaisesRegex(UnknownModelError, "Unknown RGB model"):
            segmenter.predict(self.image_path, model_name="thermal")

    def test_invalid_mock_prediction_shape_is_rejected(self):
        probabilities = np.zeros((10, 10), dtype=np.float32)
        segmenter = self.make_segmenter(probabilities)
        with self.assertRaisesRegex(
            RGBInferenceError, "Inference runner returned shape"
        ):
            segmenter.predict(self.image_path, model_name="baseline")

    def test_incompatible_checkpoint_metadata_is_rejected(self):
        checkpoint = {
            "epoch": 5,
            "model_state_dict": {},
            "val_metrics": {"dice": 0.5},
            "architecture": ARCHITECTURE,
            "image_size": IMAGE_SIZE,
        }
        with self.assertRaisesRegex(
            IncompatibleCheckpointError, "expected 19"
        ):
            _validate_checkpoint(
                checkpoint,
                CHECKPOINT_SPECS["baseline"],
                self.baseline_checkpoint,
            )

    def test_wrong_checkpoint_identity_is_rejected(self):
        checkpoint = {
            "epoch": 19,
            "model_state_dict": {},
            "val_metrics": {"dice": 0.1},
            "architecture": ARCHITECTURE,
            "image_size": IMAGE_SIZE,
        }
        with self.assertRaisesRegex(
            IncompatibleCheckpointError, "validation Dice"
        ):
            _validate_checkpoint(
                checkpoint,
                CHECKPOINT_SPECS["baseline"],
                self.baseline_checkpoint,
            )

    def test_rgb_adapter_returns_modality_neutral_result(self):
        adapter = RGBSegmentationAdapter(
            self.make_segmenter(), model_name="augmented"
        )
        result = adapter.predict(self.image_path)
        self.assertEqual(result.modality, "rgb")
        self.assertEqual(result.model_name, "augmented")
        self.assertEqual(result.metadata["checkpoint_epoch"], 7)
        self.assertEqual(result.metadata["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
