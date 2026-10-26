# Reusable RGB inference

`ghostvision.rgb_inference.RGBSegmenter` exposes the completed RGB U-Net
ResNet34 models without depending on Google Colab or Google Drive. The caller
owns checkpoint storage and supplies paths for the baseline epoch-19 and
augmented epoch-7 checkpoints.

Production inference requires PyTorch, Pillow, NumPy, and
`segmentation-models-pytorch==0.5.0`. Install the PyTorch build appropriate for
the deployment CPU or CUDA runtime.

```python
from ghostvision import RGBSegmenter

segmenter = RGBSegmenter({
    "baseline": "/models/best_resumed_epoch_19.pth",
    "augmented": "/models/best_resumed_epoch_07.pth",
})

prediction = segmenter.predict(
    "/data/user-image.jpg",
    model_name="augmented",
)

prediction.mask        # uint8 H x W array containing 0 and 1
prediction.overlay     # uint8 H x W x 3 highlighted RGB array
prediction.model_name  # "baseline" or "augmented"
prediction.checkpoint  # validated path, epoch, architecture, and Dice metadata
```

The module verifies each selected checkpoint's epoch, architecture, image
size, and recorded validation Dice before recreating U-Net with a ResNet34
encoder and strictly loading the weights. It resizes RGB inputs to 256 x 256 with bilinear
interpolation, applies ImageNet normalization, runs `model.eval()` under
`torch.inference_mode()`, and thresholds probabilities at 0.5. The mask is
resized to the original image dimensions with nearest-neighbor interpolation.
The module never saves images or modifies checkpoints.

Missing files, undecodable images, unsupported model names, malformed
checkpoint metadata, and incompatible weights raise specific subclasses of
`RGBInferenceError`.

## Application and thermal integration

No thermal module or thermal API exists on the current `feature/rgb` branch.
`ghostvision.adapters.SegmentationAdapter` is therefore an adapter contract,
not a claim about an existing thermal interface. It returns a modality-neutral
`ApplicationSegmentationResult` containing a mask, overlay, modality, model
name, and metadata.

`RGBSegmentationAdapter` already implements this contract:

```python
from ghostvision import RGBSegmentationAdapter

rgb_backend = RGBSegmentationAdapter(
    segmenter,
    model_name="augmented",
)
application_result = rgb_backend.predict("/data/user-image.jpg")
```

Once the thermal module is available, a thermal-specific adapter can implement
the same `predict(image_path)` contract. The application can then select RGB or
thermal backends without coupling either model's preprocessing, checkpoint
format, or dependencies. Full integration remains pending until the thermal
module's actual input requirements, result types, and checkpoint lifecycle are
available.
