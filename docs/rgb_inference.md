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

`ghostvision.adapters.SegmentationAdapter` returns an
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

The first application integration layer and the distinct thermal detection
result are documented in
[`application_integration.md`](application_integration.md).
