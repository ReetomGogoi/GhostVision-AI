# RGB and thermal application integration

`GhostVisionApplication` routes a user-supplied image to one configured
modality while preserving each model's real result shape:

- `rgb` returns `ApplicationSegmentationResult` with a segmentation mask,
  highlighted overlay, RGB model name, and checkpoint metadata.
- `thermal` returns `ThermalApplicationResult` with the image path, animal
  count, XYXY bounding boxes, confidence scores, thermal checkpoint path, and
  configured confidence threshold.

The integration does not convert RGB masks into boxes or fabricate thermal
masks. RGB and thermal checkpoints remain external to the repository.

```python
from ghostvision import GhostVisionApplication

application = GhostVisionApplication(
    rgb_checkpoint_paths={
        "baseline": "/models/rgb/best_resumed_epoch_19.pth",
        "augmented": "/models/rgb/best_resumed_epoch_07.pth",
    },
    rgb_model_name="augmented",
    thermal_model_path="/models/thermal/best.pt",
    thermal_confidence=0.25,
)

rgb_result = application.predict("rgb", "/images/rgb.jpg")
thermal_result = application.predict("thermal", "/images/thermal.jpg")
```

The thermal adapter calls the preserved interface from `thermal_detector.py`:

```python
detector = ThermalAnimalDetector(model_path, confidence=0.25)
payload = detector.detect(image_path)
```

`thermal_detector` and Ultralytics are imported only when the first thermal
prediction is requested. RGB-only deployments therefore do not need to import
thermal dependencies. Thermal inference requires the `ultralytics` package.
PyTorch and `segmentation-models-pytorch` likewise stay
behind the existing lazy RGB inference path.

The application router reports unsupported modalities separately from
supported but unconfigured modalities. Model loading, device management, and
preprocessing remain owned by the modality modules.
