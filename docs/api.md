# GhostVision-AI FastAPI backend

The FastAPI service exposes the existing `GhostVisionApplication` to browser or
mobile clients. It does not implement a second model pipeline: RGB requests are
routed to the existing U-Net ResNet34 segmenter, and thermal requests are routed
to the existing YOLO11n detector.

The Streamlit application in `app.py` remains available and unchanged.

## Configuration

Model checkpoints stay outside the repository. Configure one or more of these
environment variables on the backend host:

```bash
export GHOSTVISION_RGB_BASELINE=/absolute/path/to/best_resumed_epoch_19.pth
export GHOSTVISION_RGB_AUGMENTED=/absolute/path/to/best_resumed_epoch_07.pth
export GHOSTVISION_THERMAL_CHECKPOINT=/absolute/path/to/best.pt
export GHOSTVISION_THERMAL_CONFIDENCE=0.25
export GHOSTVISION_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Paths may point to any mounted local or network volume. The code contains no
Google Drive or Colab path. `GHOSTVISION_ALLOWED_ORIGINS` controls which browser
origins may call the API; the two common local Vite origins are allowed by
default.

Install dependencies and start the API from the repository root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn ghostvision.api:app --host 127.0.0.1 --port 8000
```

OpenAPI documentation is available at <http://127.0.0.1:8000/docs>.

## Endpoints

### `GET /health`

Returns backend status and whether each checkpoint is configured and present.
The health check does not load either model.

```bash
curl http://127.0.0.1:8000/health
```

### `POST /predict/rgb`

Accepts multipart form data:

- `image`: JPEG, PNG, or WebP image, up to 15 MB and 40 megapixels.
- `model_name`: optional `baseline` or `augmented`; defaults to `augmented`.

```bash
curl -X POST http://127.0.0.1:8000/predict/rgb \
  -F image=@camouflaged_animal.jpg \
  -F model_name=augmented
```

The response contains the actual binary mask and highlight overlay as base64
encoded PNG objects, plus the model name and safe checkpoint metadata:

```json
{
  "modality": "rgb",
  "model_name": "augmented",
  "mask": {
    "media_type": "image/png",
    "encoding": "base64",
    "width": 1024,
    "height": 768,
    "data": "..."
  },
  "overlay": {
    "media_type": "image/png",
    "encoding": "base64",
    "width": 1024,
    "height": 768,
    "data": "..."
  },
  "metadata": {
    "checkpoint": "best_resumed_epoch_07.pth",
    "checkpoint_epoch": 7,
    "threshold": 0.5
  }
}
```

### `POST /predict/thermal`

Accepts one `image` multipart field with the same validation rules:

```bash
curl -X POST http://127.0.0.1:8000/predict/thermal \
  -F image=@thermal_capture.png
```

The response preserves the detector's animal count, XYXY bounding boxes, and
individual confidence scores:

```json
{
  "modality": "thermal",
  "animal_count": 1,
  "detections": [
    {
      "class_name": "animal",
      "confidence": 0.91,
      "bbox": [120.0, 84.0, 420.0, 355.0]
    }
  ],
  "metadata": {
    "checkpoint": "best.pt",
    "confidence_threshold": 0.25
  }
}
```

The example values only illustrate the response schema. Production responses
come from the configured checkpoint through `GhostVisionApplication`.

## Error behavior

- `413`: upload is larger than 15 MB.
- `415`: unsupported media type or image format.
- `422`: empty, invalid, oversized-dimension image, or invalid RGB model name.
- `503`: checkpoint or inference dependency is unavailable.
- `500`: incompatible checkpoint, malformed model result, or inference failure.

Uploads are decoded before inference, stored only in an isolated temporary
directory, and removed after the request completes. Checkpoint paths are read
only; the API never writes model files or experiment results.

## Tests

API tests use lightweight application doubles and do not load model weights:

```bash
python -B -m unittest tests.test_api -v
```
