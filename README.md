# GhostVision-AI

Unified multi-modal wildlife perception integrating RGB camouflage segmentation (U-Net ResNet34) and thermal infrared detection (YOLO11n).

## Documentation
- Reusable RGB inference: [`docs/rgb_inference.md`](docs/rgb_inference.md)
- Unified application integration: [`docs/application_integration.md`](docs/application_integration.md)

## Interactive Web UI (Streamlit)

GhostVision-AI provides a responsive, professional Streamlit interface for running RGB segmentation, thermal wildlife detection, and reviewing verified held-out evaluation benchmarks.

### Installation

```bash
pip install -r requirements.txt
```

### Launching the Web Application

To launch the Streamlit app locally:

```bash
streamlit run app.py
```

### Checkpoint Configuration

Model weights are managed separately and not committed to git. Checkpoints can be configured dynamically in the Streamlit sidebar or via environment variables:

```bash
# Linux / macOS
export GHOSTVISION_RGB_BASELINE="/path/to/best_resumed_epoch_19.pth"
export GHOSTVISION_RGB_AUGMENTED="/path/to/best_resumed_epoch_07.pth"
export GHOSTVISION_THERMAL_CHECKPOINT="/path/to/best.pt"

# Windows (PowerShell)
$env:GHOSTVISION_RGB_BASELINE="C:\models\best_resumed_epoch_19.pth"
$env:GHOSTVISION_RGB_AUGMENTED="C:\models\best_resumed_epoch_07.pth"
$env:GHOSTVISION_THERMAL_CHECKPOINT="C:\models\best.pt"
```

### Running Automated Tests

Run the full automated test suite (including UI tests and application integration tests):

```bash
python -m pytest
```

## FastAPI Backend

A separate FastAPI transport is available for future React and mobile clients.
It reuses `GhostVisionApplication`, keeps inference server-side, and leaves this
Streamlit application intact.

Configuration, launch commands, endpoint examples, and response schemas are
documented in [`docs/api.md`](docs/api.md).

## React Workstation

The React 18 and TypeScript workstation in `frontend/` connects directly to
the FastAPI endpoints. It provides validated drag-and-drop uploads, RGB mask
and overlay comparison, thermal bounding-box review, confidence details, and
downloadable visualizations generated from the real API response.

Frontend setup and validation commands are documented in
[`docs/web_application.md`](docs/web_application.md). Interface design
attribution is recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
