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
