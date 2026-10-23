# GhostVision-AI: Thermal Wildlife Detection

## Developer
Nikhil — Thermal Detection Module

## Dataset
BAMBI thermal wildlife dataset.

- Training experiment: 10,000 augmented images
- Distinct training source IDs selected: 2,000
- Validation: 1,696 images
- Test: 1,826 images
- Detection classes: 1 (animal)

## Model
- Architecture: YOLO11n
- Framework: Ultralytics / PyTorch
- Initial weights: COCO pretrained
- GPU: NVIDIA Tesla T4
- Training epochs: 15
- Image size: 640
- Batch size: 8

## Held-Out Test Results
- Precision: 0.9086
- Recall: 0.8925
- mAP50: 0.9576
- mAP50-95: 0.7227

## Inference Module
File: thermal_detector.py

Class: ThermalAnimalDetector

Methods:
- Initialize using model checkpoint path
- detect(image_path)

Output:
- Image path
- Animal detection count
- Bounding boxes in XYXY pixel coordinates
- Prediction confidence scores

## Model Weights
Saved separately in Google Drive:
GhostVision-AI/thermal/yolo11n_large_v1/weights/best.pt

Model checkpoints are not stored in GitHub.

## Limitations
- Trained and evaluated on BAMBI thermal imagery
- Different thermal cameras may have different performance
- Low-confidence and small animals can be missed
- Filename-based split separation verified, but
  recording-level independence not independently confirmed
- Detection confidence is not a guarantee of correctness

## Integration
Branch: feature/thermal

The RGB and thermal modules operate independently.
Integration should occur after both modules are verified.
