
# GhostVision AI — RGB Camouflaged Object Detection

## 1. Objective
Develop and evaluate an RGB-based camouflaged object segmentation
system using the COD10K dataset.

## 2. Dataset and preprocessing
- Dataset: COD10K
- Training images: 2,736
- Validation images: 304
- Held-out test images: 2,026
- Input resolution: 256 x 256
- Input modality: RGB
- Target: binary segmentation mask
- Batch size: 8
- Training augmentation: compared against a baseline without
  training augmentation
- Validation augmentation: disabled

The held-out test set was not used for model tuning.

## 3. Model and training configuration
- Architecture: U-Net
- Encoder: ResNet34
- Encoder initialization: ImageNet-pretrained
- Optimizer: AdamW
- Learning rate: 0.0001
- Hardware: Google Colab Tesla T4 GPU
- Baseline best checkpoint: epoch 3
- Augmented best checkpoint: epoch 5

Both experiments use the same model architecture.

## 4. Best validation checkpoints

| Metric | Baseline | Augmented |
|---|---:|---:|
| Best epoch | 3 | 5 |
| Validation loss | 0.6651 | 0.6855 |
| Validation Dice | 0.5641 | 0.5447 |
| Validation IoU | 0.4430 | 0.4283 |

## 5. Final held-out test results

Each model was evaluated on 2,026 test images.

| Metric | Baseline | Augmented |
|---|---:|---:|
| Mean Dice | 0.5624 | 0.5574 |
| Mean IoU | 0.4392 | 0.4360 |
| Mean Precision | 0.5869 | 0.6215 |
| Mean Recall | 0.6520 | 0.6145 |

## 6. Comparative analysis
The baseline achieved higher Dice by 0.0050 and higher IoU
by 0.0033 on the held-out test set.

The augmented model achieved higher precision by 0.0346
(approximately 0.0347 using the displayed rounded metrics),
but lower recall by 0.0375.

For this experiment, augmentation did not improve overall
segmentation overlap. It traded recall for precision.

These findings apply to the tested configurations and do not
establish that augmentation is generally ineffective.

## 7. Limitations
- Results represent the tested training configurations.
- Only the reported metrics and checkpoint epochs are recorded;
  no unverified performance claims are made.
- Qualitative prediction examples and further error analysis can
  help explain segmentation failures.
- The results should not be compared with other papers unless
  dataset splits, evaluation protocols and metrics are compatible.

## 8. Reproducibility and artifacts
The experiment checkpoints and final metric exports are stored
under the GhostVision_AI/RGB_Checkpoints directory in Google Drive.

Expected evaluation artifacts:
- final_test_metrics.csv
- final_test_metrics.json
- final_test_comparison.png

## 9. Conclusion
The baseline U-Net with a ResNet34 encoder produced the best
overall Dice and IoU scores in this comparison. The augmented
model achieved higher precision. The baseline is therefore the
current preferred checkpoint for overall segmentation overlap,
subject to the project's intended use and further error analysis.

Report generated: 2026-10-24
