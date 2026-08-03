# Industrial Image Anomaly Detection Basics

A reproducible normal-only industrial image anomaly-detection project using
a convolutional autoencoder on the MVTec AD bottle category.

## Project Objective

The objective is to learn normal industrial appearance using defect-free
training images and identify anomalous test images using reconstruction error.

## Dataset

The project uses the MVTec AD bottle category.

- Training-normal images: 167
- Validation-normal images: 42
- Test images: 83
- Normal test images: 20
- Anomalous test images: 63
- Ground-truth anomaly masks: 63

The dataset itself is not included in this repository.

## Current Method

A convolutional autoencoder was trained using only normal images.

The anomaly map is calculated from channel-averaged squared reconstruction
error. The image anomaly score is obtained from the highest 1% of anomaly-map
pixels.

The decision threshold was selected from held-out normal validation images
using the 0.99 quantile. Test labels were not used for threshold selection.

## Current Results

| Metric | Result |
|---|---:|
| Accuracy | 0.6627 |
| Balanced accuracy | 0.7778 |
| Anomaly precision | 1.0000 |
| Anomaly recall | 0.5556 |
| Normal recall | 1.0000 |
| Anomaly F1 | 0.7143 |
| Macro-F1 | 0.6513 |
| Image ROC-AUC | 0.7714 |
| Average precision | 0.9322 |
| Pixel ROC-AUC | 0.7235 |
| Pixel average precision | 0.1582 |

## Confusion Matrix

The model correctly classified all 20 normal images but missed 28 of the
63 anomalous images.

![Confusion Matrix](figures/confusion_matrix_autoencoder.png)

## Defect-Specific Detection

| Defect type | Detection rate |
|---|---:|
| Broken small | 72.73% |
| Broken large | 65.00% |
| Contamination | 28.57% |
| Good-image false-positive rate | 0.00% |

## Visual Results

### Training Loss

![Training Loss](figures/autoencoder_training_loss.png)

### ROC Curve

![ROC Curve](figures/roc_curve_autoencoder.png)

### Precision-Recall Curve

![Precision-Recall Curve](figures/precision_recall_curve_autoencoder.png)

### Score Distribution

![Score Distribution](figures/anomaly_score_distribution.png)

### Anomaly Heatmaps

![Anomaly Heatmaps](figures/anomaly_heatmaps.png)
## Main Finding

The model is conservative. It produced no false alarms on the normal test
images, but anomaly recall was only 55.56%. Contamination was the most difficult
defect type.

## Repository Structure

```text
notebooks/   Executed Kaggle notebook
figures/     Training and evaluation plots
results/     Metrics, predictions, and experiment configuration
