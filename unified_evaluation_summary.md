# Unified Evaluation Summary

This file is the single-file summary of the project after the refreshed metric run in the conda environment.

It combines:

- project scope
- brief code summaries
- refreshed evaluation metrics
- conclusions
- practical interpretation

## 1. Project Scope

The project classifies Alzheimer's MRI scans into 4 classes:

- Mild Demented
- Moderate Demented
- Non Demented
- Very Mild Demented

Two modeling directions are implemented:

- a custom PyTorch CNN with several imbalance-handling variants
- a Swin Transformer fine-tuned from `microsoft/swin-base-patch4-window7-224`

The dataset is strongly imbalanced, especially for `Moderate Demented`, so balanced metrics matter more than raw accuracy alone.

## 2. Brief Code Summary

### CNN Notebook

File: `CNN-Pytorch.ipynb`

Main structure:

- loads the Hugging Face MRI dataset
- builds a custom grayscale MRI dataset wrapper
- defines a baseline CNN and a configurable optimized CNN
- uses hardcoded Optuna-selected hyperparameters instead of rerunning a full search
- compares 5 CNN strategies:
  - Baseline
  - Weighted Loss
  - Balanced Sampling
  - MONAI Augmentation
  - Combined Strategy
- computes classification metrics, confusion matrices, and exports `comparison_results.csv`

Metric logic now includes:

- accuracy
- balanced accuracy
- specificity macro
- F1 macro
- F1 weighted
- MCC
- kappa
- ROC-AUC OvR macro
- per-class precision, recall, specificity, F1, support

### Swin Notebook

File: `VIT-Swin.ipynb`

Main structure:

- loads the same dataset
- converts grayscale MRI images into RGB for Swin preprocessing
- uses Hugging Face image processor + pretrained Swin backbone
- applies weighted loss and stronger augmentation for minority classes
- evaluates the trained model and saves the checkpoint to `models/swin_alzheimer_classifier.pt`

The saved checkpoint contains:

- model weights
- labels
- training history
- evaluation metrics

## 3. Evaluation Setup

The refreshed CNN comparison was executed and regenerated `comparison_results.csv`.

The Swin metrics were confirmed from the saved checkpoint:

- `models/swin_alzheimer_classifier.pt`

Important note:

- the refreshed CNN numbers differ materially from the older narrative that existed earlier in the repo
- this file reflects the refreshed executed metrics, not the older pre-refresh story

## 4. Overall Metrics

### CNN Variants

| Strategy | Accuracy | Balanced Acc | Specificity Macro | F1 Macro | F1 Weighted | MCC | Kappa | ROC-AUC |
|----------|----------|--------------|-------------------|----------|-------------|-----|-------|---------|
| Baseline | 0.5312 | 0.2752 | 0.7699 | 0.2244 | 0.4173 | 0.1406 | 0.0848 | 0.7315 |
| Weighted Loss | 0.5010 | 0.2500 | 0.7500 | 0.1669 | 0.3344 | 0.0000 | 0.0000 | 0.5000 |
| Balanced Sampling | 0.4883 | 0.3189 | 0.7672 | 0.1972 | 0.3600 | 0.1002 | 0.0646 | 0.5988 |
| MONAI Augmentation | 0.5986 | 0.3503 | 0.8313 | 0.3204 | 0.5583 | 0.3189 | 0.3046 | 0.8026 |
| Combined Strategy | 0.3477 | 0.5134 | 0.7567 | 0.2687 | 0.2092 | 0.0758 | 0.0352 | 0.6502 |

### Swin Transformer

| Model | Accuracy | Balanced Acc | Specificity Macro | F1 Macro | F1 Weighted | MCC | Kappa | ROC-AUC |
|-------|----------|--------------|-------------------|----------|-------------|-----|-------|---------|
| Swin Transformer | 0.8760 | 0.9044 | 0.9503 | 0.8916 | 0.8768 | 0.7996 | 0.7977 | 0.9773 |

## 5. Detailed Per-Class Metrics

### CNN: MONAI Augmentation

This is the strongest CNN variant by raw accuracy and the best general overall CNN profile in the refreshed run.

| Class | Precision | Recall / Sensitivity | Specificity | F1 | Support |
|-------|-----------|----------------------|-------------|----|---------|
| Mild Demented | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 145 |
| Moderate Demented | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 10 |
| Non Demented | 0.7430 | 0.7271 | 0.7476 | 0.7350 | 513 |
| Very Mild Demented | 0.4598 | 0.6742 | 0.5778 | 0.5467 | 356 |

### CNN: Combined Strategy

This is the strongest CNN variant by balanced accuracy, but it behaves pathologically on some classes.

| Class | Precision | Recall / Sensitivity | Specificity | F1 | Support |
|-------|-----------|----------------------|-------------|----|---------|
| Mild Demented | 0.4651 | 0.1379 | 0.9738 | 0.2128 | 145 |
| Moderate Demented | 0.2174 | 1.0000 | 0.9645 | 0.3571 | 10 |
| Non Demented | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 513 |
| Very Mild Demented | 0.3487 | 0.9157 | 0.0883 | 0.5050 | 356 |

### Swin Transformer

| Class | Precision | Recall / Sensitivity | Specificity | F1 | Support |
|-------|-----------|----------------------|-------------|----|---------|
| Mild Demented | 0.8794 | 0.8552 | 0.9807 | 0.8671 | 145 |
| Moderate Demented | 0.9091 | 1.0000 | 0.9990 | 0.9524 | 10 |
| Non Demented | 0.9307 | 0.8635 | 0.9354 | 0.8959 | 513 |
| Very Mild Demented | 0.8081 | 0.8989 | 0.8862 | 0.8511 | 356 |

## 6. Main Conclusions

### 6.1 Accuracy Alone Is Misleading

The CNN variants illustrate the accuracy paradox clearly:

- the best CNN by accuracy is not the best CNN by balanced accuracy
- the best CNN by balanced accuracy is not clinically well-behaved across all classes

So accuracy alone is not enough for this dataset.

### 6.2 The Combined CNN Strategy Is Not a Stable Winner

The combined strategy does improve balanced accuracy to `0.5134` and reaches `100%` recall on `Moderate Demented`.

But it also collapses badly elsewhere:

- `0%` recall for `Non Demented`
- very poor specificity for `Very Mild Demented` at `0.0883`
- weak overall F1, MCC, and kappa

So it is useful as an imbalance-sensitive stress case, but not the clean best CNN model overall.

### 6.3 MONAI Augmentation Is the Strongest CNN Variant Overall

MONAI Augmentation is the best CNN variant on:

- accuracy
- specificity macro
- F1 macro
- F1 weighted
- MCC
- kappa
- ROC-AUC

Its weakness is that it still fails completely on the two rarer demented classes in this refreshed run.

### 6.4 Swin Transformer Is Clearly Better

The Swin model is substantially better than every CNN variant across the overall summary metrics.

Its advantages are visible in:

- much higher balanced accuracy: `0.9044`
- much higher specificity macro: `0.9503`
- much higher F1 macro: `0.8916`
- much higher MCC: `0.7996`
- much higher ROC-AUC: `0.9773`

It also has strong per-class performance across all four classes, including the rare `Moderate Demented` class.

## 7. Practical Interpretation

If the goal is:

- best CNN baseline: use `MONAI Augmentation`
- best CNN for minority recall emphasis: inspect `Combined Strategy`, but treat it cautiously
- best overall model: use `Swin Transformer`

## 8. What Was Added In The Metric Refresh

Compared with the older version of the project summary, the refreshed evaluation now includes:

- per-class specificity
- macro specificity
- ROC-AUC OvR macro for CNN exports
- per-class support in the exported CSV

## 9. What Is Still Missing

The main optional metrics still not implemented are:

- PR-AUC OvR macro
- calibration metrics such as Brier score or ECE

These are useful, but not necessary for a clear proof-of-concept comparison.

## 10. Recommended Use Of This File

If you want one single project summary file, use this one as the authoritative high-level report.

For raw numeric exports and notebook-level detail, the supporting sources are:

- `comparison_results.csv`
- `CNN-Pytorch.ipynb`
- `VIT-Swin.ipynb`
- `models/swin_alzheimer_classifier.pt`
