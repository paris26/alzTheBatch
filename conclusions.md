# Conclusions: Alzheimer's MRI Classification

This document summarizes what we learned from training a small CNN (with various imbalance-handling techniques) and a Swin Transformer on the same Alzheimer's MRI dataset.

---

## 1. Baseline CNN Ignores Minority Classes

- The vanilla CNN achieved **0% recall** for "Mild Demented" and "Moderate Demented".
- **Why?** The dataset is imbalanced; the model maximized accuracy by predicting only the majority classes.
- **Lesson:** Standard training fails for imbalanced medical data.

---

## 2. CNN Imbalance Strategies Show Tradeoffs, Not a Clean Win

| Strategy | Accuracy | Balanced Acc | Specificity Macro | F1 Macro | MCC | ROC-AUC |
|----------|----------|--------------|-------------------|----------|-----|---------|
| Baseline | 0.53 | 0.28 | 0.77 | 0.22 | 0.14 | 0.73 |
| Weighted Loss | 0.50 | 0.25 | 0.75 | 0.17 | 0.00 | 0.50 |
| Balanced Sampling | 0.49 | 0.32 | 0.77 | 0.20 | 0.10 | 0.60 |
| **MONAI Augmentation** | **0.60** | 0.35 | **0.83** | **0.32** | **0.32** | **0.80** |
| **Combined Strategy** | 0.35 | **0.51** | 0.76 | 0.27 | 0.08 | 0.65 |

- The **Combined Strategy** still achieved the best balanced accuracy and detected the rare "Moderate Demented" class with **100% recall**.
- But that came with a severe tradeoff: in the refreshed run it collapsed to **0% recall for "Non Demented"**, so it is not a robust overall winner.
- **MONAI Augmentation** gave the strongest overall CNN profile on raw accuracy, macro F1, MCC, specificity, and ROC-AUC, even though its balanced accuracy remained modest.

---

## 3. Swin Transformer Outperforms the CNN

Switching from a small CNN to a pretrained **Swin Transformer** (`microsoft/swin-base-patch4-window7-224`) delivered a significant jump:

| Model | Accuracy | Balanced Acc | Specificity Macro | F1 Macro | MCC | Kappa | ROC-AUC |
|-------|----------|--------------|-------------------|----------|-----|-------|---------|
| CNN (Best balanced: Combined) | 0.35 | 0.51 | 0.76 | 0.27 | 0.08 | 0.04 | 0.65 |
| CNN (Best overall: MONAI) | 0.60 | 0.35 | 0.83 | 0.32 | 0.32 | 0.30 | 0.80 |
| **Swin Transformer** | **0.88** | **0.90** | **0.95** | **0.89** | **0.80** | **0.80** | **0.98** |

The Swin Transformer improved accuracy by roughly **28 percentage points** over the strongest CNN-by-accuracy and balanced accuracy by roughly **39 percentage points** over the strongest CNN-by-balanced-accuracy. Per-class detection, specificity, and probability ranking performance also improved substantially.

- **Why does Swin win?**
  - Transfer learning from ImageNet provides strong visual features.
  - Shifted-window attention captures both local and global patterns.
  - Differential learning rates and aggressive augmentation for minority classes help further.

---

## 4. Accuracy vs Balanced Accuracy

- High **accuracy** can be misleading when classes are imbalanced.
- **Balanced accuracy** averages recall across classes, giving equal weight to rare conditions.
- In medical diagnosis, missing a positive case (false negative) is costly—prefer balanced metrics.

---

## 5. Practical Takeaways

1. **Start with imbalance-aware training** (weighted loss, balanced sampling, augmentation) before scaling up model complexity.
2. **Transfer learning** with modern vision transformers (Swin, ViT) can dramatically improve results on small medical datasets.
3. **Evaluate with balanced metrics** (balanced accuracy, specificity, F1 macro, MCC, ROC-AUC) to ensure the model doesn't ignore rare but critical classes.
4. **Visualize predictions**—confusion matrices and per-class metrics reveal where the model struggles.

---

## 6. Recommended Workflow

1. Explore data distribution and identify imbalance.
2. Train a simple CNN baseline and compare multiple imbalance-handling variants rather than assuming the combined recipe will dominate every metric.
3. Fine-tune a pretrained transformer (e.g., Swin) for better performance.
4. Evaluate on balanced metrics; inspect confusion matrix and per-class specificity/recall.
5. Save the best checkpoint for inference (`models/swin_alzheimer_classifier.pt`).
