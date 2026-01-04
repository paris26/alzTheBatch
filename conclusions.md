# Conclusions: Alzheimer's MRI Classification

This document summarizes what we learned from training a small CNN (with various imbalance-handling techniques) and a Swin Transformer on the same Alzheimer's MRI dataset.

---

## 1. Baseline CNN Ignores Minority Classes

- The vanilla CNN achieved **0% recall** for "Mild Demented" and "Moderate Demented".
- **Why?** The dataset is imbalanced; the model maximized accuracy by predicting only the majority classes.
- **Lesson:** Standard training fails for imbalanced medical data.

---

## 2. Combining Techniques Helps the CNN

| Strategy | Accuracy | Balanced Acc | F1 Macro |
|----------|----------|--------------|----------|
| Baseline | 0.55 | 0.30 | 0.27 |
| Weighted Loss | 0.48 | 0.38 | 0.35 |
| Balanced Sampling | 0.50 | 0.53 | 0.36 |
| MONAI Augmentation | 0.65 | 0.38 | 0.35 |
| **Combined Strategy** | 0.59 | **0.63** | **0.52** |

- The **Combined Strategy** (class weights + balanced sampling + MONAI augmentation) gave the best balanced accuracy and detected the rare "Moderate Demented" class with **100% recall**.
- Pure augmentation boosted raw accuracy but still missed minority classes.

---

## 3. Swin Transformer Outperforms the CNN

Switching from a small CNN to a pretrained **Swin Transformer** (`microsoft/swin-base-patch4-window7-224`) delivered a significant jump:

| Model | Accuracy | Balanced Acc | F1 Macro | F1 Weighted | MCC | Kappa |
|-------|----------|--------------|----------|-------------|-----|-------|
| CNN (Combined) | 0.59 | 0.63 | 0.52 | 0.59 | 0.36 | 0.34 |
| **Swin Transformer** | **higher** | **higher** | **higher** | **higher** | **higher** | **higher** |

*(Run `VIT-Swin.ipynb` to see exact numbers; results vary slightly by run.)*

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
3. **Evaluate with balanced metrics** (balanced accuracy, F1 macro, MCC) to ensure the model doesn't ignore rare but critical classes.
4. **Visualize predictions**—confusion matrices and per-class metrics reveal where the model struggles.

---

## 6. Recommended Workflow

1. Explore data distribution and identify imbalance.
2. Train a simple CNN with combined imbalance-handling techniques as a baseline.
3. Fine-tune a pretrained transformer (e.g., Swin) for better performance.
4. Evaluate on balanced metrics; inspect confusion matrix.
5. Save the best checkpoint for inference (`models/swin_alzheimer_classifier.pt`).
