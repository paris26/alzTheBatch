# Alzheimer's MRI Classification (CNN vs Swin Transformer)

Educational walkthrough for classifying brain MRI scans into four cognitive-impairment categories using two approaches:
1) a small PyTorch CNN with augmentation and weighted sampling, and 2) a Swin Transformer vision model that achieves stronger performance.

## Dataset
- Source: Hugging Face dataset `Falah/Alzheimer_MRI` (grayscale MRIs).
- Classes: Mild Demented, Moderate Demented, Non Demented, Very Mild Demented.
- Structure: Hugging Face download in notebooks; local data folders under `best-alzheimer-mri-dataset-99-accuracy/` mirror the class subfolders.
- Note: Images are grayscale; Swin expects RGB, so channels are triplicated during preprocessing.

## Approaches
- CNN (PyTorch): compact custom CNN; data augmentation; class-weighted loss; WeightedRandomSampler for imbalance.
- Swin Transformer: `microsoft/swin-base-patch4-window7-224` via Hugging Face; transfer learning; stronger augmentation (MONAI) tailored to minority classes; differential learning rates.

## Results (validation)
- CNN results are mixed and depend strongly on the imbalance-handling strategy:
  - Best CNN raw accuracy: MONAI Augmentation, Accuracy ≈ 0.60, Balanced Accuracy ≈ 0.35, ROC-AUC ≈ 0.80
  - Best CNN balanced accuracy: Combined Strategy, Accuracy ≈ 0.35, Balanced Accuracy ≈ 0.51, ROC-AUC ≈ 0.65
- Swin Transformer is clearly stronger overall: Accuracy ≈ 0.88, Balanced Accuracy ≈ 0.90, Specificity Macro ≈ 0.95, ROC-AUC ≈ 0.98.
- See `comparison_results.csv` and the evaluation sections in the notebooks for the full metric breakdown.

## How to Run
1) Install dependencies (Python 3.10+ recommended):
   - torch, torchvision, transformers, datasets, monai, scikit-learn, matplotlib, pandas, tqdm
2) Open the notebooks:
   - `VIT-Swin.ipynb` for the Swin Transformer pipeline
   - `CNN-Pytorch.ipynb` for the CNN experiments
3) Run cells in order. GPU is recommended; code falls back to MPS/CPU if CUDA is unavailable.

## Training Details
- Imbalance handling: class-weighted CrossEntropyLoss; WeightedRandomSampler oversamples minority classes.
- Augmentation: flips, rotations, zoom; Swin path also adds noise, contrast shifts, and Gaussian smoothing (via MONAI) with stronger transforms for minority classes.
- Optimization: AdamW; differential LRs for backbone vs classifier head (Swin); ReduceLROnPlateau scheduler; gradient clipping; early stopping.

## Evaluation and Visuals
- Metrics: Accuracy, Balanced Accuracy, Specificity (macro and per class), F1 (macro/weighted), MCC, Cohen's Kappa, ROC-AUC OvR macro, per-class precision/recall/F1/support.
- Plots (in notebooks): training curves, confusion matrix, sample predictions with confidences.

## Saved Models
- Swin checkpoint: `models/swin_alzheimer_classifier.pt` (saved from `VIT-Swin.ipynb`).


## Credits
- Dataset: `Falah/Alzheimer_MRI` on Hugging Face.
- Models: PyTorch; Hugging Face Transformers (Swin).
