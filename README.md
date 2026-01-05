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
- Best CNN variant (Combined Strategy): Accuracy ≈ 0.59, Balanced Accuracy ≈ 0.63 (see `comparison_results.csv`).
- Swin Transformer: higher accuracy/balanced accuracy and per-class F1 than the CNN runs (see `VIT-Swin.ipynb` evaluation section for metrics and plots).

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
- Metrics: Accuracy, Balanced Accuracy, F1 (macro/weighted), MCC, Cohen's Kappa, per-class precision/recall/F1.
- Plots (in notebooks): training curves, confusion matrix, sample predictions with confidences.

## Saved Models
- Swin checkpoint: `models/swin_alzheimer_classifier.pt` (saved from `VIT-Swin.ipynb`).


## Credits
- Dataset: `Falah/Alzheimer_MRI` on Hugging Face.
- Models: PyTorch; Hugging Face Transformers (Swin).
