# Alzheimer's MRI Classification (ResNet-50 and Swin Transformer)

This repository contains the experiments and paper artifacts for classifying
brain MRI scans into four cognitive-impairment categories. The main comparison
uses a ResNet-50 convolutional network and a Swin Transformer. The earlier
custom CNN experiments are retained for reference.

## Dataset

- Source: Hugging Face dataset `Falah/Alzheimer_MRI`.
- Pinned runner revision: `daac24f9597236b45837d82f7eb9c9ad1f8c60c8`.
- Classes: Mild Demented, Moderate Demented, Non Demented, Very Mild Demented.
- Images are grayscale; model inputs repeat the channel to obtain RGB images.
- The reproducible runner uses an 80/20 stratified split with split seed 42.

The historical local dataset mirror under
`best-alzheimer-mri-dataset-99-accuracy/` is retained on this recovery branch.
New experiments load the pinned Hugging Face revision directly.

## Repository contents

- `run_reproducible_experiments.py`: standalone ResNet-50 and Swin experiment
  runner using five training seeds per model.
- `CNN-ResNet50.ipynb`: ResNet-50 training, evaluation, and interpretation.
- `VIT-Swin.ipynb`: Swin training, evaluation, and interpretation.
- `CNN-Pytorch.ipynb`: retained custom CNN experiments.
- `scripts/`: scripts used to regenerate paper figures.
- `paper/figures/`: generated figures and figure documentation.
- `comparison_results.csv` and the evaluation summaries: historical experiment
  results and metric documentation.

## Results (validation)

- The earlier custom CNN results vary substantially with imbalance handling.
  The best raw accuracy was approximately 0.60, while the best balanced
  accuracy was approximately 0.51.
- The Swin Transformer achieved approximately 0.88 accuracy, 0.90 balanced
  accuracy, 0.95 macro specificity, and 0.98 ROC-AUC.
- See `comparison_results.csv`, `unified_evaluation_summary.md`, and the
  notebook evaluation sections for the complete metric breakdown.

## Running the experiments

Python 3.10 or newer is recommended. Install PyTorch for the target CPU/GPU
platform, followed by the remaining dependencies: torchvision, transformers,
datasets, MONAI, scikit-learn, matplotlib, pandas, NumPy, Pillow, and tqdm.

Run the standalone experiment with:

```bash
python run_reproducible_experiments.py
```

The default configuration performs ten training runs: five seeds for each of
ResNet-50 and Swin. It writes checkpoints, split indices, configuration,
environment metadata, per-seed results, and an aggregate summary under
`reproducibility_results/`.

The notebooks can also be executed individually. A GPU is strongly recommended
for training; the code supports CPU and available accelerator backends.

## Training details

- Class-weighted cross-entropy and weighted sampling address class imbalance.
- Augmentation includes flips, rotations, and zooms, with stronger MONAI
  transforms for minority classes.
- Optimization uses AdamW, differential classifier-head learning rates,
  gradient clipping, learning-rate reduction, and early stopping.
- The standalone runner pins the dataset and Swin base-model revisions and
  records the package environment and script checksum with each run.

## Recovered trained models

The original notebook-trained checkpoints have been recovered locally:

- `models/resnet50_alzheimer.pt` — 94,406,017 bytes.
- `models/swin_alzheimer_classifier.pt` — 347,650,683 bytes.

The binary checkpoints are intentionally ignored by ordinary Git. Their
SHA-256 hashes are recorded in `models/checksums.sha256`. They should be
distributed through a model artifact service or Git LFS rather than committed
as regular Git objects.

These recovered files correspond to the original notebook runs. They are not
the ten per-seed checkpoints that the standalone runner can produce.

## Grad-CAM

Both main notebooks include Grad-CAM visualizations:

- ResNet-50 uses the final residual block, `layer4[-1]`.
- Swin uses the final transformer's `layernorm_before` activations and reshapes
  its spatial tokens into a feature grid.

The implementations use the same samples, `jet` colormap, `[0, 1]`
normalization, overlay alpha, and labeled colorbar. After reproducible-runner
checkpoints are available, generate a side-by-side comparison with:

```bash
python scripts/generate_gradcam_comparison.py --seed 0
```

The default output is `paper/figures/gradcam_resnet50_vs_swin.png`. Pass
`--target true` to explain the ground-truth class rather than each network's
predicted class.

## Evaluation and limitations

Reported metrics include accuracy, balanced accuracy, macro specificity,
macro/weighted F1, MCC, Cohen's kappa, ROC-AUC OvR, and per-class metrics.

This is a research and educational project. The models are not validated for
clinical use and must not be used to diagnose or guide treatment.

## Credits

- Dataset: `Falah/Alzheimer_MRI` on Hugging Face.
- Models: PyTorch/torchvision ResNet-50 and Hugging Face Transformers Swin.
