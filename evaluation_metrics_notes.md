# Evaluation Metrics Notes

This note summarizes which evaluation metrics are already available for the CNN and Swin models, what was added, what was confirmed by execution, and what is still missing.

## Current Status

The notebook source has now been updated to:

- compute per-class specificity
- compute macro specificity
- export ROC-AUC more consistently in the CNN comparison pipeline
- include specificity in the per-class metric tables

The refresh was then executed in the active conda environment.

- The CNN comparison section was run end to end and regenerated `comparison_results.csv`.
- The saved Swin checkpoint was loaded successfully and its stored metrics were confirmed.

This means the key additions are now confirmed rather than only proposed.

## Execution Confirmation

Confirmed from execution or checkpoint loading:

- CNN `roc_auc_ovr_macro`: `0.6502`
- CNN `specificity_macro`: `0.7567`
- Swin `roc_auc_ovr_macro`: `0.9773`
- Swin `specificity_macro`: `0.9503`

The updated inventory CSV contains the per-class values as well.

## Important Caveat

The refreshed CNN results differ materially from the older checked-in CNN summary that had the combined strategy near `0.59` accuracy.

That means the repo now has two different stories:

- the older narrative in `README.md` and `conclusions.md`
- the newly executed CNN comparison export in `comparison_results.csv`

If you want the paper/report to be internally consistent, you should decide which result set is the authoritative one and then update the narrative text accordingly.

## What Is Already Covered

The current evaluation setup already reports a strong core set of metrics:

- Accuracy
- Balanced accuracy
- F1 macro
- F1 weighted
- MCC
- Cohen's kappa
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix
- Classification report

For this project, balanced accuracy, macro F1, MCC, and the per-class metrics are the most important ones because the dataset is imbalanced.

## Important Clarification

Sensitivity is already present.

- In the current notebooks, sensitivity is the same thing as per-class recall.
- So there is no need to add a separate "sensitivity" metric unless the goal is just to rename recall to a more clinical term.

## What Was Added

The main requested addition, specificity, is now implemented in the notebook code.

- `specificity_per_class`
- `specificity_macro`

The inventory CSV now reflects executed CNN values and confirmed Swin checkpoint values.

ROC-AUC export was also fixed in the CNN notebook code and is now confirmed in the regenerated comparison CSV.

## What Is Still Missing

Two additional optional metrics would also be useful:

- `pr_auc_ovr_macro`, which is often more informative than ROC-AUC under class imbalance
- calibration metrics if model confidence will be discussed, for example Brier score or expected calibration error, ECE

## Can This Be Done From What We Already Collected?

Partly.

### No Code Change Needed

No code change is needed to:

- explain the current metric set
- state that sensitivity is already covered by recall
- identify PR-AUC and calibration as still missing from the current reports

### Small Code Change Or Re-Export Needed

The executed refresh resolved the main missing items, so this section now only applies to optional future metrics.

#### Specificity

Specificity is now implemented in the notebooks, and the CNN export now includes it directly.

For Swin, the inventory uses specificity derived from the exact checkpoint precision, recall, and support values.

In general, specificity requires one of the following to be exported cleanly:

- the confusion matrix values
- per-sample predictions and labels
- explicit TP/FP/TN/FN counts in a one-vs-rest format

#### PR-AUC And Calibration

These are still not implemented. They need predicted probabilities to be saved and exported consistently.

## Recommendation

Keep the current metrics as the main comparison set, and add these next:

1. Per-class specificity
2. Macro specificity
3. Macro OvR ROC-AUC exported consistently
4. Macro OvR PR-AUC if you want one more imbalance-aware probability metric

That is enough to make the evaluation more complete without turning it into metric overload.
