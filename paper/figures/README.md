# Dataset class samples

`dataset_class_samples.png` and `dataset_class_samples.pdf` show one
unaugmented validation-set example from each class in the pinned
`Falah/Alzheimer_MRI` dataset revision used by the experiments:
`daac24f9597236b45837d82f7eb9c9ad1f8c60c8`.

Suggested caption:

> Example coronal brain MRI samples from the four dataset classes:
> (a) Mild Demented, (b) Moderate Demented, (c) Non Demented, and
> (d) Very Mild Demented. Images are shown without data augmentation.

The figure is regenerated from the executed output of `VIT-Swin.ipynb` by
running:

```bash
python scripts/extract_class_samples_figure.py
```

`dataset_class_samples_with_predictions.png` is the original notebook output
retained for provenance; it also includes the Swin model confidence plots.
