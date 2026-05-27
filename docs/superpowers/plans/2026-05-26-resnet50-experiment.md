# ResNet-50 vs Swin Transformer -- Matched Experiment

**Goal:** Train a pretrained ResNet-50 on the Alzheimer's MRI dataset using the exact same setup as the Swin notebook, then update metrics/docs to report both side-by-side.

**Why ResNet-50 (not ResNet-18):** Swin-Base has ~87M params. ResNet-18 has ~11M -- that's a 8x gap. ResNet-50 has ~25M params and 50 layers with bottleneck residual blocks, making it the closest standard CNN to Swin in both capacity and depth. It's still 3.5x smaller, but that's a much fairer fight than 8x.

---

## What we match to the Swin notebook

Every decision below is copied from `VIT-Swin.ipynb` so the only variable is the architecture itself.

| Setting | Swin (existing) | ResNet-50 (new) |
|---------|-----------------|-----------------|
| Input | 224x224 RGB | 224x224 RGB |
| Pretraining | ImageNet-1K | ImageNet-1K |
| Split | 80/20 stratified, random_state=42 | **same split, same indices** |
| Freeze ratio | 60% of params frozen | 60% of params frozen |
| Optimizer | AdamW | AdamW |
| LR backbone | 2e-5 | 2e-5 |
| LR head | 2e-4 (10x backbone) | 2e-4 (10x backbone) |
| Weight decay | 0.01 | 0.01 |
| Loss | CrossEntropyLoss(weight=class_weights) | CrossEntropyLoss(weight=class_weights) |
| Scheduler | ReduceLROnPlateau(patience=2, factor=0.5) | ReduceLROnPlateau(patience=2, factor=0.5) |
| Epochs | 15 | 15 |
| Early stopping | patience=5 on val accuracy | patience=5 on val accuracy |
| Grad clipping | max_norm=1.0 | max_norm=1.0 |
| Augmentation | MONAI (aggressive minority, standard majority) | same MONAI pipeline |
| Sampling | WeightedRandomSampler | WeightedRandomSampler |
| Evaluation | val set, compute_all_metrics() | val set, same function |
| Metrics | acc, bal_acc, spec, F1, MCC, kappa, ROC-AUC, per-class | identical |

---

## File structure

```
AlzTheBatch/
  CNN-ResNet50.ipynb             # NEW -- the experiment
  VIT-Swin.ipynb                 # UNCHANGED
  comparison_results.csv         # UPDATE -- add ResNet-50 row
  unified_evaluation_summary.md  # UPDATE -- add ResNet-50 section
  models/
    resnet50_alzheimer.pt        # NEW -- saved checkpoint
    swin_alzheimer_classifier.pt # UNCHANGED
```

---

## Task 1: Create the notebook -- data loading

Create `CNN-ResNet50.ipynb`.

- [ ] **Step 1: Imports**

```python
# ====== IMPORTS & DEVICE SETUP ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay, f1_score,
    matthews_corrcoef, cohen_kappa_score, precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from collections import Counter
import monai.transforms as mt
import pandas as pd
import copy
import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device | PyTorch {torch.__version__}")
```

- [ ] **Step 2: Load data -- same split as Swin**

Identical to `VIT-Swin.ipynb` cell 10: 80/20 stratified, random_state=42.

```python
# ====== DATASET -- SAME SPLIT AS SWIN ======
dataset = load_dataset('Falah/Alzheimer_MRI', split='train')
NUM_CLASSES = 4
LABELS = {0: "Mild Demented", 1: "Moderate Demented", 2: "Non Demented", 3: "Very Mild Demented"}
CLASS_NAMES = list(LABELS.values())
print(f"Loaded {len(dataset)} samples")

all_labels = [ex['label'] for ex in dataset]

# SAME split as Swin notebook -- 80/20 stratified, random_state=42
train_idx, val_idx = train_test_split(
    np.arange(len(dataset)), test_size=0.2, stratify=all_labels, random_state=42
)

train_labels = [dataset[idx]['label'] for idx in train_idx]
class_counts = Counter(train_labels)

print("Training set distribution:")
for k in sorted(class_counts):
    print(f"  {LABELS[k]}: {class_counts[k]} ({100*class_counts[k]/len(train_labels):.1f}%)")

# Class weights -- same formula as Swin notebook
total_train = len(train_labels)
class_weights = torch.tensor(
    [total_train / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)],
    dtype=torch.float32
).to(device)
print(f"\nClass weights: {class_weights.cpu().numpy().round(3)}")
print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
```

- [ ] **Step 3: Dataset class -- same augmentation as Swin**

Copied from `VIT-Swin.ipynb` cell 9 but using torchvision ImageNet normalization instead of HuggingFace image processor (ResNet expects the same normalization -- both are ImageNet-pretrained).

```python
# ====== DATASET CLASS ======
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class AlzMRIDataset(Dataset):
    """
    Grayscale MRI -> 224x224 RGB.
    MONAI augmentation matching VIT-Swin.ipynb exactly.
    """
    def __init__(self, hf_dataset, indices, class_counts=None,
                 augment=True, minority_threshold=0.15):
        self.data = hf_dataset
        self.indices = indices
        self.augment = augment

        # Identify minority classes -- same logic as Swin notebook
        self.minority_classes = set()
        if class_counts:
            total = sum(class_counts.values())
            self.minority_classes = {
                c for c, count in class_counts.items()
                if count / total < minority_threshold
            }

        # MONAI pipelines -- identical to Swin notebook
        self.monai_minority = mt.Compose([
            mt.RandFlip(prob=0.5, spatial_axis=1),
            mt.RandRotate(prob=0.8, range_x=0.35),
            mt.RandZoom(prob=0.5, min_zoom=0.85, max_zoom=1.15),
            mt.RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
            mt.RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
            mt.RandGaussianSmooth(prob=0.2, sigma_x=(0.25, 0.75)),
            mt.RandShiftIntensity(prob=0.3, offsets=0.1),
        ])
        self.monai_standard = mt.Compose([
            mt.RandFlip(prob=0.5, spatial_axis=1),
            mt.RandRotate(prob=0.5, range_x=0.26),
            mt.RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.1),
        ])

        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = np.array(self.data[real_idx]['image'], dtype=np.float32) / 255.0
        label = self.data[real_idx]['label']

        img_t = torch.tensor(img).unsqueeze(0)  # (1, H, W)

        if self.augment:
            if label in self.minority_classes:
                img_t = self.monai_minority(img_t)
            else:
                img_t = self.monai_standard(img_t)

        img_t = torch.clamp(img_t, 0, 1)

        # Resize to 224x224
        img_t = F.interpolate(
            img_t.unsqueeze(0), size=224, mode='bilinear', align_corners=False
        ).squeeze(0)

        # Grayscale -> RGB (3 channels)
        img_rgb = img_t.repeat(3, 1, 1)

        # ImageNet normalization
        img_rgb = self.normalize(img_rgb)

        return img_rgb, torch.tensor(label, dtype=torch.long)


# Create datasets
train_dataset = AlzMRIDataset(dataset, train_idx, class_counts, augment=True)
val_dataset = AlzMRIDataset(dataset, val_idx, augment=False)

print(f"Train dataset: {len(train_dataset)} | Val dataset: {len(val_dataset)}")
```

- [ ] **Step 4: DataLoaders with balanced sampler -- same as Swin**

```python
# ====== DATALOADERS ======
BATCH_SIZE = 32

# Balanced sampler -- same as Swin notebook
sample_weights = [1.0 / class_counts[all_labels[i]] for i in train_idx]
balanced_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

pin = (device == 'cuda')
train_loader = DataLoader(train_dataset, BATCH_SIZE, sampler=balanced_sampler,
                          pin_memory=pin, num_workers=2)
val_loader = DataLoader(val_dataset, BATCH_SIZE, pin_memory=pin, num_workers=2)

print(f"Batch size: {BATCH_SIZE}")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
```

- [ ] **Step 5: Commit**

```bash
git add CNN-ResNet50.ipynb
git commit -m "feat: add ResNet-50 notebook with data loading matching Swin setup"
```

---

## Task 2: Model setup and training

- [ ] **Step 1: Load pretrained ResNet-50 and freeze 60%**

```python
# ====== RESNET-50 MODEL ======
print("Loading ResNet-50 (ImageNet pretrained)...")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Replace final FC for 4 classes (Swin replaces classifier head the same way)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)  # 2048 -> 4

resnet = resnet.to(device)

# Freeze 60% of parameters -- same ratio as Swin notebook
all_params = list(resnet.named_parameters())
freeze_until = int(len(all_params) * 0.6)
for i, (name, param) in enumerate(all_params):
    param.requires_grad = (i >= freeze_until)

# Always train the FC head (same as Swin always trains classifier)
for param in resnet.fc.parameters():
    param.requires_grad = True

total_p = sum(p.numel() for p in resnet.parameters())
trainable_p = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(f"Total params: {total_p:,}")
print(f"Trainable: {trainable_p:,} ({100*trainable_p/total_p:.1f}%)")
```

- [ ] **Step 2: Training config -- identical to Swin**

```python
# ====== TRAINING CONFIG -- MATCHES SWIN EXACTLY ======
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5

# Differential LR: backbone 2e-5, head 2e-4 (10x) -- same as Swin
backbone_params = [p for n, p in resnet.named_parameters()
                   if 'fc' not in n and p.requires_grad]
head_params = list(resnet.fc.parameters())

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LEARNING_RATE},
    {'params': head_params, 'lr': LEARNING_RATE * 10},
], weight_decay=WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss(weight=class_weights)

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

print("Training Configuration (mirrors Swin):")
print(f"  LR backbone:    {LEARNING_RATE}")
print(f"  LR head:        {LEARNING_RATE * 10}")
print(f"  Weight decay:   {WEIGHT_DECAY}")
print(f"  Epochs:         {EPOCHS}")
print(f"  Early stopping: {EARLY_STOPPING_PATIENCE}")
print(f"  Grad clipping:  1.0")
```

- [ ] **Step 3: Training loop -- same structure as Swin**

```python
# ====== TRAINING LOOP ======
def train_resnet(model, train_loader, val_loader, optimizer, criterion,
                 scheduler=None, epochs=15, early_stopping_patience=5):
    """Training loop -- same structure as train_swin_model in VIT-Swin.ipynb."""
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm.tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if scheduler:
            scheduler.step(avg_val_loss)

        # Early stopping on val accuracy -- same as Swin
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *best*"
        else:
            patience_counter += 1
            marker = ""

        print(f"Epoch {epoch+1}: Train Acc={history['train_acc'][-1]:.4f} | "
              f"Val Acc={val_acc:.4f} | Val Loss={avg_val_loss:.4f}{marker}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, history

# Run training
print("=" * 60)
print("Training ResNet-50")
print("=" * 60)

resnet, history = train_resnet(
    resnet, train_loader, val_loader,
    optimizer, criterion, scheduler,
    epochs=EPOCHS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE
)
```

- [ ] **Step 4: Commit**

```bash
git add CNN-ResNet50.ipynb
git commit -m "feat: add ResNet-50 model setup and training loop"
```

---

## Task 3: Evaluation, checkpoint, and comparison

- [ ] **Step 1: Metrics function + evaluation -- same as Swin**

```python
# ====== METRICS -- SAME FUNCTION AS SWIN NOTEBOOK ======
def compute_all_metrics(y_true, y_pred, y_proba=None, class_names=None):
    """Identical to VIT-Swin.ipynb cell 14."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(precision))]

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    metrics['per_class'] = {}
    specificity_scores = []

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_scores.append(spec)
        metrics['per_class'][name] = {
            'precision': precision[i], 'recall': recall[i],
            'specificity': spec, 'f1': f1[i], 'support': int(support[i])
        }

    metrics['specificity_macro'] = float(np.mean(specificity_scores))
    if y_proba is not None:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            )
        except Exception:
            metrics['roc_auc_ovr'] = None
    return metrics


def predict_with_proba(model, loader):
    """Same as Swin notebook."""
    model.eval()
    preds, labels, probas = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(device))
            probs = F.softmax(out, dim=1)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
            probas.extend(probs.cpu().numpy())
    return np.array(preds), np.array(labels), np.array(probas)


# Evaluate on validation set -- same as Swin
preds, labels, probas = predict_with_proba(resnet, val_loader)
metrics = compute_all_metrics(labels, preds, probas, CLASS_NAMES)

print("=" * 60)
print("RESNET-50 -- EVALUATION RESULTS")
print("=" * 60)
for k in ['accuracy', 'balanced_accuracy', 'specificity_macro',
          'f1_macro', 'f1_weighted', 'mcc', 'kappa', 'roc_auc_ovr']:
    val = metrics.get(k)
    if val is not None:
        print(f"  {k:25s}: {val:.4f}")
    else:
        print(f"  {k:25s}: N/A")

print("\nPer-class:")
for cls in CLASS_NAMES:
    pc = metrics['per_class'][cls]
    print(f"  {cls:22s}  P={pc['precision']:.4f}  R={pc['recall']:.4f}  "
          f"Sp={pc['specificity']:.4f}  F1={pc['f1']:.4f}  n={pc['support']}")
```

- [ ] **Step 2: Save checkpoint -- same format as Swin**

```python
# ====== SAVE CHECKPOINT -- SAME FORMAT AS SWIN ======
torch.save({
    'model_state_dict': resnet.state_dict(),
    'metrics': metrics,
    'history': history,
    'labels': LABELS,
}, 'models/resnet50_alzheimer.pt')
print("Saved to models/resnet50_alzheimer.pt")
```

- [ ] **Step 3: Side-by-side comparison table**

```python
# ====== SIDE-BY-SIDE COMPARISON ======
# Load Swin metrics
swin_ckpt = torch.load('models/swin_alzheimer_classifier.pt',
                        map_location='cpu', weights_only=False)
swin_metrics = swin_ckpt['metrics']

print("=" * 70)
print("HEAD-TO-HEAD: ResNet-50 vs Swin Transformer")
print("=" * 70)

header = f"{'Metric':<25} {'ResNet-50':>10} {'Swin':>10} {'Delta':>10}"
print(header)
print("-" * len(header))
for k in ['accuracy', 'balanced_accuracy', 'specificity_macro',
          'f1_macro', 'f1_weighted', 'mcc', 'kappa', 'roc_auc_ovr']:
    r = metrics.get(k, 0) or 0
    s = swin_metrics.get(k, 0) or 0
    delta = r - s
    sign = "+" if delta >= 0 else ""
    print(f"  {k:<23} {r:>10.4f} {s:>10.4f} {sign}{delta:>9.4f}")

print("\nPer-class F1:")
for cls in CLASS_NAMES:
    r = metrics['per_class'][cls]['f1']
    s = swin_metrics['per_class'][cls]['f1']
    delta = r - s
    sign = "+" if delta >= 0 else ""
    print(f"  {cls:<22} {r:>10.4f} {s:>10.4f} {sign}{delta:>9.4f}")
```

- [ ] **Step 4: Visualizations**

```python
# ====== TRAINING CURVES ======
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

epochs_range = range(1, len(history['train_loss']) + 1)
ax1.plot(epochs_range, history['train_loss'], label='Train')
ax1.plot(epochs_range, history['val_loss'], label='Val')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('ResNet-50 Loss')
ax1.legend()

ax2.plot(epochs_range, history['train_acc'], label='Train')
ax2.plot(epochs_range, history['val_acc'], label='Val')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.set_title('ResNet-50 Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('resnet50_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```

```python
# ====== CONFUSION MATRICES SIDE BY SIDE ======
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ResNet-50
cm_resnet = confusion_matrix(labels, preds)
short_names = [n.replace(" Demented", "").replace("Non ", "Non-") for n in CLASS_NAMES]
ConfusionMatrixDisplay(cm_resnet, display_labels=short_names).plot(ax=ax1, cmap='Blues', values_format='d')
ax1.set_title('ResNet-50')

# Swin (from checkpoint)
swin_preds = swin_ckpt.get('predictions')
if swin_preds is not None:
    cm_swin = confusion_matrix(swin_ckpt['labels'], swin_preds)
    ConfusionMatrixDisplay(cm_swin, display_labels=short_names).plot(ax=ax2, cmap='Oranges', values_format='d')
    ax2.set_title('Swin Transformer')
else:
    ax2.text(0.5, 0.5, 'Swin predictions\nnot in checkpoint', ha='center', va='center')
    ax2.set_title('Swin Transformer (metrics only)')

plt.tight_layout()
plt.savefig('resnet50_vs_swin_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 5: Commit**

```bash
git add CNN-ResNet50.ipynb models/resnet50_alzheimer.pt resnet50_training_curves.png resnet50_vs_swin_confusion.png
git commit -m "feat: add ResNet-50 evaluation with Swin comparison"
```

---

## Task 4: Update comparison CSV and docs

- [ ] **Step 1: Append ResNet-50 to comparison_results.csv**

```python
# ====== UPDATE CSV ======
existing_df = pd.read_csv('comparison_results.csv')

row = {
    'Approach': 'ResNet-50 (Pretrained)',
    'Accuracy': metrics['accuracy'],
    'Balanced Accuracy': metrics['balanced_accuracy'],
    'Specificity Macro': metrics['specificity_macro'],
    'F1 Macro': metrics['f1_macro'],
    'F1 Weighted': metrics['f1_weighted'],
    'MCC': metrics['mcc'],
    'Kappa': metrics['kappa'],
    'ROC-AUC OvR Macro': metrics.get('roc_auc_ovr'),
}
for cls in CLASS_NAMES:
    pc = metrics['per_class'][cls]
    row[f'{cls}_Precision'] = pc['precision']
    row[f'{cls}_Recall'] = pc['recall']
    row[f'{cls}_Specificity'] = pc['specificity']
    row[f'{cls}_F1'] = pc['f1']
    row[f'{cls}_Support'] = pc['support']

# Remove any existing ResNet-50 row, then append
existing_df = existing_df[~existing_df['Approach'].astype(str).str.contains('ResNet-50', na=False)]
updated_df = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
updated_df.to_csv('comparison_results.csv', index=False)
print("Updated comparison_results.csv")
print(updated_df[['Approach', 'Accuracy', 'Balanced Accuracy', 'F1 Macro', 'MCC']].round(4).to_string(index=False))
```

- [ ] **Step 2: Update unified_evaluation_summary.md**

Add a new section after the existing Swin section with:

- Model card table (architecture, params, all hyperparameters)
- Overall metrics table
- Per-class metrics table
- Head-to-head delta table vs Swin
- One paragraph interpreting what the gap means

Use actual numbers from the run -- no placeholders.

- [ ] **Step 3: Commit**

```bash
git add comparison_results.csv unified_evaluation_summary.md
git commit -m "docs: add ResNet-50 results to comparison CSV and summary"
```

---

## What this answers for the report

| Question | Answer |
|----------|--------|
| **What model?** | ResNet-50, torchvision, ImageNet-1K pretrained |
| **What hyperparameters?** | Printed in model card -- identical to Swin |
| **What evaluation?** | Same val set, same metrics function, same split |
| **How compared?** | Head-to-head delta table, same conditions, only architecture differs |

The comparison isolates **architecture** as the single variable. If ResNet-50 underperforms Swin, the gap is attributable to transformers vs CNNs -- not to different training recipes, different data splits, or different metrics.
