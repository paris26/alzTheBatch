from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import monai.transforms as mt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import Normalize
from transformers import AutoImageProcessor, SwinForImageClassification


@dataclass(frozen=True)
class Config:
    models: tuple[str, ...] = ("resnet50", "swin")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    split_seed: int = 42

    dataset_name: str = "Falah/Alzheimer_MRI"
    dataset_split: str = "train"
    dataset_revision: str = "daac24f9597236b45837d82f7eb9c9ad1f8c60c8"
    validation_fraction: float = 0.20

    swin_name: str = "microsoft/swin-base-patch4-window7-224"
    swin_revision: str = "20d6c26ef6455d36c0a78671d787c5da57513d4bd"
    image_size: int = 224
    freeze_ratio: float = 0.60

    epochs: int = 15
    patience: int = 5
    learning_rate: float = 2e-5
    head_lr_multiplier: float = 10.0
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    resnet_batch_size: int = 32
    swin_batch_size: int = 16
    num_workers: int = 0
    minority_threshold: float = 0.15

    deterministic: bool = True
    device: str = "auto"
    print_each_epoch: bool = True
    save_checkpoints: bool = True
    output_dir: str = "reproducibility_results"


CONFIG = Config()

LABELS = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non Demented",
    3: "Very Mild Demented",
}
CLASS_NAMES = tuple(LABELS.values())
MODEL_NAMES = {"resnet50": "ResNet-50", "swin": "Swin"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REPORT_METRICS = {
    "Accuracy": "accuracy",
    "Balanced Accuracy": "balanced_accuracy",
    "Macro F1": "f1_macro",
    "ROC-AUC": "roc_auc_ovr",
}


class MRIData(Dataset):
    def __init__(
        self,
        data,
        indices,
        config,
        model_name,
        processor=None,
        class_counts=None,
        augment=False,
    ):
        self.data = data
        self.indices = indices
        self.config = config
        self.model_name = model_name
        self.processor = processor
        self.augment = augment
        self.normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)

        self.minority_classes = set()
        if class_counts:
            total = sum(class_counts.values())
            self.minority_classes = {
                label
                for label, count in class_counts.items()
                if count / total < config.minority_threshold
            }

        self.minority_transform = mt.Compose(
            [
                mt.RandFlip(prob=0.5, spatial_axis=1),
                mt.RandRotate(prob=0.8, range_x=0.35),
                mt.RandZoom(prob=0.5, min_zoom=0.85, max_zoom=1.15),
                mt.RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
                mt.RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
                mt.RandGaussianSmooth(prob=0.2, sigma_x=(0.25, 0.75)),
                mt.RandShiftIntensity(prob=0.3, offsets=0.1),
            ]
        )
        self.standard_transform = mt.Compose(
            [
                mt.RandFlip(prob=0.5, spatial_axis=1),
                mt.RandRotate(prob=0.5, range_x=0.26),
                mt.RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.1),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def set_random_state(self, seed):
        self.minority_transform.set_random_state(seed=seed)
        self.standard_transform.set_random_state(seed=seed + 1)

    def __getitem__(self, position):
        index = int(self.indices[position])
        image = self.data[index]["image"].convert("L")
        image = np.asarray(image, dtype=np.float32) / 255.0
        label = int(self.data[index]["label"])
        image = torch.from_numpy(image).unsqueeze(0)

        if self.augment:
            transform = (
                self.minority_transform
                if label in self.minority_classes
                else self.standard_transform
            )
            image = transform(image)

        image = torch.clamp(image, 0.0, 1.0)
        if self.model_name == "resnet50":
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            inputs = self.normalize(image.repeat(3, 1, 1))
        else:
            image = image.squeeze(0).numpy()
            image = np.stack([image * 255] * 3, axis=-1).astype(np.uint8)
            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)

        return inputs, torch.tensor(label, dtype=torch.long)


def choose_device(config):
    if config.device != "auto":
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    worker = torch.utils.data.get_worker_info()
    if worker:
        worker.dataset.set_random_state(seed)


def load_data(config):
    data = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        revision=config.dataset_revision,
    )
    labels = np.asarray([int(example["label"]) for example in data])
    train_indices, validation_indices = train_test_split(
        np.arange(len(data)),
        test_size=config.validation_fraction,
        stratify=labels,
        random_state=config.split_seed,
    )
    return data, labels, train_indices, validation_indices


def build_model(config, name, device):
    if name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Linear(model.fc.in_features, len(LABELS))
        processor, head = None, model.fc
    elif name == "swin":
        processor = AutoImageProcessor.from_pretrained(
            config.swin_name,
            revision=config.swin_revision,
            size={"height": config.image_size, "width": config.image_size},
            use_fast=False,
        )
        model = SwinForImageClassification.from_pretrained(
            config.swin_name,
            revision=config.swin_revision,
            num_labels=len(LABELS),
            ignore_mismatched_sizes=True,
            id2label=LABELS,
            label2id={text: label for label, text in LABELS.items()},
        )
        head = model.classifier
    else:
        raise ValueError(f"Unknown model: {name}")

    parameters = list(model.parameters())
    freeze_until = int(len(parameters) * config.freeze_ratio)
    for position, parameter in enumerate(parameters):
        parameter.requires_grad = position >= freeze_until
    for parameter in head.parameters():
        parameter.requires_grad = True
    return model.to(device), processor


def make_loaders(
    config,
    name,
    seed,
    data,
    labels,
    train_indices,
    validation_indices,
    processor,
    device,
):
    train_labels = labels[train_indices]
    counts = Counter(int(label) for label in train_labels)
    train_data = MRIData(
        data,
        train_indices,
        config,
        name,
        processor,
        counts,
        augment=True,
    )
    validation_data = MRIData(
        data,
        validation_indices,
        config,
        name,
        processor,
    )
    train_data.set_random_state(seed)

    sampler = WeightedRandomSampler(
        [1.0 / counts[int(label)] for label in train_labels],
        num_samples=len(train_labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    batch_size = (
        config.resnet_batch_size
        if name == "resnet50"
        else config.swin_batch_size
    )
    common = {
        "batch_size": batch_size,
        "pin_memory": device.type == "cuda",
        "num_workers": config.num_workers,
        "worker_init_fn": seed_worker if config.num_workers else None,
    }
    train_loader = DataLoader(
        train_data,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed + 1),
        **common,
    )
    validation_loader = DataLoader(
        validation_data,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 2),
        **common,
    )
    weights = torch.tensor(
        [
            len(train_labels) / (len(LABELS) * counts[label])
            for label in LABELS
        ],
        dtype=torch.float32,
        device=device,
    )
    return train_loader, validation_loader, weights


def forward(model, name, inputs):
    output = model(inputs)
    return output if name == "resnet50" else output.logits


def make_optimizer(config, model, name):
    head_name = "fc" if name == "resnet50" else "classifier"
    head = getattr(model, head_name)
    backbone = [
        parameter
        for parameter_name, parameter in model.named_parameters()
        if not parameter_name.startswith(f"{head_name}.")
        and parameter.requires_grad
    ]
    return AdamW(
        [
            {"params": backbone, "lr": config.learning_rate},
            {
                "params": head.parameters(),
                "lr": config.learning_rate * config.head_lr_multiplier,
            },
        ],
        weight_decay=config.weight_decay,
    )


def train(config, model, name, train_loader, validation_loader, weights):
    device = next(model.parameters()).device
    optimizer = make_optimizer(config, model, name)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    history, best_state = [], {}
    best_accuracy, best_epoch, stale_epochs = -1.0, 0, 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = train_correct = train_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = forward(model, name, inputs)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
            )
            optimizer.step()
            train_loss += loss.item() * len(targets)
            train_correct += (output.argmax(1) == targets).sum().item()
            train_count += len(targets)

        model.eval()
        validation_loss = validation_correct = validation_count = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                output = forward(model, name, inputs)
                loss = criterion(output, targets)
                validation_loss += loss.item() * len(targets)
                validation_correct += (
                    (output.argmax(1) == targets).sum().item()
                )
                validation_count += len(targets)

        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            "train_accuracy": train_correct / train_count,
            "validation_loss": validation_loss / validation_count,
            "validation_accuracy": validation_correct / validation_count,
        }
        history.append(row)
        scheduler.step(row["validation_loss"])

        improved = row["validation_accuracy"] > best_accuracy
        if improved:
            best_accuracy = row["validation_accuracy"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if config.print_each_epoch:
            marker = " best" if improved else ""
            print(
                f"  epoch {epoch:02d}/{config.epochs}  "
                f"train={row['train_accuracy']:.4f}  "
                f"val={row['validation_accuracy']:.4f}{marker}"
            )
        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    return history, best_state, best_epoch


def predict(model, name, loader):
    device = next(model.parameters()).device
    predictions, targets, probabilities = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, batch_targets in loader:
            output = forward(model, name, inputs.to(device))
            predictions.append(output.argmax(1).cpu().numpy())
            targets.append(batch_targets.numpy())
            probabilities.append(F.softmax(output, dim=1).cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(probabilities),
    )


def evaluate(targets, predictions, probabilities):
    labels = np.arange(len(CLASS_NAMES))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "f1_macro": float(f1_score(targets, predictions, average="macro")),
        "f1_weighted": float(
            f1_score(targets, predictions, average="weighted")
        ),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "kappa": float(cohen_kappa_score(targets, predictions)),
    }
    specificities = []
    for label, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[label, label]
        false_negative = matrix[label, :].sum() - true_positive
        false_positive = matrix[:, label].sum() - true_positive
        true_negative = (
            matrix.sum()
            - true_positive
            - false_negative
            - false_positive
        )
        denominator = true_negative + false_positive
        specificity = true_negative / denominator if denominator else 0.0
        specificities.append(specificity)
        prefix = f"per_class.{class_name}"
        metrics[f"{prefix}.precision"] = float(precision[label])
        metrics[f"{prefix}.recall"] = float(recall[label])
        metrics[f"{prefix}.specificity"] = float(specificity)
        metrics[f"{prefix}.f1"] = float(f1[label])
        metrics[f"{prefix}.support"] = int(support[label])

    metrics["specificity_macro"] = float(np.mean(specificities))
    try:
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(
                targets,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        metrics["roc_auc_ovr"] = None
    return metrics


def run_once(
    config,
    name,
    seed,
    data,
    labels,
    train_indices,
    validation_indices,
    device,
    output_dir,
):
    set_seed(seed, config.deterministic)
    model, processor = build_model(config, name, device)
    train_loader, validation_loader, weights = make_loaders(
        config,
        name,
        seed,
        data,
        labels,
        train_indices,
        validation_indices,
        processor,
        device,
    )
    history, state, best_epoch = train(
        config,
        model,
        name,
        train_loader,
        validation_loader,
        weights,
    )
    predictions, targets, probabilities = predict(
        model,
        name,
        validation_loader,
    )
    metrics = evaluate(targets, predictions, probabilities)
    result = {
        "model": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        **metrics,
    }

    if config.save_checkpoints:
        torch.save(
            {
                "model_state_dict": state,
                "config": asdict(config),
                "result": result,
                "history": history,
                "validation_indices": validation_indices,
                "targets": targets,
                "predictions": predictions,
                "probabilities": probabilities,
            },
            output_dir / f"{name}_seed_{seed}.pt",
        )
    del model, train_loader, validation_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def save_context(config, device, data, labels, train_indices, validation_indices):
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit = None

    save_json(output_dir / "config.json", asdict(config))
    save_json(
        output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "device": str(device),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "git_commit": git_commit,
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "dataset_fingerprint": getattr(data, "_fingerprint", None),
            "class_counts": {
                LABELS[label]: int((labels == label).sum())
                for label in LABELS
            },
            "packages": {
                name: package_version(name)
                for name in (
                    "datasets",
                    "monai",
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "torch",
                    "torchvision",
                    "transformers",
                )
            },
        },
    )
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train=train_indices,
        validation=validation_indices,
    )
    return output_dir


def save_summary(results, config):
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(results)
    results.to_csv(output_dir / "results_by_seed.csv", index=False)

    metric_names = list(REPORT_METRICS.values())
    numeric = results[metric_names].apply(pd.to_numeric, errors="coerce")
    numeric["model"] = results["model"]
    summary = numeric.groupby("model")[metric_names].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]
    summary = summary.reindex(config.models)
    summary.to_csv(output_dir / "summary.csv")

    table = {}
    for model in config.models:
        table[MODEL_NAMES[model]] = {}
        for heading, metric in REPORT_METRICS.items():
            mean = summary.loc[model, f"{metric}_mean"]
            std = summary.loc[model, f"{metric}_std"]
            table[MODEL_NAMES[model]][heading] = (
                f"{mean:.4f} ± {std:.4f}"
                if pd.notna(std)
                else f"{mean:.4f}"
            )
    print("\nFinal validation results across seeds")
    print(pd.DataFrame.from_dict(table, orient="index").to_string())


def main():
    config = CONFIG
    unknown_models = set(config.models) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")
    if not config.seeds:
        raise ValueError("Config.seeds must contain at least one seed")

    device = choose_device(config)
    data, labels, train_indices, validation_indices = load_data(config)
    output_dir = save_context(
        config,
        device,
        data,
        labels,
        train_indices,
        validation_indices,
    )
    total_runs = len(config.models) * len(config.seeds)
    print(
        f"Device: {device} | Runs: {total_runs} | "
        f"Split seed: {config.split_seed}"
    )

    results = []
    for run_number, (name, seed) in enumerate(
        (
            (name, seed)
            for name in config.models
            for seed in config.seeds
        ),
        start=1,
    ):
        print(
            f"\n[{run_number}/{total_runs}] "
            f"{MODEL_NAMES[name]} | seed {seed}"
        )
        results.append(
            run_once(
                config,
                name,
                seed,
                data,
                labels,
                train_indices,
                validation_indices,
                device,
                output_dir,
            )
        )
        pd.DataFrame(results).to_csv(
            output_dir / "results_by_seed.csv",
            index=False,
        )

    save_summary(results, config)
    print(f"\nSaved detailed results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
