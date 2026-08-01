#!/usr/bin/env python3
"""Generate a same-sample Grad-CAM comparison for ResNet-50 and Swin.

Run this after ``run_reproducible_experiments.py`` has produced checkpoints:

    python scripts/generate_gradcam_comparison.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gradcam_utils import (
    GradCAM,
    first_sample_positions_per_class,
    overlay_cam,
    swin_reshape_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("reproducibility_results"),
        help="Directory containing resnet50_seed_N.pt and swin_seed_N.pt.",
    )
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        help="Explicit ResNet checkpoint (overrides --results-dir and --seed).",
    )
    parser.add_argument(
        "--swin-checkpoint",
        type=Path,
        help="Explicit Swin checkpoint (overrides --results-dir and --seed).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target",
        choices=("predicted", "true"),
        default="predicted",
        help="Class score used to generate each heatmap.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/gradcam_resnet50_vs_swin.png"),
    )
    parser.add_argument(
        "--dataset-arrow",
        type=Path,
        help=(
            "Optional cached Hugging Face Arrow file. This bypasses cache "
            "locking and network access."
        ),
    )
    parser.add_argument(
        "--swin-base-dir",
        type=Path,
        help=(
            "Optional local Hugging Face Swin snapshot containing config.json, "
            "preprocessor_config.json, and the base weights."
        ),
    )
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def checkpoint_path(results_dir: Path, model_name: str, seed: int) -> Path:
    path = results_dir / f"{model_name}_seed_{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run run_reproducible_experiments.py first."
        )
    return path


def resolve_checkpoint_paths(args: argparse.Namespace) -> dict[str, Path]:
    explicit = {
        "resnet50": args.resnet_checkpoint,
        "swin": args.swin_checkpoint,
    }
    if any(path is not None for path in explicit.values()):
        if not all(path is not None for path in explicit.values()):
            raise ValueError(
                "Pass both --resnet-checkpoint and --swin-checkpoint together."
            )
        missing = [str(path) for path in explicit.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint(s): {', '.join(missing)}")
        return explicit

    return {
        name: checkpoint_path(args.results_dir, name, args.seed)
        for name in ("resnet50", "swin")
    }


def main() -> None:
    args = parse_args()

    # Importing here keeps --help usable in lightweight environments.
    from run_reproducible_experiments import (
        CLASS_NAMES,
        CONFIG,
        LABELS,
        MRIData,
        build_model,
        choose_device,
        load_data,
    )

    runtime_config = CONFIG
    if args.swin_base_dir is not None:
        if not args.swin_base_dir.is_dir():
            raise FileNotFoundError(
                f"Missing Swin snapshot directory: {args.swin_base_dir}"
            )
        runtime_config = replace(
            CONFIG,
            swin_name=str(args.swin_base_dir),
            swin_revision=None,
        )

    device = choose_device(runtime_config)
    if args.dataset_arrow is None:
        data, _labels, _train_indices, default_validation_indices = load_data(
            runtime_config
        )
    else:
        if not args.dataset_arrow.is_file():
            raise FileNotFoundError(f"Missing dataset file: {args.dataset_arrow}")
        from datasets import Dataset
        from sklearn.model_selection import train_test_split

        data = Dataset.from_file(str(args.dataset_arrow))
        _labels = np.asarray([int(example["label"]) for example in data])
        _train_indices, default_validation_indices = train_test_split(
            np.arange(len(data)),
            test_size=runtime_config.validation_fraction,
            stratify=_labels,
            random_state=runtime_config.split_seed,
        )
    checkpoint_paths = resolve_checkpoint_paths(args)
    checkpoints = {
        name: torch.load(
            checkpoint_paths[name],
            map_location="cpu",
            weights_only=False,
        )
        for name in ("resnet50", "swin")
    }
    validation_indices = np.asarray(
        checkpoints["resnet50"].get(
            "validation_indices", default_validation_indices
        )
    )
    swin_validation_indices = np.asarray(
        checkpoints["swin"].get("validation_indices", default_validation_indices)
    )
    if not np.array_equal(validation_indices, swin_validation_indices):
        raise ValueError(
            "The two checkpoints use different validation splits; a "
            "same-sample Grad-CAM comparison would be misleading."
        )

    sample_specs = first_sample_positions_per_class(
        data,
        validation_indices,
        len(LABELS),
    )
    original_images = [
        np.asarray(data[dataset_index]["image"].convert("L"), dtype=np.float32)
        / 255.0
        for _position, dataset_index, _class_id in sample_specs
    ]

    models: dict[str, torch.nn.Module] = {}
    processors = {}
    for model_name in ("resnet50", "swin"):
        model, processor = build_model(runtime_config, model_name, device)
        model.load_state_dict(checkpoints[model_name]["model_state_dict"])
        model.eval()
        models[model_name] = model
        processors[model_name] = processor

    validation_datasets = {
        model_name: MRIData(
            data,
            validation_indices,
            runtime_config,
            model_name,
            processors[model_name],
        )
        for model_name in ("resnet50", "swin")
    }
    gradcams = {
        "resnet50": GradCAM(
            models["resnet50"],
            models["resnet50"].layer4[-1],
        ),
        "swin": GradCAM(
            models["swin"],
            models["swin"].swin.encoder.layers[-1].blocks[-1].layernorm_before,
            reshape_transform=swin_reshape_transform,
            logits_getter=lambda output: output.logits,
        ),
    }

    try:
        results = {"resnet50": [], "swin": []}
        for model_name in ("resnet50", "swin"):
            for validation_position, _dataset_index, true_class in sample_specs:
                input_tensor, _label = validation_datasets[model_name][
                    validation_position
                ]
                target_class = true_class if args.target == "true" else None
                results[model_name].append(
                    gradcams[model_name](
                        input_tensor.unsqueeze(0).to(device),
                        target_class=target_class,
                    )
                )
    finally:
        for gradcam in gradcams.values():
            gradcam.close()

    figure, axes = plt.subplots(
        len(sample_specs),
        3,
        figsize=(12.5, 3.2 * len(sample_specs)),
        squeeze=False,
        constrained_layout=True,
    )
    normalization = Normalize(vmin=0.0, vmax=1.0)
    display_names = {"resnet50": "ResNet-50", "swin": "Swin"}
    short_class_names = [
        (
            "Non-demented"
            if class_name == "Non Demented"
            else class_name.removesuffix(" Demented")
        )
        for class_name in CLASS_NAMES
    ]

    for row, (
        (_validation_position, _dataset_index, true_class),
        original_image,
    ) in enumerate(zip(sample_specs, original_images)):
        axes[row, 0].imshow(original_image, cmap="gray")
        axes[row, 0].set_title(
            f"Original\nTrue: {short_class_names[true_class]}"
        )
        for column, model_name in enumerate(("resnet50", "swin"), start=1):
            result = results[model_name][row]
            axes[row, column].imshow(
                overlay_cam(
                    original_image,
                    result.cam,
                    cmap=args.cmap,
                    alpha=args.alpha,
                )
            )
            axes[row, column].set_title(
                f"{display_names[model_name]}\n"
                f"Pred: {short_class_names[result.predicted_class]} "
                f"({result.confidence:.1%})\n"
                f"Target: {short_class_names[result.target_class]}"
            )
        for axis in axes[row]:
            axis.axis("off")

    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=args.cmap),
        ax=axes[:, 1:].ravel().tolist(),
        label="Normalized Grad-CAM intensity",
        shrink=0.82,
        pad=0.02,
    )
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    figure.suptitle(
        f"ResNet-50 vs Swin Grad-CAM — same samples, "
        f"{args.cmap!r} scale [0, 1]",
        fontsize=14,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
