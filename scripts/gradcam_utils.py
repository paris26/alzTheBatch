"""Shared Grad-CAM utilities for the ResNet-50 and Swin notebooks.

The implementation is intentionally dependency-light: it uses the PyTorch and
Matplotlib packages that the notebooks already require instead of adding
``pytorch-grad-cam``. Both architectures therefore use exactly the same CAM
normalization, Matplotlib colormap, overlay alpha, and colorbar scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


TensorTransform = Callable[[torch.Tensor], torch.Tensor]
LogitsGetter = Callable[[Any], torch.Tensor]


@dataclass(frozen=True)
class GradCAMResult:
    """Result of one class-targeted Grad-CAM calculation."""

    cam: np.ndarray
    predicted_class: int
    target_class: int
    confidence: float
    probabilities: np.ndarray


def _first_tensor(value: Any) -> torch.Tensor:
    """Return the first tensor from the common module-output containers."""

    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    raise TypeError(
        "The Grad-CAM target layer must return a tensor or a tuple/list "
        "containing a tensor."
    )


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Convert Swin token activations from ``B x tokens x C`` to ``B x C x H x W``.

    Hugging Face Swin has no class token, so the final stage normally contains
    49 tokens (7 x 7) for a 224 x 224 input. The one-token fallback also makes
    the helper safe for closely related transformer backbones that do use a
    class token.
    """

    if tensor.ndim == 4:
        # Some transformer implementations expose B x H x W x C tensors.
        if tensor.shape[-1] > tensor.shape[1]:
            return tensor.permute(0, 3, 1, 2).contiguous()
        return tensor
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected 3D Swin tokens or a 4D feature map, got {tensor.shape}."
        )

    batch_size, token_count, channels = tensor.shape
    side = math.isqrt(token_count)
    if side * side != token_count:
        side = math.isqrt(token_count - 1)
        if side * side != token_count - 1:
            raise ValueError(
                f"Cannot reshape {token_count} tokens into a square feature map."
            )
        tensor = tensor[:, 1:, :]

    return (
        tensor.transpose(1, 2)
        .reshape(batch_size, channels, side, side)
        .contiguous()
    )


class GradCAM:
    """Hook-based Grad-CAM that supports CNN feature maps and Swin tokens."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        *,
        reshape_transform: TensorTransform | None = None,
        logits_getter: LogitsGetter | None = None,
    ) -> None:
        self.model = model
        self.reshape_transform = reshape_transform
        self.logits_getter = logits_getter or (lambda output: output)
        self._activation: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None
        self._hook = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        activation = _first_tensor(output)
        self._activation = activation
        if not activation.requires_grad:
            raise RuntimeError(
                "The target activation does not track gradients. Ensure the "
                "input tensor has requires_grad=True."
            )
        activation.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self._gradient = gradient

    def close(self) -> None:
        """Remove the forward hook."""

        self._hook.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> GradCAMResult:
        """Calculate a normalized CAM for one image.

        If ``target_class`` is omitted, the model's predicted class is used.
        The returned CAM is always normalized to the shared colorbar range
        ``[0, 1]``.
        """

        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                "GradCAM expects one batched image with shape 1 x C x H x W."
            )

        self._activation = None
        self._gradient = None
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Requiring input gradients keeps the graph available even if every
        # parameter before the selected target layer is frozen.
        input_tensor = input_tensor.detach().requires_grad_(True)
        output = self.model(input_tensor)
        logits = self.logits_getter(output)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(
                f"Expected classifier logits shaped 1 x classes, got {logits.shape}."
            )

        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(logits.argmax(dim=1).item())
        selected_class = (
            predicted_class if target_class is None else int(target_class)
        )
        if not 0 <= selected_class < logits.shape[1]:
            raise ValueError(
                f"Target class {selected_class} is outside the classifier range."
            )

        logits[0, selected_class].backward()
        if self._activation is None or self._gradient is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture both activations and gradients."
            )

        activation = self._activation.detach()
        gradient = self._gradient.detach()
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
            gradient = self.reshape_transform(gradient)
        if activation.ndim != 4 or gradient.shape != activation.shape:
            raise ValueError(
                "Grad-CAM requires matching B x C x H x W activations and "
                f"gradients, got {activation.shape} and {gradient.shape}."
            )

        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min).clamp_min(1e-8)

        return GradCAMResult(
            cam=cam.cpu().numpy(),
            predicted_class=predicted_class,
            target_class=selected_class,
            confidence=float(probabilities[0, predicted_class].detach().cpu()),
            probabilities=probabilities[0].detach().cpu().numpy(),
        )


def overlay_cam(
    image: np.ndarray,
    cam: np.ndarray,
    *,
    cmap: str = "jet",
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a normalized CAM with a grayscale or RGB image."""

    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected H x W or H x W x 3 image, got {image.shape}.")
    if image.shape[:2] != cam.shape:
        image = (
            F.interpolate(
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .unsqueeze(0),
                size=cam.shape,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )

    image_min, image_max = float(image.min()), float(image.max())
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image)
    heatmap = plt.get_cmap(cmap)(np.clip(cam, 0.0, 1.0))[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heatmap, 0.0, 1.0)


def first_sample_positions_per_class(
    dataset: Any,
    indices: Sequence[int],
    number_of_classes: int,
) -> list[tuple[int, int, int]]:
    """Return ``(validation_position, dataset_index, class_id)`` per class."""

    samples: dict[int, tuple[int, int, int]] = {}
    for position, dataset_index in enumerate(indices):
        dataset_index = int(dataset_index)
        class_id = int(dataset[dataset_index]["label"])
        samples.setdefault(class_id, (position, dataset_index, class_id))
        if len(samples) == number_of_classes:
            break

    missing = sorted(set(range(number_of_classes)) - set(samples))
    if missing:
        raise ValueError(f"No validation sample found for classes {missing}.")
    return [samples[class_id] for class_id in range(number_of_classes)]


def plot_gradcam_rows(
    images: Sequence[np.ndarray],
    results: Sequence[GradCAMResult],
    true_classes: Sequence[int],
    class_names: Sequence[str],
    *,
    model_name: str,
    cmap: str = "jet",
    alpha: float = 0.45,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot original, CAM, and overlay columns with one shared ``[0,1]`` bar."""

    if not (len(images) == len(results) == len(true_classes)):
        raise ValueError("Images, Grad-CAM results, and labels must have equal length.")

    row_count = len(images)
    figure, axes = plt.subplots(
        row_count,
        3,
        figsize=(11, 3.2 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    normalization = Normalize(vmin=0.0, vmax=1.0)

    for row, (image, result, true_class) in enumerate(
        zip(images, results, true_classes)
    ):
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"Original\nTrue: {class_names[true_class]}")
        axes[row, 1].imshow(
            result.cam,
            cmap=cmap,
            norm=normalization,
            interpolation="bilinear",
        )
        axes[row, 1].set_title(
            f"Grad-CAM\nTarget: {class_names[result.target_class]}"
        )
        axes[row, 2].imshow(
            overlay_cam(image, result.cam, cmap=cmap, alpha=alpha)
        )
        axes[row, 2].set_title(
            f"Overlay\nPred: {class_names[result.predicted_class]} "
            f"({result.confidence:.1%})"
        )
        for axis in axes[row]:
            axis.axis("off")

    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=cmap),
        ax=axes[:, 1:].ravel().tolist(),
        label="Normalized Grad-CAM intensity",
        shrink=0.82,
        pad=0.02,
    )
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    figure.suptitle(
        f"{model_name} Grad-CAM — shared {cmap!r} scale [0, 1]",
        fontsize=14,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return figure, axes
