#!/usr/bin/env python3
"""Create a paper-ready one-sample-per-class figure from the executed notebook."""

import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "VIT-Swin.ipynb"
OUTPUT_DIR = ROOT / "paper" / "figures"
SOURCE_FIGURE = OUTPUT_DIR / "dataset_class_samples_with_predictions.png"
PAPER_PNG = OUTPUT_DIR / "dataset_class_samples.png"
PAPER_PDF = OUTPUT_DIR / "dataset_class_samples.pdf"

CLASSES = (
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented",
)

# Bounds of the four unaugmented MRI panels in the notebook's saved output.
CROPS = (
    (63, 57, 389, 381),
    (459, 57, 785, 381),
    (856, 57, 1182, 381),
    (1252, 57, 1578, 381),
)


def extract_notebook_figure() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    marker = "visualize_class_samples(model, dataset, val_idx, LABELS)"

    for cell in notebook["cells"]:
        if marker not in "".join(cell.get("source", [])):
            continue
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png:
                encoded = "".join(png) if isinstance(png, list) else png
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                SOURCE_FIGURE.write_bytes(base64.b64decode(encoded))
                return

    raise RuntimeError("The executed class-sample PNG was not found in the notebook.")


def make_paper_figure() -> None:
    source = Image.open(SOURCE_FIGURE).convert("RGB")
    samples = [source.crop(bounds) for bounds in CROPS]

    width, height = 2400, 1800
    margin, gap = 120, 90
    label_height = 115
    panel_width = (width - 2 * margin - gap) // 2
    panel_height = (height - 2 * margin - gap) // 2
    image_side = min(panel_width, panel_height - label_height)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf", 48)

    for index, (sample, class_name) in enumerate(zip(samples, CLASSES)):
        row, column = divmod(index, 2)
        panel_x = margin + column * (panel_width + gap)
        panel_y = margin + row * (panel_height + gap)
        image_x = panel_x + (panel_width - image_side) // 2

        resized = sample.resize((image_side, image_side), Image.Resampling.LANCZOS)
        canvas.paste(resized, (image_x, panel_y))

        label = f"({chr(97 + index)}) {class_name}"
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        text_x = panel_x + (panel_width - text_width) // 2
        text_y = panel_y + image_side + 28
        draw.text((text_x, text_y), label, fill="black", font=font)

    canvas.save(PAPER_PNG, dpi=(300, 300), optimize=True)
    canvas.save(PAPER_PDF, "PDF", resolution=300)


def main() -> None:
    extract_notebook_figure()
    make_paper_figure()
    print(PAPER_PNG)
    print(PAPER_PDF)


if __name__ == "__main__":
    main()
