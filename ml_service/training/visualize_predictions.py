"""Write a reproducible INPUT | TARGET | PREDICTION contact sheet."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_prediction_sheet(examples, output_path, *, tile_size=256):
    """Examples contain CPU image/target/probability tensors and metadata."""
    if not examples:
        raise ValueError("prediction sheet requires at least one example")
    row_height = tile_size + 32
    sheet = Image.new("RGB", (tile_size * 3, row_height * len(examples) + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(("INPUT", "TARGET", "PREDICTION (probability)")):
        draw.text((col * tile_size + 8, 8), title, fill="black")
    for row, example in enumerate(examples):
        image = example["image"].detach().cpu().permute(1, 2, 0).numpy()
        target = example["target"].detach().cpu().squeeze(0).numpy()
        probability = example["probability"].detach().cpu().squeeze(0).numpy()
        metadata = example["metadata"]
        label = str(metadata.get("sample", metadata.get("source", row)))
        clean = " CLEAN" if not metadata.get("defects", []) else ""
        label += f"{clean}  mean p={probability.mean():.3f}  >0.5={100 * (probability > .5).mean():.1f}%"
        y = 28 + row * row_height
        draw.text((8, y + 8), label, fill="black")
        for col, array in enumerate((image, target, probability)):
            tile = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255 + .5)).convert("RGB")
            resampling = Image.Resampling.NEAREST if col == 1 else Image.Resampling.BILINEAR
            tile = tile.resize((tile_size, tile_size), resampling)
            sheet.paste(tile, (col * tile_size, y + 32))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main():
    # Evaluation owns checkpoint/dataset loading; this entry point reuses it.
    from .evaluate import main as evaluate_main

    evaluate_main()


if __name__ == "__main__":
    main()
