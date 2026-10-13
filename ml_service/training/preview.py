"""CLI for reproducible sample export and visual contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dataset import DocumentDefectDataset, find_images, load_clean_image, require_split_leaf


def _save_rgb(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.uint8(np.clip(image * 255 + 0.5, 0, 255)), "RGB").save(path)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.uint8(np.clip(mask * 255 + 0.5, 0, 255)), "L").save(path)


def export_samples(clean_dir: str | Path, output_dir: str | Path, *, count=500, seed=0, size=(512, 512), grid=True) -> None:
    require_split_leaf(clean_dir)
    paths = find_images(clean_dir)
    dataset = DocumentDefectDataset(paths, size=size, seed=seed, samples_per_epoch=count, dynamic=False, source_root=clean_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet = None
    tile_w = min(size[0], 256)
    tile_h = max(1, round(size[1] * tile_w / size[0]))
    per_sheet = 20
    for index in range(count):
        degraded, mask, metadata = dataset[index]
        sample_dir = output_dir / f"sample_{index:05d}"
        sample_dir.mkdir(exist_ok=True)
        _save_rgb(degraded, sample_dir / "input.png")
        _save_mask(mask, sample_dir / "mask.png")
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if grid and index < 200:
            if index % per_sheet == 0:
                sheet = Image.new("RGB", (2 * 3 * tile_w, 10 * (tile_h + 28)), "white")
                draw = ImageDraw.Draw(sheet)
            clean = load_clean_image(paths[index % len(paths)], size)
            row, col = divmod(index % per_sheet, 2)
            x0, y0 = col * 3 * tile_w, row * (tile_h + 28)
            for offset, array in enumerate((clean, degraded, np.repeat(mask[..., None], 3, axis=2))):
                tile = Image.fromarray(np.uint8(np.clip(array * 255 + 0.5, 0, 255)))
                sheet.paste(tile.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x0 + offset * tile_w, y0 + 28))
            draw.text((x0 + 8, y0 + 7), f"{index:05d}  clean | degraded | mask", fill="black")
            if index % per_sheet == per_sheet - 1 or index == min(count, 200) - 1:
                sheet.save(output_dir / f"preview_grid_{index // per_sheet:02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean_dir", help="directory containing clean document images")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512, help="square output size")
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.size <= 0:
        parser.error("--count and --size must be positive")
    try:
        export_samples(args.clean_dir, args.output_dir, count=args.count, seed=args.seed, size=(args.size, args.size), grid=not args.no_grid)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
