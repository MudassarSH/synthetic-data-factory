"""CLI for reproducible sample export and visual contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from .dataset import DocumentDefectDataset, find_images, load_clean_image, require_split_leaf


def _save_rgb(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.uint8(np.clip(image * 255 + 0.5, 0, 255)), "RGB").save(path)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.uint8(np.clip(mask * 255 + 0.5, 0, 255)), "L").save(path)


def _validate_output(destination: Path, replace: bool) -> None:
    if destination.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if destination.exists() and not destination.is_dir():
        raise ValueError("output destination must be a directory")
    if destination.exists() and any(destination.iterdir()):
        if not replace:
            raise FileExistsError("output directory is not empty; use --replace for a managed export")
        manifest_path = destination / "export_manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("replacement requires an export_manifest.json from this exporter")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("format") != "document-degradation-export" or manifest.get("version") != 1:
            raise ValueError("unrecognized export manifest")
        for entry in destination.iterdir():
            if entry.is_symlink():
                raise ValueError("replacement refuses symlink entries")
            if entry.name == "export_manifest.json":
                continue
            if re.fullmatch(r"sample_\d{5,}", entry.name) and entry.is_dir():
                if any(child.is_symlink() or not child.is_file() or child.name not in {"input.png", "mask.png", "metadata.json"} for child in entry.iterdir()):
                    raise ValueError("replacement refuses unrelated sample contents")
            elif not (re.fullmatch(r"preview_grid_\d{2,}\.png", entry.name) and entry.is_file()):
                raise ValueError("replacement refuses unrelated output contents")


def export_samples(clean_dir: str | Path, output_dir: str | Path, *, count=500, seed=0,
                   size=(512, 512), grid=True, replace=False, defect_weights=None,
                   mask_mode="max") -> None:
    """Publish a complete export; replacing requires a previously managed export."""
    require_split_leaf(clean_dir)
    source_root = Path(clean_dir).resolve()
    destination = Path(output_dir)
    if destination.is_symlink():
        raise ValueError("output directory must not be a symlink")
    destination = destination.resolve()
    if source_root == destination or source_root in destination.parents or destination in source_root.parents:
        raise ValueError("source and output directories must be separate")
    paths = find_images(clean_dir)
    dataset = DocumentDefectDataset(paths, size=size, seed=seed, samples_per_epoch=count,
                                   dynamic=False, source_root=clean_dir,
                                   defect_weights=defect_weights, mask_mode=mask_mode)
    _validate_output(destination, replace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Generate first: a failed render leaves the previous validation set intact.
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "export"
        _write_export(dataset, paths, stage, count=count, seed=seed, size=size, grid=grid)
        _validate_output(destination, replace)
        backup = destination.parent / f".{destination.name}-backup-{uuid4().hex}"
        existed = destination.exists()
        if existed:
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if existed:
                backup.rename(destination)
            raise
        if existed:
            shutil.rmtree(backup)


def _write_export(dataset, paths, output_dir, *, count, seed, size, grid):
    output_dir = Path(output_dir)
    output_dir.mkdir()
    records = []
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
        records.append({"directory": sample_dir.name, "source": metadata["source"],
                        "seed": metadata["seed"], "variant": metadata["variant"]})
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
    manifest = {
        "format": "document-degradation-export", "version": 1,
        "sample_count": count, "base_seed": int(seed), "size": list(size),
        "grid": grid, "mask_mode": dataset.mask_mode,
        "defect_weights": dataset.defect_weights, "geometry": "synthesize-page-then-pad",
        "runtime_versions": {"numpy": np.__version__, "pillow": pillow_version},
        "sources": [{"path": path.resolve().relative_to(dataset.source_root).as_posix(),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "samples": records,
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean_dir", help="directory containing clean document images")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512, help="square output size")
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--replace", action="store_true", help="replace a managed export after generating its replacement")
    parser.add_argument("--mask-mode", choices=("max", "attenuation"), default="max")
    parser.add_argument("--defect-weights", type=json.loads, help='JSON weights, e.g. {"haze": 0.2, "ring": 2}')
    args = parser.parse_args()
    if args.count <= 0 or args.size <= 0:
        parser.error("--count and --size must be positive")
    try:
        export_samples(args.clean_dir, args.output_dir, count=args.count, seed=args.seed, size=(args.size, args.size), grid=not args.no_grid,
                       replace=args.replace, mask_mode=args.mask_mode, defect_weights=args.defect_weights)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
