"""Clean-page loading and deterministic per-epoch sample generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

from .synthesis import synthesize

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_clean_image(path: str | Path, size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Apply EXIF orientation and contain the page on a white RGB canvas."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("size must be positive")
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255


def find_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"no document images found in {directory}")
    return paths


def require_split_leaf(directory: str | Path) -> None:
    """Reject a root that would recursively mix train and validation pages."""
    root = Path(directory)
    if (root / "train").is_dir() and (root / "validation").is_dir():
        raise ValueError(
            "Input appears to be a split root.\n"
            "Use either:\n"
            f"{root / 'train'}\n"
            "or\n"
            f"{root / 'validation'}"
        )


def fixed_sample_seed(base_seed: int, source_identity: str, variant: int) -> int:
    """Stable 64-bit seed for one page and one validation variant."""
    if base_seed < 0 or variant < 0:
        raise ValueError("base_seed and variant must be nonnegative")
    payload = f"{base_seed}\0{source_identity.replace(chr(92), '/')}\0{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class DocumentDefectDataset:
    """PyTorch-compatible dataset; call set_epoch before each training epoch.

    Dynamic seeds derive from (base seed, epoch, index). Fixed seeds derive
    from (base seed, source-relative path, variant) and survive insertion of
    additional pages into the sorted source list.
    """

    def __init__(self, clean_paths: Iterable[str | Path], *, size=(512, 512), seed=0, samples_per_epoch=None, dynamic=True, as_torch=False, source_root=None):
        self.paths = [Path(path) for path in clean_paths]
        if not self.paths:
            raise ValueError("clean_paths must not be empty")
        self.size = size
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        self.source_root = Path(source_root).resolve() if source_root is not None else None
        self.samples_per_epoch = len(self.paths) if samples_per_epoch is None else int(samples_per_epoch)
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.dynamic = dynamic
        self.as_torch = as_torch
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        path = self.paths[index % len(self.paths)]
        if self.source_root is not None:
            identity = path.resolve().relative_to(self.source_root).as_posix()
        else:
            identity = path.resolve().as_posix()
        variant = index // len(self.paths)
        if self.dynamic:
            sample_seed = int(np.random.SeedSequence([self.seed, self.epoch, index]).generate_state(1)[0])
        else:
            sample_seed = fixed_sample_seed(self.seed, identity, variant)
        rng = np.random.default_rng(sample_seed)
        if self.dynamic and len(self.paths) > 1:
            other_index = int(rng.integers(0, len(self.paths) - 1))
            if other_index >= index % len(self.paths):
                other_index += 1
            reverse_path = self.paths[other_index]
        else:
            # Fixed validation must remain identical if the source list grows.
            reverse_path = path
        clean = load_clean_image(path, self.size)
        reverse = load_clean_image(reverse_path, self.size)
        degraded, mask, metadata = synthesize(clean, sample_seed, source=identity, reverse=reverse)
        metadata["variant"] = variant
        if self.as_torch:
            import torch

            return torch.from_numpy(degraded.transpose(2, 0, 1).copy()), torch.from_numpy(mask[None].copy()), metadata
        return degraded, mask, metadata
