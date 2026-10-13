"""Shared array operations. Images are float RGB in [0, 1]."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def blur(field: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.uint8(np.clip(field, 0, 1) * 255), "L")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255


def noise_field(rng: np.random.Generator, shape: tuple[int, int], cells: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ch, cw = cells
    small = rng.random((max(2, ch), max(2, cw)), dtype=np.float32)
    image = Image.fromarray(np.uint8(small * 255), "L")
    return np.asarray(image.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32) / 255


def grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    return np.mgrid[0:h, 0:w].astype(np.float32)


def darken(image: np.ndarray, mask: np.ndarray, tint: tuple[float, float, float] = (1, 1, 1)) -> np.ndarray:
    """Attenuate reflected light using the exact opacity field returned as mask."""
    return np.clip(image * (1 - mask[..., None] * np.asarray(tint, dtype=np.float32)), 0, 1)


def finish(image: np.ndarray, mask: np.ndarray, kind: str, **params):
    return image.astype(np.float32), np.clip(mask, 0, 1).astype(np.float32), {"type": kind, **params}
