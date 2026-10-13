from __future__ import annotations

import numpy as np

from .common import blur, darken, finish, grid


def generate(image: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    y, x = grid((h, w))
    vertical = bool(rng.integers(0, 2))
    across, along = (x, y) if vertical else (y, x)
    cross_size, along_size = (w, h) if vertical else (h, w)
    mask = np.zeros((h, w), np.float32)
    count = int(rng.integers(2, 8))
    for _ in range(count):
        center = float(rng.uniform(0, cross_size))
        width = float(rng.uniform(0.002, 0.025) * cross_size)
        start = float(rng.uniform(-0.2, 0.5) * along_size)
        length = float(rng.uniform(0.35, 1.2) * along_size)
        line = np.exp(-0.5 * ((across - center) / max(1, width)) ** 2)
        taper = np.clip((along - start) / max(1, width * 8), 0, 1) * np.clip((start + length - along) / max(1, width * 8), 0, 1)
        mask = np.maximum(mask, line * taper * float(rng.uniform(0.2, 1)))
    strength = float(rng.uniform(0.2, 0.6))
    mask = blur(mask * strength, max(0.5, min(h, w) * 0.002))
    return finish(darken(image, mask), mask, "streak", orientation="vertical" if vertical else "horizontal", count=count, strength=strength)
