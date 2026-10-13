from __future__ import annotations

import numpy as np

from .common import darken, finish, grid, noise_field


def generate(image: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    y, x = grid((h, w))
    widths = rng.uniform(0.035, 0.2, 4)
    weights = rng.uniform(0, 1, 4)
    weights[int(rng.integers(0, 4))] = 1.0
    distances = [x / w, (w - 1 - x) / w, y / h, (h - 1 - y) / h]
    field = np.zeros((h, w), np.float32)
    for distance, width, weight in zip(distances, widths, weights):
        field = np.maximum(field, weight * np.exp(-((distance / width) ** 1.7)))
    irregular = noise_field(rng, (h, w), (8, 8))
    strength = float(rng.uniform(0.2, 0.65))
    mask = np.clip(field * (0.7 + 0.55 * irregular) * strength, 0, 1)
    return finish(darken(image, mask), mask, "edge_shadow", widths=widths.tolist(), weights=weights.tolist(), strength=strength)
