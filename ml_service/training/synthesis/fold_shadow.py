from __future__ import annotations

import numpy as np

from .common import darken, finish, grid, noise_field


def generate(image: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    y, x = grid((h, w))
    angle = float(rng.uniform(0, np.pi))
    offset = float(rng.uniform(-0.22, 0.22))
    distance = ((x / w - 0.5) * np.cos(angle) + (y / h - 0.5) * np.sin(angle) - offset)
    distance += (noise_field(rng, (h, w), (4, 5)) - 0.5) * 0.025
    width = float(rng.uniform(0.012, 0.07))
    strength = float(rng.uniform(0.18, 0.58))
    core = np.exp(-0.5 * (distance / width) ** 2)
    side = 0.3 * np.exp(-0.5 * ((distance - width * 2.2) / (width * 2)) ** 2)
    mask = np.clip((core + side) * strength, 0, 1)
    return finish(darken(image, mask), mask, "fold_shadow", angle=angle, offset=offset, width=width, strength=strength)
