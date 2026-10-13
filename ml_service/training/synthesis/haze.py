from __future__ import annotations

import numpy as np

from .common import darken, finish, noise_field


def generate(image: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    strength = float(rng.uniform(0.08, 0.32))
    broad = noise_field(rng, (h, w), (3, 4))
    detail = noise_field(rng, (h, w), (8, 9))
    field = np.clip(0.35 + 0.6 * broad + 0.2 * detail, 0, 1)
    mask = strength * field
    return finish(darken(image, mask), mask, "haze", strength=strength)
