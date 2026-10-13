from __future__ import annotations

import numpy as np

from .common import blur, darken, finish, grid, noise_field


def generate(image: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    y, x = grid((h, w))
    cx, cy = float(rng.uniform(0.2, 0.8) * w), float(rng.uniform(0.2, 0.8) * h)
    rx = float(rng.uniform(0.09, 0.26) * w)
    ry = rx * float(rng.uniform(0.75, 1.3))
    theta = np.arctan2((y - cy) / ry, (x - cx) / rx)
    radial = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
    distortion = noise_field(rng, (h, w), (15, 15))
    radial += (distortion - 0.5) * 0.23 + 0.055 * np.sin(5 * theta + float(rng.uniform(0, 6.28)))
    thickness = float(rng.uniform(0.08, 0.2))
    edge = np.clip((thickness / 2 - np.abs(radial - 1)) / (thickness * 0.25), 0, 1)
    angular = np.sin(theta * int(rng.integers(3, 7)) + float(rng.uniform(0, 6.28)))
    breaks = noise_field(rng, (h, w), (7, 7))
    continuity = np.clip((angular + 0.8) * 1.8, 0.1, 1) * np.clip((breaks - 0.22) * 3, 0, 1)
    strength = float(rng.uniform(0.22, 0.7))
    mask = blur(edge * continuity * strength, max(0.8, min(h, w) * 0.004))
    return finish(darken(image, mask, (1.0, 0.95, 0.9)), mask, "ring", center=[cx / w, cy / h], radii=[rx / w, ry / h], thickness=thickness, strength=strength)
