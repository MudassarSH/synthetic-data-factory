from __future__ import annotations

import numpy as np

from .common import blur, darken, finish, grid, noise_field


def generate(image: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    y, x = grid((h, w))
    cx, cy = float(rng.uniform(0.15, 0.85) * w), float(rng.uniform(0.15, 0.85) * h)
    rx, ry = float(rng.uniform(0.12, 0.38) * w), float(rng.uniform(0.1, 0.35) * h)
    angle = float(rng.uniform(0, np.pi))
    dx, dy = x - cx, y - cy
    u = (dx * np.cos(angle) + dy * np.sin(angle)) / rx
    v = (-dx * np.sin(angle) + dy * np.cos(angle)) / ry
    irregular = noise_field(rng, (h, w), (12, 12))
    distance = np.sqrt(u * u + v * v) + (irregular - 0.5) * 0.55
    soft = float(rng.uniform(0.1, 0.25))
    strength = float(rng.uniform(0.18, 0.62))
    mask = strength * np.clip((1.1 - distance) / soft, 0, 1) * (0.55 + 0.45 * irregular)
    mask = blur(mask, max(1, min(h, w) * 0.008))
    tint = (1.0, float(rng.uniform(0.92, 0.99)), float(rng.uniform(0.85, 0.95)))
    return finish(darken(image, mask, tint), mask, "stain", center=[cx / w, cy / h], radii=[rx / w, ry / h], strength=strength, tint=tint)
