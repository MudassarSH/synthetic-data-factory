from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .common import darken, finish, noise_field


def generate(image: np.ndarray, rng: np.random.Generator, reverse: np.ndarray | None = None):
    h, w = image.shape[:2]
    source = image if reverse is None else reverse
    source_image = Image.fromarray(np.uint8(np.clip(source, 0, 1) * 255), "RGB")
    source_image = source_image.resize((w, h), Image.Resampling.BICUBIC).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    angle = float(rng.uniform(-1.8, 1.8))
    source_image = source_image.rotate(angle, Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    source_image = source_image.filter(ImageFilter.GaussianBlur(max(1, min(h, w) * float(rng.uniform(0.003, 0.009)))))
    ink = 1 - np.asarray(source_image, dtype=np.float32).mean(axis=2) / 255
    ink = np.clip((ink - 0.08) * 1.6, 0, 1)
    variation = 0.55 + 0.45 * noise_field(rng, (h, w), (5, 5))
    strength = float(rng.uniform(0.12, 0.38))
    mask = ink * variation * strength
    result = darken(image, mask, (0.9, 0.95, 1.0))
    return finish(result, mask, "bleed", strength=strength, angle=angle, source="current_page" if reverse is None else "supplied_page", tint=(0.9, 0.95, 1.0))
