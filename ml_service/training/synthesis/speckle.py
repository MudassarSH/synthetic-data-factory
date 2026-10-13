from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .common import darken, finish


def generate(image: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    canvas = Image.new("L", (w, h))
    draw = ImageDraw.Draw(canvas)
    count = int(rng.integers(60, 300) * h * w / (512 * 512))
    for _ in range(max(5, count)):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        radius = max(1, int(rng.exponential(max(1, min(h, w) * 0.003))))
        value = int(rng.integers(70, 255))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
    canvas = canvas.filter(ImageFilter.GaussianBlur(max(0.3, min(h, w) * 0.001)))
    strength = float(rng.uniform(0.25, 0.8))
    mask = np.asarray(canvas, dtype=np.float32) / 255 * strength
    return finish(darken(image, mask), mask, "speckle", count=count, strength=strength)
