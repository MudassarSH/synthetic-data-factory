"""Sample and apply zero to four defects to one normalized page."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import bleed, edge_shadow, fold_shadow, haze, ring, speckle, stain, streak

GENERATORS = {
    "haze": haze.generate,
    "stain": stain.generate,
    "ring": ring.generate,
    "edge_shadow": edge_shadow.generate,
    "fold_shadow": fold_shadow.generate,
    "streak": streak.generate,
    "speckle": speckle.generate,
    "bleed": bleed.generate,
}
DEFECT_COUNTS = np.array([0.20, 0.35, 0.30, 0.12, 0.03])


def synthesize(
    image: np.ndarray,
    seed: int,
    *,
    source: str | None = None,
    defect_types: Sequence[str] | None = None,
    reverse: np.ndarray | None = None,
    count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return degraded RGB, soft union mask and JSON-ready metadata.

    Both arrays use float32 values in [0, 1]. ``defect_types`` restricts the
    available generators; ``count`` fixes the number of defects for inspection.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image must be finite RGB float data in [0, 1]")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    kinds = tuple(GENERATORS if defect_types is None else defect_types)
    if not kinds or len(set(kinds)) != len(kinds) or any(kind not in GENERATORS for kind in kinds):
        raise ValueError("defect_types must contain unique known generator names")
    rng = np.random.default_rng(seed)
    if count is None:
        probabilities = DEFECT_COUNTS[: min(4, len(kinds)) + 1].copy()
        probabilities /= probabilities.sum()
        count = int(rng.choice(len(probabilities), p=probabilities))
    if not 0 <= count <= min(4, len(kinds)):
        raise ValueError("count must be between zero and min(4, number of types)")
    selected = rng.choice(kinds, size=count, replace=False)
    output = image.copy()
    combined = np.zeros(image.shape[:2], np.float32)
    details = []
    for kind in selected:
        if kind == "bleed":
            output, mask, data = GENERATORS[kind](output, rng, reverse=reverse)
        else:
            output, mask, data = GENERATORS[kind](output, rng)
        combined = np.maximum(combined, mask)
        data["coverage_at_0.05"] = float(np.mean(mask >= 0.05))
        details.append(data)
    return output, combined, {"source": source, "seed": int(seed), "defects": details}
