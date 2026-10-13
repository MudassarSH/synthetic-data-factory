"""Sample and apply zero to four defects to one normalized page."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def validate_defect_weights(weights):
    if weights is None:
        return None
    if not isinstance(weights, Mapping) or any(kind not in GENERATORS for kind in weights):
        raise ValueError("defect_weights must map known generator names to weights")
    try:
        result = {kind: float(weight) for kind, weight in weights.items()}
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("defect_weights must be numeric") from error
    if any(not np.isfinite(weight) or weight < 0 for weight in result.values()):
        raise ValueError("defect_weights must be finite and nonnegative")
    if not any(result.get(kind, 1.0) > 0 for kind in GENERATORS):
        raise ValueError("at least one defect weight must be positive")
    return result


def synthesize(
    image: np.ndarray,
    seed: int,
    *,
    source: str | None = None,
    defect_types: Sequence[str] | None = None,
    reverse: np.ndarray | None = None,
    count: int | None = None,
    defect_weights: Mapping[str, float] | None = None,
    mask_mode: str = "max",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return degraded RGB, a maximum-field or cumulative attenuation mask, and metadata.

    Both arrays use float32 values in [0, 1]. ``defect_types`` restricts the
    available generators; ``count`` fixes the number of defects for inspection.
    ``defect_weights`` controls sampling frequency; ``mask_mode='attenuation'``
    returns mean channel attenuation after overlapping fields and tints.
    """
    if mask_mode not in {"max", "attenuation"}:
        raise ValueError("mask_mode must be max or attenuation")
    defect_weights = validate_defect_weights(defect_weights)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image must be finite RGB float data in [0, 1]")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    kinds = tuple(GENERATORS if defect_types is None else defect_types)
    if not kinds or len(set(kinds)) != len(kinds) or any(kind not in GENERATORS for kind in kinds):
        raise ValueError("defect_types must contain unique known generator names")
    probabilities_by_type = None
    if defect_weights is not None:
        weights = np.array([defect_weights.get(kind, 1.0) for kind in kinds], dtype=np.float64)
        if not (weights > 0).any():
            raise ValueError("at least one available defect type must have a positive weight")
        kinds = tuple(kind for kind, weight in zip(kinds, weights) if weight > 0)
        weights = weights[weights > 0]
        weights /= weights.max()
        probabilities_by_type = weights / weights.sum()
    rng = np.random.default_rng(seed)
    if count is None:
        probabilities = DEFECT_COUNTS[: min(4, len(kinds)) + 1].copy()
        probabilities /= probabilities.sum()
        count = int(rng.choice(len(probabilities), p=probabilities))
    if not 0 <= count <= min(4, len(kinds)):
        raise ValueError("count must be between zero and min(4, number of types)")
    selected = rng.choice(kinds, size=count, replace=False, p=probabilities_by_type)
    output = image.copy()
    combined = np.zeros(image.shape[:2], np.float32)
    details = []
    transmission = np.ones_like(image) if mask_mode == "attenuation" else None
    for kind in selected:
        if kind == "bleed":
            output, mask, data = GENERATORS[kind](output, rng, reverse=image if reverse is None else reverse)
            if reverse is None:
                data["source"] = "current_page"
        else:
            output, mask, data = GENERATORS[kind](output, rng)
        combined = np.maximum(combined, mask)
        if transmission is not None:
            transmission *= 1 - mask[..., None] * np.asarray(data.get("tint", (1, 1, 1)), dtype=np.float32)
        data["coverage_at_0.05"] = float(np.mean(mask >= 0.05))
        details.append(data)
    if mask_mode == "attenuation":
        combined = (1 - transmission).mean(axis=2).astype(np.float32)
    return output, combined, {"source": source, "seed": int(seed), "mask_mode": mask_mode, "defects": details}
