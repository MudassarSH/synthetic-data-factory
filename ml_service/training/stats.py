"""Summarize synthesized dataset balance without writing sample images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import DocumentDefectDataset, find_images, require_split_leaf
from .synthesis.compose import GENERATORS

THRESHOLDS = (0.05, 0.15, 0.30, 0.50)


def _percent_summary(values: list[float]) -> dict:
    numbers = np.asarray(values, dtype=np.float64)
    return {
        "mean": 100 * float(numbers.mean()),
        "p50": 100 * float(np.percentile(numbers, 50)),
        "p95": 100 * float(np.percentile(numbers, 95)),
    }


def summarize(clean_dir: str | Path, *, count=5000, seed=0, size=(512, 512)) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    require_split_leaf(clean_dir)
    paths = find_images(clean_dir)
    dataset = DocumentDefectDataset(paths, size=size, seed=seed, samples_per_epoch=count, dynamic=False, source_root=clean_dir)
    coverages = []
    threshold_coverages = {threshold: [] for threshold in THRESHOLDS}
    mean_intensities = []
    type_counts = Counter()
    type_coverages = {kind: [] for kind in GENERATORS}
    defect_counts = Counter()
    for index in range(count):
        _, mask, metadata = dataset[index]
        defects = metadata["defects"]
        defect_counts[len(defects)] += 1
        type_counts.update(defect["type"] for defect in defects)
        for defect in defects:
            type_coverages[defect["type"]].append(defect["coverage_at_0.05"])
        for threshold in THRESHOLDS:
            threshold_coverages[threshold].append(float(np.mean(mask >= threshold)))
        coverages.append(threshold_coverages[0.05][-1])
        mean_intensities.append(float(mask.mean()))
    coverage = np.asarray(coverages)
    bins = [0, 0.01, 0.1, 0.3, 0.6, 1.000001]
    names = ["<1%", "1-10%", "10-30%", "30-60%", ">60%"]
    return {
        "sample_count": count,
        "coverage_threshold": 0.05,
        "high_defect_threshold": 0.5,
        "clean_sample_ratio": defect_counts[0] / count,
        "single_defect_ratio": defect_counts[1] / count,
        "multi_defect_ratio": sum(defect_counts[n] for n in (2, 3, 4)) / count,
        "defect_count_percent": {str(n): 100 * defect_counts[n] / count for n in range(5)},
        "type_percent": {kind: 100 * type_counts[kind] / count for kind in GENERATORS},
        "per_defect_coverage_percent": {
            kind: {"sample_count": len(values), **(_percent_summary(values) if values else {"mean": None, "p50": None, "p95": None})}
            for kind, values in type_coverages.items()
        },
        "coverage_percent": {"mean": 100 * float(coverage.mean()), "p05": 100 * float(np.percentile(coverage, 5)), "p50": 100 * float(np.percentile(coverage, 50)), "p95": 100 * float(np.percentile(coverage, 95))},
        "coverage_by_threshold_percent": {f"{threshold:.2f}": _percent_summary(values) for threshold, values in threshold_coverages.items()},
        "mean_mask_intensity": float(np.mean(mean_intensities)),
        "high_defect_coverage_percent": 100 * float(np.mean(threshold_coverages[0.5])),
        "coverage_buckets_percent": {name: 100 * float(np.mean((coverage >= low) & (coverage < high))) for name, low, high in zip(names, bins[:-1], bins[1:])},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean_dir")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.count <= 0 or args.size <= 0:
        parser.error("--count and --size must be positive")
    try:
        report = summarize(args.clean_dir, count=args.count, seed=args.seed, size=(args.size, args.size))
    except ValueError as error:
        parser.error(str(error))
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
