"""CPU summaries shared by fixed validation and real-document acceptance."""

import hashlib
import json

import numpy as np
from scipy.ndimage import label


PROBABILITY_BINS = np.linspace(0, 1, 21)
STRENGTH_BINS = np.array([0, .05, .1, .2, .4, .6, .8, 1])


def source_document(record):
    return str(record.get("source", "unknown")).replace("\\", "/").split("/")[0]


def largest_region(predicted):
    """Area in pixels of the largest 8-connected component."""
    regions, count = label(predicted, structure=np.ones((3, 3), dtype=bool))
    return int(np.bincount(regions.ravel())[1:].max()) if count else 0


def pixel_diagnostics(probability, actual, strength, image, bce, threshold):
    predicted = probability > threshold
    missed = actual & ~predicted
    luminance = image.mean(axis=0)
    foreground = int(actual.sum())
    background = actual.size - foreground
    hist = lambda values, bins: np.histogram(values, bins=bins)[0].tolist()
    return {
        "foreground_pixels": foreground, "background_pixels": background,
        "foreground_bce_sum": float(bce[actual].sum()),
        "background_bce_sum": float(bce[~actual].sum()),
        "foreground_bce": float(bce[actual].mean()) if foreground else None,
        "background_bce": float(bce[~actual].mean()) if background else None,
        "probability_histogram": hist(probability, PROBABILITY_BINS),
        "foreground_probability_histogram": hist(probability[actual], PROBABILITY_BINS),
        "background_probability_histogram": hist(probability[~actual], PROBABILITY_BINS),
        "positive_strength_histogram": hist(strength[actual], STRENGTH_BINS),
        "false_negative_strength_histogram": hist(strength[missed], STRENGTH_BINS),
        "false_negative_pixels": int(missed.sum()),
        "dark_foreground_pixels": int((actual & (luminance < .1)).sum()),
        "dark_false_negative_pixels": int((missed & (luminance < .1)).sum()),
        "largest_predicted_region_pixels": largest_region(predicted),
    }


def summarize_records(records):
    """Macro sample metrics, pooled pixel diagnostics; no images retained."""
    n = len(records)
    if not n:
        return {"sample_count": 0, "loss": None, "bce": None, "dice_loss": None,
                "dice": None, "iou": None, "defect_dice": None,
                "clean_sample_count": 0, "defect_sample_count": 0,
                "effective_clean_sample_count": 0,
                "fingerprint": hashlib.sha256(b"[]").hexdigest()}
    mean = lambda items, key: float(np.mean([r[key] for r in items])) if items else None
    clean = [r for r in records if r["synthetically_clean"]]
    defect = [r for r in records if not r["synthetically_clean"]]
    effective_clean = [r for r in records if r["is_effectively_clean"]]
    effective_defect = [r for r in records if not r["is_effectively_clean"]]
    tp, fp, fn = (sum(r[k] for r in records) for k in ("tp", "fp", "fn"))
    result = {"sample_count": n, **{k: mean(records, k) for k in ("loss", "bce", "dice_loss", "dice", "iou")},
              "micro_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
              "precision": tp / (tp + fp) if tp + fp else (1. if not fn else 0.),
              "recall": tp / (tp + fn) if tp + fn else 1.,
              "defect_sample_count": len(defect), "defect_dice": mean(defect, "dice"),
              "defect_bce": mean(defect, "bce"), "clean_bce": mean(clean, "bce"),
              "effective_defect_sample_count": len(effective_defect),
              "effective_defect_dice": mean(effective_defect, "dice"),
              "worst_clean_samples": [{k: r[k] for k in ("sample", "source", "mean_probability", "coverage_gt_0_5", "bce")} for r in sorted(clean, key=lambda r: r["bce"], reverse=True)[:10]]}
    for prefix, group in (("clean", clean), ("effective_clean", effective_clean)):
        result.update({f"{prefix}_sample_count": len(group),
                       f"{prefix}_mean_predicted_probability": mean(group, "mean_probability"),
                       f"{prefix}_predicted_coverage_gt_0_5": mean(group, "coverage_gt_0_5"),
                       f"{prefix}_largest_region_coverage_mean": mean(group, "largest_predicted_region_coverage"),
                       f"{prefix}_largest_region_pixels_max": max((r["largest_predicted_region_pixels"] for r in group), default=None)})
        for suffix, cutoff in (("", 0), ("_coverage_gt_0_001", .001), ("_coverage_gt_0_005", .005), ("_coverage_gt_0_01", .01)):
            result[f"{prefix}_false_positive_sample_fraction{suffix}"] = (
                sum(r["coverage_gt_0_5"] > cutoff for r in group) / len(group) if group else None)
        for percentile in (50, 95):
            result[f"{prefix}_coverage_p{percentile}"] = float(np.percentile([r["coverage_gt_0_5"] for r in group], percentile)) if group else None
    for prefix in ("foreground", "background"):
        pixels = sum(r[f"{prefix}_pixels"] for r in records)
        result[f"{prefix}_bce"] = sum(r[f"{prefix}_bce_sum"] for r in records) / pixels if pixels else None
    for key in ("probability_histogram", "foreground_probability_histogram", "background_probability_histogram",
                "positive_strength_histogram", "false_negative_strength_histogram"):
        result[key] = np.sum([r[key] for r in records], axis=0).tolist()
    result["probability_bin_edges"] = PROBABILITY_BINS.tolist()
    result["strength_bin_edges"] = STRENGTH_BINS.tolist()
    result["false_negative_rate_by_strength_bin"] = [miss / total if total else None for miss, total in
        zip(result["false_negative_strength_histogram"], result["positive_strength_histogram"])]
    for key in ("false_negative_pixels", "dark_foreground_pixels", "dark_false_negative_pixels"):
        result[key] = sum(r[key] for r in records)
    # Bind membership AND original fixed sample bytes; independent of prediction.
    members = sorted((r["sample"], r.get("content_fingerprint")) for r in records)
    result["fingerprint"] = hashlib.sha256(json.dumps(members).encode()).hexdigest()
    return result


def domain_report(records, excluded_documents):
    excluded = set(excluded_documents)
    full = summarize_records(records)
    primary = summarize_records([r for r in records if r["document"] not in excluded])
    stress = summarize_records([r for r in records if r["document"] in excluded])
    domains = {"full": full, "supported": primary, "excluded": stress}
    for name, summary in domains.items():
        members = records if name == "full" else [r for r in records if (r["document"] in excluded) == (name == "excluded")]
        summary["by_source"] = {source: summarize_records([r for r in members if r["source"] == source])
                                for source in sorted({r["source"] for r in members})}
        summary["by_document"] = {doc: summarize_records([r for r in members if r["document"] == doc])
                                  for doc in sorted({r["document"] for r in members})}
        families = sorted({family for r in members for family in r["defect_families"]})
        summary["by_defect_family"] = {family: summarize_records([r for r in members if family in r["defect_families"]]) for family in families}
    return domains
