"""Run a frozen checkpoint on real pages and compare supplied restoration outputs.

This module predicts regions. It does not invent an attenuation-to-correction
mapping: model probabilities are binary-region confidence, not defect strength.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .dataset import load_clean_page, pad_page
from .diagnostics import largest_region


REQUIRED_TAGS = {"bleed_through", "shadow", "stain", "gray_paper", "clean_text"}
REQUIRED_CAPTURES = {"photograph", "photocopy"}


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32) / 255


def restoration_metrics(original, baseline, guided, reference=None, text_mask=None):
    """Aligned RGB arrays in [0,1]; measure change and reference error separately."""
    arrays = [original, baseline, guided] + ([reference] if reference is not None else [])
    if any(array.shape != original.shape for array in arrays):
        raise ValueError("restoration/reference images must be registered and have identical oriented dimensions")
    result = {"guided_changed_pixel_fraction": float((np.abs(guided - original).max(axis=2) > 1/255).mean())}
    if reference is not None:
        mse = lambda image: float(np.square(image - reference).mean())
        result.update({"input_mse": mse(original), "baseline_mse": mse(baseline), "guided_mse": mse(guided),
                       "mse_gain_over_baseline": mse(baseline) - mse(guided)})
    if text_mask is not None:
        if text_mask.shape != original.shape[:2] or not text_mask.any():
            raise ValueError("text mask must contain foreground text pixels and match oriented dimensions")
        result["text_changed_fraction"] = float((np.abs(guided - original).max(axis=2)[text_mask] > 1/255).mean())
        if reference is not None:
            result["baseline_text_mse"] = float(np.square(baseline - reference)[text_mask].mean())
            result["guided_text_mse"] = float(np.square(guided - reference)[text_mask].mean())
    return result


def character_error_rate(reference, predicted):
    """Exact Unicode edit distance / reference length; no OCR normalization."""
    if not reference:
        raise ValueError("reference transcript cannot be empty")
    previous = list(range(len(predicted) + 1))
    for i, character in enumerate(reference, 1):
        current = [i]
        for j, candidate in enumerate(predicted, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (character != candidate)))
        previous = current
    return previous[-1] / len(reference)


@torch.inference_mode()
def predict_page(model, path, device, size):
    original = read_rgb(path)
    page = load_clean_page(path, (size, size))
    h, w = page.shape[:2]
    x, y = (size - w) // 2, (size - h) // 2
    canvas = pad_page(page, (size, size))
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy())[None].to(device)
    probability = model(tensor).float().sigmoid()[0, 0, y:y+h, x:x+w].cpu().numpy()
    probability = np.asarray(Image.fromarray(probability).resize((original.shape[1], original.shape[0]), Image.Resampling.BILINEAR))
    return original, probability


def run_acceptance(model, manifest_path, output_dir, device, *, size=512, threshold=.5):
    if not 0 < threshold < 1:
        raise ValueError("prediction threshold must be in (0,1)")
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    specification = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not specification.get("cases"):
        raise ValueError("acceptance manifest has no real cases")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("acceptance output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(manifest_path.read_bytes())
    def resolve(case, key):
        path = (manifest_path.parent / case[key]).resolve()
        digest.update(key.encode() + hashlib.sha256(path.read_bytes()).digest())
        return path
    rows, seen = [], set()
    was_training = model.training
    model.eval()
    try:
        for case in specification["cases"]:
            identity = case["id"]
            if not isinstance(identity, str) or not identity or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identity) or identity in seen:
                raise ValueError("case ids must be unique safe filenames containing letters, digits, underscores or hyphens")
            seen.add(identity)
            tags, capture = set(case["tags"]), case["capture"]
            if not tags or not tags <= REQUIRED_TAGS or capture not in REQUIRED_CAPTURES:
                raise ValueError("invalid real-page tags or capture type")
            original, probability = predict_page(model, resolve(case, "input"), device, size)
            prediction = probability > threshold
            np.save(output_dir / f"{identity}_probability.npy", probability)
            Image.fromarray(np.uint8(probability * 255 + .5)).save(output_dir / f"{identity}_probability.png")
            Image.fromarray(np.uint8(prediction) * 255).save(output_dir / f"{identity}_regions.png")
            alpha = .65 * probability[..., None]
            overlay = original * (1 - alpha) + np.array([1., 0., 0.]) * alpha
            Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255 + .5)).save(output_dir / f"{identity}_overlay.png")
            row = {"id": identity, "tags": sorted(tags), "capture": capture,
                   "mean_probability": float(probability.mean()), "predicted_coverage": float(prediction.mean()),
                   "largest_region_pixels": largest_region(prediction), "status": "needs_restoration_outputs"}
            if case.get("defect_mask"):
                with Image.open(resolve(case, "defect_mask")) as image:
                    actual = np.asarray(ImageOps.exif_transpose(image).convert("L")) >= 128
                if actual.shape != prediction.shape:
                    raise ValueError("real defect mask must match oriented input dimensions")
                tp = int((actual & prediction).sum())
                denominator = int(actual.sum() + prediction.sum())
                row["segmentation_dice"] = 2 * tp / denominator if denominator else 1.
            if case.get("baseline_restored") and case.get("guided_restored"):
                baseline = read_rgb(resolve(case, "baseline_restored"))
                guided = read_rgb(resolve(case, "guided_restored"))
                reference = read_rgb(resolve(case, "reference")) if case.get("reference") else None
                text_mask = None
                if case.get("text_mask"):
                    with Image.open(resolve(case, "text_mask")) as image:
                        text_mask = np.asarray(ImageOps.exif_transpose(image).convert("L")) >= 128
                row.update(restoration_metrics(original, baseline, guided, reference, text_mask))
                row["status"] = "needs_reference_and_text_mask_or_human_review"
                if reference is not None and text_mask is not None:
                    gain = row["mse_gain_over_baseline"]
                    text_increase = row["guided_text_mse"] - row["baseline_text_mse"]
                    gate = specification.get("gates", {})
                    minimum = gate.get("min_mse_gain", 0.)
                    # Healthy/gray pages can pass by preserving an already good
                    # baseline. Defect cases must improve it, not merely tie it.
                    needs_improvement = bool(tags & {"bleed_through", "shadow", "stain"})
                    gain_pass = gain > minimum if needs_improvement else gain >= minimum
                    passed = gain_pass and text_increase <= gate.get("max_text_mse_increase", 0.)
                    row["status"] = "pass" if passed else "fail"
                review = case.get("review", {})
                if type(review.get("readability_improved")) is bool and type(review.get("text_damage")) is bool:
                    row["human_review"] = review
                    healthy = not bool(tags & {"bleed_through", "shadow", "stain"})
                    preservation = healthy and review.get("acceptable_preservation") is True
                    human_pass = (review["readability_improved"] or preservation) and not review["text_damage"]
                    if row["status"] != "fail":
                        row["status"] = "pass" if human_pass else "fail"
                if case.get("transcripts"):
                    transcript = case["transcripts"]
                    row["baseline_cer"] = character_error_rate(transcript["reference"], transcript["baseline"])
                    row["guided_cer"] = character_error_rate(transcript["reference"], transcript["guided"])
                    if row["guided_cer"] > row["baseline_cer"]:
                        row["status"] = "fail"
            rows.append(row)
    finally:
        model.train(was_training)
    missing_tags = sorted(REQUIRED_TAGS - {tag for row in rows for tag in row["tags"]})
    missing_captures = sorted(REQUIRED_CAPTURES - {row["capture"] for row in rows})
    complete = not missing_tags and not missing_captures and specification.get("purpose") == "held_out_acceptance"
    status = "pass" if complete and all(row["status"] == "pass" for row in rows) else "fail" if any(row["status"] == "fail" for row in rows) else "incomplete"
    report = {"status": status, "purpose": specification.get("purpose"), "fingerprint": digest.hexdigest(),
              "prediction_threshold": threshold, "missing_tags": missing_tags, "missing_captures": missing_captures,
              "cases": rows, "restoration_scope": "external baseline and model-guided outputs; model predicts regions only"}
    (output_dir / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    from .model import DocumentDefectModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    device = torch.device(args.device)
    model = DocumentDefectModel(pretrained=False, decoder_dropout=config.get("decoder_dropout", 0.)).to(device)
    model.load_state_dict(checkpoint["model"])
    report = run_acceptance(model, args.manifest, args.output_dir, device,
                            size=config["size"], threshold=config["prediction_threshold"])
    report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report["checkpoint_epoch"] = checkpoint["epoch"] + 1
    (args.output_dir / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
