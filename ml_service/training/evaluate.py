"""Read fixed exported validation samples and report segmentation/clean-page metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .dataset import collate_document_samples
from .losses import BCEDiceLoss, segmentation_targets
from .visualize_predictions import save_prediction_sheet


class FixedExportDataset(Dataset):
    """Read input.png/mask.png/metadata.json verbatim; never regenerate or resize."""

    def __init__(self, directory, *, size=(512, 512), mask_mode="max"):
        self.root = Path(directory)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.samples = sorted(path for path in self.root.glob("sample_*") if path.is_dir())
        if not self.samples:
            raise ValueError(f"no exported validation samples in {self.root}")
        self.size = tuple(size)
        self.metadata = []
        digest = hashlib.sha256()
        for index, sample in enumerate(self.samples):
            for name in ("input.png", "mask.png", "metadata.json"):
                file = sample / name
                digest.update(f"{sample.name}/{name}\0".encode())
                digest.update(hashlib.sha256(file.read_bytes()).digest())
            record = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
            if record.get("mask_mode", "max") != mask_mode:
                raise ValueError(f"mask mode mismatch in {sample}")
            for name in ("input.png", "mask.png"):
                with Image.open(sample / name) as image:
                    if image.size != self.size:
                        raise ValueError(f"{sample / name} has size {image.size}; expected {self.size}")
            self.metadata.append({**record, "sample": sample.name, "sample_index": index})
        manifest_path = self.root / "export_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("sample_count") != len(self.samples):
                raise ValueError("validation sample count disagrees with export manifest")
            if manifest.get("mask_mode", "max") != mask_mode or tuple(manifest.get("size", self.size)) != self.size:
                raise ValueError("validation settings disagree with export manifest")
            declared = [sample["directory"] for sample in manifest.get("samples", [])]
            if declared != [sample.name for sample in self.samples]:
                raise ValueError("validation sample directories disagree with export manifest")
        # Legacy exports are supported; byte fingerprints still protect resume.
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample / "input.png") as image:
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255
        with Image.open(sample / "mask.png") as image:
            mask = np.array(image.convert("L"), dtype=np.float32) / 255
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()), torch.from_numpy(mask[None].copy()), self.metadata[index]


def prediction_indices(dataset, count=24, seed=2026):
    """Keep the same examples each epoch, reserving a quarter for clean pages."""
    if count <= 0:
        raise ValueError("preview count must be positive")
    count = min(count, len(dataset))
    clean = [i for i, record in enumerate(dataset.metadata) if not record.get("defects", [])]
    defective = [i for i, record in enumerate(dataset.metadata) if record.get("defects", [])]
    rng = np.random.default_rng(seed)
    clean_count = min(len(clean), max(1, count // 4))
    selected = list(rng.choice(clean, clean_count, replace=False)) if clean_count else []
    defect_count = min(len(defective), count - len(selected))
    if defect_count:
        selected.extend(rng.choice(defective, defect_count, replace=False))
    remaining = sorted(set(range(len(dataset))) - set(selected))
    if len(selected) < count:
        selected.extend(rng.choice(remaining, count - len(selected), replace=False))
    return sorted(int(index) for index in selected)


@torch.inference_mode()
def evaluate(model, loader, device, *, criterion=None, target_threshold=.05,
             prediction_threshold=.5, amp=False, preview_indices=(), preview_path=None):
    if not 0 < prediction_threshold < 1:
        raise ValueError("prediction threshold must be in (0,1)")
    criterion = criterion or BCEDiceLoss()
    was_training = model.training
    model.eval()
    count = clean_count = defect_count = 0
    tp = fp = fn = 0
    dice_sum = iou_sum = defect_dice_sum = 0.0
    clean_probability = clean_coverage = 0.0
    clean_bce = defect_bce = 0.0
    clean_records = []
    losses = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0}
    previews = {}
    requested = set(preview_indices)
    try:
        for images, masks, metadata in loader:
            images = images.to(device, non_blocking=True)
            targets = segmentation_targets(masks.to(device, non_blocking=True), target_threshold)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
            parts = criterion.components(logits, targets)
            probabilities = logits.float().sigmoid()
            predicted = probabilities > prediction_threshold
            actual = targets.bool()
            sample_bce = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none").mean(dim=(1, 2, 3))
            intersection = (predicted & actual).sum(dim=(1, 2, 3)).double()
            pred_count = predicted.sum(dim=(1, 2, 3)).double()
            true_count = actual.sum(dim=(1, 2, 3)).double()
            denominator = pred_count + true_count
            union = denominator - intersection
            dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1)
            iou = torch.where(union > 0, intersection / union.clamp_min(1), 1)
            n = len(metadata)
            count += n
            for key in losses:
                losses[key] += float(parts[key]) * n
            tp += int(intersection.sum())
            fp += int((pred_count - intersection).sum())
            fn += int((true_count - intersection).sum())
            dice_sum += float(dice.sum())
            iou_sum += float(iou.sum())
            for i, record in enumerate(metadata):
                if not record.get("defects", []):
                    clean_count += 1
                    clean_probability += float(probabilities[i].mean())
                    clean_coverage += float((probabilities[i] > .5).float().mean())
                    clean_bce += float(sample_bce[i])
                    clean_records.append({"sample": record.get("sample"), "source": record.get("source"),
                                          "mean_probability": float(probabilities[i].mean()),
                                          "coverage_gt_0_5": float((probabilities[i] > .5).float().mean()),
                                          "bce": float(sample_bce[i])})
                else:
                    defect_count += 1
                    defect_dice_sum += float(dice[i])
                    defect_bce += float(sample_bce[i])
                index = record["sample_index"]
                if index in requested:
                    previews[index] = {"image": images[i].cpu(), "target": targets[i].cpu(),
                                       "probability": probabilities[i].cpu(), "metadata": record}
        if not count:
            raise ValueError("validation loader is empty")
        if preview_path is not None:
            save_prediction_sheet([previews[i] for i in sorted(previews)], preview_path)
        return {
            "sample_count": count, **{key: value / count for key, value in losses.items()},
            "dice": dice_sum / count, "iou": iou_sum / count,
            "micro_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0),
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "defect_sample_count": defect_count,
            "defect_dice": defect_dice_sum / defect_count if defect_count else None,
            "clean_sample_count": clean_count,
            "clean_mean_predicted_probability": clean_probability / clean_count if clean_count else None,
            "clean_predicted_coverage_gt_0_5": clean_coverage / clean_count if clean_count else None,
            "clean_bce": clean_bce / clean_count if clean_count else None,
            "defect_bce": defect_bce / defect_count if defect_count else None,
            "clean_false_positive_sample_fraction": sum(item["coverage_gt_0_5"] > 0 for item in clean_records) / clean_count if clean_count else None,
            "clean_coverage_p50": float(np.percentile([item["coverage_gt_0_5"] for item in clean_records], 50)) if clean_records else None,
            "clean_coverage_p95": float(np.percentile([item["coverage_gt_0_5"] for item in clean_records], 95)) if clean_records else None,
            "worst_clean_samples": sorted(clean_records, key=lambda item: item["bce"], reverse=True)[:10],
            "target_threshold": target_threshold, "prediction_threshold": prediction_threshold,
        }
    finally:
        model.train(was_training)


def main():
    from .model import DocumentDefectModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=24)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.preview_count <= 0:
        parser.error("batch size/preview count must be positive; workers must be nonnegative")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    dataset = FixedExportDataset(args.validation_dir, size=(config["size"], config["size"]),
                                 mask_mode=config["mask_mode"])
    model = DocumentDefectModel(pretrained=False, decoder_dropout=config.get("decoder_dropout", 0.0)).to(device)
    model.load_state_dict(checkpoint["model"])
    options = {"multiprocessing_context": "spawn"} if args.workers else {}
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate_document_samples, pin_memory=device.type == "cuda", **options)
    criterion = BCEDiceLoss(config["bce_weight"], config["dice_weight"])
    metrics = evaluate(model, loader, device, criterion=criterion,
                       target_threshold=config["target_threshold"],
                       prediction_threshold=config["prediction_threshold"], amp=device.type == "cuda",
                       preview_indices=prediction_indices(dataset, args.preview_count, config["seed"]),
                       preview_path=args.output_dir / "predictions.png")
    metrics["validation_fingerprint"] = dataset.fingerprint
    metrics["checkpoint_epoch"] = checkpoint["epoch"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
