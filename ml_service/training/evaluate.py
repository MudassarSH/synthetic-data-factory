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
from .visualize_predictions import save_prediction_sheet, save_mask_diagnostic_sheet
from .diagnostics import pixel_diagnostics, domain_report, source_document


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
        self.sample_fingerprints = {}
        digest = hashlib.sha256()
        for index, sample in enumerate(self.samples):
            sample_digest = hashlib.sha256()
            for name in ("input.png", "mask.png", "metadata.json"):
                file = sample / name
                digest.update(f"{sample.name}/{name}\0".encode())
                content_hash = hashlib.sha256(file.read_bytes()).digest()
                digest.update(content_hash)
                sample_digest.update(name.encode() + content_hash)
            self.sample_fingerprints[sample.name] = sample_digest.hexdigest()
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
             prediction_threshold=.5, amp=False, preview_indices=(), preview_path=None,
             excluded_documents=(), diagnostics_dir=None):
    """Legacy top-level metrics remain full-set; domain summaries are additive.

    Exclusions affect reporting only here. Training explicitly monitors supported
    metrics. Family breakdowns measure the entire combined target on samples
    selecting that family; overlapping families are not isolated ground truth.
    """
    if not 0 < prediction_threshold < 1:
        raise ValueError("prediction threshold must be in (0,1)")
    criterion = criterion or BCEDiceLoss()
    was_training = model.training
    model.eval()
    rows, previews = [], {}
    requested = set(preview_indices)
    fingerprints = getattr(getattr(loader, "dataset", None), "sample_fingerprints", {})
    try:
        for images, masks, metadata in loader:
            images = images.to(device, non_blocking=True)
            targets = segmentation_targets(masks.to(device, non_blocking=True), target_threshold)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
            probabilities = logits.float().sigmoid()
            bce_map = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
            # Per-sample components preserve custom criterion weights/smoothing.
            for i, record in enumerate(metadata):
                parts = criterion.components(logits[i:i+1], targets[i:i+1])
                probability = probabilities[i, 0].cpu().numpy()
                actual = targets[i, 0].bool().cpu().numpy()
                predicted = probability > prediction_threshold
                tp = int((predicted & actual).sum())
                fp = int((predicted & ~actual).sum())
                fn = int((~predicted & actual).sum())
                strength = masks[i, 0].cpu().numpy()
                rgb = images[i].cpu().numpy()
                pixel = pixel_diagnostics(probability, actual, strength, rgb, bce_map[i, 0].cpu().numpy(), prediction_threshold)
                index = record.get("sample_index", len(rows))
                sample = str(record.get("sample", index))
                row = {"sample": sample, "source": str(record.get("source", "unknown")),
                       "document": source_document(record), "seed": record.get("seed"),
                       "content_fingerprint": fingerprints.get(sample),
                       "selected_defects": record.get("defects", []),
                       "defect_families": sorted({d["type"] for d in record.get("defects", [])}),
                       "synthetically_clean": not record.get("defects", []),
                       "is_effectively_clean": not bool(actual.any()),
                       "target_positive_coverage": float(actual.mean()),
                       "mean_probability": float(probability.mean()),
                       "coverage_gt_0_5": float((probability > .5).mean()),
                       "predicted_coverage": float(predicted.mean()),
                       "tp": tp, "fp": fp, "fn": fn,
                       "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                       "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.,
                       **{key: float(value) for key, value in parts.items()}, **pixel,
                       "largest_predicted_region_coverage": pixel["largest_predicted_region_pixels"] / actual.size}
                rows.append(row)
                if index in requested:
                    previews[index] = {"image": images[i].cpu(), "target": targets[i].cpu(),
                                       "mask": masks[i].cpu(), "probability": probabilities[i].cpu(), "metadata": {**record, "prediction_threshold": prediction_threshold}}
        if not rows:
            raise ValueError("validation loader is empty")
        domains = domain_report(rows, excluded_documents)
        # Do not attach domains to their own full object (would be recursive).
        result = dict(domains["full"])
        result.update({"domains": domains, "excluded_documents": sorted(set(excluded_documents)),
                       "target_threshold": target_threshold, "prediction_threshold": prediction_threshold,
                       "metric_region": "full_canvas_including_padding",
                       "family_metric_scope": "combined_target_on_samples_selecting_family"})
        examples = [previews[i] for i in sorted(previews)]
        if preview_path is not None:
            save_prediction_sheet(examples, preview_path)
        if diagnostics_dir is not None:
            destination = Path(diagnostics_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            if examples:
                save_mask_diagnostic_sheet(examples, destination / "mask_overlays.png")
        return result
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
    parser.add_argument("--exclude-document", action="append", default=None, help="exact top-level source document; repeatable; defaults to checkpoint policy")
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
                       preview_path=args.output_dir / "predictions.png",
                       excluded_documents=args.exclude_document if args.exclude_document is not None else config.get("exclude_document", []),
                       diagnostics_dir=args.output_dir / "diagnostics")
    metrics["validation_fingerprint"] = dataset.fingerprint
    metrics["checkpoint_epoch"] = checkpoint["epoch"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
