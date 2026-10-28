"""Pilot training: dynamic synthesis, fixed exported validation, and checkpoints."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augmentations import TrainingAugmentation
from .dataset import DocumentDefectDataset, collate_document_samples, find_images, require_split_leaf
from .evaluate import FixedExportDataset, evaluate, prediction_indices
from .losses import BCEDiceLoss, segmentation_targets
from .model import DocumentDefectModel


def create_scheduler(optimizer, *, name="plateau", factor=.5, patience=2, min_lr=1e-6):
    if name == "none":
        return None
    if name != "plateau":
        raise ValueError("scheduler must be none or plateau")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience,
        threshold=1e-4, threshold_mode="abs", min_lr=min_lr,
    )


def train_epoch(model, loader, optimizer, criterion, scaler, device, *, target_threshold=.05,
                amp=False, progress=True, augmentation=None):
    model.train()
    totals = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0}
    count = 0
    iterator = tqdm(loader, desc="train", leave=False, disable=not progress)
    for images, masks, metadata in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if augmentation is not None:
            images, masks = augmentation(images, masks, metadata)
        targets = segmentation_targets(masks, target_threshold)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
            parts = criterion.components(logits, targets)
        if not torch.isfinite(parts["loss"]):
            raise FloatingPointError("nonfinite training loss")
        scaler.scale(parts["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        n = images.shape[0]
        count += n
        for key in totals:
            totals[key] += float(parts[key].detach()) * n
        iterator.set_postfix(loss=f"{totals['loss'] / count:.4f}")
    if not count:
        raise ValueError("training loader is empty")
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def update_metric_selection(selection, primary, epoch):
    """Mutate resumable score/epoch state; ties retain the earlier checkpoint.

    Defect Dice deliberately retains the legacy selected-defect definition;
    effective_defect_dice is separately reported for zero-coverage diagnosis.
    """
    improved = []
    for name, value in (("loss", primary["loss"]), ("primary_dice", primary["dice"]),
                        ("defect_dice", primary["defect_dice"])):
        if value is None:
            continue
        if not math.isfinite(value):
            raise FloatingPointError(f"nonfinite selection metric: {name}")
        previous = selection.get(name)
        if previous is None or (value < previous["value"] if name == "loss" else value > previous["value"]):
            selection[name] = {"value": value, "epoch": epoch}
            improved.append(name)
    return improved


def _rng_state():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, values.numpy().astype(np.uint32), position, gaussian, cached))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--augmentation", choices=("none", "standard", "strong"), default="standard")
    parser.add_argument("--decoder-dropout", type=float, default=.1)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="plateau")
    parser.add_argument("--lr-factor", type=float, default=.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--target-threshold", type=float, default=.05)
    parser.add_argument("--prediction-threshold", type=float, default=.5)
    parser.add_argument("--mask-mode", choices=("max", "attenuation"), default="max")
    parser.add_argument("--sampling", choices=("page", "document"), default="page")
    parser.add_argument("--defect-weights", type=json.loads)
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--exclude-document", action="append", default=[], help="exclude exact validation source document from primary selection, retaining it as stress data; repeatable")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true", help="random encoder initialization for offline smoke tests")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, help="resume last.pt, preserving epoch seeds and optimizer state")
    initialization.add_argument("--init-checkpoint", type=Path, help="start a new experiment from model weights only; reset optimizer/epoch/scheduler")
    args = parser.parse_args()
    if min(args.epochs, args.samples_per_epoch, args.batch_size, args.preview_count, args.cpu_threads) <= 0:
        parser.error("epochs, samples, batch size, preview count, and CPU threads must be positive")
    if args.size < 64 or args.workers < 0 or args.seed < 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("size must be >=64; workers/seed/weight decay nonnegative; lr positive")
    if not 0 < args.target_threshold <= 1 or not 0 < args.prediction_threshold < 1:
        parser.error("target threshold must be in (0,1]; prediction threshold in (0,1)")
    if not 0 <= args.decoder_dropout < 1 or not 0 < args.lr_factor < 1 or args.lr_patience < 0:
        parser.error("dropout must be in [0,1), LR factor in (0,1), patience nonnegative")
    if not 0 <= args.min_lr <= args.lr or not all(math.isfinite(value) for value in
            (args.lr, args.weight_decay, args.decoder_dropout, args.lr_factor, args.min_lr)):
        parser.error("learning rates/regularization must be finite and 0 <= min-lr <= lr")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    require_split_leaf(args.train_dir)
    paths = find_images(args.train_dir)
    train_dataset = DocumentDefectDataset(paths, seed=args.seed, samples_per_epoch=args.samples_per_epoch,
                                          dynamic=True, as_torch=True, size=(args.size, args.size),
                                          source_root=args.train_dir, sampling=args.sampling,
                                          defect_weights=args.defect_weights, mask_mode=args.mask_mode)
    validation = FixedExportDataset(args.validation_dir, size=(args.size, args.size), mask_mode=args.mask_mode)
    args.exclude_document = sorted(set(args.exclude_document))
    from .diagnostics import source_document
    validation_documents = {source_document(r) for r in validation.metadata}
    unknown = set(args.exclude_document) - validation_documents
    if unknown:
        parser.error(f"excluded validation documents not found: {sorted(unknown)}")
    if not validation_documents - set(args.exclude_document):
        parser.error("supported-domain validation would be empty")
    train_groups = {path.resolve().relative_to(args.train_dir.resolve()).parts[0] for path in paths}
    validation_groups = {str(record.get("source", "")).replace("\\", "/").split("/")[0]
                         for record in validation.metadata}
    overlap = train_groups & validation_groups
    if overlap:
        parser.error(f"train/validation source documents overlap: {sorted(overlap)}")
    source_digest = hashlib.sha256()
    for path in paths:
        source_digest.update(path.resolve().relative_to(args.train_dir.resolve()).as_posix().encode())
        source_digest.update(hashlib.sha256(path.read_bytes()).digest())
    train_fingerprint = source_digest.hexdigest()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["defect_weights"] = train_dataset.defect_weights
    config["amp_enabled"] = amp
    config["shuffle"] = True
    config["architecture"] = "mobilenet_v3_large_160-96-64-32-16-1"
    config["validation_fingerprint"] = validation.fingerprint
    config["train_fingerprint"] = train_fingerprint
    preview_selection = prediction_indices(validation, args.preview_count, args.seed)
    config["preview_indices"] = preview_selection
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None:
        if args.resume.resolve() != (args.output_dir / "last.pt").resolve():
            parser.error("resume the run's last.pt in the same --output-dir")
        previous = checkpoint["config"]
        compatible = ("size", "seed", "samples_per_epoch", "batch_size", "mask_mode", "sampling",
                      "defect_weights", "target_threshold", "prediction_threshold", "bce_weight", "dice_weight",
                      "lr", "weight_decay", "train_fingerprint", "validation_fingerprint", "preview_indices", "shuffle")
        for key in compatible:
            if previous.get(key) != config[key]:
                parser.error(f"resume setting/data mismatch: {key}")
        legacy_defaults = {"augmentation": "none", "decoder_dropout": 0.0, "lr_scheduler": "none", "exclude_document": []}
        for key, default in legacy_defaults.items():
            if previous.get(key, default) != config[key]:
                parser.error(f"resume setting mismatch: {key}; use --init-checkpoint in a new output directory to change training policy")
        if args.lr_scheduler == "plateau":
            for key in ("lr_factor", "lr_patience", "min_lr"):
                if previous.get(key) != config[key]:
                    parser.error(f"resume setting mismatch: {key}")
        if args.epochs <= checkpoint["epoch"] + 1:
            parser.error("--epochs must exceed completed checkpoint epochs")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is not empty; choose a new run directory or --resume")
    warm_start = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True) if args.init_checkpoint else None
    model = DocumentDefectModel(pretrained=not args.no_pretrained and checkpoint is None and warm_start is None,
                                decoder_dropout=args.decoder_dropout).to(device)
    if warm_start is not None:
        model.load_state_dict(warm_start["model"])
        config["initial_checkpoint_epoch"] = warm_start["epoch"] + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, name=args.lr_scheduler, factor=args.lr_factor,
                                 patience=args.lr_patience, min_lr=args.min_lr)
    criterion = BCEDiceLoss(args.bce_weight, args.dice_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_loss = 0, float("inf")
    selection = {}
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_loss = checkpoint["epoch"] + 1, checkpoint["best_loss"]
        _restore_rng(checkpoint["rng"])
        selection = checkpoint.get("metric_selection", {"loss": {"value": best_loss, "epoch": None}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "metrics.jsonl"
    if checkpoint is not None and history_path.exists():
        # A crash between logging and checkpoint publication can leave one
        # uncommitted epoch in the history; replay it from the saved state.
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        history = [record for record in history if record["epoch"] <= start_epoch]
        temporary_history = history_path.with_suffix(".tmp")
        temporary_history.write_text("".join(json.dumps(record) + "\n" for record in history), encoding="utf-8")
        temporary_history.replace(history_path)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
                      "collate_fn": collate_document_samples, "pin_memory": device.type == "cuda",
                      "persistent_workers": False}
    if args.workers:
        # Spawn avoids forking an initialized CUDA runtime and works on Windows.
        loader_options["multiprocessing_context"] = "spawn"
    shuffle_generator = torch.Generator()
    train_loader = DataLoader(train_dataset, shuffle=True, generator=shuffle_generator, **loader_options)
    validation_loader = DataLoader(validation, shuffle=False, **loader_options)
    print(f"Device: {device}; AMP: {amp}; train sources: {len(paths)}; fixed validation: {len(validation)}", flush=True)
    augmentation = TrainingAugmentation(args.augmentation)
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        shuffle_generator.manual_seed(args.seed + epoch)
        learning_rate = optimizer.param_groups[0]["lr"]
        training = train_epoch(model, train_loader, optimizer, criterion, scaler, device,
                               target_threshold=args.target_threshold, amp=amp, progress=not args.no_progress,
                               augmentation=augmentation)
        metrics = evaluate(model, validation_loader, device, criterion=criterion,
                           target_threshold=args.target_threshold, prediction_threshold=args.prediction_threshold,
                           amp=amp, preview_indices=preview_selection,
                           preview_path=args.output_dir / "previews" / f"epoch_{epoch + 1:03d}.png",
                           excluded_documents=args.exclude_document,
                           diagnostics_dir=args.output_dir / "diagnostics" / f"epoch_{epoch + 1:03d}")
        primary = metrics["domains"]["supported"]
        if not math.isfinite(primary["loss"]):
            raise FloatingPointError("nonfinite validation loss")
        if scheduler is not None:
            scheduler.step(primary["loss"])
        record = {"epoch": epoch + 1, "train": training, "validation": metrics,
                  "selection_domain": "supported", "selection_fingerprint": primary["fingerprint"],
                  "learning_rate": learning_rate, "next_learning_rate": optimizer.param_groups[0]["lr"]}
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        improvements = update_metric_selection(selection, primary, epoch + 1)
        best_loss = selection["loss"]["value"]
        state = {"format_version": 1, "epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "best_loss": best_loss, "metric_selection": selection,
                 "selection_domain": "supported", "selection_fingerprint": primary["fingerprint"],
                 "config": config, "rng": _rng_state(), "metrics": record}
        save_checkpoint(args.output_dir / "last.pt", state)
        for metric in improvements:
            filename = {"loss": "best_loss.pt", "primary_dice": "best_primary_dice.pt", "defect_dice": "best_defect_dice.pt"}[metric]
            save_checkpoint(args.output_dir / filename, state)
        if "loss" in improvements:
            save_checkpoint(args.output_dir / "best.pt", state)
        # Full source/family diagnostics are in metrics.jsonl; keep notebook
        # output bounded even for hundreds of validation source pages.
        print(json.dumps({"epoch": epoch + 1, "train_loss": training["loss"],
                          "primary_loss": primary["loss"], "primary_dice": primary["dice"],
                          "primary_defect_dice": primary["defect_dice"],
                          "full_loss": metrics["loss"], "next_learning_rate": optimizer.param_groups[0]["lr"],
                          "improved_checkpoints": improvements}), flush=True)


if __name__ == "__main__":
    main()
