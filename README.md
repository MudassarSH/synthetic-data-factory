# Document degradation synthesis

A standalone, seeded generator for document defect segmentation data. It produces a degraded RGB page, an aligned soft grayscale mask, and JSON metadata. No model code is required.

## Install

```powershell
python -m pip install -e .
```

The only runtime dependencies are NumPy and Pillow. PyTorch is optional (`pip install -e ".[train]"`).

## Split clean source documents

Start with unsplit clean pages. Create the split once, before generating any samples:

```powershell
python -m ml_service.training.split ml-data/clean-unsplit ml-data/clean --validation-fraction 0.18 --seed 2026
```

This copies pages into `ml-data/clean/train/` and `ml-data/clean/validation/` and writes `split_manifest.json`. Byte-identical pages stay together. Files in the same top-level source subfolder are treated as one document by default. If several files belong to one document, group them in a subfolder before splitting. The utility requires an empty destination so an existing split cannot be silently mixed with a new one. Keep `ml-data/real-test/` untouched and separate from both synthetic sets.

## Export previews or a fixed validation set

Use the split-specific directories:

```powershell
python -m ml_service.training.preview ml-data/clean/train ml-data/generated-preview --count 100 --seed 12345

python -m ml_service.training.preview ml-data/clean/validation ml-data/fixed-validation --count 500 --seed 2026 --no-grid
```

Each `sample_00000/` contains `input.png`, `mask.png`, and `metadata.json`. Contact sheets are saved in groups of 20 (`preview_grid_00.png`, etc.), up to the first 200 samples. The source page is loaded with EXIF orientation and resized to fit the requested square. Defects are applied to the physical page before padding: artificial margins remain white with zero mask values. `page_bounds` in metadata records `[x, y, width, height]` inside the final canvas. `--size` defaults to 512.

Exports require an empty destination by default. Use `--replace` to deliberately replace an export created by this version. The exporter generates the replacement in a staging directory before publishing it, removes stale samples/contact sheets, and refuses to replace unmanaged directories or unrelated files. A failed generation preserves the previous export. `export_manifest.json` records source checksums, sample identities/seeds, dimensions, geometry, mask interpretation, and defect weights. Legacy exports without this manifest must be regenerated in a new directory.

The corrected page geometry changes generated pixels even with the same seeds. Keep existing validation sets for historical comparisons; create a new versioned validation export for the corrected generator.

For fixed validation, a sample seed is a hash of the base seed, relative source path, and variant number. Fixed bleed-through mirrors the same source page so adding another source does not alter an existing page's result. Dynamic training uses a different random page when available. The sample's output index can still move; keep the split manifest and exported validation set fixed for comparisons across model runs.

## Inspect dataset balance

```powershell
python -m ml_service.training.stats ml-data/clean/train --count 5000 --seed 12345 --output ml-data/train-stats.json
```

This generates masks in memory and reports clean, single, and multi-defect ratios; each defect type's prevalence and its own mask coverage mean/P50/P95; combined-mask coverage at thresholds 0.05, 0.15, 0.30, and 0.50; mean mask intensity; and coverage buckets. Per-defect coverage uses each defect's own mask at 0.05, including samples where that defect overlaps another. It saves no sample images. Both preview and stats reject a directory with immediate `train/` and `validation/` children, so pass one split leaf.

Coverage statistics now exclude artificial padding. The report also measures document sampling, effective defects with nonzero coverage at 0.05, and each family's share of summed positive pixels (overlaps count once per family). `clean_sample_ratio` still means no selected defects; an ineffective bleed on a blank page can have a selected defect but no positive pixels.

To inspect document-balanced dynamic training, including other-page bleed-through:

```powershell
python -m ml_service.training.stats ml-data/clean/train --count 5000 --seed 12345 --dynamic --sampling document --output ml-data/balanced-train-stats.json
```

Large-area haze/shadows and small rings/speckles need different sampling frequencies if you want more balanced positive-pixel contributions. `suggested_defect_weights` provides inverse-square-root conditional-coverage weights as a starting point. Pass a weights dictionary to the dataset or `synthesize`, or its JSON representation through `--defect-weights` in preview/stats. Unspecified families have weight 1; zero disables a family. Selection remains without replacement, and the defect-count distribution is renormalized if fewer than four families remain. Re-measure after applying weights: the suggestion does not guarantee equal pixel shares or represent a calibrated real-scan distribution.

## Stream into training

```python
from torch.utils.data import DataLoader
from ml_service.training.dataset import (
    DocumentDefectDataset, collate_document_samples, find_images,
)

dataset = DocumentDefectDataset(
    find_images("ml-data/clean/train"),
    seed=12345,
    samples_per_epoch=10000,
    dynamic=True,
    as_torch=True,
    source_root="ml-data/clean/train",
    sampling="document",
    # Illustrative starting weights; tune against statistics and real scans.
    defect_weights={"haze": 0.2, "edge_shadow": 0.3, "fold_shadow": 0.4},
    mask_mode="max",
)
loader = DataLoader(
    dataset, batch_size=16, num_workers=2,
    collate_fn=collate_document_samples, persistent_workers=False,
)
dataset.set_epoch(0)
for degraded, mask, metadata in loader:
    # degraded: [B, 3, H, W]; mask: [B, 1, H, W]
    # metadata is a list of dictionaries, preserving ragged defects/64-bit seeds.
    pass
```

The dataset returns channel-first tensors only when `as_torch=True`; otherwise it returns NumPy arrays. Set the epoch before creating each new `DataLoader` iterator. Use nonpersistent workers, since persistent worker copies do not see later calls to `set_epoch`. Dynamic sample seeds depend on base seed, epoch, and index, so processing order does not affect a sample. For fixed validation, use `dynamic=False` and pass `source_root="ml-data/clean/validation"`.

Use `collate_document_samples` for batches: it stacks images/masks and leaves metadata as Python objects, so variable defect lists and unsigned 64-bit seeds are safe. It also supports NumPy batches. `sampling="page"` retains the original cycling policy. For dynamic training, `sampling="document"` selects a document uniformly, then a page uniformly within it. With `source_root`, top-level document folders define groups and root-level files are independent documents; without it, immediate parent folders define groups. Document balancing is opt-in and requires `dynamic=True`. On platforms using spawned workers, create the loader/iterate inside an `if __name__ == "__main__":` guard.

## Mask contract

Each generator returns its applied local attenuation field in `[0, 1]`. Choose the combined training target explicitly with `mask_mode` (or `--mask-mode` in preview/stats):

- `max` (default): the pixelwise maximum of individual fields. This is a defect-region/maximum-strength target, not total accumulated darkening. Two overlapping neutral defects of 0.3 give a mask of 0.3.
- `attenuation`: the mean RGB-channel attenuation after composing all fields and tints, `mean(1 - product(1 - mask_i * tint_i))`. Two overlapping neutral defects of 0.3 give 0.51. Use this for accumulated-strength regression.

Both targets retain defects over black text, where visible image differences can be zero. Values encode applied strength, not binary class probabilities. For binary segmentation, define and keep a threshold consistent across training/evaluation; for strength regression, evaluate against the selected soft target. Color tints vary channel attenuation by at most 15%. Speckles and bleed-through encode local dot or ink density rather than whole bounding regions. Exported 8-bit PNG masks are quantized. Metadata and export manifests record the selected mask mode.

The generator currently includes haze, irregular stains, broken elliptical rings, edge shadows, fold shadows, toner streaks, speckles, and mirrored reverse-page bleed-through. A sampled page has 0–4 distinct defect types with weights 20%, 35%, 30%, 12%, and 3%. The optional `reverse` image in `synthesize` supplies the reverse-page content for bleed-through; otherwise the current page is mirrored.

```python
from ml_service.training.dataset import load_clean_page
from ml_service.training.synthesis import synthesize

clean = load_clean_page("ml-data/clean/train/page.png")
degraded, soft_mask, metadata = synthesize(
    clean, seed=12345, source="page.png", mask_mode="attenuation",
)
```

Before using the output for model training, inspect the contact sheets and full-resolution masks against a representative collection of your clean pages. In particular, check text overlap, rings, scanner borders, haze, and bleed-through. Realism still requires comparison with real scans; seeded synthesis alone cannot establish that match.

## Tests

```powershell
python -m unittest discover -v
```

The core suite needs only NumPy and Pillow. Install the optional training dependency to also run real PyTorch collation and DataLoader checks with zero/two workers, spawned processes, and changing epochs. GitHub Actions runs core tests on Python 3.10/3.12 and a separate CPU-PyTorch integration job.

## Pilot segmentation training

For the complete Google Colab upload, GPU setup, training, resume, evaluation, and download workflow, see [GOOGLE_COLAB_GUIDE.md](GOOGLE_COLAB_GUIDE.md).

For Kaggle background jobs, input/output setup, and checkpoint recovery across sessions, see [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md).

Install the training stack (PyTorch, torchvision, NumPy, Pillow, tqdm, matplotlib, and scikit-learn):

```powershell
python -m pip install -e ".[train]"
```

Optional Albumentations/TensorBoard packages are available through `pip install -e ".[train,train-extras]"`. The built-in paired augmentation uses the existing torch/torchvision dependencies; TensorBoard logging is not enabled by installing a package. Choose a PyTorch build appropriate for your GPU before installing the project extras.

The five training modules are `model.py`, `losses.py`, `train.py`, `evaluate.py`, and `visualize_predictions.py`. The model uses the [torchvision MobileNetV3-Large ImageNet V2 encoder](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html), with skip features at strides 2/4/8/16/32 and a depthwise-separable decoder of 160 → 96 → 64 → 32 → 16 → 1 channels. RGB normalization happens inside the model; inputs are not classification-cropped. It outputs `[B, 1, 512, 512]` logits for 512-square inputs, without applying sigmoid.

Train with the existing clean training sources and the **exported** 500-sample validation set:

```powershell
python -m ml_service.training.train ml-data/clean/train ml-data/fixed-validation --output-dir ml-data/runs/pilot-01 --epochs 10 --samples-per-epoch 10000 --batch-size 8 --workers 2
```

Defaults are seed 12345, 512-square inputs, page sampling, maximum-field masks, AdamW with learning rate 0.0003 and weight decay 0.001, and equally weighted BCE + Dice. New runs use standard augmentation, decoder dropout 0.1, and a validation-loss plateau scheduler. The encoder is pretrained by default; its weights download on the first run. `--no-pretrained` is available for offline smoke tests. `--sampling document` and `--defect-weights` expose the existing dataset's balancing controls without changing them.

`augmentations.py` applies shared rotation/flip/scale/translation/elastic grids to images and soft masks, then image-only brightness/contrast/blur. Masks are binarized afterward; geometric fill is white for images and zero for masks. Sample-derived seeds make transformations independent of batching and worker order. Use `--augmentation strong` for wider variation or `--augmentation none` for a baseline. Strong augmentation includes mirrored/upside-down documents and small elastic warps; inspect whether these match your deployment documents. Brightness/contrast are global nuisance variations, not new labeled local stains. Validation is never augmented. Diversity is increased, not guaranteed to make every generated image unique.

The [ReduceLROnPlateau scheduler](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) monitors validation loss once per epoch. Defaults: `--lr-factor 0.5 --lr-patience 2 --min-lr 0.000001`. With patience 2, reduction occurs after three consecutive non-improving epochs (absolute improvement threshold 0.0001). `learning_rate` and `next_learning_rate` in each metrics record show the rate used and the rate selected for the next epoch. Scheduler state is saved/restored with checkpoints. Disable with `--lr-scheduler none`. `--decoder-dropout` controls spatial dropout immediately before the output head; it is inactive during evaluation and adds no checkpoint parameters.

The attenuation mask is binarized at **≥0.05** for both training and validation. Predictions are thresholded at **>0.5** for reported binary metrics. These thresholds can be configured, are saved with the run, and are reused by standalone evaluation. Padding remains a negative target. Legacy exported validation sets without a manifest are accepted; their actual dimensions and mask mode are checked, and their image/mask/metadata bytes are fingerprinted. Validation is never regenerated or resized by the training stack. If you choose attenuation-mode training, export matching attenuation-mode validation first.

CUDA automatically enables [autocast and gradient scaling](https://docs.pytorch.org/docs/stable/notes/amp_examples.html); CPU runs use float32. BCE and Dice reductions use float32 even under autocast. Use `--no-amp` to disable mixed precision. Worker processes are nonpersistent and spawned; the dataset epoch is set before each iterator. Training batches are shuffled with a seed derived from the base seed and epoch, preserving their ordering when resuming.

Every epoch saves:

- `previews/epoch_001.png`, etc.: 24 fixed **INPUT | TARGET | PREDICTION** examples, with a quarter reserved for clean pages where available. Prediction panels show continuous probabilities. Labels include mean predicted probability and coverage above 0.5.
- `metrics.jsonl`: training/validation loss components, per-page Dice/IoU, micro Dice, pixel precision/recall, defect-only Dice, and clean-page diagnostics.
- `last.pt`: model, optimizer, scaler, scheduler, epoch, random states, configuration, and data fingerprints for resuming.
- `best.pt`: the checkpoint with the lowest validation BCE + Dice loss.
- `config.json`: the architecture, thresholds, data fingerprints, preview indices, and training settings.

`clean_mean_predicted_probability` and `clean_predicted_coverage_gt_0_5` are measured over validation pages whose metadata has no selected defects, across the full canvas. Low values indicate fewer false alarms on those pages. Metrics also report the clean sample count; if no clean samples exist, these values are `null`. Empty prediction/target pairs have Dice/IoU 1, so inspect defect-only Dice and clean diagnostics alongside overall scores. Lower false positives on synthetic clean pages do not establish performance on real scans.

Additional diagnostics separate `clean_bce` from `defect_bce`, report clean coverage P50/P95 and the fraction of clean samples with any false positive, and list `worst_clean_samples` (ranked by BCE) with source identities and coverage. Eighteen percent coverage means eighteen percent of clean pixels exceed 0.5; it does not mean harmless probabilities near zero. Inspect these samples and the existing BCE/Dice loss components when validation loss and thresholded Dice move in opposite directions.

Resume with a larger total epoch count and otherwise matching data/settings:

```powershell
python -m ml_service.training.train ml-data/clean/train ml-data/fixed-validation --output-dir ml-data/runs/pilot-01 --epochs 20 --samples-per-epoch 10000 --batch-size 8 --workers 2 --resume ml-data/runs/pilot-01/last.pt
```

Changed augmentation/dropout/scheduler policies require a **new experiment**. To keep weights learned in the first pilot while resetting the optimizer, scheduler, and epoch counter:

```powershell
python -m ml_service.training.train ml-data/clean/train ml-data/fixed-validation --output-dir ml-data/runs/regularized-01 --init-checkpoint ml-data/runs/pilot-01/best.pt --epochs 10 --samples-per-epoch 10000 --batch-size 8 --workers 2 --augmentation strong --decoder-dropout 0.1 --weight-decay 0.001 --lr-scheduler plateau
```

Preserve the original pilot for comparison. `--init-checkpoint` loads only model weights and is mutually exclusive with `--resume`. New `best.pt` selection starts afresh. For exact continuation of a pre-regularization checkpoint, supply its old weight decay plus `--augmentation none --decoder-dropout 0 --lr-scheduler none`, retain its other settings, and use `--resume`. Older checkpoints remain readable for evaluation/inference; see the Colab guide's checkpoint-driven resume cell for automatic legacy defaults.

Evaluate a saved checkpoint and write a fresh prediction sheet:

```powershell
python -m ml_service.training.evaluate ml-data/runs/pilot-01/best.pt ml-data/fixed-validation --output-dir ml-data/runs/pilot-01/evaluation
```

`python -m ml_service.training.visualize_predictions` accepts the same arguments as evaluation. Standalone evaluation loads model weights without downloading ImageNet weights again. Installed equivalents are `document-defect-train`, `document-defect-evaluate`, and `document-defect-previews`.
