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

Each `sample_00000/` contains `input.png`, `mask.png`, and `metadata.json`. Contact sheets are saved in groups of 20 (`preview_grid_00.png`, etc.), up to the first 200 samples. The source page is loaded with EXIF orientation, fitted inside the requested square, and padded white. `--size` defaults to 512.

For fixed validation, a sample seed is a hash of the base seed, relative source path, and variant number. Fixed bleed-through mirrors the same source page so adding another source does not alter an existing page's result. Dynamic training uses a different random page when available. The sample's output index can still move; keep the split manifest and exported validation set fixed for comparisons across model runs.

## Inspect dataset balance

```powershell
python -m ml_service.training.stats ml-data/clean/train --count 5000 --seed 12345 --output ml-data/train-stats.json
```

This generates masks in memory and reports clean, single, and multi-defect ratios; each defect type's prevalence and its own mask coverage mean/P50/P95; combined-mask coverage at thresholds 0.05, 0.15, 0.30, and 0.50; mean mask intensity; and coverage buckets. Per-defect coverage uses each defect's own mask at 0.05, including samples where that defect overlaps another. It saves no sample images. Both preview and stats reject a directory with immediate `train/` and `validation/` children, so pass one split leaf.

## Stream into training

```python
from ml_service.training.dataset import DocumentDefectDataset, find_images

dataset = DocumentDefectDataset(
    find_images("ml-data/clean/train"),
    seed=12345,
    samples_per_epoch=10000,
    dynamic=True,
    as_torch=True,
)
dataset.set_epoch(0)
degraded, mask, metadata = dataset[0]
```

The dataset returns channel-first tensors only when `as_torch=True`; otherwise it returns NumPy arrays. Set the epoch before creating each new `DataLoader` iterator. Use nonpersistent workers, since persistent worker copies do not see later calls to `set_epoch`. Dynamic sample seeds depend on base seed, epoch, and index, so processing order does not affect a sample. For fixed validation, use `dynamic=False` and pass `source_root="ml-data/clean/validation"`.

## Mask contract

Each generator returns its applied local attenuation field, in `[0, 1]`; the composer takes the pixelwise maximum across defects. `0` means no applied defect, `0.1` is faint, and `0.5` is strong. Color tints vary channel attenuation by at most 15%, so mask values have roughly comparable meaning across families. Speckles and bleed-through also encode local dot or ink density in this field, rather than labeling whole bounding regions. The exported 8-bit PNG is a quantized version. The mask describes the generated defect region, including pixels over text where dark ink can hide the visible difference. It is therefore more useful for training than a thresholded absolute difference between images.

The generator currently includes haze, irregular stains, broken elliptical rings, edge shadows, fold shadows, toner streaks, speckles, and mirrored reverse-page bleed-through. A sampled page has 0–4 distinct defect types with weights 20%, 35%, 30%, 12%, and 3%. The optional `reverse` image in `synthesize` supplies the reverse-page content for bleed-through; otherwise the current page is mirrored.

```python
from ml_service.training.dataset import load_clean_image
from ml_service.training.synthesis import synthesize

clean = load_clean_image("ml-data/clean/train/page.png")
degraded, soft_mask, metadata = synthesize(clean, seed=12345, source="page.png")
```

Before using the output for model training, inspect the contact sheets and full-resolution masks against a representative collection of your clean pages. In particular, check text overlap, rings, scanner borders, haze, and bleed-through. Realism still requires comparison with real scans; seeded synthesis alone cannot establish that match.
