"""Seeded, paired tensor augmentation for training only; no extra dependencies."""

import math

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur


class TrainingAugmentation:
    """Apply shared geometric warps and image-only lighting/blur nuisances.

    Seeds come from sample metadata (already epoch-dependent), so results do
    not depend on batching or worker order. Soft masks use bilinear sampling
    before the training loop binarizes them. Uncovered areas are white/zero.
    Flips and quarter turns model orientation variation, not readable text
    augmentation for an OCR task. Elastic displacement stays small.
    """

    def __init__(self, profile="standard", *, photometric=True):
        if profile not in {"none", "standard", "strong"}:
            raise ValueError("augmentation must be none, standard, or strong")
        self.profile = profile
        self.photometric = photometric

    @torch.no_grad()
    def __call__(self, images, masks, metadata):
        if self.profile == "none":
            return images, masks
        if images.ndim != 4 or masks.shape != (images.shape[0], 1, *images.shape[-2:]):
            raise ValueError("expected aligned [B,3,H,W] images and [B,1,H,W] masks")
        if len(metadata) != images.shape[0]:
            raise ValueError("one metadata record is required per sample")
        strong = self.profile == "strong"
        height, width = images.shape[-2:]
        matrices, displacements, photo = [], [], []
        for record in metadata:
            rng = np.random.default_rng([int(record["seed"]), 0xA091])
            angle = float(rng.uniform(-15 if strong else -7, 15 if strong else 7))
            if rng.random() < (.3 if strong else .1):
                angle += int(rng.integers(0, 4)) * 90
            radians = math.radians(angle)
            scale = float(rng.uniform(.85 if strong else .95, 1.15 if strong else 1.05))
            sx = scale * (-1 if rng.random() < (.5 if strong else .15) else 1)
            sy = scale * (-1 if rng.random() < (.2 if strong else .05) else 1)
            shift = .06 if strong else .02
            matrices.append([
                [math.cos(radians) * sx, -math.sin(radians) * sy * height / width, float(rng.uniform(-shift, shift))],
                [math.sin(radians) * sx * width / height, math.cos(radians) * sy, float(rng.uniform(-shift, shift))],
            ])
            if rng.random() < (.5 if strong else .2):
                # Smooth low-frequency displacement, in normalized grid coordinates.
                field = torch.tensor(rng.uniform(-1, 1, (1, 2, 7, 7)), device=images.device, dtype=torch.float32)
                field = F.interpolate(field, size=(height, width), mode="bicubic", align_corners=False)
                field = field.clamp(-1, 1) * (.025 if strong else .01)
            else:
                field = torch.zeros((1, 2, height, width), device=images.device)
            displacements.append(field)
            lighting = .2 if strong else .1
            photo.append((float(rng.uniform(1 - lighting, 1 + lighting)),
                          float(rng.uniform(1 - lighting, 1 + lighting)),
                          float(rng.uniform(.2, 1.2 if strong else .7)) if rng.random() < (.35 if strong else .15) else None))
        theta = torch.tensor(matrices, device=images.device, dtype=torch.float32)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        grid = grid + torch.cat(displacements).permute(0, 2, 3, 1)
        images = F.grid_sample(images.float() - 1, grid, align_corners=False, padding_mode="zeros") + 1
        masks = F.grid_sample(masks.float(), grid, align_corners=False, padding_mode="zeros").clamp(0, 1)
        if self.photometric:
            adjusted = []
            for image, (brightness, contrast, sigma) in zip(images, photo):
                center = image.mean(dim=(-2, -1), keepdim=True)
                image = ((image - center) * contrast + center) * brightness
                image = image.clamp(0, 1)
                if sigma is not None:
                    image = gaussian_blur(image, [5, 5], [sigma, sigma])
                adjusted.append(image)
            images = torch.stack(adjusted)
        return images.clamp(0, 1), masks
