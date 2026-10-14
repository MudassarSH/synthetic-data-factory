"""Numerically stable BCE + foreground Dice for one-channel logits."""

import torch
from torch import nn
from torch.nn import functional as F


def segmentation_targets(masks, threshold=0.05):
    """Convert attenuation fields to binary defect regions consistently."""
    if not 0 < threshold <= 1:
        raise ValueError("target threshold must be in (0,1]")
    return (masks >= threshold).float()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0 or smooth <= 0:
            raise ValueError("loss weights must be nonnegative with a positive sum; smooth must be positive")
        self.bce_weight, self.dice_weight, self.smooth = bce_weight, dice_weight, smooth

    def components(self, logits, targets):
        if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("logits and targets must have equal [B,1,H,W] shapes")
        # Float32 reductions avoid fp16 overflow at 512x512 under autocast.
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probabilities = torch.sigmoid(logits)
        axes = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=axes)
        denominator = probabilities.sum(dim=axes) + targets.sum(dim=axes)
        dice = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return {"loss": self.bce_weight * bce + self.dice_weight * dice, "bce": bce, "dice_loss": dice}

    def forward(self, logits, targets):
        return self.components(logits, targets)["loss"]
