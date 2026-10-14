"""MobileNetV3-Large encoder with a small skip-connected segmentation decoder."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels, bias=False),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.GroupNorm(8, output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, groups=output_channels, bias=False),
            nn.Conv2d(output_channels, output_channels, 1, bias=False),
            nn.GroupNorm(8, output_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((x, skip), dim=1))


class DocumentDefectModel(nn.Module):
    """Raw RGB [0,1] in; one-channel logits at the original resolution out.

    Encoder taps: 16/24/40/112/160 channels at strides 2/4/8/16/32.
    Decoder: 160 -> 96 -> 64 -> 32 -> 16 -> 1. ImageNet normalization
    happens internally, without classification cropping/resizing.
    """

    def __init__(self, *, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.encoder = backbone.features[:16]
        self.decoder = nn.ModuleList([
            DecoderBlock(160 + 112, 96), DecoderBlock(96 + 40, 64),
            DecoderBlock(64 + 24, 32), DecoderBlock(32 + 16, 16),
        ])
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("expected images shaped [B,3,H,W]")
        output_size = images.shape[-2:]
        x = (images - self.image_mean) / self.image_std
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in (1, 3, 6, 12, 15):
                features.append(x)
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            x = block(x, skip)
        return F.interpolate(self.head(x), size=output_size, mode="bilinear", align_corners=False)
