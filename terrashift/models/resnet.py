"""Bi-temporal ResNet-50 change classifier (early fusion)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from ..data.bands import BAND_INDEX, NUM_BANDS, NUM_CLASSES


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    """Replace a 3-channel conv with an `in_channels` conv.

    RGB channels of both dates reuse the pretrained RGB filters; the remaining
    channels (NIR / SWIR) get the mean of the RGB filters. All weights are rescaled
    so activation magnitude roughly matches the 3-channel original.
    """
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            w = conv.weight  # (out, 3, k, k) in RGB order
            mean_w = w.mean(dim=1, keepdim=True)
            new_w = mean_w.repeat(1, in_channels, 1, 1).clone()
            rgb_map = {"red": 0, "green": 1, "blue": 2}
            for t in range(in_channels // NUM_BANDS):
                for band, src in rgb_map.items():
                    new_w[:, t * NUM_BANDS + BAND_INDEX[band]] = w[:, src]
            new_w *= 3.0 / in_channels
            new.weight.copy_(new_w)
    return new


class BiTemporalResNet50(nn.Module):
    name = "resnet50"

    def __init__(self, num_bands: int = NUM_BANDS, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.conv1 = _adapt_first_conv(backbone.conv1, 2 * num_bands, pretrained)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, num_classes))
        self.backbone = backbone
        self.num_bands = num_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "num_bands": self.num_bands, **(extra or {})}, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "BiTemporalResNet50":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(num_bands=ckpt.get("num_bands", NUM_BANDS), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


def resolve_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
