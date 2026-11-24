"""CNN definitions for Task 2.

`PlantCNN` is a compact Conv-BN-ReLU-Pool network trained from scratch. It is deliberately small
(~0.47M parameters, ~100M MACs/image) so it trains on a CPU in minutes. `build_model("mobilenet_v3_small",
pretrained=True)` offers the transfer-learning alternative when pretrained weights can be downloaded.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn(c_in, c_out, stride=1):
    # bias=False because BatchNorm's shift makes the conv bias redundant
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class PlantCNN(nn.Module):
    def __init__(self, num_classes: int, width: int = 1, dropout: float = 0.3):
        super().__init__()
        w = lambda c: int(c * width)
        self.features = nn.Sequential(
            conv_bn(3, w(16), stride=2),            # 128 -> 64   (stride-2 stem: cheap early downsampling)
            conv_bn(w(16), w(32), stride=2),        # 64  -> 32
            conv_bn(w(32), w(64)),
            nn.MaxPool2d(2),                        # 32  -> 16
            conv_bn(w(64), w(128)),
            conv_bn(w(128), w(128)),
            nn.MaxPool2d(2),                        # 16  -> 8
            conv_bn(w(128), w(192)),                # last conv: Grad-CAM target layer
        )
        self.pool = nn.AdaptiveAvgPool2d(1)         # global average pooling: no big FC layer, fewer parameters
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(w(192), num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))

    @property
    def cam_layer(self):
        return self.features[-1]


def build_model(num_classes: int, arch: str = "plantcnn", pretrained: bool = False, width: int = 1):
    if arch == "plantcnn":
        return PlantCNN(num_classes, width)
    if arch == "mobilenet_v3_small":
        import torchvision
        m = torchvision.models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        m.cam_layer = m.features[-1]
        return m
    raise ValueError(arch)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
