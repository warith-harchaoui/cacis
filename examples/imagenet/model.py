"""
examples.imagenet.model
=======================

Build a torchvision ResNet from scratch (no pretrained weights).
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


_RESNET_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


def build_resnet(arch: str, num_classes: int = 1000) -> nn.Module:
    """
    Construct a randomly initialized ResNet.

    Parameters
    ----------
    arch:
        One of ``resnet18 / resnet34 / resnet50 / resnet101 / resnet152``.
    num_classes:
        Output classes (default 1000 for ImageNet).
    """
    if arch not in _RESNET_BUILDERS:
        raise ValueError(
            f"Unknown ResNet arch: {arch}. Choose from {sorted(_RESNET_BUILDERS)}."
        )
    return _RESNET_BUILDERS[arch](weights=None, num_classes=num_classes)
