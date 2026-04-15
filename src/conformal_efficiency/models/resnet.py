from __future__ import annotations

from torch import nn
from torchvision.models import resnet18


def build_resnet18_cifar(num_classes: int):
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    return model
