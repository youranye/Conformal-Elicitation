from __future__ import annotations

import torch

from conformal_efficiency.models.resnet import build_resnet18_cifar


def test_resnet18_cifar_output_shape() -> None:
    model = build_resnet18_cifar(num_classes=10)
    inputs = torch.randn(4, 3, 32, 32)
    outputs = model(inputs)
    assert outputs.shape == (4, 10)
