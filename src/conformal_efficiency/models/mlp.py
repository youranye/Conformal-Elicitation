from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Small MLP that maps fixed-point coordinates to class logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(inputs), dim=-1)
