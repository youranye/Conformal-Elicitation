from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from conformal_efficiency.models.mlp import MLPClassifier


def train_indirect(
    model: MLPClassifier,
    fit_inputs: torch.Tensor,
    fit_labels: torch.Tensor,
    learning_rate: float,
    epochs: int,
    weight_decay: float = 0.0,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(fit_inputs)
        loss = F.cross_entropy(logits, fit_labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=1)
            acc = (preds == fit_labels).float().mean().item()
        history.append({"epoch": float(epoch), "loss": loss.item(), "fit_accuracy": acc})

    return history
