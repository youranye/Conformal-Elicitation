from __future__ import annotations

from typing import Dict, List

import torch

from conformal_efficiency.conformal.pvalues import candidate_probability_scores
from conformal_efficiency.models.mlp import MLPClassifier
from conformal_efficiency.objectives.direct import soft_set_size_loss


def train_direct(
    model: MLPClassifier,
    fit_inputs: torch.Tensor,
    inner_calibration_inputs: torch.Tensor,
    inner_calibration_labels: torch.Tensor,
    alpha: float,
    learning_rate: float,
    epochs: int,
    tau_rank: float,
    tau_set: float,
    weight_decay: float = 0.0,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        fit_probs = model.predict_proba(fit_inputs)
        fit_scores = candidate_probability_scores(fit_probs)

        inner_probs = model.predict_proba(inner_calibration_inputs)
        calibration_scores = inner_probs[
            torch.arange(inner_calibration_labels.numel()), inner_calibration_labels
        ]

        loss = soft_set_size_loss(
            candidate_scores=fit_scores,
            calibration_scores=calibration_scores,
            alpha=alpha,
            tau_rank=tau_rank,
            tau_set=tau_set,
        )
        loss.backward()
        optimizer.step()

        history.append({"epoch": float(epoch), "loss": loss.item()})

    return history
