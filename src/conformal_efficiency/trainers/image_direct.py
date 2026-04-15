from __future__ import annotations

from typing import Dict, List

import torch

from conformal_efficiency.conformal.pvalues import candidate_probability_scores
from conformal_efficiency.objectives.direct import soft_set_size_loss
from conformal_efficiency.trainers.image_indirect import _build_optimizer_and_scheduler


@torch.no_grad()
def collect_calibration_scores(model, loader, device: torch.device) -> torch.Tensor:
    model.eval()
    scores = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        probs = torch.softmax(model(images), dim=-1)
        batch_scores = probs[torch.arange(labels.size(0), device=device), labels]
        scores.append(batch_scores)
    return torch.cat(scores, dim=0)


def train_image_direct(
    model,
    fit_loader,
    inner_calibration_loader,
    device: torch.device,
    method_config: Dict,
    alpha: float,
) -> List[Dict[str, float]]:
    optimizer, scheduler = _build_optimizer_and_scheduler(model, method_config)
    epochs = int(method_config["epochs"])
    tau_rank = float(method_config["tau_rank"])
    tau_set = float(method_config["tau_set"])
    history: List[Dict[str, float]] = []
    model.to(device)

    for epoch in range(epochs):
        calibration_scores = collect_calibration_scores(model, inner_calibration_loader, device=device)
        model.train()
        total_loss = 0.0
        total_examples = 0

        for images, _labels in fit_loader:
            images = images.to(device)
            optimizer.zero_grad()
            probs = torch.softmax(model(images), dim=-1)
            candidate_scores = candidate_probability_scores(probs)
            loss = soft_set_size_loss(
                candidate_scores=candidate_scores,
                calibration_scores=calibration_scores,
                alpha=alpha,
                tau_rank=tau_rank,
                tau_set=tau_set,
            )
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        if scheduler is not None:
            scheduler.step()

        history.append(
            {
                "epoch": float(epoch),
                "loss": total_loss / max(total_examples, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "num_inner_calibration": float(calibration_scores.numel()),
            }
        )

    return history
