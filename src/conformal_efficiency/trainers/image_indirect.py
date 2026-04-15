from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def _build_optimizer_and_scheduler(model, method_config: Dict):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(method_config["learning_rate"]),
        momentum=float(method_config.get("momentum", 0.9)),
        weight_decay=float(method_config.get("weight_decay", 0.0)),
    )
    schedule_name = method_config.get("lr_schedule", "cosine")
    if schedule_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(method_config["epochs"])
        )
    else:
        scheduler = None
    return optimizer, scheduler


def train_image_indirect(
    model,
    fit_loader,
    device: torch.device,
    method_config: Dict,
) -> List[Dict[str, float]]:
    optimizer, scheduler = _build_optimizer_and_scheduler(model, method_config)
    epochs = int(method_config["epochs"])
    label_smoothing = float(method_config.get("label_smoothing", 0.0))
    history: List[Dict[str, float]] = []
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0.0
        total_examples = 0

        for images, labels in fit_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).float().sum().item()
            total_examples += batch_size

        if scheduler is not None:
            scheduler.step()

        history.append(
            {
                "epoch": float(epoch),
                "loss": total_loss / max(total_examples, 1),
                "fit_accuracy": total_correct / max(total_examples, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    return history
