from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature


def fit_temperature_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
) -> Tuple[float, Dict[str, float]]:
    """
    Fit a scalar temperature by minimizing NLL on held-out logits/labels.
    """
    logits = logits.detach()
    labels = labels.detach()
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=logits.dtype, device=logits.device))

    with torch.no_grad():
        nll_before = F.cross_entropy(logits, labels).item()

    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        temperature = torch.exp(log_temperature).item()
        nll_after = F.cross_entropy(logits / temperature, labels).item()

    return temperature, {
        "temperature": temperature,
        "temperature_nll_before": nll_before,
        "temperature_nll_after": nll_after,
    }
