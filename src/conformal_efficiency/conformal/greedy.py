from __future__ import annotations

import math

import torch


def greedy_cumulative_mass_sets(probabilities: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Build top-ranked prefix sets by accumulating probability mass until tau.

    Returns a boolean mask of shape (batch, num_classes).
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be 2D")

    sorted_probs, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
    csum = torch.cumsum(sorted_probs, dim=1)
    included_count = (csum < tau).sum(dim=1) + 1
    included_count = included_count.clamp(max=probabilities.shape[1])

    prefix_mask_sorted = (
        torch.arange(probabilities.shape[1], device=probabilities.device)
        .unsqueeze(0)
        .expand_as(sorted_probs)
        < included_count.unsqueeze(1)
    )
    set_mask = torch.zeros_like(prefix_mask_sorted, dtype=torch.bool)
    set_mask.scatter_(1, sorted_indices, prefix_mask_sorted)
    return set_mask


def greedy_preceding_mass(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    For each example, return the cumulative probability mass strictly before the
    true label in the descending probability ranking.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be 2D")

    sorted_probs, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
    csum = torch.cumsum(sorted_probs, dim=1)

    ranks = torch.empty_like(sorted_indices)
    ranks.scatter_(
        1,
        sorted_indices,
        torch.arange(probabilities.shape[1], device=probabilities.device).unsqueeze(0).expand_as(sorted_indices),
    )
    true_ranks = ranks[torch.arange(labels.numel(), device=labels.device), labels]
    preceding = torch.zeros(labels.numel(), dtype=probabilities.dtype, device=probabilities.device)

    positive_rank = true_ranks > 0
    if positive_rank.any():
        preceding[positive_rank] = csum[
            torch.arange(labels.numel(), device=labels.device)[positive_rank],
            true_ranks[positive_rank] - 1,
        ]
    return preceding


def calibrate_greedy_tau(probabilities: torch.Tensor, labels: torch.Tensor, alpha: float) -> float:
    """
    Pick the smallest global tau such that empirical coverage on the calibration
    set is at least 1 - alpha for the greedy cumulative-mass sets.
    """
    preceding = greedy_preceding_mass(probabilities, labels)
    sorted_preceding, _ = torch.sort(preceding)
    n = sorted_preceding.numel()
    target_rank = max(math.ceil((1.0 - alpha) * n) - 1, 0)
    threshold = sorted_preceding[target_rank]
    tau = torch.nextafter(threshold, torch.tensor(float("inf"), device=threshold.device, dtype=threshold.dtype))
    return float(tau.item())
