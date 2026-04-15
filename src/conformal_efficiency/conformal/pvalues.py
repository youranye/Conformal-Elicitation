from __future__ import annotations

from typing import Optional

import torch


def candidate_probability_scores(probs: torch.Tensor) -> torch.Tensor:
    """Return S(x, y) = p_theta(y | x) for every candidate label."""
    return probs


def exact_randomized_pvalues(
    candidate_scores: torch.Tensor,
    calibration_scores: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Compute exact randomized split-conformal p-values.

    candidate_scores: shape (m, K)
    calibration_scores: shape (n,)
    returns: shape (m, K)
    """
    if calibration_scores.ndim != 1:
        raise ValueError("calibration_scores must be one-dimensional")

    expanded_candidates = candidate_scores.unsqueeze(-1)
    expanded_cal = calibration_scores.view(1, 1, -1)
    lt = (expanded_cal < expanded_candidates).sum(dim=-1)
    eq = (expanded_cal == expanded_candidates).sum(dim=-1)
    uniform = torch.rand(
        candidate_scores.shape,
        generator=generator,
        device=candidate_scores.device,
        dtype=candidate_scores.dtype,
    )
    return (lt + uniform * (1 + eq)) / (calibration_scores.numel() + 1)


def exact_prediction_sets(
    candidate_scores: torch.Tensor,
    calibration_scores: torch.Tensor,
    alpha: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pvalues = exact_randomized_pvalues(
        candidate_scores=candidate_scores,
        calibration_scores=calibration_scores,
        generator=generator,
    )
    return (pvalues > alpha), pvalues


def soft_pvalues(
    candidate_scores: torch.Tensor,
    calibration_scores: torch.Tensor,
    tau_rank: float,
) -> torch.Tensor:
    """
    Smooth training approximation to randomized p-values.

    This smooths the rank comparison only and omits the explicit randomized
    tie-breaking term during training.
    """
    expanded_candidates = candidate_scores.unsqueeze(-1)
    expanded_cal = calibration_scores.view(1, 1, -1)
    comparisons = torch.sigmoid((expanded_candidates - expanded_cal) / tau_rank)
    return comparisons.sum(dim=-1) / (calibration_scores.numel() + 1)
