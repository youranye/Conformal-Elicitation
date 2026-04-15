from __future__ import annotations

import torch

from conformal_efficiency.conformal.pvalues import soft_pvalues


def soft_set_size_loss(
    candidate_scores: torch.Tensor,
    calibration_scores: torch.Tensor,
    alpha: float,
    tau_rank: float,
    tau_set: float,
) -> torch.Tensor:
    soft_p = soft_pvalues(
        candidate_scores=candidate_scores,
        calibration_scores=calibration_scores,
        tau_rank=tau_rank,
    )
    soft_membership = torch.sigmoid((soft_p - alpha) / tau_set)
    return soft_membership.sum(dim=1).mean()
