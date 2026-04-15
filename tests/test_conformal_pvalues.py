from __future__ import annotations

import torch

from conformal_efficiency.conformal.pvalues import exact_randomized_pvalues, soft_pvalues


def test_exact_pvalues_in_unit_interval() -> None:
    calibration_scores = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32)
    candidates = torch.tensor([[0.1, 0.5], [0.8, 0.9]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    pvalues = exact_randomized_pvalues(candidates, calibration_scores, generator)
    assert torch.all(pvalues >= 0.0)
    assert torch.all(pvalues <= 1.0)


def test_soft_pvalues_shape() -> None:
    calibration_scores = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32)
    candidates = torch.tensor([[0.1, 0.5], [0.8, 0.9]], dtype=torch.float32)
    pvalues = soft_pvalues(candidates, calibration_scores, tau_rank=0.1)
    assert pvalues.shape == candidates.shape
