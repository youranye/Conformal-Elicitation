from __future__ import annotations

import torch

from conformal_efficiency.calibration.temperature import apply_temperature, fit_temperature_from_logits
from conformal_efficiency.conformal.pvalues import exact_randomized_pvalues, soft_pvalues
from conformal_efficiency.evaluation.metrics import expected_calibration_error


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


def test_ece_in_valid_range() -> None:
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    metrics = expected_calibration_error(probabilities=probs, labels=labels, num_bins=5)
    assert 0.0 <= metrics["ece_top1_5bin"] <= 1.0


def test_temperature_scaling_positive_and_shape_preserving() -> None:
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.0], [1.5, 0.2]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    temperature, stats = fit_temperature_from_logits(logits=logits, labels=labels, max_iter=10)
    scaled = apply_temperature(logits, temperature)
    assert temperature > 0.0
    assert scaled.shape == logits.shape
    assert "temperature_nll_after" in stats
