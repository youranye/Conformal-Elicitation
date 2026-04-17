from __future__ import annotations

import torch

from conformal_efficiency.conformal.greedy import (
    calibrate_greedy_tau,
    greedy_cumulative_mass_sets,
    greedy_preceding_mass,
)


def test_greedy_cumulative_mass_builds_prefix_set() -> None:
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    set_mask = greedy_cumulative_mass_sets(probs, tau=0.8)
    expected = torch.tensor([[True, True, True, False]])
    assert torch.equal(set_mask, expected)


def test_greedy_preceding_mass_matches_rank_prefix() -> None:
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    labels = torch.tensor([2], dtype=torch.long)
    preceding = greedy_preceding_mass(probs, labels)
    assert torch.allclose(preceding, torch.tensor([0.7]))


def test_calibrated_tau_hits_reasonable_range() -> None:
    probs = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    tau = calibrate_greedy_tau(probs, labels, alpha=0.1)
    assert tau > 0.0
