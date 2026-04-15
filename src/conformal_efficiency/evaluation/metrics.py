from __future__ import annotations

from typing import Dict

import torch


def summarize_prediction_sets(
    set_mask: torch.Tensor,
    labels: torch.Tensor,
    pvalues: torch.Tensor,
) -> Dict[str, float]:
    sizes = set_mask.sum(dim=1)
    covered = set_mask[torch.arange(labels.numel()), labels]
    top1 = pvalues.argmax(dim=1)
    accuracy = (top1 == labels).float().mean()

    hist = torch.bincount(sizes, minlength=set_mask.shape[1] + 1).float()
    hist = hist / hist.sum()

    results: Dict[str, float] = {
        "coverage": covered.float().mean().item(),
        "average_set_size": sizes.float().mean().item(),
        "singleton_rate": (sizes == 1).float().mean().item(),
        "empty_rate": (sizes == 0).float().mean().item(),
        "top1_accuracy_from_pvalues": accuracy.item(),
    }
    for size, prob in enumerate(hist.tolist()):
        results[f"set_size_prob_{size}"] = prob
    return results
