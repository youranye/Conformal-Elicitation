from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> Dict[str, float]:
    confidences, predictions = probabilities.max(dim=1)
    correctness = (predictions == labels).float()

    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, dtype=probabilities.dtype)
    ece = torch.tensor(0.0, dtype=probabilities.dtype)
    results: Dict[str, float] = {}

    for idx in range(num_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == num_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        mass = mask.float().mean()
        if mask.any():
            bin_acc = correctness[mask].mean()
            bin_conf = confidences[mask].mean()
            ece = ece + mass * torch.abs(bin_acc - bin_conf)

        # Optional per-bin classifier calibration diagnostics kept available for
        # future use if we want a more detailed calibration report.
        # key = f"ece_bin_{idx}"
        # results[f"{key}_count"] = float(mask.sum().item())
        # if mask.any():
        #     results[f"{key}_accuracy"] = bin_acc.item()
        #     results[f"{key}_confidence"] = bin_conf.item()
        # else:
        #     results[f"{key}_accuracy"] = 0.0
        #     results[f"{key}_confidence"] = 0.0

    results[f"ece_top1_{num_bins}bin"] = ece.item()
    return results


def negative_log_likelihood(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    eps = torch.finfo(probabilities.dtype).eps
    return {
        "nll": F.nll_loss(torch.log(probabilities.clamp_min(eps)), labels).item(),
    }


def _quantiles(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_q10": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_q90": 0.0,
            f"{prefix}_max": 0.0,
        }
    quantiles = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=values.dtype, device=values.device)
    q_values = torch.quantile(values, quantiles)
    return {
        f"{prefix}_min": q_values[0].item(),
        f"{prefix}_q10": q_values[1].item(),
        f"{prefix}_median": q_values[2].item(),
        f"{prefix}_q90": q_values[3].item(),
        f"{prefix}_max": q_values[4].item(),
    }


def summarize_calibration(
    calibration_scores: torch.Tensor,
    alpha: float,
) -> Dict[str, float]:
    results: Dict[str, float] = {
        "target_coverage": 1.0 - alpha,
        "num_calibration": float(calibration_scores.numel()),
        "calibration_score_mean": calibration_scores.mean().item(),
        "calibration_score_std": calibration_scores.std(unbiased=False).item(),
    }
    results.update(_quantiles(calibration_scores, prefix="calibration_score"))
    return results


def summarize_prediction_sets(
    set_mask: torch.Tensor,
    labels: torch.Tensor,
    pvalues: torch.Tensor,
    probabilities: torch.Tensor,
    alpha: float,
) -> Dict[str, float]:
    sizes = set_mask.sum(dim=1)
    covered = set_mask[torch.arange(labels.numel()), labels]
    true_label_pvalues = pvalues[torch.arange(labels.numel()), labels]
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
    results.update(expected_calibration_error(probabilities=probabilities, labels=labels, num_bins=15))
    results.update(negative_log_likelihood(probabilities=probabilities, labels=labels))

    # Optional detailed diagnostics kept available for future experiments.
    # results["coverage_gap_to_target"] = covered.float().mean().item() - (1.0 - alpha)
    # results.update(_quantiles(true_label_pvalues, prefix="true_label_pvalue"))
    # results.update(_quantiles(sizes.float(), prefix="set_size"))

    for size, prob in enumerate(hist.tolist()):
        results[f"set_size_prob_{size}"] = prob

    num_classes = pvalues.shape[1]
    for cls in range(num_classes):
        mask = labels == cls
        count = int(mask.sum().item())
        if count == 0:
            results[f"class_{cls}_coverage"] = 0.0
            continue
        results[f"class_{cls}_coverage"] = covered[mask].float().mean().item()

    # Optional p-value-binned diagnostics kept available for later use.
    # bin_edges = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001], dtype=true_label_pvalues.dtype)
    # for idx in range(len(bin_edges) - 1):
    #     left = bin_edges[idx].item()
    #     right = bin_edges[idx + 1].item()
    #     if idx == len(bin_edges) - 2:
    #         mask = (true_label_pvalues >= left) & (true_label_pvalues <= right)
    #     else:
    #         mask = (true_label_pvalues >= left) & (true_label_pvalues < right)
    #     count = int(mask.sum().item())
    #     key = f"pvalue_bin_{left:.1f}_{min(right, 1.0):.1f}"
    #     results[f"{key}_count"] = float(count)
    #     if count == 0:
    #         results[f"{key}_coverage"] = 0.0
    #         results[f"{key}_avg_set_size"] = 0.0
    #     else:
    #         results[f"{key}_coverage"] = covered[mask].float().mean().item()
    #         results[f"{key}_avg_set_size"] = sizes[mask].float().mean().item()
    return results
