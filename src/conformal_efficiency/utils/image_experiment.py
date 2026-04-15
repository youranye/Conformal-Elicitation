from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from conformal_efficiency.conformal.pvalues import candidate_probability_scores, exact_prediction_sets
from conformal_efficiency.data.cifar10 import build_cifar10_splits, make_loader
from conformal_efficiency.evaluation.metrics import summarize_prediction_sets
from conformal_efficiency.models.resnet import build_resnet18_cifar
from conformal_efficiency.trainers.image_direct import train_image_direct
from conformal_efficiency.trainers.image_indirect import train_image_indirect
from conformal_efficiency.utils.io import ensure_dir, save_json, save_yaml
from conformal_efficiency.utils.seed import set_seed


def _build_cifar_model(model_config: Dict):
    model_name = model_config["name"]
    if model_name != "resnet18_cifar":
        raise ValueError(f"Unsupported image model config: {model_name}")
    return build_resnet18_cifar(num_classes=int(model_config["num_classes"]))


@torch.no_grad()
def evaluate_image_model(
    model,
    outer_calibration_loader,
    test_loader,
    alpha: float,
    tie_seed: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    calibration_scores = []
    for images, labels in outer_calibration_loader:
        images = images.to(device)
        labels = labels.to(device)
        probs = torch.softmax(model(images), dim=-1)
        calibration_scores.append(probs[torch.arange(labels.size(0), device=device), labels])
    calibration_scores = torch.cat(calibration_scores, dim=0)

    all_set_masks = []
    all_labels = []
    all_pvalues = []
    generator = torch.Generator(device=device.type).manual_seed(tie_seed)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        probs = torch.softmax(model(images), dim=-1)
        candidate_scores = candidate_probability_scores(probs)
        set_mask, pvalues = exact_prediction_sets(
            candidate_scores=candidate_scores,
            calibration_scores=calibration_scores,
            alpha=alpha,
            generator=generator,
        )
        all_set_masks.append(set_mask.cpu())
        all_labels.append(labels.cpu())
        all_pvalues.append(pvalues.cpu())

    return summarize_prediction_sets(
        set_mask=torch.cat(all_set_masks, dim=0),
        labels=torch.cat(all_labels, dim=0),
        pvalues=torch.cat(all_pvalues, dim=0),
    )


def run_cifar10_method(config: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    set_seed(int(config["seed"]))
    device_name = config.get("device", "cpu")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    splits = build_cifar10_splits(config["data_config"], config["splits"], seed=int(config["seed"]))
    method_config = config["method_configs"][method_name]
    batch_size = int(method_config["batch_size"])

    fit_loader = make_loader(splits.fit, batch_size=batch_size, shuffle=True)
    inner_loader = make_loader(splits.inner_calibration, batch_size=batch_size, shuffle=False)
    outer_loader = make_loader(splits.outer_calibration, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(splits.test, batch_size=batch_size, shuffle=False)

    model = _build_cifar_model(config["model_config"])

    if method_name == "indirect":
        history = train_image_indirect(
            model=model,
            fit_loader=fit_loader,
            device=device,
            method_config=method_config,
        )
    elif method_name == "direct":
        history = train_image_direct(
            model=model,
            fit_loader=fit_loader,
            inner_calibration_loader=inner_loader,
            device=device,
            method_config=method_config,
            alpha=float(config["alpha"]),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    metrics = evaluate_image_model(
        model=model,
        outer_calibration_loader=outer_loader,
        test_loader=test_loader,
        alpha=float(config["alpha"]),
        tie_seed=int(config["evaluation"]["tie_seed"]),
        device=device,
    )
    return {
        "model": model.cpu(),
        "history": history,
        "metrics": metrics,
        "indices": splits.indices,
    }


def save_cifar10_run_artifacts(
    config: Dict[str, Any],
    method_name: str,
    results: Dict[str, Any],
) -> Path:
    run_dir = ensure_dir(
        Path(config["output_dir"]) / config["experiment_name"] / method_name / f"seed_{config['seed']}"
    )
    save_yaml(run_dir / "config.yaml", config)
    save_json(run_dir / "metrics.json", results["metrics"])
    save_json(run_dir / "train_history.json", results["history"])
    save_json(run_dir / "split_indices.json", results["indices"])
    torch.save(results["model"].state_dict(), run_dir / "checkpoint.pt")
    return run_dir
