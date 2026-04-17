from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from conformal_efficiency.calibration.temperature import apply_temperature, fit_temperature_from_logits
from conformal_efficiency.conformal.greedy import calibrate_greedy_tau, greedy_cumulative_mass_sets
from conformal_efficiency.conformal.pvalues import candidate_probability_scores, exact_prediction_sets
from conformal_efficiency.data.cifar10 import build_cifar10_splits, make_loader as make_cifar10_loader
from conformal_efficiency.data.cifar100 import build_cifar100_splits, make_loader as make_cifar100_loader
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


def _build_image_splits(data_config: Dict, split_cfg: Dict, seed: int):
    dataset_name = data_config["name"]
    if dataset_name == "cifar10":
        return build_cifar10_splits(data_config, split_cfg, seed=seed), make_cifar10_loader
    if dataset_name == "cifar100":
        return build_cifar100_splits(data_config, split_cfg, seed=seed), make_cifar100_loader
    raise ValueError(f"Unsupported image dataset: {dataset_name}")


@torch.no_grad()
def evaluate_image_model(
    model,
    outer_calibration_loader,
    test_loader,
    alpha: float,
    tie_seed: int,
    device: torch.device,
    temperature: float = 1.0,
    method_name: str = "indirect",
    greedy_tau: float | None = None,
) -> Dict[str, float]:
    model.eval()
    calibration_scores = []
    for images, labels in outer_calibration_loader:
        images = images.to(device)
        labels = labels.to(device)
        probs = torch.softmax(apply_temperature(model(images), temperature), dim=-1)
        calibration_scores.append(probs[torch.arange(labels.size(0), device=device), labels])
    calibration_scores = torch.cat(calibration_scores, dim=0)

    all_set_masks = []
    all_labels = []
    all_ranking_scores = []
    all_probs = []
    generator = torch.Generator(device=device.type).manual_seed(tie_seed)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        probs = torch.softmax(apply_temperature(model(images), temperature), dim=-1)

        if method_name in {"indirect", "direct"}:
            candidate_scores = candidate_probability_scores(probs)
            set_mask, ranking_scores = exact_prediction_sets(
                candidate_scores=candidate_scores,
                calibration_scores=calibration_scores,
                alpha=alpha,
                generator=generator,
            )
        elif method_name in {"greedy_mass_fixed", "greedy_mass_calibrated"}:
            tau = (1.0 - alpha) if method_name == "greedy_mass_fixed" else float(greedy_tau)
            set_mask = greedy_cumulative_mass_sets(probs, tau=tau)
            ranking_scores = probs
        else:
            raise ValueError(f"Unsupported evaluation method: {method_name}")

        all_set_masks.append(set_mask.cpu())
        all_labels.append(labels.cpu())
        all_ranking_scores.append(ranking_scores.cpu())
        all_probs.append(probs.cpu())

    return summarize_prediction_sets(
        set_mask=torch.cat(all_set_masks, dim=0),
        labels=torch.cat(all_labels, dim=0),
        pvalues=torch.cat(all_ranking_scores, dim=0),
        probabilities=torch.cat(all_probs, dim=0),
        alpha=alpha,
    )


def _train_image_indirect_family(
    model,
    fit_loader,
    inner_loader,
    device: torch.device,
    method_config: Dict[str, Any],
) -> tuple[list[dict[str, float]], float, dict[str, float]]:
    history = train_image_indirect(
        model=model,
        fit_loader=fit_loader,
        device=device,
        method_config=method_config,
    )
    if bool(method_config.get("posthoc_temperature_scaling", False)):
        model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in inner_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits_list.append(model(images))
                labels_list.append(labels)
        temperature, posthoc = fit_temperature_from_logits(
            logits=torch.cat(logits_list, dim=0),
            labels=torch.cat(labels_list, dim=0),
            max_iter=int(method_config.get("temperature_max_iter", 50)),
        )
    else:
        temperature = 1.0
        posthoc = {"temperature": 1.0}
    return history, float(temperature), posthoc


def run_image_method(config: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    set_seed(int(config["seed"]))
    device_name = config.get("device", "cpu")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    splits, make_loader = _build_image_splits(config["data_config"], config["splits"], seed=int(config["seed"]))
    method_config = config["method_configs"][method_name]
    batch_size = int(method_config["batch_size"])

    fit_loader = make_loader(splits.fit, batch_size=batch_size, shuffle=True)
    inner_loader = make_loader(splits.inner_calibration, batch_size=batch_size, shuffle=False)
    outer_loader = make_loader(splits.outer_calibration, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(splits.test, batch_size=batch_size, shuffle=False)

    model = _build_cifar_model(config["model_config"])
    posthoc = {"temperature": 1.0}
    greedy_tau = None

    if method_name in {"indirect", "greedy_mass_fixed", "greedy_mass_calibrated"}:
        history, temperature, posthoc = _train_image_indirect_family(
            model=model,
            fit_loader=fit_loader,
            inner_loader=inner_loader,
            device=device,
            method_config=method_config,
        )
        if method_name == "greedy_mass_calibrated":
            model.eval()
            probs_list = []
            labels_list = []
            with torch.no_grad():
                for images, labels in outer_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    probs_list.append(torch.softmax(apply_temperature(model(images), temperature), dim=-1))
                    labels_list.append(labels)
            greedy_tau = calibrate_greedy_tau(
                probabilities=torch.cat(probs_list, dim=0),
                labels=torch.cat(labels_list, dim=0),
                alpha=float(config["alpha"]),
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
        temperature = 1.0
    else:
        raise ValueError(f"Unknown method: {method_name}")

    metrics = evaluate_image_model(
        model=model,
        outer_calibration_loader=outer_loader,
        test_loader=test_loader,
        alpha=float(config["alpha"]),
        tie_seed=int(config["evaluation"]["tie_seed"]),
        device=device,
        temperature=temperature,
        method_name=method_name,
        greedy_tau=greedy_tau,
    )
    metrics["temperature"] = float(temperature)
    if greedy_tau is not None:
        metrics["greedy_tau"] = float(greedy_tau)
        posthoc["greedy_tau"] = float(greedy_tau)
    return {
        "model": model.cpu(),
        "history": history,
        "metrics": metrics,
        "indices": splits.indices,
        "posthoc": posthoc,
    }


def save_image_run_artifacts(
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
    save_json(run_dir / "posthoc_calibration.json", results.get("posthoc", {"temperature": 1.0}))
    torch.save(results["model"].state_dict(), run_dir / "checkpoint.pt")
    return run_dir


def run_cifar10_method(config: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    return run_image_method(config, method_name)


def save_cifar10_run_artifacts(
    config: Dict[str, Any],
    method_name: str,
    results: Dict[str, Any],
) -> Path:
    return save_image_run_artifacts(config, method_name, results)
