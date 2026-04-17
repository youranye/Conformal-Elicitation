from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from conformal_efficiency.calibration.temperature import apply_temperature, fit_temperature_from_logits
from conformal_efficiency.conformal.greedy import calibrate_greedy_tau, greedy_cumulative_mass_sets
from conformal_efficiency.conformal.pvalues import candidate_probability_scores, exact_prediction_sets
from conformal_efficiency.data.fixed_points import build_fixed_point_dataset, make_experiment_splits
from conformal_efficiency.evaluation.metrics import summarize_prediction_sets
from conformal_efficiency.models.mlp import MLPClassifier
from conformal_efficiency.trainers.direct import train_direct
from conformal_efficiency.trainers.indirect import train_indirect
from conformal_efficiency.utils.io import ensure_dir, save_json, save_yaml
from conformal_efficiency.utils.seed import set_seed


def build_model_from_config(dataset, model_config) -> MLPClassifier:
    input_dim = int(model_config.get("input_dim", dataset.input_dim))
    if input_dim != dataset.input_dim:
        raise ValueError("model input_dim must match dataset coordinate dimension")
    return MLPClassifier(
        input_dim=input_dim,
        hidden_dims=list(model_config["hidden_dims"]),
        num_classes=dataset.num_classes,
    )


def evaluate_model(
    model: MLPClassifier,
    splits: Dict[str, Any],
    alpha: float,
    tie_seed: int,
    temperature: float = 1.0,
    method_name: str = "indirect",
    greedy_tau: float | None = None,
) -> Dict[str, float]:
    with torch.no_grad():
        outer_logits = model(splits["outer_calibration"].coordinates)
        outer_probs = torch.softmax(apply_temperature(outer_logits, temperature), dim=-1)
        outer_labels = splits["outer_calibration"].labels
        calibration_scores = outer_probs[torch.arange(outer_labels.numel()), outer_labels]

        test_logits = model(splits["test"].coordinates)
        test_probs = torch.softmax(apply_temperature(test_logits, temperature), dim=-1)

        if method_name in {"indirect", "direct"}:
            candidate_scores = candidate_probability_scores(test_probs)
            generator = torch.Generator().manual_seed(tie_seed)
            set_mask, ranking_scores = exact_prediction_sets(
                candidate_scores=candidate_scores,
                calibration_scores=calibration_scores,
                alpha=alpha,
                generator=generator,
            )
        elif method_name in {"greedy_mass_fixed", "greedy_mass_calibrated"}:
            tau = (1.0 - alpha) if method_name == "greedy_mass_fixed" else float(greedy_tau)
            set_mask = greedy_cumulative_mass_sets(test_probs, tau=tau)
            ranking_scores = test_probs
        else:
            raise ValueError(f"Unsupported evaluation method: {method_name}")

        metrics = summarize_prediction_sets(
            set_mask=set_mask,
            labels=splits["test"].labels,
            pvalues=ranking_scores,
            probabilities=test_probs,
            alpha=alpha,
        )
    return metrics


def _train_indirect_family(
    model: MLPClassifier,
    splits: Dict[str, Any],
    method_config: Dict[str, Any],
) -> tuple[list[dict[str, float]], float, dict[str, float]]:
    history = train_indirect(
        model=model,
        fit_inputs=splits["fit"].coordinates,
        fit_labels=splits["fit"].labels,
        learning_rate=float(method_config["learning_rate"]),
        epochs=int(method_config["epochs"]),
        weight_decay=float(method_config.get("weight_decay", 0.0)),
    )

    if bool(method_config.get("posthoc_temperature_scaling", False)):
        with torch.no_grad():
            inner_logits = model(splits["inner_calibration"].coordinates)
        temperature, posthoc = fit_temperature_from_logits(
            logits=inner_logits,
            labels=splits["inner_calibration"].labels,
            max_iter=int(method_config.get("temperature_max_iter", 50)),
        )
    else:
        temperature = 1.0
        posthoc = {"temperature": 1.0}
    return history, float(temperature), posthoc


def run_method(
    config: Dict[str, Any],
    method_name: str,
) -> Dict[str, Any]:
    set_seed(int(config["seed"]))
    dataset = build_fixed_point_dataset(config["data_config"])
    splits = make_experiment_splits(dataset, config["splits"], seed=int(config["seed"]))
    method_config = config["method_configs"][method_name]
    model = build_model_from_config(dataset, config["model_config"])
    posthoc = {"temperature": 1.0}
    greedy_tau = None

    if method_name in {"indirect", "greedy_mass_fixed", "greedy_mass_calibrated"}:
        history, temperature, posthoc = _train_indirect_family(
            model=model,
            splits=splits,
            method_config=method_config,
        )
        if method_name == "greedy_mass_calibrated":
            with torch.no_grad():
                outer_logits = model(splits["outer_calibration"].coordinates)
                outer_probs = torch.softmax(apply_temperature(outer_logits, temperature), dim=-1)
            greedy_tau = calibrate_greedy_tau(
                probabilities=outer_probs,
                labels=splits["outer_calibration"].labels,
                alpha=float(config["alpha"]),
            )
    elif method_name == "direct":
        history = train_direct(
            model=model,
            fit_inputs=splits["fit"].coordinates,
            inner_calibration_inputs=splits["inner_calibration"].coordinates,
            inner_calibration_labels=splits["inner_calibration"].labels,
            alpha=float(config["alpha"]),
            learning_rate=float(method_config["learning_rate"]),
            epochs=int(method_config["epochs"]),
            tau_rank=float(method_config["tau_rank"]),
            tau_set=float(method_config["tau_set"]),
            weight_decay=float(method_config.get("weight_decay", 0.0)),
        )
        temperature = 1.0
    else:
        raise ValueError(f"Unknown method: {method_name}")

    metrics = evaluate_model(
        model=model,
        splits=splits,
        alpha=float(config["alpha"]),
        tie_seed=int(config["evaluation"]["tie_seed"]),
        temperature=temperature,
        method_name=method_name,
        greedy_tau=greedy_tau,
    )
    metrics["temperature"] = float(temperature)
    if greedy_tau is not None:
        metrics["greedy_tau"] = float(greedy_tau)
        posthoc["greedy_tau"] = float(greedy_tau)
    return {"model": model, "history": history, "metrics": metrics, "splits": splits, "posthoc": posthoc}


def save_run_artifacts(
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
    save_json(run_dir / "posthoc_calibration.json", results.get("posthoc", {"temperature": 1.0}))
    torch.save(results["model"].state_dict(), run_dir / "checkpoint.pt")

    split_payload = {
        split_name: {
            "point_ids": split.point_ids.tolist(),
            "coordinates": split.coordinates.tolist(),
            "labels": split.labels.tolist(),
        }
        for split_name, split in results["splits"].items()
    }
    save_json(run_dir / "splits.json", split_payload)
    return run_dir
