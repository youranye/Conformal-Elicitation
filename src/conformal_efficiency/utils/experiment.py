from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from conformal_efficiency.conformal.pvalues import (
    candidate_probability_scores,
    exact_prediction_sets,
)
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
) -> Dict[str, float]:
    with torch.no_grad():
        outer_probs = model.predict_proba(splits["outer_calibration"].coordinates)
        outer_labels = splits["outer_calibration"].labels
        calibration_scores = outer_probs[torch.arange(outer_labels.numel()), outer_labels]

        test_probs = model.predict_proba(splits["test"].coordinates)
        candidate_scores = candidate_probability_scores(test_probs)
        generator = torch.Generator().manual_seed(tie_seed)
        set_mask, pvalues = exact_prediction_sets(
            candidate_scores=candidate_scores,
            calibration_scores=calibration_scores,
            alpha=alpha,
            generator=generator,
        )
        metrics = summarize_prediction_sets(set_mask=set_mask, labels=splits["test"].labels, pvalues=pvalues)
    return metrics


def run_method(
    config: Dict[str, Any],
    method_name: str,
) -> Dict[str, Any]:
    set_seed(int(config["seed"]))
    dataset = build_fixed_point_dataset(config["data_config"])
    splits = make_experiment_splits(dataset, config["splits"], seed=int(config["seed"]))
    method_config = config["method_configs"][method_name]
    model = build_model_from_config(dataset, config["model_config"])

    if method_name == "indirect":
        history = train_indirect(
            model=model,
            fit_inputs=splits["fit"].coordinates,
            fit_labels=splits["fit"].labels,
            learning_rate=float(method_config["learning_rate"]),
            epochs=int(method_config["epochs"]),
            weight_decay=float(method_config.get("weight_decay", 0.0)),
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
    else:
        raise ValueError(f"Unknown method: {method_name}")

    metrics = evaluate_model(
        model=model,
        splits=splits,
        alpha=float(config["alpha"]),
        tie_seed=int(config["evaluation"]["tie_seed"]),
    )
    return {"model": model, "history": history, "metrics": metrics, "splits": splits}


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
