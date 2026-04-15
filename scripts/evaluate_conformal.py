#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conformal_efficiency.data.fixed_points import build_fixed_point_dataset, make_experiment_splits
from conformal_efficiency.models.mlp import MLPClassifier
from conformal_efficiency.utils.config import load_experiment_config
from conformal_efficiency.utils.experiment import evaluate_model
from conformal_efficiency.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=["indirect", "direct"], required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    set_seed(int(config["seed"]))
    dataset = build_fixed_point_dataset(config["data_config"])
    splits = make_experiment_splits(dataset, config["splits"], seed=int(config["seed"]))
    model = MLPClassifier(
        input_dim=int(config["model_config"]["input_dim"]),
        hidden_dims=list(config["model_config"]["hidden_dims"]),
        num_classes=dataset.num_classes,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    metrics = evaluate_model(
        model=model,
        splits=splits,
        alpha=float(config["alpha"]),
        tie_seed=int(config["evaluation"]["tie_seed"]),
    )
    print(metrics)


if __name__ == "__main__":
    main()
