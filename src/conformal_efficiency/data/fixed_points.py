from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class ToySplit:
    point_ids: torch.Tensor
    coordinates: torch.Tensor
    labels: torch.Tensor


@dataclass
class FixedPointDataset:
    coordinates: np.ndarray
    px: np.ndarray
    p_true: np.ndarray
    num_points: int
    num_classes: int
    input_dim: int


def build_fixed_point_dataset(config: Dict) -> FixedPointDataset:
    coordinates = np.asarray(config["point_coordinates"], dtype=np.float32)
    px = np.asarray(config["px"], dtype=np.float64)
    p_true = np.asarray(config["p_true"], dtype=np.float64)
    num_points = int(config["num_points"])
    num_classes = int(config["num_classes"])

    if coordinates.shape[0] != num_points:
        raise ValueError("point_coordinates length must match num_points")
    if px.shape[0] != num_points:
        raise ValueError("px length must match num_points")
    if p_true.shape != (num_points, num_classes):
        raise ValueError("p_true must have shape (num_points, num_classes)")
    if not np.isclose(px.sum(), 1.0):
        raise ValueError("px must sum to 1")
    if not np.allclose(p_true.sum(axis=1), 1.0):
        raise ValueError("each p_true row must sum to 1")

    return FixedPointDataset(
        coordinates=coordinates,
        px=px,
        p_true=p_true,
        num_points=num_points,
        num_classes=num_classes,
        input_dim=int(coordinates.shape[1]),
    )


def sample_split(dataset: FixedPointDataset, size: int, rng: np.random.Generator) -> ToySplit:
    point_ids = rng.choice(dataset.num_points, size=size, p=dataset.px)
    labels = np.array(
        [rng.choice(dataset.num_classes, p=dataset.p_true[idx]) for idx in point_ids],
        dtype=np.int64,
    )
    coordinates = dataset.coordinates[point_ids]
    return ToySplit(
        point_ids=torch.tensor(point_ids, dtype=torch.long),
        coordinates=torch.tensor(coordinates, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def make_experiment_splits(dataset: FixedPointDataset, split_cfg: Dict, seed: int) -> Dict[str, ToySplit]:
    rng = np.random.default_rng(seed)
    return {
        "fit": sample_split(dataset, int(split_cfg["fit_size"]), rng),
        "inner_calibration": sample_split(dataset, int(split_cfg["inner_calibration_size"]), rng),
        "outer_calibration": sample_split(dataset, int(split_cfg["outer_calibration_size"]), rng),
        "test": sample_split(dataset, int(split_cfg["test_size"]), rng),
    }
