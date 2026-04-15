from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class CIFAR10Splits:
    fit: Subset
    inner_calibration: Subset
    outer_calibration: Subset
    test: datasets.CIFAR10
    indices: Dict[str, list[int]]


def _build_transforms(mean: list[float], std: list[float]) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_cifar10_splits(config: Dict, split_cfg: Dict, seed: int) -> CIFAR10Splits:
    root = Path(config["root"]).expanduser()
    mean = list(config["mean"])
    std = list(config["std"])
    download = bool(config.get("download", False))
    train_transform, eval_transform = _build_transforms(mean, std)

    train_aug = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=download)
    train_eval = datasets.CIFAR10(root=root, train=True, transform=eval_transform, download=download)
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=eval_transform, download=download)

    fit_size = int(split_cfg["fit_size"])
    inner_size = int(split_cfg["inner_calibration_size"])
    outer_size = int(split_cfg["outer_calibration_size"])
    total_requested = fit_size + inner_size + outer_size
    total_train = len(train_aug)
    if total_requested > total_train:
        raise ValueError(
            f"Requested {total_requested} training examples but CIFAR-10 train split has only {total_train}"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_train)
    fit_idx = perm[:fit_size].tolist()
    inner_idx = perm[fit_size : fit_size + inner_size].tolist()
    outer_idx = perm[fit_size + inner_size : fit_size + inner_size + outer_size].tolist()

    return CIFAR10Splits(
        fit=Subset(train_aug, fit_idx),
        inner_calibration=Subset(train_eval, inner_idx),
        outer_calibration=Subset(train_eval, outer_idx),
        test=test_dataset,
        indices={
            "fit": fit_idx,
            "inner_calibration": inner_idx,
            "outer_calibration": outer_idx,
        },
    )


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
