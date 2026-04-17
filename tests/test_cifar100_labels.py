from __future__ import annotations

from conformal_efficiency.data.cifar100 import _extract_cifar100_targets


def test_extract_cifar100_targets_for_both_modes() -> None:
    entry = {"fine_labels": [1, 2, 3], "coarse_labels": [4, 5, 6]}
    assert _extract_cifar100_targets(entry, label_mode="fine") == [1, 2, 3]
    assert _extract_cifar100_targets(entry, label_mode="coarse") == [4, 5, 6]
