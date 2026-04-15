from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    root = Path(path).resolve().parent.parent.parent
    config = load_yaml(path)
    data_cfg = load_yaml(root / config["data"]["path"])
    config["data_config"] = data_cfg
    model_cfg = load_yaml(root / config["model"]["path"])
    config["model_config"] = model_cfg

    method_cfgs: Dict[str, Dict[str, Any]] = {}
    for name, spec in config["methods"].items():
        method_cfgs[name] = load_yaml(root / spec["path"])
    config["method_configs"] = method_cfgs
    return config
