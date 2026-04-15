from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_yaml(path: str | Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
