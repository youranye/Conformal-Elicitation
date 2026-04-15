#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conformal_efficiency.utils.config import load_experiment_config
from conformal_efficiency.utils.image_experiment import (
    run_cifar10_method,
    save_cifar10_run_artifacts,
)
from conformal_efficiency.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    summary = {}
    for method_name in ["indirect", "direct"]:
        results = run_cifar10_method(config, method_name=method_name)
        run_dir = save_cifar10_run_artifacts(config, method_name=method_name, results=results)
        summary[method_name] = {
            "run_dir": str(run_dir),
            "metrics": results["metrics"],
        }

    summary_dir = ensure_dir(Path(config["output_dir"]) / config["experiment_name"])
    with open(summary_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
