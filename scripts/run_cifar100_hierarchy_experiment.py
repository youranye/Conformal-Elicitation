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
from conformal_efficiency.utils.image_experiment import run_image_method, save_image_run_artifacts
from conformal_efficiency.utils.io import ensure_dir


DEFAULT_CONFIGS = [
    "configs/experiment/cifar100_fine_resnet18_v1.yaml",
    "configs/experiment/cifar100_coarse_resnet18_v1.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    hierarchy_summary = {}
    for config_path in args.configs:
        config = load_experiment_config(config_path)
        experiment_summary = {}
        for method_name in config["methods"].keys():
            results = run_image_method(config, method_name=method_name)
            run_dir = save_image_run_artifacts(config, method_name=method_name, results=results)
            experiment_summary[method_name] = {
                "run_dir": str(run_dir),
                "metrics": results["metrics"],
            }
        hierarchy_summary[config["experiment_name"]] = experiment_summary

        summary_dir = ensure_dir(Path(config["output_dir"]) / config["experiment_name"])
        with open(summary_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(experiment_summary, handle, indent=2)

    hierarchy_dir = ensure_dir(Path("outputs") / "cifar100_hierarchy_v1")
    with open(hierarchy_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(hierarchy_summary, handle, indent=2)
    print(json.dumps(hierarchy_summary, indent=2))


if __name__ == "__main__":
    main()
