#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conformal_efficiency.utils.config import load_experiment_config
from conformal_efficiency.utils.experiment import run_method, save_run_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    results = run_method(config, method_name="indirect")
    run_dir = save_run_artifacts(config, method_name="indirect", results=results)
    print(f"saved indirect run to {run_dir}")
    print(results["metrics"])


if __name__ == "__main__":
    main()
