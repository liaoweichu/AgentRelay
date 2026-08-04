#!/usr/bin/env python3
"""Download only the pinned public Hugging Face datasets in a locked config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import (  # noqa: E402
    StorageLayout,
    load_json_config,
    validate_experiment_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_json_config(args.config)
    validate_experiment_config(config)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install AgentRelay with the 'ml' extra") from exc

    storage = StorageLayout(Path(config["data_root"]).resolve())
    storage.create()
    for dataset in config["datasets"]:
        for split in dataset["splits"]:
            loaded = load_dataset(
                dataset["hf_id"],
                dataset.get("config_name"),
                split=split,
                revision=dataset["revision"],
                cache_dir=str(storage.datasets),
                trust_remote_code=False,
            )
            print(
                f"{dataset['name']} split={split} rows={len(loaded)} "
                f"config={dataset.get('config_name')} revision={dataset['revision']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
