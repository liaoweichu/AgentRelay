#!/usr/bin/env python3
"""Fail closed if a local/formal runtime violates declared constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import os


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
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_json_config(args.config)
    validate_experiment_config(config)
    storage = StorageLayout(Path(config["data_root"]).resolve())
    storage.create()

    try:
        import torch
        import transformers
        import datasets
    except ImportError as exc:
        raise RuntimeError("formal preflight requires AgentRelay[ml]") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native model inference")
    gpu_name = torch.cuda.get_device_name(0)
    if config["run_mode"] == "formal_autodl" and "4090" not in gpu_name:
        raise RuntimeError(f"formal configuration requires a 4090-class GPU, found {gpu_name}")
    if config["run_mode"] == "formal_autodl":
        root = storage.root.resolve()
        for variable in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "PIP_CACHE_DIR", "TMPDIR"):
            value = os.environ.get(variable)
            if not value:
                raise RuntimeError(f"formal preflight requires {variable} to be set")
            try:
                Path(value).resolve().relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"{variable} must stay under {root}, found {value}") from exc
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        if total_memory < 20 * 1024**3:
            raise RuntimeError(
                f"formal configuration requires at least 20 GiB GPU memory, found {total_memory}"
            )
        if any(str(model.get("quantization", "")) in {"bnb_4bit", "bnb_8bit"} for model in config["models"].values()):
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("formal quantized models require bitsandbytes") from exc
    total, used, free = shutil.disk_usage(storage.root)
    report = {
        "config": str(Path(args.config).resolve()),
        "mode": config["run_mode"],
        "paper_evidence": config["paper_evidence"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "gpu": gpu_name,
        "cuda": torch.version.cuda,
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "data_root": str(storage.root),
        "disk_free_bytes": free,
        "disk_used_bytes": used,
        "disk_total_bytes": total,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(target)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
