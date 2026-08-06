#!/usr/bin/env python3
"""Software and cloud preflight for the pinned tau2/Gemma experiment path."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import (
    GEMMA4_FORMAL_MODEL_PAIR,
    load_json_config,
    validate_experiment_config,
    validate_gemma4_model_pair,
)
from agentrelay.schema import sha256_json
from agentrelay.tau2_adapter import (
    TAU2_DOMAINS,
    TAU2_PINNED_REVISION,
    Tau2UserSimulatorConfig,
    official_tau2_split_ids,
    verify_tau2_repository,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("tau2_repository")
    parser.add_argument("user_config")
    parser.add_argument("--software-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_json_config(args.config)
    validate_experiment_config(config)
    validate_gemma4_model_pair(config["models"])
    model_ids = {role: value["model_id"] for role, value in config["models"].items()}
    if model_ids != GEMMA4_FORMAL_MODEL_PAIR:
        raise ValueError("preflight requires the frozen Gemma 4 E4B/12B pair")
    if any(model.get("model_source") != "modelscope" for model in config["models"].values()):
        raise ValueError("both formal models must use ModelScope")
    repository_entry = next(
        (item for item in config.get("repositories", ()) if item.get("name") == "tau2-bench"),
        None,
    )
    if not repository_entry or repository_entry.get("revision") != TAU2_PINNED_REVISION:
        raise ValueError("locked config does not pin the authorized tau2-bench revision")
    repository = verify_tau2_repository(args.tau2_repository)
    user = Tau2UserSimulatorConfig.load(
        args.user_config,
        require_secret=not args.software_only,
    )
    split_ids = official_tau2_split_ids(args.tau2_repository)
    counts = {
        domain: {split: len(ids) for split, ids in split_ids[domain].items()}
        for domain in TAU2_DOMAINS
    }
    if set(counts) != set(TAU2_DOMAINS) or any(
        value["train"] <= 0 or value["test"] <= 0 for value in counts.values()
    ):
        raise ValueError("tau2 three-domain official splits are unavailable")
    try:
        modelscope_version = importlib.metadata.version("modelscope")
    except importlib.metadata.PackageNotFoundError:
        modelscope_version = None
    if not args.software_only and modelscope_version != "1.39.1":
        raise RuntimeError(f"cloud preflight requires modelscope==1.39.1, found {modelscope_version}")
    report = {
        "software_only": args.software_only,
        "config_hash": sha256_json(config),
        "model_ids": model_ids,
        "model_revisions": {
            role: value["revision"] for role, value in config["models"].items()
        },
        "model_precisions": {
            role: {
                "dtype": value["dtype"],
                "quantization": value["quantization"],
            }
            for role, value in config["models"].items()
        },
        "modelscope_version": modelscope_version,
        "tau2_repository": repository,
        "tau2_counts": counts,
        "user_simulator": user.provenance(),
        "held_out_test_accessed_for_router": False,
    }
    if args.output:
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
