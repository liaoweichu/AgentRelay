#!/usr/bin/env python3
"""Resolve mutable HF/Git references and emit an immutable run config."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import load_json_config, validate_experiment_config  # noqa: E402


def resolve_hf(repo_id: str, requested: str, repo_type: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required; install AgentRelay[ml]") from exc
    info = HfApi().repo_info(repo_id=repo_id, revision=requested, repo_type=repo_type)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit for {repo_id}")
    return str(info.sha)


def resolve_git(url: str, requested: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", url, requested],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"cannot resolve {requested!r} in {url}")
    commits = {row[0] for row in rows}
    if len(commits) != 1:
        raise RuntimeError(f"reference {requested!r} in {url} is ambiguous")
    return commits.pop()


def lock_config(config: dict[str, Any]) -> dict[str, Any]:
    for model in config["models"].values():
        model["revision"] = resolve_hf(
            str(model["model_id"]),
            str(model.get("requested_revision", "main")),
            "model",
        )
    for dataset in config["datasets"]:
        dataset["revision"] = resolve_hf(
            str(dataset["hf_id"]),
            str(dataset.get("requested_revision", "main")),
            "dataset",
        )
    for repository in config.get("repositories", []):
        repository["revision"] = resolve_git(
            str(repository["url"]),
            str(repository.get("requested_revision", "HEAD")),
        )
    config["locked_at"] = datetime.now(timezone.utc).isoformat()
    validate_experiment_config(config)
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("template")
    parser.add_argument("output")
    args = parser.parse_args()
    source = load_json_config(args.template)
    validate_experiment_config(source, allow_unlocked=True)
    locked = lock_config(source)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(locked, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
