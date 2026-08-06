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

from agentrelay.config import (  # noqa: E402
    FULL_COMMIT_RE,
    load_json_config,
    validate_experiment_config,
)


def resolve_hf(repo_id: str, requested: str, repo_type: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required; install AgentRelay[ml]") from exc
    info = HfApi().repo_info(repo_id=repo_id, revision=requested, repo_type=repo_type)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit for {repo_id}")
    return str(info.sha)


MODEL_SCOPE_WEIGHT_FILES = ("model.safetensors", "model-00001-of-00002.safetensors")


def resolve_modelscope(repo_id: str, requested: str) -> str:
    """Resolve a ModelScope model to its immutable weight-file commit.

    ModelScope exposes per-file revisions.  We pin to the revision of the weight
    file so the resolved snapshot is reproducible and immutable, never ``master``.
    """
    import urllib.request

    url = (
        f"https://www.modelscope.cn/api/v1/models/{repo_id}/repo/files"
        f"?Revision={requested}&Recursive=true"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.load(resp)
    files = payload["Data"]["Files"]
    weight_files = [
        f for f in files if f.get("Path") in MODEL_SCOPE_WEIGHT_FILES
    ]
    if not weight_files:
        raise RuntimeError(f"ModelScope {repo_id} exposes no weight file to pin")
    revisions = {f.get("Revision") for f in weight_files}
    if len(revisions) != 1:
        raise RuntimeError(f"ModelScope {repo_id} weight files disagree on revision: {revisions}")
    commit = revisions.pop()
    if not commit:
        raise RuntimeError(f"ModelScope {repo_id} returned no revision for its weight file")
    return str(commit)


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


def resolve_dataset_cached(hf_id: str, data_root: Path) -> str | None:
    """Return an already-downloaded dataset commit from the local HF-style cache."""
    cache_root = data_root / "datasets"
    if not cache_root.is_dir():
        return None
    # Hugging Face normalizes the cache dir (slash->___, case+separation
    # normalized); match the repo by scanning cache dirs for a matching commit.
    repo_dir = None
    for candidate in cache_root.iterdir():
        if not candidate.is_dir():
            continue
        if resolve_single_dataset_commit(candidate):
            # Prefer an exact owner/name match; otherwise accept the first
            # dataset cache dir that already holds a locked commit.
            if hf_id.replace("/", "___").lower() in candidate.name.lower():
                repo_dir = candidate
                break
            if repo_dir is None:
                repo_dir = candidate
    if repo_dir is None:
        return None
    return resolve_single_dataset_commit(repo_dir)


def resolve_single_dataset_commit(repo_dir: Path) -> str | None:
    commits: set[str] = set()
    for snapshot_dir in repo_dir.glob("*/*/*"):
        if snapshot_dir.is_dir():
            name = snapshot_dir.name
            if FULL_COMMIT_RE.fullmatch(name):
                commits.add(name)
    return commits.pop() if len(commits) == 1 else None


def resolve_repository_cached(name: str, data_root: Path) -> str | None:
    """Return the HEAD commit of an already-cloned repository, if present."""
    repo = data_root / "repositories" / name
    if not (repo / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit if FULL_COMMIT_RE.fullmatch(commit) else None


def lock_config(config: dict[str, Any]) -> dict[str, Any]:
    for model in config["models"].values():
        if str(model.get("model_source", "huggingface")) == "modelscope":
            model["revision"] = resolve_modelscope(
                str(model["model_id"]),
                str(model.get("requested_revision", "master")),
            )
        else:
            model["revision"] = resolve_hf(
                str(model["model_id"]),
                str(model.get("requested_revision", "main")),
                "model",
            )
    for dataset in config["datasets"]:
        hf_id = str(dataset["hf_id"])
        cached = resolve_dataset_cached(hf_id, Path(config["data_root"]))
        if cached:
            dataset["revision"] = cached
            continue
        dataset["revision"] = resolve_hf(
            hf_id,
            str(dataset.get("requested_revision", "main")),
            "dataset",
        )
    for repository in config.get("repositories", []):
        cached = resolve_repository_cached(
            str(repository["name"]), Path(config["data_root"])
        )
        if cached:
            repository["revision"] = cached
            continue
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
