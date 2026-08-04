#!/usr/bin/env python3
"""Clone official benchmark repositories at the locked commits only."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import load_json_config, validate_experiment_config  # noqa: E402


def run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_json_config(args.config)
    validate_experiment_config(config)
    destination_root = Path(config["data_root"]).resolve() / "repositories"
    destination_root.mkdir(parents=True, exist_ok=True)
    for repository in config.get("repositories", []):
        target = destination_root / repository["name"]
        if not target.exists():
            run(["git", "clone", "--no-checkout", repository["url"], str(target)])
        if not (target / ".git").is_dir():
            raise RuntimeError(f"existing destination is not a git repository: {target}")
        origin = run(["git", "remote", "get-url", "origin"], cwd=target)
        if origin.rstrip("/") != str(repository["url"]).rstrip("/"):
            raise RuntimeError(f"origin mismatch for {target}: {origin}")
        revision = str(repository["revision"])
        run(["git", "fetch", "--depth", "1", "origin", revision], cwd=target)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target)
        actual = run(["git", "rev-parse", "HEAD"], cwd=target)
        if actual != revision:
            raise RuntimeError(f"revision mismatch for {target}: {actual} != {revision}")
        print(f"{repository['name']} revision={actual} path={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
