#!/usr/bin/env python3
"""Sequential tau2 endpoint collection and reward-aware router learnability gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.schema import canonical_json, sha256_json


def _run(command: list[str], *, allow_gate_failure: bool = False) -> int:
    print(" ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode and not (allow_gate_failure and result.returncode == 2):
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def _rows(path: Path) -> list[dict]:
    return list(json.loads(path.read_text(encoding="utf-8"))["rows"])


def _write_rows(path: Path, rows: list[dict]) -> None:
    payload = {"rows": rows, "rows_hash": sha256_json(rows)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("profile")
    parser.add_argument("task_manifest")
    parser.add_argument("tau2_repository")
    parser.add_argument("user_config")
    parser.add_argument("output_dir")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--minimum-oracle-capture", type=float, default=0.30)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    endpoint_script = PROJECT_ROOT / "scripts" / "run_tau2_endpoint.py"
    for split in ("train", "dev"):
        for role in ("edge", "cloud"):
            command = [
                sys.executable,
                str(endpoint_script),
                args.config,
                args.profile,
                args.task_manifest,
                args.tau2_repository,
                args.user_config,
                str(root / split / role),
                "--role",
                role,
                "--split",
                split,
            ]
            if args.validate_only:
                command.append("--validate-only")
            _run(command)
    if args.validate_only:
        print("tau2 train/dev gate inputs validated; no model/API call was made")
        return 0
    train_rows = _rows(root / "train" / "edge" / "episodes.json") + _rows(
        root / "train" / "cloud" / "episodes.json"
    )
    dev_rows = _rows(root / "dev" / "edge" / "episodes.json") + _rows(
        root / "dev" / "cloud" / "episodes.json"
    )
    train_artifact = root / "train-episodes.json"
    dev_artifact = root / "dev-episodes.json"
    _write_rows(train_artifact, train_rows)
    _write_rows(dev_artifact, dev_rows)
    gate_code = _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_router_learnability_gate.py"),
            str(train_artifact),
            str(dev_artifact),
            str(root / "reward-aware-router.json"),
            str(root / "learnability-gate.json"),
            "--minimum-train-tasks",
            "120",
            "--minimum-paired-tasks",
            "50",
            "--minimum-oracle-capture",
            str(args.minimum_oracle_capture),
        ],
        allow_gate_failure=True,
    )
    gate = json.loads((root / "learnability-gate.json").read_text(encoding="utf-8"))
    receipt = {
        "scope": "tau2_train_dev_reward_router",
        "paper_evidence": False,
        "train_endpoint_episodes": len(train_rows),
        "dev_endpoint_episodes": len(dev_rows),
        "train_rows_hash": sha256_json(train_rows),
        "dev_rows_hash": sha256_json(dev_rows),
        "gate_hash": gate["gate_hash"],
        "gate_pass": gate["gate_pass"],
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    target = root / "run-receipt.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return gate_code


if __name__ == "__main__":
    raise SystemExit(main())
