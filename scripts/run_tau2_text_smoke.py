#!/usr/bin/env python3
"""Sequential three-domain text smoke for both frozen Gemma endpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.router_data import pair_endpoint_episodes
from agentrelay.schema import canonical_json, sha256_json
from agentrelay.tau2_adapter import TAU2_DOMAINS


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _rows(path: Path) -> list[dict]:
    return list(json.loads(path.read_text(encoding="utf-8"))["rows"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("profile")
    parser.add_argument("task_manifest")
    parser.add_argument("tau2_repository")
    parser.add_argument("user_config")
    parser.add_argument("output_dir")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    endpoint = PROJECT_ROOT / "scripts" / "run_tau2_endpoint.py"
    for role in ("edge", "cloud"):
        command = [
            sys.executable,
            str(endpoint),
            args.config,
            args.profile,
            args.task_manifest,
            args.tau2_repository,
            args.user_config,
            str(root / role),
            "--role",
            role,
            "--split",
            "dev",
            "--tasks-per-domain",
            "1",
            "--as-smoke",
        ]
        if args.validate_only:
            command.append("--validate-only")
        _run(command)
    if args.validate_only:
        print("three-domain text smoke inputs validated; no model/API call was made")
        return 0
    rows = _rows(root / "edge" / "episodes.json") + _rows(
        root / "cloud" / "episodes.json"
    )
    pairs = pair_endpoint_episodes(rows)
    domains = {str(edge["domain"]) for edge, _ in pairs}
    if len(pairs) != 3 or domains != set(TAU2_DOMAINS):
        raise RuntimeError("three-domain smoke did not produce one paired task per domain")
    for edge, cloud in pairs:
        edge_features = edge["steps"][0]["router_features"]
        cloud_features = cloud["steps"][0]["router_features"]
        if edge_features != cloud_features:
            raise RuntimeError("fixed user simulator produced different step-zero inputs")
        if not edge["steps"][0]["prompt_hash"] or not cloud["steps"][0]["prompt_hash"]:
            raise RuntimeError("text-mode smoke contains an empty native prompt")
    report = {
        "scope": "tau2_three_domain_text_smoke",
        "paper_evidence": False,
        "domains": sorted(domains),
        "paired_tasks": len(pairs),
        "edge_mean_reward": sum(edge["reward"] for edge, _ in pairs) / len(pairs),
        "cloud_mean_reward": sum(cloud["reward"] for _, cloud in pairs) / len(pairs),
        "rows_hash": sha256_json(rows),
    }
    report["report_hash"] = sha256_json(report)
    target = root / "smoke-report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(canonical_json(report) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
