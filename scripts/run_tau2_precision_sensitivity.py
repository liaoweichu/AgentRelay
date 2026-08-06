#!/usr/bin/env python3
"""Sequential E4B NF4/BF16-compute versus FP16 diagnostic on identical tau2 tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.schema import canonical_json, sha256_json


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _rows(path: Path) -> list[dict]:
    return list(json.loads(path.read_text(encoding="utf-8"))["rows"])


def _mean(rows: list[dict], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def _step_mean(rows: list[dict], field: str) -> float:
    values = [float(step[field]) for row in rows for step in row["steps"]]
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("profile")
    parser.add_argument("task_manifest")
    parser.add_argument("tau2_repository")
    parser.add_argument("user_config")
    parser.add_argument("output_dir")
    parser.add_argument("--tasks-per-domain", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    endpoint = PROJECT_ROOT / "scripts" / "run_tau2_endpoint.py"
    arms = (("e4b-bnb4bit", "formal"), ("e4b-fp16", "fp16"))
    for name, precision in arms:
        command = [
            sys.executable,
            str(endpoint),
            args.config,
            args.profile,
            args.task_manifest,
            args.tau2_repository,
            args.user_config,
            str(root / name),
            "--role",
            "edge",
            "--split",
            "dev",
            "--tasks-per-domain",
            str(args.tasks_per_domain),
            "--as-smoke",
            "--edge-precision",
            precision,
        ]
        if args.validate_only:
            command.append("--validate-only")
        _run(command)
    if args.validate_only:
        print("E4B precision sensitivity inputs validated; no weights were loaded")
        return 0
    quantized = _rows(root / "e4b-bnb4bit" / "episodes.json")
    fp16 = _rows(root / "e4b-fp16" / "episodes.json")
    by_key = lambda rows: {row["sample_id"]: row for row in rows}
    quantized_by_key, fp16_by_key = by_key(quantized), by_key(fp16)
    if set(quantized_by_key) != set(fp16_by_key):
        raise RuntimeError("E4B precision arms contain different task sets")
    for key in quantized_by_key:
        if (
            quantized_by_key[key]["steps"][0]["router_features"]
            != fp16_by_key[key]["steps"][0]["router_features"]
        ):
            raise RuntimeError(f"precision arms received different visible input for {key}")
    report = {
        "scope": "tau2_e4b_precision_sensitivity",
        "paper_evidence": False,
        "tasks": len(quantized),
        "bnb4bit_bfloat16_compute": {
            "mean_reward": _mean(quantized, "reward"),
            "mean_inference_ms": _step_mean(quantized, "inference_ms"),
            "peak_cuda_memory_bytes": max(
                step["peak_cuda_memory_bytes"] for row in quantized for step in row["steps"]
            ),
        },
        "fp16": {
            "mean_reward": _mean(fp16, "reward"),
            "mean_inference_ms": _step_mean(fp16, "inference_ms"),
            "peak_cuda_memory_bytes": max(
                step["peak_cuda_memory_bytes"] for row in fp16 for step in row["steps"]
            ),
        },
        "paired_reward_difference": _mean(fp16, "reward") - _mean(quantized, "reward"),
        "quantized_rows_hash": sha256_json(quantized),
        "fp16_rows_hash": sha256_json(fp16),
    }
    report["report_hash"] = sha256_json(report)
    target = root / "precision-sensitivity-report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(canonical_json(report) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
