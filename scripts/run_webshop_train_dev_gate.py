#!/usr/bin/env python3
"""Run the complete leakage-safe WebShop train/dev router gate sequentially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.formal_matrix import OfficialTaskManifest  # noqa: E402
from agentrelay.config import (  # noqa: E402
    GEMMA4_FORMAL_MODEL_PAIR,
    load_json_config,
    validate_experiment_config,
)
from agentrelay.schema import canonical_json, sha256_json, sha256_text  # noqa: E402


def _run_streaming(
    command: Iterable[str],
    *,
    accepted_returncodes: tuple[int, ...] = (0,),
) -> tuple[int, tuple[str, ...]]:
    rendered = tuple(str(item) for item in command)
    print(f"command={canonical_json(rendered)}", flush=True)
    process = subprocess.Popen(
        rendered,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\r\n")
        print(line, flush=True)
        lines.append(line)
    returncode = process.wait()
    if returncode not in accepted_returncodes:
        raise RuntimeError(
            f"command failed with status {returncode}: {canonical_json(rendered)}"
        )
    return returncode, tuple(lines)


def _extract_run_directory(lines: Iterable[str]) -> Path:
    matches = [line.split("=", 1)[1] for line in lines if line.startswith("run_directory=")]
    if len(matches) != 1:
        raise ValueError("matrix command did not emit exactly one run_directory")
    root = Path(matches[0]).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        raise ValueError(f"matrix run has no completed manifest: {root}")
    return root


def _load_run_manifest(root: Path, *, expected_method: str) -> dict:
    path = root / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded_hash = str(payload.pop("manifest_hash", ""))
    if not recorded_hash or sha256_json(payload) != recorded_hash:
        raise ValueError(f"run manifest hash mismatch: {path}")
    if value.get("methods") != [expected_method]:
        raise ValueError(f"resume run method mismatch: {root}")
    if value.get("paper_evidence") is not False:
        raise ValueError(f"resume run is not a train/dev diagnostic: {root}")
    if value.get("model_ids") != GEMMA4_FORMAL_MODEL_PAIR:
        raise ValueError(f"resume run does not use the frozen Gemma 4 model pair: {root}")
    return value


def _write_receipt(path: Path, value: dict) -> None:
    payload = dict(value)
    payload.pop("receipt_hash", None)
    payload["receipt_hash"] = sha256_json(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)
    value.clear()
    value.update(payload)


def _load_receipt(path: Path, *, plan_hash: str) -> dict:
    if not path.exists():
        return {
            "schema_version": "1.0",
            "scope": "webshop_official_train_dev_learnability_gate",
            "paper_evidence": False,
            "plan_hash": plan_hash,
            "runs": {},
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    recorded_hash = str(payload.pop("receipt_hash", ""))
    if not recorded_hash or sha256_json(payload) != recorded_hash:
        raise ValueError("gate receipt hash mismatch")
    if value.get("plan_hash") != plan_hash:
        raise ValueError(
            "existing gate receipt belongs to different inputs; use a new output directory"
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--webshop-file", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", default="results/webshop-train-dev-gate")
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--dev-count", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=20260805)
    parser.add_argument("--dev-seed", type=int, default=20260806)
    args = parser.parse_args()
    if args.train_count < 200:
        raise ValueError("the predeclared learnability gate requires at least 200 train tasks")
    if args.dev_count < 100:
        raise ValueError("the predeclared learnability gate requires at least 100 dev tasks")

    config = Path(args.config).expanduser().resolve()
    profile = Path(args.profile).expanduser().resolve()
    webshop_file = Path(args.webshop_file).expanduser().resolve()
    for required in (config, profile, webshop_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    locked_config = load_json_config(config)
    validate_experiment_config(locked_config)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "revision": args.revision,
        "config_hash": sha256_text(config.read_text(encoding="utf-8")),
        "profile_hash": sha256_text(profile.read_text(encoding="utf-8")),
        "model_ids": dict(GEMMA4_FORMAL_MODEL_PAIR),
        "webshop_file": str(webshop_file),
        "train_count": args.train_count,
        "dev_count": args.dev_count,
        "train_seed": args.train_seed,
        "dev_seed": args.dev_seed,
    }
    plan_hash = sha256_json(plan)
    receipt_path = output / "receipt.json"
    receipt = _load_receipt(receipt_path, plan_hash=plan_hash)
    receipt["plan"] = plan
    _write_receipt(receipt_path, receipt)

    manifests = {
        "train": output / f"webshop-train-{args.train_count}.json",
        "dev": output / f"webshop-dev-{args.dev_count}.json",
    }
    specifications = {
        "train": ("train", args.train_count, args.train_seed),
        "dev": ("tune", args.dev_count, args.dev_seed),
    }
    for split, (purpose, count, seed) in specifications.items():
        _run_streaming(
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_official_task_manifest.py"),
                "--benchmark",
                "webshop",
                "--split",
                split,
                "--purpose",
                purpose,
                "--revision",
                args.revision,
                "--webshop-file-path",
                str(webshop_file),
                "--sample-count",
                str(count),
                "--sample-seed",
                str(seed),
                "--output",
                str(manifests[split]),
            )
        )
        loaded = OfficialTaskManifest.load(manifests[split])
        if len(loaded.tasks) != count or any(task.split != split for task in loaded.tasks):
            raise ValueError(f"generated {split} manifest failed scope validation")

    run_roots: dict[str, Path] = {}
    for split in ("train", "dev"):
        for method in ("edge_only", "cloud_only"):
            key = f"{split}_{method}"
            recorded = receipt["runs"].get(key)
            if recorded:
                candidate = Path(recorded).expanduser().resolve()
                _load_run_manifest(candidate, expected_method=method)
                print(f"resume_run={key} run_directory={candidate}", flush=True)
                run_roots[key] = candidate
                continue
            _, lines = _run_streaming(
                (
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_autodl_matrix.py"),
                    str(config),
                    str(manifests[split]),
                    str(profile),
                    "--method",
                    method,
                    "--resume-key",
                    f"webshop-gate-{plan_hash[:12]}-{split}-{method}",
                )
            )
            candidate = _extract_run_directory(lines)
            _load_run_manifest(candidate, expected_method=method)
            run_roots[key] = candidate
            receipt["runs"][key] = str(candidate)
            _write_receipt(receipt_path, receipt)

    paired = {
        "train": output / f"webshop-train-{args.train_count}-paired.jsonl",
        "dev": output / f"webshop-dev-{args.dev_count}-paired.jsonl",
    }
    for split in ("train", "dev"):
        _run_streaming(
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "collect_endpoint_episodes.py"),
                str(run_roots[f"{split}_edge_only"]),
                str(run_roots[f"{split}_cloud_only"]),
                "--split",
                split,
                "--output",
                str(paired[split]),
            )
        )

    router = output / "router-webshop-task-reward.joblib"
    gate = output / "router-webshop-train-dev-gate.json"
    returncode, _ = _run_streaming(
        (
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_router_learnability_gate.py"),
            str(paired["train"]),
            str(paired["dev"]),
            str(router),
            str(gate),
            "--minimum-train-tasks",
            str(args.train_count),
            "--minimum-paired-tasks",
            str(args.dev_count),
        ),
        accepted_returncodes=(0, 2),
    )
    receipt["gate_output"] = str(gate)
    receipt["gate_status"] = returncode
    _write_receipt(receipt_path, receipt)
    print(f"gate_output={gate}", flush=True)
    print(f"gate_pass={str(returncode == 0).lower()}", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
