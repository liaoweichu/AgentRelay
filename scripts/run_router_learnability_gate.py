#!/usr/bin/env python3
"""Fit on paired official-train endpoints and gate on paired official-dev endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.gating import evaluate_router_learnability  # noqa: E402
from agentrelay.learning import JointRouterEstimator  # noqa: E402
from agentrelay.router_data import (  # noqa: E402
    read_episode_records,
    task_router_training_rows,
)
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402


def _read(path: str) -> tuple[dict, ...]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return tuple(
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return tuple(read_episode_records(json.loads(source.read_text(encoding="utf-8"))))


def _provenance_scope(episodes: tuple[dict, ...], *, label: str) -> dict:
    records = []
    for index, episode in enumerate(episodes, 1):
        provenance = episode.get("endpoint_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"{label} episode {index} has no endpoint provenance")
        required = (
            "run_id",
            "manifest_hash",
            "code_revision",
            "config_hash",
            "profile_hash",
            "model_revisions",
            "task_manifest_hash",
        )
        if any(not provenance.get(field) for field in required):
            raise ValueError(f"{label} episode {index} has incomplete endpoint provenance")
        records.append(provenance)
    code_revisions = {str(item["code_revision"]) for item in records}
    config_hashes = {str(item["config_hash"]) for item in records}
    profile_hashes = {str(item["profile_hash"]) for item in records}
    task_manifests = {str(item["task_manifest_hash"]) for item in records}
    run_ids = {str(item["run_id"]) for item in records}
    endpoint_manifests = {str(item["manifest_hash"]) for item in records}
    model_payloads = {
        canonical_json(dict(item["model_revisions"])) for item in records
    }
    if (
        len(code_revisions) != 1
        or len(config_hashes) != 1
        or len(profile_hashes) != 1
        or len(model_payloads) != 1
    ):
        raise ValueError(f"{label} endpoints disagree on code/config/profile/models")
    if len(task_manifests) != 1:
        raise ValueError(f"{label} endpoints disagree on task manifest")
    if len(run_ids) != 2 or len(endpoint_manifests) != 2:
        raise ValueError(f"{label} must contain exactly two fixed-endpoint runs")
    return {
        "code_revision": next(iter(code_revisions)),
        "config_hash": next(iter(config_hashes)),
        "profile_hash": next(iter(profile_hashes)),
        "model_revisions": json.loads(next(iter(model_payloads))),
        "task_manifest_hash": next(iter(task_manifests)),
        "run_ids": sorted(run_ids),
        "endpoint_manifest_hashes": sorted(endpoint_manifests),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_episodes")
    parser.add_argument("dev_episodes")
    parser.add_argument("router_output")
    parser.add_argument("gate_output")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", choices=("dev",), default="dev")
    parser.add_argument("--minimum-train-tasks", type=int, default=200)
    parser.add_argument("--minimum-paired-tasks", type=int, default=100)
    parser.add_argument("--minimum-oracle-capture", type=float, default=0.30)
    parser.add_argument("--minimum-cloud-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-cloud-fraction", type=float, default=0.90)
    args = parser.parse_args()

    train = _read(args.train_episodes)
    dev = _read(args.dev_episodes)
    train_provenance = _provenance_scope(train, label="train")
    dev_provenance = _provenance_scope(dev, label="dev")
    for field in (
        "code_revision",
        "config_hash",
        "profile_hash",
        "model_revisions",
    ):
        if train_provenance[field] != dev_provenance[field]:
            raise ValueError(f"train/dev provenance mismatch for {field}")
    rows = task_router_training_rows(
        train,
        authorized_train_splits=(args.train_split,),
    )
    router = JointRouterEstimator().fit(
        rows,
        authorized_train_splits=(args.train_split,),
    )
    train_tasks = {
        (
            str(episode.get("benchmark", "")),
            str(episode.get("dataset_revision", "")),
            str(episode.get("sample_id", episode.get("task_id", ""))),
        )
        for episode in train
    }
    dev_tasks = {
        (
            str(episode.get("benchmark", "")),
            str(episode.get("dataset_revision", "")),
            str(episode.get("sample_id", episode.get("task_id", ""))),
        )
        for episode in dev
    }
    if train_tasks & dev_tasks:
        raise ValueError("train and dev endpoint inputs contain overlapping tasks")
    train_domains = {(item[0], item[1]) for item in train_tasks}
    dev_domains = {(item[0], item[1]) for item in dev_tasks}
    if len(train_domains) != 1 or train_domains != dev_domains:
        raise ValueError(
            "train and dev endpoint inputs must share one benchmark and dataset revision"
        )
    gate = evaluate_router_learnability(
        router,
        dev,
        minimum_paired_tasks=args.minimum_paired_tasks,
        minimum_oracle_capture=args.minimum_oracle_capture,
        minimum_cloud_fraction=args.minimum_cloud_fraction,
        maximum_cloud_fraction=args.maximum_cloud_fraction,
        authorized_dev_splits=(args.dev_split,),
    )
    router.save(args.router_output)
    independent_train_tasks = len(train_tasks)
    gate["checks"] = {
        "enough_train_tasks": independent_train_tasks >= args.minimum_train_tasks,
        **gate["checks"],
    }
    gate["thresholds"] = {
        "minimum_train_tasks": args.minimum_train_tasks,
        **gate["thresholds"],
    }
    gate["gate_pass"] = all(gate["checks"].values())
    gate.update(
        {
            "paper_evidence": False,
            "scope": "offline_task_router_train_dev_gate",
            "train_rows": len(rows),
            "independent_train_tasks": independent_train_tasks,
            "train_episodes_hash": sha256_json(train),
            "dev_episodes_hash": sha256_json(dev),
            "benchmark": next(iter(train_domains))[0],
            "dataset_revision": next(iter(train_domains))[1],
            "endpoint_provenance": {
                "train": train_provenance,
                "dev": dev_provenance,
            },
            "router_metadata": dict(router.metadata),
        }
    )
    gate["gate_hash"] = sha256_json(gate)
    target = Path(args.gate_output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(canonical_json(gate) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(
        f"train_tasks={independent_train_tasks} dev_tasks={gate['paired_dev_tasks']} "
        f"pass={gate['gate_pass']} output={target}"
    )
    return 0 if gate["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
