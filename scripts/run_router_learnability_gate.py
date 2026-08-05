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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_episodes")
    parser.add_argument("dev_episodes")
    parser.add_argument("router_output")
    parser.add_argument("gate_output")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--minimum-paired-tasks", type=int, default=100)
    parser.add_argument("--minimum-oracle-capture", type=float, default=0.30)
    parser.add_argument("--minimum-cloud-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-cloud-fraction", type=float, default=0.90)
    args = parser.parse_args()

    train = _read(args.train_episodes)
    dev = _read(args.dev_episodes)
    rows = task_router_training_rows(
        train,
        authorized_train_splits=(args.train_split,),
    )
    router = JointRouterEstimator().fit(
        rows,
        authorized_train_splits=(args.train_split,),
    )
    router.save(args.router_output)
    gate = evaluate_router_learnability(
        router,
        dev,
        minimum_paired_tasks=args.minimum_paired_tasks,
        minimum_oracle_capture=args.minimum_oracle_capture,
        minimum_cloud_fraction=args.minimum_cloud_fraction,
        maximum_cloud_fraction=args.maximum_cloud_fraction,
    )
    gate.update(
        {
            "paper_evidence": False,
            "scope": "offline_task_router_train_dev_gate",
            "train_rows": len(rows),
            "independent_train_tasks": len(rows) // 2,
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
        f"train_tasks={len(rows) // 2} dev_tasks={gate['paired_dev_tasks']} "
        f"pass={gate['gate_pass']} output={target}"
    )
    return 0 if gate["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
