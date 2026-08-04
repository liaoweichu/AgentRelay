#!/usr/bin/env python3
"""Fit the joint policy estimator from immutable native training rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.learning import JointRouterEstimator, read_training_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", help="JSONL produced by native official-train rollouts")
    parser.add_argument("output")
    parser.add_argument("--train-split", action="append", required=True)
    args = parser.parse_args()
    rows = read_training_rows(args.rows)
    estimator = JointRouterEstimator().fit(
        rows,
        authorized_train_splits=args.train_split,
    )
    estimator.save(args.output)
    print(f"rows={len(rows)} actions={len(estimator.predictors)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
