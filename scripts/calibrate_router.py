#!/usr/bin/env python3
"""Fit train/dev-only conservative routing offsets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.calibration import ConformalRiskCalibrator, read_calibration_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows")
    parser.add_argument("output")
    parser.add_argument("--validation-split", action="append", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--minimum-group-size", type=int, default=20)
    args = parser.parse_args()
    rows = read_calibration_rows(args.rows)
    calibrator = ConformalRiskCalibrator(
        alpha=args.alpha,
        minimum_group_size=args.minimum_group_size,
    ).fit(rows, authorized_validation_splits=args.validation_split)
    calibrator.save(args.output)
    print(f"rows={len(rows)} output={args.output} guarantee=empirical_validation_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

