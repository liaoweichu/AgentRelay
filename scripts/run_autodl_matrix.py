#!/usr/bin/env python3
"""Run implemented methods over one complete official task manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import load_json_config  # noqa: E402
from agentrelay.formal_matrix import FormalMatrixRunner, OfficialTaskManifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="locked formal AutoDL config")
    parser.add_argument("task_manifest")
    parser.add_argument("profile", help="measured service/network profile JSON")
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--router")
    parser.add_argument("--calibrator")
    args = parser.parse_args()
    runner = FormalMatrixRunner(
        config=load_json_config(args.config),
        task_manifest=OfficialTaskManifest.load(args.task_manifest),
        methods=args.method,
        profile_path=args.profile,
        router_path=args.router,
        calibrator_path=args.calibrator,
    )
    run_root = runner.run()
    print(f"run_directory={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

