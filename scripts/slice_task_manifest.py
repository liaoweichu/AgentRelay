#!/usr/bin/env python3
"""Slice an immutable task manifest down to a deterministic prefix.

Used to keep router/calibration training samples small on the AutoDL GPU.
Train/tune-purpose manifests need not attest the complete official split, so
the sliced copies are written with complete_official_split=False.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.formal_matrix import OfficialTaskManifest, write_task_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="source full task manifest")
    parser.add_argument("output")
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    source = OfficialTaskManifest.load(args.manifest)
    selected = source.tasks[: args.count]
    if not selected:
        raise ValueError(f"slice count must be positive, got {args.count}")
    target = write_task_manifest(
        args.output,
        dataset_revision=source.dataset_revision,
        tasks=selected,
        complete_official_split=False,
    )
    print(f"slice={len(selected)}/{len(source.tasks)} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())