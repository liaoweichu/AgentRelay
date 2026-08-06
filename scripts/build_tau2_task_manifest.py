#!/usr/bin/env python3
"""Build deterministic internal train/dev splits from official tau2 train tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.schema import canonical_json
from agentrelay.tau2_adapter import build_tau2_router_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tau2_repository")
    parser.add_argument("output")
    parser.add_argument("--split-seed", type=int, default=20260806)
    parser.add_argument("--dev-fraction", type=float, default=0.30)
    args = parser.parse_args()
    manifest = build_tau2_router_manifest(
        args.tau2_repository,
        split_seed=args.split_seed,
        dev_fraction=args.dev_fraction,
    )
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    print(f"manifest_hash={manifest['manifest_hash']} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
