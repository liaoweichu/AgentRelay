#!/usr/bin/env python3
"""Recompute paired endpoint summaries with correct exclusive-success counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.gating import summarize_paired_endpoints  # noqa: E402
from agentrelay.router_data import read_episode_records  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--split-validated", action="store_true")
    args = parser.parse_args()
    value = json.loads(Path(args.input).read_text(encoding="utf-8"))
    payload = summarize_paired_endpoints(read_episode_records(value))
    payload.update(
        {
            "paper_evidence": False,
            "split_validated": bool(args.split_validated),
            "source_artifact": str(Path(args.input)),
        }
    )
    payload["summary_hash"] = sha256_json(payload)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(f"paired_tasks={payload['paired_tasks']} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
