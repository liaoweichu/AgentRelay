#!/usr/bin/env python3
"""Combine separately resident edge/cloud matrix runs into paired episode JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.router_data import pair_endpoint_episodes  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    episodes: list[dict] = []
    identities: set[tuple[str, str]] = set()
    for root_value in args.run_root:
        root = Path(root_value)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest.get("results", ()):
            if str(item.get("method")) not in {"edge_only", "cloud_only"}:
                continue
            path = root / str(item["path"])
            episode = json.loads(path.read_text(encoding="utf-8"))
            recorded_hash = str(episode.pop("result_hash", ""))
            if not recorded_hash or sha256_json(episode) != recorded_hash:
                raise ValueError(f"episode result hash mismatch: {path}")
            if str(episode.get("split")) != args.split:
                continue
            identity = (str(episode.get("sample_id")), str(episode.get("method")))
            if identity in identities:
                raise ValueError(f"duplicate endpoint episode: {identity}")
            identities.add(identity)
            episodes.append(episode)
    pairs = pair_endpoint_episodes(episodes)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        "".join(
            canonical_json(episode) + "\n"
            for pair in pairs
            for episode in pair
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    print(f"paired_tasks={len(pairs)} episodes={len(episodes)} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
