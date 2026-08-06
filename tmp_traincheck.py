#!/usr/bin/env python3
"""Summarize the ALFWorld train rollout that fed router fitting."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_episodes")
    args = parser.parse_args()

    by_method: dict[str, dict] = {}
    n = 0
    for line in Path(args.train_episodes).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        n += 1
        m = rec.get("method", "?")
        d = by_method.setdefault(m, {"episodes": 0, "success": 0.0, "steps": 0,
                                     "executors": Counter(), "splits": Counter()})
        d["episodes"] += 1
        d["success"] += float(rec.get("success", 0.0))
        for s in rec.get("steps", ()):
            d["steps"] += 1
            d["executors"][str(s.get("selected_executor"))] += 1
        d["splits"][rec.get("split")] += 1
    print(f"total_records={n}")
    for m, d in by_method.items():
        print(f"\nmethod={m}  episodes={d['episodes']}  "
              f"success_rate={d['success']/d['episodes']:.3f}  steps={d['steps']}  "
              f"splits={dict(d['splits'])}")
        print(f"  executors={dict(d['executors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())