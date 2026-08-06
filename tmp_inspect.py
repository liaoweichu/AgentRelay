#!/usr/bin/env python3
"""Inspect per-method routing behaviour: executor distribution, commit modes,
transfer modes, step counts, and whether the router ever switches executor."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    for method in args.methods:
        method_dir = run_dir / "webshop" / "test" / method
        files = sorted(method_dir.glob("*.json"))[: args.limit]
        executor = Counter()
        commit = Counter()
        transfer = Counter()
        steps_total = 0
        switch_episodes = 0
        max_steps_hit = 0
        for f in files:
            rec = json.loads(f.read_text(encoding="utf-8"))
            steps = rec.get("steps", ())
            steps_total += len(steps)
            if len(steps) >= int(rec.get("max_steps", 20)):
                max_steps_hit += 1
            # executor/commit/transfer across steps
            prev = None
            switched = False
            for s in steps:
                executor[str(s.get("selected_executor"))] += 1
                commit[str(s.get("commit_mode"))] += 1
                transfer[str(s.get("transfer_mode"))] += 1
                if prev is not None and s.get("selected_executor") != prev:
                    switched = True
                prev = s.get("selected_executor")
            if switched:
                switch_episodes += 1
        n = len(files)
        print(f"\n=== {method} (n={n}) ===")
        print(f"  steps_total={steps_total}  avg_steps={steps_total/max(1,n):.1f}")
        print(f"  episodes_with_switch={switch_episodes}/{n}  max_steps_hit={max_steps_hit}")
        print(f"  executor={dict(executor)}")
        print(f"  commit={dict(commit)}")
        print(f"  transfer={dict(transfer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())