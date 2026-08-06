#!/usr/bin/env python3
"""Dump one full ALFWorld trajectory to assess action validity & progress."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_episodes")
    parser.add_argument("--method", default="cloud_only")
    parser.add_argument("--sample", default=None)
    args = parser.parse_args()

    for line in Path(args.train_episodes).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("method") != args.method:
            continue
        if args.sample is not None and str(rec.get("sample_id")) != args.sample:
            continue
        print(f"method={rec.get('method')} sample={rec.get('sample_id')} "
              f"success={rec.get('success')} reward={rec.get('reward')} "
              f"steps={len(rec.get('steps', ()))}")
        for i, s in enumerate(rec.get("steps", ())[:30]):
            print(f"  {i:>2} ex={s.get('selected_executor'):>5} "
                  f"action={str(s.get('action_text'))[:45]!r} done={s.get('done')}")
        print("  ...")
        return 0
    print("no matching episode")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())