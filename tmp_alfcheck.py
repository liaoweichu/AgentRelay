#!/usr/bin/env python3
"""Inspect the ALFWorld train rollout episodes: rewards, step counts, episode
length vs max_steps, and whether tasks look impossible (e.g., all truncated)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_episodes")
    parser.add_argument("manifest")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", ())
    print(f"manifest tasks={len(tasks)}  revision={manifest.get('dataset_revision')}")
    print("  task_ids:", [t.get("task_id") for t in tasks][:20])
    print("  configs:", sorted({t.get("alfworld_config") for t in tasks}))

    by_method: dict[str, list] = {}
    for line in Path(args.train_episodes).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        by_method.setdefault(rec.get("method"), []).append(rec)

    for method, recs in by_method.items():
        rewards = [float(r.get("reward", 0.0)) for r in recs]
        successes = [bool(r.get("success")) for r in recs]
        steps = [len(r.get("steps", ())) for r in recs]
        max_steps = Counter(int(r.get("max_steps", 0) or 0) for r in recs)
        step_index_at_end = Counter()
        for r in recs:
            st = r.get("steps", ())
            step_index_at_end[len(st)] += 1
        print(f"\nmethod={method} n={len(recs)}")
        print(f"  success={sum(successes)}/{len(recs)}  reward_dist={Counter(round(x,3) for x in rewards)}")
        print(f"  avg_steps={sum(steps)/len(steps):.1f}  len_dist={Counter(steps)}")
        print(f"  max_steps_field={dict(max_steps)}")
        # show first 3 episodes' final step
        for r in recs[:3]:
            st = r.get("steps", ())
            last = st[-1] if st else {}
            print(f"    sample={r.get('sample_id')} success={r.get('success')} "
                  f"reward={r.get('reward')} done={last.get('done')} steps={len(st)} "
                  f"last_action={str(last.get('action_text'))[:40]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())