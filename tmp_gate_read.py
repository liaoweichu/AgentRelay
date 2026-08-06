#!/usr/bin/env python3
import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"=== {path} ===")
    print("edge:", json.dumps(d.get("edge"), sort_keys=True))
    print("cloud:", json.dumps(d.get("cloud"), sort_keys=True))
    print("reward_diverged_tasks:", d.get("reward_diverged_tasks"), "/", d.get("total_tasks"))
    for r in d.get("rows", []):
        print(f"  {r.get('benchmark')} task={r.get('task_id')} role={r.get('role')} "
              f"success={r.get('success')} reward={r.get('reward'):.4f} steps={r.get('steps')}")