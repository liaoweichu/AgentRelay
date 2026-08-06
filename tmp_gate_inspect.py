#!/usr/bin/env python3
"""Inspect the gate manifests for the Gemma/Qwen capability gate."""
import json
from pathlib import Path

for name in ("webshop-gate50.json", "alfworld-gate20.json"):
    d = json.loads(Path("results/manifests", name).read_text(encoding="utf-8"))
    tasks = d["tasks"]
    print(f"=== {name} ===")
    print("  revision:", d["dataset_revision"])
    print("  complete_official_split:", d["complete_official_split"])
    print("  n:", len(tasks))
    print("  purposes:", sorted({t["purpose"] for t in tasks}))
    print("  splits:", sorted({t["split"] for t in tasks}))
    print("  benchmarks:", sorted({t["benchmark"] for t in tasks}))
    print("  first:", json.dumps(tasks[0], ensure_ascii=False, sort_keys=True))
    print("  task_ids:", [t["task_id"] for t in tasks[:5]])