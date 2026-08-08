#!/usr/bin/env python3
"""Build a disjoint-by-library train/dev matrix from the expanded hard/extra pool.

Consumes the merged 314-task episodes (results/intercode-sql-gate-v2-hardextra340)
and the Spider-dev source records, assigns each library to either train or dev,
and writes the four fixed-endpoint episode files plus the two manifests in the
format expected by run_intersql_router_learnability_gate.py:

  <out>/intercode-sql-{edge,cloud}-{train,dev}-episodes.json
  <out>/ic_spider_{train,dev}_subset.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POOL_DIR = PROJECT_ROOT / "results/intercode-sql-gate-v2-hardextra340"
SPIDER_DEV = PROJECT_ROOT / "repositories/InterCode/data/sql/spider/ic_spider_dev.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev-dbs",
        nargs="+",
        default=None,
        help="libraries assigned to the held-out dev split (disjoint from train)",
    )
    parser.add_argument(
        "--by-task",
        action="store_true",
        help="split tasks within each library (train/dev share libraries)",
    )
    parser.add_argument("--dev-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results/intercode-sql-g14-matrix-exp340"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = {}
    for role in ("edge", "cloud"):
        episodes[role] = json.loads(
            (POOL_DIR / f"intercode-sql-{role}-episodes.json").read_text(encoding="utf-8")
        )

    # Source records for db_tables / hardness by (db, query).
    src_by_key = {}
    for r in json.loads(Path(SPIDER_DEV).read_text(encoding="utf-8")):
        src_by_key[(r["db"], str(r["query"]).strip())] = r

    # Assign each task to a split.
    split_of = {}
    if args.by_task:
        import random
        rng = random.Random(args.seed)
        from collections import defaultdict
        by_db = defaultdict(list)
        for e_ep, c_ep in zip(episodes["edge"], episodes["cloud"]):
            by_db[e_ep["db"]].append(str(e_ep["task_id"]))
        for db, ids in by_db.items():
            rng.shuffle(ids)
            n_dev = max(1, int(round(len(ids) * args.dev_fraction)))
            if n_dev >= len(ids):
                n_dev = len(ids) // 2 if len(ids) > 1 else 0
            for tid in ids:
                split_of[tid] = "dev" if ids.index(tid) < n_dev else "train"
    else:
        dev_dbs = set(args.dev_dbs)
        for e_ep, c_ep in zip(episodes["edge"], episodes["cloud"]):
            split_of[str(e_ep["task_id"])] = "dev" if e_ep["db"] in dev_dbs else "train"

    # Build per-split episode lists (edge/cloud aligned by pool order).
    per_split = {"train": {"edge": [], "cloud": []}, "dev": {"edge": [], "cloud": []}}
    task_ids = {"train": set(), "dev": set()}
    for e_ep, c_ep in zip(episodes["edge"], episodes["cloud"]):
        split = split_of[str(e_ep["task_id"])]
        per_split[split]["edge"].append(e_ep)
        per_split[split]["cloud"].append(c_ep)
        task_ids[split].add(str(e_ep["task_id"]))

    for split in ("train", "dev"):
        manifest = []
        seen = set()
        for role in ("edge", "cloud"):
            for ep in per_split[split][role]:
                task_id = str(ep["task_id"])
                if task_id in seen:
                    continue
                seen.add(task_id)
                key = (ep["db"], str(ep["query"]).strip())
                src = src_by_key.get(key, {})
                manifest.append(
                    {
                        "db": ep["db"],
                        "gold": ep.get("gold", src.get("gold", "")),
                        "query": ep["query"],
                        "hardness": ep.get("hardness", src.get("hardness", "unknown")),
                        "db_tables": src.get("db_tables", {}),
                        "task_id": task_id,
                    }
                )
        # persist episodes + manifest
        for role in ("edge", "cloud"):
            path = out_dir / f"intercode-sql-{role}-{split}-episodes.json"
            path.write_text(json.dumps(per_split[split][role], ensure_ascii=False) + "\n", encoding="utf-8")
        (out_dir / f"ic_spider_{split}_subset.json").write_text(
            json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        dbs = sorted(set(e["db"] for e in per_split[split]["edge"]))
        print(
            f"{split}: n={len(seen)} dbs={len(dbs)} "
            f"hard={dict(Counter(m['hardness'] for m in manifest))}",
            flush=True,
        )

    summary = {
        "mode": "by_task" if args.by_task else "by_library",
        "dev_dbs": sorted(set(e["db"] for e in per_split["dev"]["edge"])) if not args.by_task else "shared",
        "dev_fraction": args.dev_fraction if args.by_task else None,
        "seed": args.seed if args.by_task else None,
        "n_train": len(task_ids["train"]),
        "n_dev": len(task_ids["dev"]),
        "n_dbs_train": len(set(e["db"] for e in per_split["train"]["edge"])),
        "n_dbs_dev": len(set(e["db"] for e in per_split["dev"]["edge"])),
    }
    (out_dir / "matrix-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
