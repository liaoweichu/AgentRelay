#!/usr/bin/env python3
"""Confidence-gated (selective) router OOF test on the 314-task pool.

A selective router only commits to a non-default arm when its model margin is
above a threshold tau; otherwise it falls back to the best-fixed arm learned on
the training libraries (no dev leakage). This is a legitimate real-world routing
design (avoid risky choices). Evaluated as leave-one-library-out OOF capture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.intercode_sql import intercode_task_features  # noqa: E402

POOL = PROJECT_ROOT / "results/intercode-sql-gate-v2-hardextra340"
SPIDER_DEV = PROJECT_ROOT / "repositories/InterCode/data/sql/spider/ic_spider_dev.json"

BASE_FEATURES = (
    "hardness_ordinal",
    "query_char_count",
    "query_token_count",
    "query_numeric_count",
    "n_tables",
    "n_columns",
)
DB_HASH = tuple(f"db_hash_{i:02d}" for i in range(8))
FEATURES = BASE_FEATURES + DB_HASH


def main() -> int:
    edge = json.loads((POOL / "intercode-sql-edge-episodes.json").read_text(encoding="utf-8"))
    cloud = json.loads((POOL / "intercode-sql-cloud-episodes.json").read_text(encoding="utf-8"))
    src_by_key = {}
    for r in json.loads(Path(SPIDER_DEV).read_text(encoding="utf-8")):
        src_by_key[(r["db"], str(r["query"]).strip())] = r

    tasks = []
    for e, c in zip(edge, cloud):
        src = src_by_key.get((e["db"], str(e["query"]).strip()), {})
        feats = intercode_task_features(
            db=e["db"],
            hardness=str(e.get("hardness", src.get("hardness", "unknown"))),
            query=str(e.get("query", src.get("query", ""))),
            db_tables=src.get("db_tables"),
        )
        tasks.append((e["db"], feats, float(e["success"]), float(c["success"])))

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dbs = sorted({db for db, *_ in tasks})
    # evaluate over a range of tau
    for tau in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        per_lib = {}
        for held in dbs:
            tr = [t for t in tasks if t[0] != held]
            te = [t for t in tasks if t[0] == held]
            if not te:
                continue
            models = {}
            for role in ("edge", "cloud"):
                x = np.asarray([[t[1][n] for n in FEATURES] for t in tr], dtype=float)
                y = np.asarray([t[2] if role == "edge" else t[3] for t in tr], dtype=float)
                models[role] = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
                ).fit(x, y)
            train_edge = np.mean([t[2] for t in tr])
            train_cloud = np.mean([t[3] for t in tr])
            fallback = "cloud" if train_cloud >= train_edge else "edge"
            router_vals = []
            for t in te:
                x = np.asarray([[t[1][n] for n in FEATURES]], dtype=float)
                pe = models["edge"].predict_proba(x)[0, 1]
                pc = models["cloud"].predict_proba(x)[0, 1]
                margin = pc - pe
                if margin > tau:
                    pick = "cloud"
                elif margin < -tau:
                    pick = "edge"
                else:
                    pick = fallback
                router_vals.append(t[3] if pick == "cloud" else t[2])
            edge_vals = [t[2] for t in te]
            cloud_vals = [t[3] for t in te]
            best = max(np.mean(edge_vals), np.mean(cloud_vals))
            oracle = np.mean([max(t[2], t[3]) for t in te])
            og = oracle - best
            router = np.mean(router_vals)
            cap = (router - best) / og if og > 0 else 0.0
            per_lib[held] = cap
        n = sum(len([t for t in tasks if t[0] == db]) for db in dbs)
        wcap = sum(per_lib[db] * len([t for t in tasks if t[0] == db]) for db in dbs) / n
        print(f"tau={tau:.2f}  weighted_mean_capture={wcap:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
