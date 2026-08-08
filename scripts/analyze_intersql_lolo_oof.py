#!/usr/bin/env python3
"""Leave-one-library-out OOF learnability on the 314-task hard/extra pool.

For each library, train a per-executor router on all other libraries and evaluate
oracle capture on that library. Aggregates to an overall cross-library capture,
mirroring the earlier full-pool OOF diagnostic (which was 0.28 on 200 tasks).
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

FEATURE_NAMES = (
    "hardness_ordinal",
    "query_char_count",
    "query_token_count",
    "query_numeric_count",
    "n_tables",
    "n_columns",
    "db_hash_00",
    "db_hash_01",
    "db_hash_02",
    "db_hash_03",
    "db_hash_04",
    "db_hash_05",
    "db_hash_06",
    "db_hash_07",
)


def main() -> int:
    target = "success"
    edge = json.loads((POOL / "intercode-sql-edge-episodes.json").read_text(encoding="utf-8"))
    cloud = json.loads((POOL / "intercode-sql-cloud-episodes.json").read_text(encoding="utf-8"))
    src_by_key = {}
    for r in json.loads(Path(SPIDER_DEV).read_text(encoding="utf-8")):
        src_by_key[(r["db"], str(r["query"]).strip())] = r

    def value(ep):
        return float(ep["success"]) if target == "success" else float(ep["reward"])

    # build task list with features + labels
    tasks = []
    for e, c in zip(edge, cloud):
        src = src_by_key.get((e["db"], str(e["query"]).strip()), {})
        feats = intercode_task_features(
            db=e["db"],
            hardness=str(e.get("hardness", src.get("hardness", "unknown"))),
            query=str(e.get("query", src.get("query", ""))),
            db_tables=src.get("db_tables"),
        )
        tasks.append((e["db"], feats, value(e), value(c)))

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dbs = sorted({db for db, _, _, _ in tasks})
    captures = {}
    per_lib_router_w = 0.0
    per_lib_best_w = 0.0
    per_lib_oracle_w = 0.0
    per_lib_cap_w = 0.0
    total_weight = 0
    for held in dbs:
        tr = [(f, ve, "edge") for db, f, ve, vc in tasks if db != held]
        tr += [(f, vc, "cloud") for db, f, ve, vc in tasks if db != held]
        te = [(f, ve, vc) for db, f, ve, vc in tasks if db == held]
        if not tr or not te:
            continue
        models = {}
        for role in ("edge", "cloud"):
            x = np.asarray([[f[n] for n in FEATURE_NAMES] for f, v, r in tr if r == role], dtype=float)
            y = np.asarray([v for f, v, r in tr if r == role], dtype=float)
            models[role] = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
            ).fit(x, y)
        n = len(te)
        edge_vals = [ve for _, ve, _ in te]
        cloud_vals = [vc for _, vc, _ in te]
        best_fixed = max(np.mean(edge_vals), np.mean(cloud_vals))
        oracle = np.mean([max(ve, vc) for _, ve, vc in te])
        router_vals = []
        for f, ve, vc in te:
            x = np.asarray([[f[n] for n in FEATURE_NAMES]], dtype=float)
            pe = models["edge"].predict_proba(x)[0, 1]
            pc = models["cloud"].predict_proba(x)[0, 1]
            router_vals.append(vc if pc > pe else ve)
        router = np.mean(router_vals)
        og = oracle - best_fixed
        cap = (router - best_fixed) / og if og > 0 else 0.0
        captures[held] = {"n": n, "oracle_gap": float(og), "capture": float(cap)}
        per_lib_router_w += router * n
        per_lib_best_w += best_fixed * n
        per_lib_oracle_w += oracle * n
        per_lib_cap_w += cap * n
        total_weight += n

    overall_router = per_lib_router_w / total_weight
    overall_best = per_lib_best_w / total_weight
    overall_oracle = per_lib_oracle_w / total_weight
    overall_gap = overall_oracle - overall_best
    overall_capture = (overall_router - overall_best) / overall_gap if overall_gap > 0 else 0.0
    agg = {
        "target": target,
        "pool_n": len(tasks),
        "n_libs": len(dbs),
        "weighted_mean_capture": per_lib_cap_w / total_weight,
        "pooled_router": float(overall_router),
        "pooled_best_fixed": float(overall_best),
        "pooled_oracle": float(overall_oracle),
        "pooled_oracle_gap": float(overall_gap),
        "pooled_capture": float(overall_capture),
        "per_library": captures,
    }
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
