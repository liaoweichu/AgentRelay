#!/usr/bin/env python3
"""Diagnostic: can a redesigned router (pairwise + richer query features) learn
cross-library edge/cloud routing on the 314-task pool?

Compares leave-one-library-out OOF capture for:
  A) per-executor logistic (success) with rich features
  B) pairwise logistic (cloud_better vs edge_better) rich features
Richer features add SQL structural flags (GROUP BY / ORDER BY / JOIN / subquery /
aggregate / DISTINCT / LIMIT, approx join & condition counts). db_hash is dropped
so only library-independent task features are used.
"""

from __future__ import annotations

import json
import re
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
STRUCT_FEATURES = (
    "has_group_by",
    "has_order_by",
    "has_join",
    "has_subquery",
    "has_distinct",
    "has_limit",
    "has_aggregate",
    "n_conditions",
    "n_joins",
)
FEATURES = BASE_FEATURES + STRUCT_FEATURES

_STRUCT_RE = {
    "has_group_by": re.compile(r"\bgroup\s+by\b", re.I),
    "has_order_by": re.compile(r"\border\s+by\b", re.I),
    "has_join": re.compile(r"\bjoin\b", re.I),
    "has_subquery": re.compile(r"\(\s*select\b", re.I),
    "has_distinct": re.compile(r"\bdistinct\b", re.I),
    "has_limit": re.compile(r"\blimit\b", re.I),
    "has_aggregate": re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.I),
}


def _features(db, hardness, query, db_tables) -> dict[str, float]:
    base = intercode_task_features(db=db, hardness=hardness, query=query, db_tables=db_tables)
    out = {k: base[k] for k in BASE_FEATURES}
    q = query or ""
    for k, pat in _STRUCT_RE.items():
        out[k] = float(bool(pat.search(q)))
    out["n_conditions"] = float(q.count("AND") + q.count("and"))
    out["n_joins"] = float(len(re.findall(r"\bjoin\b", q, re.I)))
    return out


def _lolo(tasks, mode: str) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dbs = sorted({db for db, *_ in tasks})
    captures = {}
    for held in dbs:
        tr = [t for t in tasks if t[0] != held]
        te = [t for t in tasks if t[0] == held]
        if not te:
            continue
        edge_vals = [t[2] for t in te]
        cloud_vals = [t[3] for t in te]
        best = max(np.mean(edge_vals), np.mean(cloud_vals))
        oracle = np.mean([max(t[2], t[3]) for t in te])
        if mode == "per_executor":
            models = {}
            for role in ("edge", "cloud"):
                x = np.asarray([[t[1][n] for n in FEATURES] for t in tr], dtype=float)
                y = np.asarray([t[2] if role == "edge" else t[3] for t in tr], dtype=float)
                models[role] = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
                ).fit(x, y)
            router_vals = []
            for t in te:
                x = np.asarray([[t[1][n] for n in FEATURES]], dtype=float)
                pe = models["edge"].predict_proba(x)[0, 1]
                pc = models["cloud"].predict_proba(x)[0, 1]
                router_vals.append(t[3] if pc > pe else t[2])
        else:  # pairwise, cloud-better vs edge-better on non-tie train tasks
            X, y = [], []
            for t in tr:
                ve, vc = t[2], t[3]
                if ve != vc:
                    X.append([t[1][n] for n in FEATURES])
                    y.append(1 if vc and not ve else 0)
            router_vals = []
            if X:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
                ).fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
                for t in te:
                    x = np.asarray([[t[1][n] for n in FEATURES]], dtype=float)
                    router_vals.append(t[3] if model.predict_proba(x)[0, 1] > 0.5 else t[2])
            else:
                router_vals = [t[3] for t in te] if np.mean(cloud_vals) >= best else [t[2] for t in te]
        router = np.mean(router_vals)
        og = oracle - best
        cap = (router - best) / og if og > 0 else 0.0
        captures[held] = {"n": len(te), "oracle_gap": float(og), "capture": float(cap)}
    return captures


def main() -> int:
    edge = json.loads((POOL / "intercode-sql-edge-episodes.json").read_text(encoding="utf-8"))
    cloud = json.loads((POOL / "intercode-sql-cloud-episodes.json").read_text(encoding="utf-8"))
    src_by_key = {}
    for r in json.loads(Path(SPIDER_DEV).read_text(encoding="utf-8")):
        src_by_key[(r["db"], str(r["query"]).strip())] = r

    tasks = []
    for e, c in zip(edge, cloud):
        src = src_by_key.get((e["db"], str(e["query"]).strip()), {})
        hardness = str(e.get("hardness", src.get("hardness", "unknown")))
        query = str(e.get("query", src.get("query", "")))
        f = _features(e["db"], hardness, query, src.get("db_tables"))
        tasks.append((e["db"], f, float(e["success"]), float(c["success"])))

    import numpy as np
    results = {}
    for mode in ("per_executor", "pairwise"):
        caps = _lolo(tasks, mode)
        n = sum(v["n"] for v in caps.values())
        wcap = sum(v["capture"] * v["n"] for v in caps.values()) / n
        results[mode] = {"weighted_mean_capture": round(float(wcap), 4), "per_library": caps}
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
