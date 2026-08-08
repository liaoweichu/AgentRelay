#!/usr/bin/env python3
"""Build the selected-libraries InterCode-SQL task manifests for G14.

Train subset: bike_1, music_1, baseball_1, sakila_1 (from Spider train_spider).
Dev subset:   student_transcripts_tracking, network_1 (from Spider dev / ic_spider_dev).

Produces ic_spider format records (db, gold, query, hardness, db_tables) with a
stable task id, written to the G14 results dir. One-time data build, not protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SPIDER = PROJECT_ROOT / "repositories/spider_official"
IC_DEV = PROJECT_ROOT / "repositories/InterCode/data/sql/spider/ic_spider_dev.json"

TRAIN_DBS = ["bike_1", "music_1", "baseball_1", "sakila_1"]
DEV_DBS = ["student_transcripts_tracking", "network_1"]

cut_spaces = lambda val: " ".join(str(val).split())


def _hardness_text(sql: str) -> str:
    """Self-contained hardness proxy mirroring the official component counts.

    nltk/process_sql are unavailable offline; hardness is metadata only (not used
    by the endpoint or the router gate). This classifier follows the official
    component1/component2/others counting rules over SQL text.
    """
    u = sql.upper()
    n_select = u.count("SELECT")
    n_from = u.count(" FROM ")
    joins = max(0, n_from - 1)
    comp1 = 0
    comp1 += 1 if "WHERE" in u else 0
    comp1 += 1 if "GROUP BY" in u else 0
    comp1 += 1 if "ORDER BY" in u else 0
    comp1 += 1 if "LIMIT" in u else 0
    comp1 += joins
    comp1 += u.count(" OR ")
    comp1 += u.count("LIKE")
    nested = max(0, n_select - 1)
    comp2 = nested
    comp2 += 1 if "UNION" in u else 0
    comp2 += 1 if "INTERSECT" in u else 0
    comp2 += 1 if "EXCEPT" in u else 0
    agg = len([k for k in ("MAX(", "MIN(", "COUNT(", "SUM(", "AVG(") if k in u])
    others = 0
    others += 1 if agg > 1 else 0
    others += 1 if u.count("SELECT") > 1 else 0
    others += 1 if u.count("WHERE") > 1 else 0
    others += 1 if u.count("GROUP BY") > 1 else 0
    if comp1 <= 1 and others == 0 and comp2 == 0:
        return "easy"
    if (others <= 2 and comp1 <= 1 and comp2 == 0) or (comp1 <= 2 and others < 2 and comp2 == 0):
        return "medium"
    if (others > 2 and comp1 <= 2 and comp2 == 0) or (
        2 < comp1 <= 3 and others <= 2 and comp2 == 0
    ) or (comp1 <= 1 and others == 0 and comp2 <= 1):
        return "hard"
    return "extra"


def get_hardness(db_dir: Path, db: str, gold: str) -> str:
    return _hardness_text(gold)


class _MySqlCheck:
    """Minimal executor to confirm each gold query runs in MySQL (SQLite->MySQL
    portability filter). Gold queries that reference reserved words or use
    SQLite-only syntax cannot be scored, so they are excluded from the manifest."""

    def __init__(self):
        import mysql.connector
        self._cfg = dict(host="127.0.0.1", port=3306, user="admin", password="admin")
        self._conn = mysql.connector.connect(**self._cfg)
        self._err = mysql.connector.Error
        self._cur = self._conn.cursor(buffered=True)

    def ok(self, db: str, sql: str) -> bool:
        try:
            self._cur.execute(f"USE `{db}`")
            self._cur.execute(sql)
            self._cur.fetchall()
            return True
        except self._err:
            return False


def build_train(out_path: Path, db_dir: Path) -> list[dict]:
    json_spider = json.loads((SPIDER / "train_spider.json").read_text(encoding="utf-8"))
    # train_gold.sql covers train_spider + train_others (8659 lines) and is not
    # index-aligned; match each task's gold SQL by (db_id, query) instead.
    gold_lookup: dict[str, set[str]] = {}
    for l in (SPIDER / "train_gold.sql").read_text(encoding="utf-8").splitlines():
        if not len(l.strip()):
            continue
        parts = l.strip().split("\t")
        if len(parts) >= 2:
            gold_lookup.setdefault(parts[1], set()).add(cut_spaces(parts[0]))
    tables = json.loads((SPIDER / "tables.json").read_text(encoding="utf-8"))
    db_to_tables = {}
    for db in tables:
        db_to_tables[db["db_id"]] = {}
        for idx, table_name in enumerate(db["table_names_original"]):
            db_to_tables[db["db_id"]][table_name] = [
                x[1] for x in db["column_names_original"] if x[0] == idx
            ]
    records = []
    for idx, value in enumerate(json_spider):
        if value["db_id"] not in TRAIN_DBS:
            continue
        gold = cut_spaces(value["query"])
        if gold not in gold_lookup.get(value["db_id"], set()):
            continue
        rec = {
            "db": value["db_id"],
            "gold": gold,
            "query": value["question"],
            "hardness": get_hardness(db_dir, value["db_id"], gold),
            "db_tables": db_to_tables[value["db_id"]],
            "task_id": f"train-{value['db_id']}-{idx:05d}",
        }
        records.append(rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def build_dev(out_path: Path) -> list[dict]:
    dev = json.loads(IC_DEV.read_text(encoding="utf-8"))
    records = []
    for idx, value in enumerate(dev):
        if value["db"] not in DEV_DBS:
            continue
        rec = {**value, "task_id": f"dev-{value['db']}-{idx:05d}"}
        records.append(rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/intercode-sql-g14-matrix"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    db_dir = SPIDER / "database"
    train = build_train(out_dir / "ic_spider_train_subset.json", db_dir)
    dev = build_dev(out_dir / "ic_spider_dev_subset.json")
    from collections import Counter

    # Drop tasks whose gold query does not execute in MySQL (cannot be scored).
    checker = _MySqlCheck()
    train_keep = [t for t in train if checker.ok(t["db"], t["gold"])]
    dev_keep = [t for t in dev if checker.ok(t["db"], t["gold"])]
    (out_dir / "ic_spider_train_subset.json").write_text(
        json.dumps(train_keep, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "ic_spider_dev_subset.json").write_text(
        json.dumps(dev_keep, indent=2) + "\n", encoding="utf-8"
    )
    print(f"train gold-filtered: {len(train)} -> {len(train_keep)} "
          f"(dropped {len(train) - len(train_keep)})")
    print(f"dev gold-filtered: {len(dev)} -> {len(dev_keep)} "
          f"(dropped {len(dev) - len(dev_keep)})")
    print("train:", len(train_keep), Counter(t["db"] for t in train_keep))
    print("train hardness:", Counter(t["hardness"] for t in train_keep))
    print("dev:", len(dev_keep), Counter(t["db"] for t in dev_keep))
    print("dev hardness:", Counter(t["hardness"] for t in dev_keep))
    print("TOTAL:", len(train_keep) + len(dev_keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
