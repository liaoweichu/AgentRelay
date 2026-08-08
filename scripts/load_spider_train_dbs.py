#!/usr/bin/env python3
"""Load the selected Spider TRAIN databases (sqlite) into MySQL.

The InterCode dev databases are already in MySQL (via ic_spider_dbs.sql). The
train split databases only ship as sqlite, so we convert them here with plain
sqlite3 + pymysql/sqlite3, emitting MySQL-compatible DDL + INSERTs and loading
each as its own schema. One-time data migration, not protocol logic.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

MYSQL_CREDS = ("127.0.0.1", 3306, "admin", "admin")


def _quote_ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _sqlite_type(t: str) -> str:
    t = t.upper()
    if "INT" in t:
        return "INT"
    if "CHAR" in t or "TEXT" in t or "CLOB" in t or "VARCHAR" in t:
        return "VARCHAR(255)"
    if "REAL" in t or "FLOA" in t or "DOUB" in t or "DECI" in t or "NUMERIC" in t:
        return "DOUBLE"
    if "BLOB" in t or "BINARY" in t:
        return "BLOB"
    if "DATE" in t or "TIME" in t:
        return "VARCHAR(255)"
    if "BOOL" in t:
        return "TINYINT(1)"
    return "VARCHAR(255)"


def _load_with_mysql(db: str, sql: str) -> None:
    proc = subprocess.run(
        [
            "mysql",
            f"-h{MYSQL_CREDS[0]}",
            f"-P{MYSQL_CREDS[1]}",
            f"-u{MYSQL_CREDS[2]}",
            f"-p{MYSQL_CREDS[3]}",
            f"--default-character-set=utf8mb4",
        ],
        input=sql,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mysql load failed for {db}: {proc.stderr[-2000:]}")


def convert(db_path: Path, db_name: str) -> None:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = cur.fetchall()
    if not tables:
        raise SystemExit(f"no tables in {db_path}")

    statements = ["SET SESSION sql_mode='';"]
    statements.append(f"DROP DATABASE IF EXISTS {_quote_ident(db_name)};")
    statements.append(f"CREATE DATABASE IF NOT EXISTS {_quote_ident(db_name)} CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;")
    statements.append(f"USE {_quote_ident(db_name)};")

    for table_row in tables:
        tname = table_row["name"]
        cur.execute(f'PRAGMA table_info("{tname}")')
        info = cur.fetchall()  # cid, name, type, notnull, dflt_value, pk
        cols = []
        pk_cols = [row[1] for row in info if row[5]]
        for row in info:
            cname, ctype, notnull, pk_flag = row[1], row[2], row[3], row[5]
            col = f"{_quote_ident(cname)} {_sqlite_type(ctype or 'TEXT')}"
            if pk_flag and len(pk_cols) == 1:
                col += " PRIMARY KEY"
            cols.append(col)
        if len(pk_cols) > 1:
            cols.append("PRIMARY KEY (" + ", ".join(_quote_ident(c) for c in pk_cols) + ")")
        create = f"CREATE TABLE IF NOT EXISTS {_quote_ident(tname)} (\n  " + ",\n  ".join(cols) + "\n);"
        statements.append(create)

        # Data
        cur.execute(f'SELECT * FROM "{tname}"')
        rows = cur.fetchall()
        colnames = [d[0] for d in cur.description]
        if not rows:
            continue
        quoted_cols = ", ".join(_quote_ident(c) for c in colnames)
        insert_prefix = f"INSERT INTO {_quote_ident(tname)} ({quoted_cols}) VALUES "
        # Batch inserts of 200 rows
        batch = []
        for row in rows:
            vals = []
            for v in row:
                if v is None:
                    vals.append("NULL")
                elif isinstance(v, (int, float)):
                    vals.append(repr(v))
                else:
                    s = str(v).replace("\\", "\\\\").replace("'", "''")
                    vals.append(f"'{s}'")
            batch.append("(" + ", ".join(vals) + ")")
            if len(batch) >= 200:
                statements.append(insert_prefix + ",\n".join(batch) + ";")
                batch = []
        if batch:
            statements.append(insert_prefix + ",\n".join(batch) + ";")

    conn.close()
    _load_with_mysql(db_name, "\n".join(statements))
    print(f"loaded {db_name}: {len(tables)} tables from {db_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_dir", help="path to spider_official/database")
    parser.add_argument("--dbs", nargs="+", required=True)
    args = parser.parse_args()
    db_dir = Path(args.db_dir)
    for db in args.dbs:
        db_path = db_dir / db / f"{db}.sqlite"
        if not db_path.exists():
            raise SystemExit(f"missing {db_path}")
        convert(db_path, db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
