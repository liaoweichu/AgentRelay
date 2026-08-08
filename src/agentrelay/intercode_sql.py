"""Protocol and sampling helpers for reproducible InterCode-SQL runs.

The helpers in this module are deliberately independent from model and database
dependencies so the action contract can be tested locally before a cloud run.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

SqlActionKind = Literal["sql", "submit", "sql_submit", "invalid"]

_THINK_BLOCK = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_SQL_CODE_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_STANDALONE_SUBMIT = re.compile(r"submit\s*", re.IGNORECASE)
_SQL_PREFIXES = ("SELECT", "SHOW", "DESC", "DESCRIBE", "WITH", "EXPLAIN")


@dataclass(frozen=True)
class ParsedSqlAction:
    """One normalized action emitted by an InterCode-SQL model wrapper."""

    kind: SqlActionKind
    sql: str | None = None


def parse_sql_action(text: str | None) -> ParsedSqlAction:
    """Parse a strict SQL, submit, or atomic SQL-plus-submit response.

    Gemma may emit a SQL code block followed by a standalone ``submit`` in one
    model turn. Treat that exact shape as two sequential environment actions:
    execute the SQL, then request evaluation. Natural-language text around the
    code block is not accepted as an atomic submit.
    """

    stripped = _THINK_BLOCK.sub("", text or "").strip()
    if not stripped:
        return ParsedSqlAction("invalid")

    match = _SQL_CODE_BLOCK.search(stripped)
    if match:
        sql = match.group(1).strip().rstrip(";")
        if not sql or sql.upper() == "SUBMIT":
            return ParsedSqlAction("invalid")
        remainder = (stripped[: match.start()] + stripped[match.end() :]).strip()
        if not remainder:
            return ParsedSqlAction("sql", sql)
        if _STANDALONE_SUBMIT.fullmatch(remainder):
            return ParsedSqlAction("sql_submit", sql)
        return ParsedSqlAction("invalid")

    if _STANDALONE_SUBMIT.fullmatch(stripped):
        return ParsedSqlAction("submit")

    # Retain a conservative raw-SQL fallback for prompt variants without
    # Markdown fences. Reject surrounding prose and multi-line explanations.
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) == 1 and lines[0].upper().startswith(_SQL_PREFIXES):
        return ParsedSqlAction("sql", lines[0].rstrip(";"))
    return ParsedSqlAction("invalid")


def proportional_stratified_indices(
    records: Sequence[Mapping[str, Any]],
    *,
    n_samples: int,
    seed: int,
    field: str,
) -> list[int]:
    """Return a deterministic proportional stratified sample.

    Every bucket is shuffled before selection. Integer allocations use largest
    remainders, so the selected distribution is as close as possible to the
    source distribution without depending on input ordering inside a bucket.
    """

    total = len(records)
    if not 1 <= n_samples <= total:
        raise ValueError(f"n_samples must be in [1, {total}], got {n_samples}")

    buckets: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        buckets[str(record.get(field, "unknown"))].append(index)

    rng = random.Random(seed)
    for label in sorted(buckets):
        rng.shuffle(buckets[label])

    exact = {
        label: n_samples * len(indices) / total for label, indices in buckets.items()
    }
    allocation = {label: math.floor(value) for label, value in exact.items()}
    remaining = n_samples - sum(allocation.values())
    remainder_order = sorted(
        buckets,
        key=lambda label: (-(exact[label] - allocation[label]), label),
    )
    for label in remainder_order[:remaining]:
        allocation[label] += 1

    selected = [
        index
        for label in sorted(buckets)
        for index in buckets[label][: allocation[label]]
    ]
    rng.shuffle(selected)
    return selected


_HARDNESS_ORDINAL = {"easy": 0.0, "medium": 1.0, "hard": 2.0, "extra": 3.0}

_DB_HASH_DIMENSIONS = 8
_DB_HASH_NAMES = tuple(f"db_hash_{index:02d}" for index in range(_DB_HASH_DIMENSIONS))


def intercode_task_features(
    *,
    db: str,
    hardness: str,
    query: str,
    db_tables: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    """Pre-action, label-free features that drive the edge/cloud reward router.

    All features are computable before any rollout reward is known: task
    difficulty label, question surface statistics, schema size, and a stable
    library identity hash. No feature depends on model output or episode reward.
    """

    text = query or ""
    words = [word for word in re.split(r"\s+", text.strip()) if word]
    db_tables = db_tables or {}
    n_columns = sum(len(columns) for columns in db_tables.values())
    features: dict[str, float] = {
        "hardness_ordinal": float(_HARDNESS_ORDINAL.get(hardness, 1.0)),
        "query_char_count": float(len(text)),
        "query_token_count": float(len(words)),
        "query_numeric_count": float(len(re.findall(r"\d", text))),
        "n_tables": float(len(db_tables)),
        "n_columns": float(n_columns),
    }
    digest = hashlib.sha256(str(db).encode("utf-8")).digest()
    for index in range(_DB_HASH_DIMENSIONS):
        features[_DB_HASH_NAMES[index]] = float(digest[index] / 255.0)
    return features


def paired_reward_summary(
    edge_rewards: Sequence[float], cloud_rewards: Sequence[float]
) -> dict[str, float]:
    """Summarize fixed endpoints and the per-task oracle on paired rewards."""

    if len(edge_rewards) != len(cloud_rewards) or not edge_rewards:
        raise ValueError("paired rewards must be non-empty and have equal length")
    n_tasks = len(edge_rewards)
    edge_mean = sum(float(value) for value in edge_rewards) / n_tasks
    cloud_mean = sum(float(value) for value in cloud_rewards) / n_tasks
    best_fixed = max(edge_mean, cloud_mean)
    oracle = sum(
        max(float(edge), float(cloud))
        for edge, cloud in zip(edge_rewards, cloud_rewards, strict=True)
    ) / n_tasks
    return {
        "edge": edge_mean,
        "cloud": cloud_mean,
        "best_fixed": best_fixed,
        "oracle": oracle,
        "oracle_gap": oracle - best_fixed,
    }
