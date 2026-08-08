#!/usr/bin/env python3
"""CPU/MySQL replay of the saved 50-task InterCode-SQL raw responses.

The reported INTERCODE-SQL-GATE decision was invalidated because the old action
extractor misread an atomic ``SQL-block + submit`` turn as a plain ``sql`` action,
then asked the model for another turn, manufacturing repeated-SQL loops. This
script does NOT call E4B/12B again.  It replays the already-saved per-step raw
responses through the corrected protocol parser (``parse_sql_action``), executes
each SQL statement against the native MySQL server, and recomputes the gate with
official InterCode semantics:

  - ``sql_submit`` (SQL block plus standalone submit in one turn) executes the
    SQL and requests evaluation immediately.
  - ``submit`` scores the last executed observation.
  - Running out of steps without a recognized submit yields official reward 0.0
    (the last-query score is retained only as a diagnostic signal).

The official Spider test split is sealed and never used here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from contextlib import suppress
from io import StringIO
from itertools import chain, groupby
from operator import itemgetter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.intercode_sql import (  # noqa: E402
    paired_reward_summary,
    parse_sql_action,
)
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402

RESULTS = PROJECT_ROOT / "results/intercode-sql-gate"
MAX_STEPS = 10


def _iou_reward(agent_rows, gold_rows):
    """Exact InterCode SqlEnv reward: row IoU scaled by sort correlation."""
    from scipy.stats import kendalltau
    if not isinstance(agent_rows, list) or not isinstance(gold_rows, list):
        return 0.0
    list_agent = [str(x) for x in agent_rows]
    list_eval = [str(x) for x in gold_rows]
    dist_agent = Counter(list_agent)
    dist_eval = Counter(list_eval)
    intersection = dist_agent & dist_eval
    get_key, get_val = itemgetter(0), itemgetter(1)
    merged = sorted(chain(dist_agent.items(), dist_eval.items()), key=get_key)
    union = {k: max(map(get_val, g)) for k, g in groupby(merged, key=get_key)}
    if len(union) == 0:
        return 1.0
    total_intersect = sum(v for v in intersection.values())
    total_union = sum(v for v in union.values())
    reward = total_intersect * 1.0 / total_union
    if len(intersection) > 0:
        def get_intersect_items(my_list, my_dict):
            result = []
            keep = dict(my_dict)
            for item in my_list:
                if item in keep:
                    keep[item] -= 1
                    if keep[item] == 0:
                        del keep[item]
                    result.append(item)
            return result
        agent_intx = get_intersect_items(list_agent.copy(), intersection.copy())
        eval_intx = get_intersect_items(list_eval.copy(), intersection.copy())
        if len(agent_intx) == len(eval_intx) and len(agent_intx) > 0:
            try:
                corr = kendalltau(agent_intx, eval_intx, nan_policy="omit").statistic
            except (TypeError, ValueError):
                corr = 1.0
            if corr is not None and not math.isnan(corr):
                reward = round(corr * reward, 2)
    return reward


class SqlExecutor:
    """Deterministic MySQL executor backed by the native MySQL server."""

    def __init__(self, host="127.0.0.1", port=3306, user="admin", password="admin"):
        import mysql.connector
        self._ctor = mysql.connector.connect
        self._error_type = mysql.connector.Error
        self._cfg = {"host": host, "port": port, "user": user, "password": password}
        self._conn = None

    def _ensure(self):
        if self._conn is None or not self._conn.is_connected():
            self._conn = self._ctor(**self._cfg)
        return self._conn

    def run(self, db, sql):
        conn = self._ensure()
        cur = conn.cursor(buffered=True)
        error = None
        rows = None
        try:
            cur.execute(f"USE `{db}`")
            cur.execute(sql)
            if cur.description is not None:
                rows = cur.fetchall()
        except self._error_type as exc:
            error = str(exc)
        finally:
            with suppress(self._error_type):
                cur.close()
        return rows, error

    def gold_rows(self, db, gold_sql):
        for cand in reversed([s.strip() for s in gold_sql.split(";") if s.strip()]):
            rows, error = self.run(db, cand)
            if error is None and rows is not None:
                return rows
        return None


def _max_duplicate_run(sql_actions):
    max_run, cur_run = 0, 1
    for i in range(1, len(sql_actions)):
        if sql_actions[i] == sql_actions[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return max(max_run, cur_run) if sql_actions else 0


def _replay_episode(executor, saved_episode):
    """Replay one saved episode's raw responses with corrected protocol semantics."""
    db = saved_episode["db"]
    gold = saved_episode["gold"]
    last_obs = None
    last_obs_str = None
    reward = 0.0
    done = False
    termination = "max_steps_without_submit"
    actions = []

    for step in range(MAX_STEPS):
        if step >= len(saved_episode["actions"]):
            break
        raw = saved_episode["actions"][step].get("raw", "")
        parsed = parse_sql_action(raw)
        kind, sql = parsed.kind, parsed.sql
        action = {
            "step": step,
            "kind": kind,
            "raw": raw,
            "prompt_hash": saved_episode["actions"][step].get("prompt_hash"),
            "response_hash": saved_episode["actions"][step].get("response_hash"),
            "output_tokens": saved_episode["actions"][step].get("output_tokens"),
        }
        if kind == "submit":
            action["sql"] = None
            reward = _iou_reward(last_obs, executor.gold_rows(db, gold))
            action.update({"observation": last_obs_str, "reward": reward})
            actions.append(action)
            done = True
            termination = "standalone_submit"
            break
        if kind == "sql" or kind == "sql_submit":
            action["sql"] = sql
            rows, error = executor.run(db, sql)
            if error is not None:
                obs_str = f"Error executing query: {error}"
                last_obs = None
            else:
                last_obs = rows
                obs_str = str(rows) if rows is not None else "Success (no rows)"
            last_obs_str = obs_str
            action.update(
                {"observation": obs_str, "exec_error": error is not None}
            )
            if kind == "sql_submit":
                reward = _iou_reward(last_obs, executor.gold_rows(db, gold))
                action["reward"] = reward
                actions.append(action)
                done = True
                termination = "combined_sql_submit"
                break
            actions.append(action)
        else:
            action.update({"error": "unparseable", "sql": None, "observation": None})
            actions.append(action)
            continue

    if not done:
        last_query_reward = _iou_reward(last_obs, executor.gold_rows(db, gold))
        reward = 0.0
    else:
        last_query_reward = reward

    sql_seq = [a.get("sql") for a in actions if a.get("kind") in {"sql", "sql_submit"}]
    return {
        "db": db,
        "query": saved_episode["query"],
        "gold": gold,
        "task_id": saved_episode["task_id"],
        "reward": float(reward),
        "last_query_reward": float(last_query_reward),
        "success": float(reward >= 0.999),
        "termination": termination,
        "max_duplicate_run": _max_duplicate_run(sql_seq),
        "n_sql_actions": len(sql_seq),
        "n_steps": len(actions),
        "actions": actions,
    }


def _arm_stats(role, episodes):
    n = len(episodes)
    success = sum(e["success"] for e in episodes) / n
    parse_ok = sum(1 for e in episodes if e["n_sql_actions"] > 0) / n
    loop = sum(1 for e in episodes if e["max_duplicate_run"] >= 4) / n
    submitted = sum(
        1 for e in episodes if e["termination"] in {"standalone_submit", "combined_sql_submit"}
    ) / n
    combined = sum(1 for e in episodes if e["termination"] == "combined_sql_submit") / n
    max_steps = sum(1 for e in episodes if e["termination"] == "max_steps_without_submit") / n
    return {
        "n": n,
        "success_rate": round(success, 4),
        "parse_rate": round(parse_ok, 4),
        "submission_rate": round(submitted, 4),
        "combined_submit_rate": round(combined, 4),
        "max_steps_without_submit_rate": round(max_steps, 4),
        "duplicate_loop_rate": round(loop, 4),
        "mean_max_duplicate_run": round(sum(e["max_duplicate_run"] for e in episodes) / n, 2),
        "mean_steps": round(sum(e["n_steps"] for e in episodes) / n, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/intercode-sql-gate-v2-replay"))
    args = parser.parse_args()

    executor = SqlExecutor()
    episodes = {}
    for role in ("edge", "cloud"):
        saved = json.loads(
            (RESULTS / f"intercode-sql-{role}-episodes.json").read_text(encoding="utf-8")
        )
        episodes[role] = [_replay_episode(executor, ep) for ep in saved]

    edge_stats = _arm_stats("edge", episodes["edge"])
    cloud_stats = _arm_stats("cloud", episodes["cloud"])

    paired = []
    for e_ep, c_ep in zip(episodes["edge"], episodes["cloud"]):
        paired.append(
            {
                "task_id": e_ep["task_id"],
                "reward_edge": e_ep["reward"],
                "reward_cloud": c_ep["reward"],
                "success_edge": bool(e_ep["success"]),
                "success_cloud": bool(c_ep["success"]),
                "termination_edge": e_ep["termination"],
                "termination_cloud": c_ep["termination"],
            }
        )
    reward_non_tie = sum(1 for p in paired if abs(p["reward_edge"] - p["reward_cloud"]) > 1e-9)
    success_non_tie = sum(1 for p in paired if p["success_edge"] != p["success_cloud"])
    edge_excl = sum(1 for p in paired if p["success_edge"] and not p["success_cloud"])
    cloud_excl = sum(1 for p in paired if p["success_cloud"] and not p["success_edge"])
    both = sum(1 for p in paired if p["success_edge"] and p["success_cloud"])
    success_summary = paired_reward_summary(
        [float(p["success_edge"]) for p in paired],
        [float(p["success_cloud"]) for p in paired],
    )
    reward_summary = paired_reward_summary(
        [float(p["reward_edge"]) for p in paired],
        [float(p["reward_cloud"]) for p in paired],
    )
    oracle_gap_pp = round(success_summary["oracle_gap"] * 100, 2)

    checks = {
        "enough_reward_non_ties": reward_non_tie >= 20,
        "oracle_gap_ge_3pp": oracle_gap_pp >= 3.0,
        "edge_success_non_saturated": 0.05 <= edge_stats["success_rate"] <= 0.95,
        "cloud_success_non_saturated": 0.05 <= cloud_stats["success_rate"] <= 0.95,
        "edge_parse_rate": edge_stats["parse_rate"] >= 0.5,
        "cloud_parse_rate": cloud_stats["parse_rate"] >= 0.5,
        "edge_submission_rate": edge_stats["submission_rate"] >= 0.95,
        "cloud_submission_rate": cloud_stats["submission_rate"] >= 0.95,
        "edge_duplicate_loop_rate": edge_stats["duplicate_loop_rate"] <= 0.3,
        "cloud_duplicate_loop_rate": cloud_stats["duplicate_loop_rate"] <= 0.3,
        "edge_exclusive_success": edge_excl >= 1,
        "cloud_exclusive_success": cloud_excl >= 1,
    }

    gate = {
        "scope": "intercode_sql_50task_model_gate_replay",
        "gate_pass": all(checks.values()),
        "checks": checks,
        "arms": {"edge": edge_stats, "cloud": cloud_stats},
        "oracle_gap_pp": oracle_gap_pp,
        "success_summary": success_summary,
        "reward_summary": reward_summary,
        "paired": {
            "n_tasks": len(paired),
            "reward_non_tie": reward_non_tie,
            "success_non_tie": success_non_tie,
            "edge_exclusive": edge_excl,
            "cloud_exclusive": cloud_excl,
            "both_success": both,
        },
        "protocol": "raw_sql_text",
        "official_test_sealed": True,
        "execution": "cpu_mysql_replay_no_model_inference",
        "source_episodes_dir": str(RESULTS),
        "episodes_hash_edge": sha256_json(episodes["edge"]),
        "episodes_hash_cloud": sha256_json(episodes["cloud"]),
    }
    gate["gate_hash"] = sha256_json(gate)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intercode-sql-gate.json").write_text(canonical_json(gate) + "\n", encoding="utf-8")
    (out_dir / "intercode-sql-edge-episodes.json").write_text(
        canonical_json(episodes["edge"]) + "\n", encoding="utf-8"
    )
    (out_dir / "intercode-sql-cloud-episodes.json").write_text(
        canonical_json(episodes["cloud"]) + "\n", encoding="utf-8"
    )
    csv_buffer = StringIO(newline="")
    csv_fields = [
        "task_id",
        "reward_edge",
        "reward_cloud",
        "success_edge",
        "success_cloud",
        "termination_edge",
        "termination_cloud",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=csv_fields, lineterminator="\n")
    writer.writeheader()
    for row in paired:
        writer.writerow({field: row[field] for field in csv_fields})
    (out_dir / "intercode-sql-paired.csv").write_text(csv_buffer.getvalue(), encoding="utf-8")

    # termination breakdown across both arms
    termination_breakdown = {}
    for role in ("edge", "cloud"):
        term_counts = Counter(e["termination"] for e in episodes[role])
        termination_breakdown[role] = {
            "counts": dict(term_counts),
            "rates": {
                term: round(n / len(episodes[role]), 4) for term, n in term_counts.items()
            },
        }
    (out_dir / "termination-breakdown.json").write_text(
        canonical_json(termination_breakdown) + "\n", encoding="utf-8"
    )

    # replay receipt: provenance of the offline replay (no model inference)
    receipt = {
        "mode": "cpu_mysql_replay_of_saved_responses",
        "model_inference": "none",
        "gpu_used": False,
        "source_episodes_dir": str(RESULTS),
        "parser_module": "agentrelay.intercode_sql.parse_sql_action",
        "parser_revision": "g10_corrected_semantics",
        "max_steps": MAX_STEPS,
        "mysql": "native_server_8.0.46",
        "official_test_sealed": True,
        "n_tasks_per_arm": {role: len(episodes[role]) for role in ("edge", "cloud")},
        "gate_pass": gate["gate_pass"],
        "gate_hash": gate["gate_hash"],
        "episodes_hash_edge": gate["episodes_hash_edge"],
        "episodes_hash_cloud": gate["episodes_hash_cloud"],
    }
    (out_dir / "replay-receipt.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )

    print(json.dumps(gate, indent=2))
    print(
        f"GATE_PASS={gate['gate_pass']} oracle_gap_pp={oracle_gap_pp} "
        f"reward_non_tie={reward_non_tie}"
    )
    return 0 if gate["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())