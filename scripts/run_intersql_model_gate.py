#!/usr/bin/env python3
"""InterCode-SQL 50-task model gate for the C3 reward-aware router.

Runs the frozen Gemma 4 edge/cloud pair (E4B vs 12B) as interactive SQL
agents on a deterministic subset of the InterCode Spider-dev tasks.  The
action protocol is raw SQL text (no function-calling / text-JSON schema).
Reward is the execution-based row IoU scaled by sort correlation, identical
to the official InterCode SqlEnv.

Gate metrics (mirroring the tau2 fork decision):
  - per-arm action parse rate
  - per-arm duplicate-loop rate (no large identical-action loops)
  - per-arm success rate (reward == 1.0), non-zero and non-saturated
  - bidirectional exclusive wins
  - oracle gap (12B - E4B success), target >= 3-5 pp
  - non-tie task count (sufficient for OOF, not single digits)

Only after the gate passes may the full train/dev endpoint matrix be built.
The official Spider test split is sealed and never used here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from itertools import chain, groupby
from operator import itemgetter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.inference import HFModelExecutor, NativeGenerationConfig  # noqa: E402
from agentrelay.config import StorageLayout  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402

SPIDER_DEV = PROJECT_ROOT / "repositories/InterCode/data/sql/spider/ic_spider_dev.json"
SPIDER_DBS = PROJECT_ROOT / "repositories/InterCode/data/sql/spider/ic_spider_dbs.sql"

MODEL_REVISIONS = {
    "edge": {
        "model_id": "google/gemma-4-E4B-it",
        "revision": "da2249f601f308b930979e20827bb464761fed65",
    },
    "cloud": {
        "model_id": "google/gemma-4-12b-it",
        "revision": "a69a4a15fe6a1b5a51373df662c1472be9b67683",
    },
}

THINK_STRIP = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
CODE_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL)


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
            # kendalltau needs numeric input; string tuples (e.g. text rows)
            # fall back to IoU-only (no sort penalty) to match the original
            # semantics where sort scaling only applies to numeric rows.
            try:
                corr = kendalltau(agent_intx, eval_intx, nan_policy="omit").statistic
            except (TypeError, ValueError):
                corr = 1.0
            if corr == corr and corr is not None:  # not NaN
                reward = round(corr * reward, 2)
    return reward


class SqlExecutor:
    """Deterministic MySQL executor backed by the native MySQL server."""

    def __init__(self, host="127.0.0.1", port=3306, user="admin", password="admin"):
        import mysql.connector
        self._ctor = mysql.connector.connect
        self._cfg = dict(host=host, port=port, user=user, password=password)
        self._conn = None

    def _ensure(self):
        if self._conn is None or not self._conn.is_connected():
            self._conn = self._ctor(**self._cfg)
        return self._conn

    def run(self, db, sql):
        """Execute `sql` against database `db`. Returns (rows_or_None, error_str_or_None)."""
        conn = self._ensure()
        cur = conn.cursor(buffered=True)
        error = None
        rows = None
        try:
            cur.execute(f"USE `{db}`")
            cur.execute(sql)
            if cur.description is not None:
                rows = cur.fetchall()
        except Exception as exc:
            error = str(exc)
        finally:
            try:
                cur.close()
            except Exception:
                pass
        return rows, error

    def gold_rows(self, db, gold_sql):
        # gold may be a single statement; split on ';' and take the last
        # non-empty statement (InterCode gold is a single query).
        for cand in reversed([s.strip() for s in gold_sql.split(";") if s.strip()]):
            rows, error = self.run(db, cand)
            if error is None and rows is not None:
                return rows
        return None


SYSTEM_PROMPT = """You are an expert SQL analyst. You are given a natural language question and the schema of a MySQL database. Write a SQL query that answers the question.

Rules:
1. You may run exploratory queries (SHOW TABLES, DESCRIBE <table>, SELECT ...) to inspect the data.
2. Output ONLY a valid SQL statement, enclosed in a ```sql code block.
3. When you are confident your last SQL query produces the correct answer, reply with the single word: submit
4. Do not ask questions. Do not output anything other than the SQL block or the word submit.
"""


def _extract_action(text):
    """Return ('submit', None) or ('sql', statement) from a model response.

    If a SQL code block is present, execute it even if the model also wrote
    ``submit`` (Gemma 4 models tend to emit ``sql`` then ``submit`` in the same
    turn; the SQL block is the intended answer query).  A bare ``submit`` with
    no SQL is the real submit action.
    """
    stripped = THINK_STRIP.sub("", text or "").strip()
    m = CODE_BLOCK.search(stripped)
    if m:
        sql = m.group(1).strip().rstrip(";")
        if sql and sql.upper() != "SUBMIT":
            return "sql", sql
    if re.search(r"\bsubmit\b", stripped, re.IGNORECASE):
        return "submit", None
    # Fallback: last line that looks like SQL
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    for ln in reversed(lines):
        head = ln.lstrip().upper()
        if head.startswith(("SELECT", "SHOW", "DESC", "DESCRIBE", "WITH", "EXPLAIN")):
            return "sql", ln.rstrip(";")
    return None, None


def _build_task_list(n_tasks, seed):
    data = json.loads(Path(SPIDER_DEV).read_text(encoding="utf-8"))
    rng = random.Random(seed)
    # Stratify across hardness bins to keep the 50-task subset representative.
    by_hardness = {}
    for idx, rec in enumerate(data):
        by_hardness.setdefault(rec.get("hardness", "unknown"), []).append(idx)
    selected = []
    buckets = sorted(by_hardness)
    # round-robin across hardness buckets
    max_len = max(len(v) for v in by_hardness.values())
    for round_idx in range(max_len):
        for b in buckets:
            bucket = by_hardness[b]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
        if len(selected) >= n_tasks:
            break
    selected = selected[:n_tasks]
    rng.shuffle(selected)
    return [data[i] for i in selected], selected


# Cap the observation fed back to the model so the multi-step interaction loop
# cannot grow the context without bound.  This mirrors the official InterCode
# SqlEnv, which also truncates observations.  It does NOT change the model,
# precision, thinking mode, or max_new_tokens (the generation budget) -- it only
# bounds the accumulated context length, which is the OOM root cause for the
# 12B arm on long SQL result sets.
MAX_OBS_LEN = 1500


def _truncate_obs(obs: str) -> str:
    if obs is None:
        return "None"
    if len(obs) > MAX_OBS_LEN:
        return obs[:MAX_OBS_LEN] + f" ... [truncated {len(obs) - MAX_OBS_LEN} chars]"
    return obs


def _run_episode(executor, model, task, max_steps):
    """Run one agent episode. Returns episode dict with per-step actions."""
    db = task["db"]
    query = task["query"]
    gold = task["gold"]
    schema = task.get("db_tables", {})
    schema_str = "\n".join(
        f"Table {tname} columns: {', '.join(cols)}" for tname, cols in schema.items()
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Database: {db}\nSchema:\n{schema_str}\n\n"
                f"Question: {query}\n\n"
                "Write a SQL query to answer the question. Explore if needed, then submit."
            ),
        },
    ]
    actions = []
    last_obs = None
    last_obs_str = None
    reward = 0.0
    done = False
    for step in range(max_steps):
        result = model.generate(messages)
        text = result.text
        kind, sql = _extract_action(text)
        action = {
            "step": step,
            "kind": kind,
            "raw": text,
            "prompt_hash": result.prompt_hash,
            "response_hash": result.response_hash,
            "output_tokens": result.output_tokens,
        }
        if kind == "submit":
            action["sql"] = None
            reward = _iou_reward(last_obs, executor.gold_rows(db, gold))
            action.update({"observation": _truncate_obs(last_obs_str), "reward": reward})
            actions.append(action)
            done = True
            break
        if kind == "sql":
            action["sql"] = sql
        else:
            action.update({"error": "unparseable", "observation": None})
            actions.append(action)
            # Appending the model text keeps the conversation going; no exec.
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": "Your last response contained no SQL and no `submit`. Output a ```sql block or the word `submit`.",
                }
            )
            continue
        # execute sql
        rows, error = executor.run(db, sql)
        if error is not None:
            obs_str = f"Error executing query: {error}"
            last_obs = None
        else:
            last_obs = rows
            obs_str = str(rows) if rows is not None else "Success (no rows)"
        last_obs_str = obs_str
        action.update({"observation": _truncate_obs(obs_str), "exec_error": error is not None})
        actions.append(action)
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": f"SQL Output: {_truncate_obs(obs_str)}\n\nContinue. Output another ```sql block or `submit` if correct.",
            }
        )
    if not done:
        reward = _iou_reward(last_obs, executor.gold_rows(db, gold))
    # duplicate-loop detection: count max run of identical sql actions
    sql_seq = [a.get("sql") for a in actions if a.get("kind") == "sql"]
    max_run = 0
    cur_run = 1
    for i in range(1, len(sql_seq)):
        if sql_seq[i] == sql_seq[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    max_run = max(max_run, cur_run) if sql_seq else 0
    return {
        "db": db,
        "query": query,
        "gold": gold,
        "task_id": task.get("id", f"{db}-{query}"),
        "reward": float(reward),
        "success": float(reward >= 0.999),
        "max_duplicate_run": max_run,
        "n_sql_actions": len(sql_seq),
        "n_steps": len(actions),
        "actions": actions,
    }


def _emit_provenance():
    code_rev = None
    try:
        import subprocess
        code_rev = (
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        code_rev = "unknown"
    return {
        "code_revision": code_rev,
        "intercode_revision": "c3e46d827cfc9d4c704ec078f7abf9f41e3191d8",
        "dataset": "InterCode-Spider-dev",
        "dataset_path": str(SPIDER_DEV),
        "official_test_sealed": True,
        "execution": "native_mysql_8.0.46_server",
        "protocol": "raw_sql_text_no_function_calling",
        "model_pair": MODEL_REVISIONS,
        "quantization": "bnb_4bit",
        "dtype": "bfloat16",
        "reward": "intercode_sql_iou_sort_scaled",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/intercode-sql-gate"))
    parser.add_argument("--edges-only-smoke", action="store_true", help="run 2 tasks as a smoke test")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = StorageLayout.from_env()
    executor = SqlExecutor()

    tasks, selected_idx = _build_task_list(args.n_tasks, args.seed)
    if args.edges_only_smoke:
        tasks = tasks[:2]
        args.max_steps = 4

    # Deterministic model order; load one at a time to bound memory.
    episodes = {}
    manifest = []
    for role in ("edge", "cloud"):
        cfg = MODEL_REVISIONS[role]
        ngen = NativeGenerationConfig(
            model_id=cfg["model_id"],
            revision=cfg["revision"],
            model_source="modelscope",
            dtype="bfloat16",
            quantization="bnb_4bit",
            device_map="auto",
            max_new_tokens=1024,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            seed=0,
            architecture="multimodal_lm",
            enable_thinking=False,
            local_files_only=True,
            trust_remote_code=True,
        )
        model = HFModelExecutor(ngen, storage)
        role_episodes = []
        for task in tasks:
            ep = _run_episode(executor, model, task, args.max_steps)
            ep["role"] = role
            role_episodes.append(ep)
        episodes[role] = role_episodes
        # free before loading the other arm
        del model
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    # ---- gate metrics ----
    def arm_stats(role):
        eps = episodes[role]
        n = len(eps)
        success = sum(e["success"] for e in eps) / n
        parse_ok = sum(1 for e in eps if e["n_sql_actions"] > 0) / n
        loop = sum(1 for e in eps if e["max_duplicate_run"] >= 4) / n
        return {
            "n": n,
            "success_rate": round(success, 4),
            "parse_rate": round(parse_ok, 4),
            "duplicate_loop_rate": round(loop, 4),
            "mean_max_duplicate_run": round(sum(e["max_duplicate_run"] for e in eps) / n, 2),
            "mean_steps": round(sum(e["n_steps"] for e in eps) / n, 2),
        }

    edge_stats = arm_stats("edge")
    cloud_stats = arm_stats("cloud")
    oracle_gap_pp = round((cloud_stats["success_rate"] - edge_stats["success_rate"]) * 100, 2)

    # paired / exclusive wins over shared task order
    paired = []
    for e_ep, c_ep in zip(episodes["edge"], episodes["cloud"]):
        paired.append(
            {
                "query": e_ep["query"],
                "reward_edge": e_ep["reward"],
                "reward_cloud": c_ep["reward"],
                "success_edge": bool(e_ep["success"]),
                "success_cloud": bool(c_ep["success"]),
            }
        )
    non_tie = sum(1 for p in paired if p["success_edge"] != p["success_cloud"])
    edge_excl = sum(1 for p in paired if p["success_edge"] and not p["success_cloud"])
    cloud_excl = sum(1 for p in paired if p["success_cloud"] and not p["success_edge"])
    both = sum(1 for p in paired if p["success_edge"] and p["success_cloud"])

    gate = {
        "scope": "intercode_sql_50task_model_gate",
        "gate_pass": (
            non_tie >= 20
            and oracle_gap_pp >= 3.0
            and 0.05 <= edge_stats["success_rate"] <= 0.95
            and 0.05 <= cloud_stats["success_rate"] <= 0.95
            and edge_stats["parse_rate"] >= 0.5
            and cloud_stats["parse_rate"] >= 0.5
            and edge_stats["duplicate_loop_rate"] <= 0.3
            and cloud_stats["duplicate_loop_rate"] <= 0.3
            and edge_excl >= 1
            and cloud_excl >= 1
        ),
        "arms": {"edge": edge_stats, "cloud": cloud_stats},
        "oracle_gap_pp": oracle_gap_pp,
        "paired": {
            "n_tasks": len(paired),
            "non_tie": non_tie,
            "edge_exclusive": edge_excl,
            "cloud_exclusive": cloud_excl,
            "both_success": both,
        },
        "protocol": "raw_sql_text",
        "official_test_sealed": True,
        "provenance": _emit_provenance(),
        "task_selection": {"n_tasks": args.n_tasks, "seed": args.seed, "indices": selected_idx},
        "episodes_hash_edge": sha256_json(episodes["edge"]),
        "episodes_hash_cloud": sha256_json(episodes["cloud"]),
    }
    gate["gate_hash"] = sha256_json(gate)

    # artifacts
    (out_dir / "intercode-sql-gate.json").write_text(canonical_json(gate) + "\n", encoding="utf-8")
    (out_dir / "intercode-sql-edge-episodes.json").write_text(
        canonical_json(episodes["edge"]) + "\n", encoding="utf-8"
    )
    (out_dir / "intercode-sql-cloud-episodes.json").write_text(
        canonical_json(episodes["cloud"]) + "\n", encoding="utf-8"
    )
    (out_dir / "intercode-sql-paired.csv").write_text(
        "query,reward_edge,reward_cloud,success_edge,success_cloud\n"
        + "\n".join(
            f"{json.dumps(p['query'])},{p['reward_edge']},{p['reward_cloud']},{int(p['success_edge'])},{int(p['success_cloud'])}"
            for p in paired
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(gate, indent=2))
    print(f"GATE_PASS={gate['gate_pass']}  oracle_gap_pp={oracle_gap_pp} non_tie={non_tie}")
    return 0 if gate["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())