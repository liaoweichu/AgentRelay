#!/usr/bin/env python3
"""G14b: run the four InterCode-SQL fixed endpoints (edge/cloud x train/dev).

Runs E4B (edge) and 12B (cloud) over the selected-libraries task manifests
(built by build_intersql_subset_manifest.py) and writes paired episode sets per
role+split. Intended to run on the 4090D (Gemma inference); MySQL must be up.

Output layout (out_dir):
  intercode-sql-<role>-<split>-episodes.json   (native intercode episodes)
  intercode-sql-matrix-receipt.json            (run provenance + arm stats)
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from contextlib import suppress
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import StorageLayout  # noqa: E402
from agentrelay.inference import HFModelExecutor, NativeGenerationConfig  # noqa: E402
from agentrelay.intercode_sql import parse_sql_action  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402

RESULTS = PROJECT_ROOT / "results/intercode-sql-g14-matrix"
TRAIN_MANIFEST = RESULTS / "ic_spider_train_subset.json"
DEV_MANIFEST = RESULTS / "ic_spider_dev_subset.json"

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

MAX_OBS_LEN = 1500
SYSTEM_PROMPT = """You are an expert SQL analyst. You are given a natural language question and the schema of a MySQL database. Write a SQL query that answers the question.

Rules:
1. You may run exploratory queries (SHOW TABLES, DESCRIBE <table>, SELECT ...) to inspect the data.
2. Output ONLY a valid SQL statement, enclosed in a ```sql code block.
3. When you are confident your last SQL query produces the correct answer, reply with the single word: submit
4. Do not ask questions. Do not output anything other than the SQL block or the word submit.
"""


def _iou_reward(agent_rows, gold_rows):
    from scipy.stats import kendalltau
    from collections import Counter as _C
    from itertools import chain, groupby
    from operator import itemgetter
    import math
    if not isinstance(agent_rows, list) or not isinstance(gold_rows, list):
        return 0.0
    list_agent = [str(x) for x in agent_rows]
    list_eval = [str(x) for x in gold_rows]
    dist_agent = _C(list_agent)
    dist_eval = _C(list_eval)
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


def _truncate_obs(obs):
    if obs is None:
        return "None"
    if len(obs) > MAX_OBS_LEN:
        return obs[:MAX_OBS_LEN] + f" ... [truncated {len(obs) - MAX_OBS_LEN} chars]"
    return obs


def _run_episode(executor, model, task, max_steps):
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
    termination = "max_steps_without_submit"

    def score(rows):
        return _iou_reward(rows, executor.gold_rows(db, gold))

    for step in range(max_steps):
        result = model.generate(messages)
        text = result.text
        parsed = parse_sql_action(text)
        kind, sql = parsed.kind, parsed.sql
        action = {
            "step": step,
            "kind": kind,
            "raw": text,
            "prompt_hash": result.prompt_hash,
            "response_hash": result.response_hash,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "peak_cuda_memory_bytes": result.peak_cuda_memory_bytes,
        }
        if kind == "submit":
            action["sql"] = None
            reward = score(last_obs)
            action.update({"observation": _truncate_obs(last_obs_str), "reward": reward})
            actions.append(action)
            done = True
            termination = "standalone_submit"
            break
        if kind in {"sql", "sql_submit"}:
            action["sql"] = sql
        else:
            action.update({"error": "unparseable", "observation": None})
            actions.append(action)
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": "Your last response contained no SQL and no `submit`. Output a ```sql block or the word `submit`.",
                }
            )
            continue
        rows, error = executor.run(db, sql)
        if error is not None:
            obs_str = f"Error executing query: {error}"
            last_obs = None
        else:
            last_obs = rows
            obs_str = str(rows) if rows is not None else "Success (no rows)"
        last_obs_str = obs_str
        action.update({"observation": _truncate_obs(obs_str), "exec_error": error is not None})
        if kind == "sql_submit":
            reward = score(last_obs)
            action["reward"] = reward
            actions.append(action)
            done = True
            termination = "combined_sql_submit"
            break
        actions.append(action)
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": f"SQL Output: {_truncate_obs(obs_str)}\n\nContinue. Output another ```sql block or `submit` if correct.",
            }
        )
    if done:
        last_query_reward = reward
    else:
        last_query_reward = score(last_obs)
        reward = 0.0
    sql_seq = [a.get("sql") for a in actions if a.get("kind") in {"sql", "sql_submit"}]
    max_run, cur_run = 0, 1
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
        "task_id": task["task_id"],
        "hardness": task.get("hardness", "unknown"),
        "reward": float(reward),
        "last_query_reward": float(last_query_reward),
        "success": float(reward >= 0.999),
        "termination": termination,
        "max_duplicate_run": max_run,
        "n_sql_actions": len(sql_seq),
        "n_steps": len(actions),
        "actions": actions,
    }


def arm_stats(eps):
    n = len(eps)
    if n == 0:
        return {}
    success = sum(e["success"] for e in eps) / n
    parse_ok = sum(1 for e in eps if e["n_sql_actions"] > 0) / n
    loop = sum(1 for e in eps if e["max_duplicate_run"] >= 4) / n
    submitted = sum(
        1 for e in eps if e["termination"] in {"standalone_submit", "combined_sql_submit"}
    ) / n
    combined = sum(1 for e in eps if e["termination"] == "combined_sql_submit") / n
    max_steps = sum(1 for e in eps if e["termination"] == "max_steps_without_submit") / n
    return {
        "n": n,
        "success_rate": round(success, 4),
        "parse_rate": round(parse_ok, 4),
        "submission_rate": round(submitted, 4),
        "combined_submit_rate": round(combined, 4),
        "max_steps_without_submit_rate": round(max_steps, 4),
        "duplicate_loop_rate": round(loop, 4),
        "mean_max_duplicate_run": round(sum(e["max_duplicate_run"] for e in eps) / n, 2),
        "mean_steps": round(sum(e["n_steps"] for e in eps) / n, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("edge", "cloud"), required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--out-dir", default=str(RESULTS))
    parser.add_argument("--manifest")
    args = parser.parse_args()

    manifest = Path(args.manifest) if args.manifest else (
        TRAIN_MANIFEST if args.split == "train" else DEV_MANIFEST
    )
    tasks = json.loads(manifest.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = StorageLayout.from_env()
    executor = SqlExecutor()

    cfg = MODEL_REVISIONS[args.role]
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
    episodes = []
    for i, task in enumerate(tasks):
        ep = _run_episode(executor, model, task, args.max_steps)
        ep["role"] = args.role
        ep["split"] = args.split
        ep["benchmark"] = "intercode/sql/spider"
        ep["dataset_revision"] = str(PROJECT_ROOT / "repositories/InterCode/data/sql/spider")
        episodes.append(ep)
        print(f"  [{args.role}:{args.split}] {i + 1}/{len(tasks)}", flush=True)
    target = out_dir / f"intercode-sql-{args.role}-{args.split}-episodes.json"
    target.write_text(canonical_json(episodes) + "\n", encoding="utf-8")

    stats = arm_stats(episodes)
    receipt = {
        "role": args.role,
        "split": args.split,
        "manifest": str(manifest),
        "n_tasks": len(tasks),
        "max_steps": args.max_steps,
        "model": cfg,
        "arm_stats": stats,
        "hardness_counts": dict(Counter(e["hardness"] for e in episodes)),
        "episodes_hash": sha256_json(episodes),
        "output": str(target),
    }
    (out_dir / f"intercode-sql-{args.role}-{args.split}-receipt.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )
    del model
    gc.collect()
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
