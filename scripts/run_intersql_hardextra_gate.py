#!/usr/bin/env python3
"""G13: disjoint hard/extra gate for the C3 reward-aware router.

Reuses the corrected InterCode-SQL protocol (parse_sql_action, official submit
semantics, IoU reward) but samples 100 FRESHLY UNRUN tasks dominated by hard /
extra Spider-dev instances, so the two frozen Gemma 4 arms are expected to
differentiate more than on the easy/medium-heavy 50-task gate.

This is the first step that re-invokes E4B/12B model inference (on the 4090D).
The previously-saved 50 raw responses are NOT reused; the 100-task set is
disjoint from those 50 by (db, query) identity.

Gate criteria mirror G12: enough reward non-ties, oracle gap >= 3 pp, both-arm
parse/submission/non-saturation, no duplicate loops, bidirectional exclusive
wins. Only if this passes may the full train/dev matrix be built (G14).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import gc
from collections import Counter
from contextlib import suppress
from io import StringIO
from itertools import chain, groupby
from operator import itemgetter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import StorageLayout  # noqa: E402
from agentrelay.inference import HFModelExecutor, NativeGenerationConfig  # noqa: E402
from agentrelay.intercode_sql import (  # noqa: E402
    paired_reward_summary,
    parse_sql_action,
)
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402

SPIDER_DEV = PROJECT_ROOT / "repositories/InterCode/data/sql/spider/ic_spider_dev.json"
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


def _truncate_obs(obs: str) -> str:
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
        "task_id": task.get("id", f"{db}-{query}"),
        "task_index": task.get("_selection_index"),
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


def _load_episodes(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--out-dir", default=str(PROJECT_ROOT / "results/intercode-sql-gate-v2-hardextra")
    )
    parser.add_argument(
        "--edges-only-smoke", action="store_true", help="run 2 tasks as a smoke test"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = StorageLayout.from_env()
    executor = SqlExecutor()

    data = json.loads(Path(SPIDER_DEV).read_text(encoding="utf-8"))

    # Disjoint from the 50 already-run tasks by (db, query).
    run_keys = set()
    for role in ("edge", "cloud"):
        for ep in _load_episodes(
            PROJECT_ROOT / "results/intercode-sql-gate" / f"intercode-sql-{role}-episodes.json"
        ):
            run_keys.add((ep["db"], ep["query"].strip()))
    candidates = [
        (i, r)
        for i, r in enumerate(data)
        if (r.get("db"), str(r.get("query", "")).strip()) not in run_keys
    ]
    # hard + extra only
    hard_extra = [(i, r) for i, r in candidates if r.get("hardness") in {"hard", "extra"}]
    if args.edges_only_smoke:
        hard_extra = hard_extra[:2]
        args.max_steps = 4

    # deterministic stratified selection across hardness buckets
    import random
    rng = random.Random(args.seed)
    by_hard = {}
    for i, r in hard_extra:
        by_hard.setdefault(r["hardness"], []).append((i, r))
    for k in by_hard:
        rng.shuffle(by_hard[k])
    selected = []
    total_he = len(hard_extra)
    exact = {k: args.n_tasks * len(v) / total_he for k, v in by_hard.items()}
    alloc = {k: math.floor(v) for k, v in exact.items()}
    remaining = args.n_tasks - sum(alloc.values())
    order = sorted(by_hard, key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in order[:remaining]:
        alloc[k] += 1
    for k in sorted(by_hard):
        selected.extend(by_hard[k][: alloc[k]])
    if len(selected) > args.n_tasks:
        selected = selected[:args.n_tasks]
    rng.shuffle(selected)

    tasks = [{**r, "_selection_index": i} for i, r in selected]
    print(f"G13 tasks: n={len(tasks)} hardness={dict(Counter(t['hardness'] for t in tasks))}", flush=True)

    episodes = {}
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
            print(f"  [{role}] {len(role_episodes)}/{len(tasks)} done", flush=True)
        episodes[role] = role_episodes
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    def arm_stats(role):
        eps = episodes[role]
        n = len(eps)
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

    edge_stats = arm_stats("edge")
    cloud_stats = arm_stats("cloud")

    paired = []
    for e_ep, c_ep in zip(episodes["edge"], episodes["cloud"]):
        paired.append(
            {
                "task_id": e_ep["task_id"],
                "task_index": e_ep["task_index"],
                "hardness": e_ep["hardness"],
                "query": e_ep["query"],
                "reward_edge": e_ep["reward"],
                "reward_cloud": c_ep["reward"],
                "success_edge": bool(e_ep["success"]),
                "success_cloud": bool(c_ep["success"]),
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
        "scope": "intercode_sql_100task_hardextra_disjoint_gate",
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
        "execution": "model_inference_4090D",
        "official_test_sealed": True,
        "task_selection": {
            "n_tasks": len(tasks),
            "seed": args.seed,
            "strategy": "disjoint_hard_extra_proportional_stratified",
            "hardness_counts": dict(Counter(t["hardness"] for t in tasks)),
            "excluded_run_tasks": len(run_keys),
        },
        "episodes_hash_edge": sha256_json(episodes["edge"]),
        "episodes_hash_cloud": sha256_json(episodes["cloud"]),
    }
    gate["gate_hash"] = sha256_json(gate)

    (out_dir / "intercode-sql-gate.json").write_text(canonical_json(gate) + "\n", encoding="utf-8")
    (out_dir / "intercode-sql-edge-episodes.json").write_text(
        canonical_json(episodes["edge"]) + "\n", encoding="utf-8"
    )
    (out_dir / "intercode-sql-cloud-episodes.json").write_text(
        canonical_json(episodes["cloud"]) + "\n", encoding="utf-8"
    )
    csv_buffer = StringIO(newline="")
    csv_fields = ["task_id", "task_index", "hardness", "query", "reward_edge", "reward_cloud", "success_edge", "success_cloud"]
    writer = csv.DictWriter(csv_buffer, fieldnames=csv_fields, lineterminator="\n")
    writer.writeheader()
    for row in paired:
        writer.writerow({field: row[field] for field in csv_fields})
    (out_dir / "intercode-sql-paired.csv").write_text(csv_buffer.getvalue(), encoding="utf-8")

    print(json.dumps(gate, indent=2))
    print(
        f"GATE_PASS={gate['gate_pass']} oracle_gap_pp={oracle_gap_pp} "
        f"reward_non_tie={reward_non_tie}"
    )
    return 0 if gate["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())