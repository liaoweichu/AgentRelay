#!/usr/bin/env python3
"""Constrained diagnostic for cloud-arm InterCode-SQL action formatting.

Motivation
----------
This historical diagnostic varies prompt text while freezing the model. The
shared parser recognizes SQL-only, submit-only, and atomic SQL-plus-submit
responses so adapter behavior matches the formal gate.

Scope
-----
- Single-turn generation (no execution loop): classify the raw first response.
- Levers varied: prompt text / few-shot demo / format enforcement only.
- Frozen (must stay identical to the formal gate): model, revision, dtype
  (bfloat16), quantization (bnb_4bit), architecture (multimodal_lm),
  enable_thinking=False, max_new_tokens, do_sample=False, seed.
- Offline, small sample (n_tasks x n_variants generations).  No full matrix.
- The official Spider test split is sealed and never used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import StorageLayout
from agentrelay.inference import HFModelExecutor, NativeGenerationConfig
from agentrelay.intercode_sql import (
    parse_sql_action,
    proportional_stratified_indices,
)
from agentrelay.schema import canonical_json, sha256_json

SPIDER_DEV = PROJECT_ROOT / "repositories/InterCode/data/sql/spider/ic_spider_dev.json"

CLOUD_CFG = {
    "model_id": "google/gemma-4-12b-it",
    "revision": "a69a4a15fe6a1b5a51373df662c1472be9b67683",
    "model_source": "modelscope",
    "dtype": "bfloat16",
    "quantization": "bnb_4bit",
    "device_map": "auto",
    "max_new_tokens": 1024,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "seed": 0,
    "architecture": "multimodal_lm",
    "enable_thinking": False,
    "local_files_only": True,
    "trust_remote_code": True,
}

# ---- prompt variants (the ONLY lever varied) ----

BASE = (
    "You are an expert SQL analyst. You are given a natural language question "
    "and the schema of a MySQL database. Write a SQL query that answers the question.\n"
    "Rules:\n"
    "1. You may run exploratory queries (SHOW TABLES, DESCRIBE <table>, SELECT ...) to inspect the data.\n"
    "2. Output ONLY a valid SQL statement, enclosed in a ```sql code block.\n"
    "3. When you are confident your last SQL query produces the correct answer, reply with the single word: submit\n"
    "4. Do not ask questions. Do not output anything other than the SQL block or the word submit."
)

FEWSHOT_DEMO = """Here is an example of the interaction format.

Question: What are the names and grades for each high schooler?
Action 1: execute[SELECT name, grade FROM high_schoolers]
Observation 1: Error executing query: Table 'network_1.high_schoolers' doesn't exist
Action 2: execute[SHOW TABLES]
Observation 2: [('friend',), ('highschooler',), ('likes',)]
Action 3: execute[SELECT name, grade FROM highschooler]
Observation 3: [('John', 12), ('Haley', 10)]
Action 4: submit

Rules:
- Use `execute[<a single SQL statement>]` to run a query. Output exactly that action line, nothing else.
- Use `submit` as your final action when the answer is ready.
- Do not output any natural-language prose, explanations, or questions."""

MINIMAL = (
    "You query a MySQL database. Reply with exactly one of: a SQL statement, "
    "or the single word `submit` (when done). No explanation."
)

STRICT_FORMAT = (
    "Your output must be machine-parsed. You have two and only two allowed outputs:\n"
    "  A) A single SQL statement (no prose, no commentary, no question).\n"
    "  B) The exact token `submit`.\n"
    "Schema and question are below. Do not output anything else. Never use markdown fences around the SQL."
)

JSON_SQL = (
    "Respond in strict JSON with exactly one key: either {\"sql\": \"<SQL statement>\"} "
    "or {\"submit\": true}. No other text, no markdown, no prose."
)

PROMPT_VARIANTS = {
    "baseline": BASE,
    "fewshot_react": FEWSHOT_DEMO,
    "minimal": MINIMAL,
    "strict_format": STRICT_FORMAT,
    "json_sql": JSON_SQL,
}

def _classify(text: str) -> str:
    """Classify a raw single-turn response into an action-emission label."""
    parsed = parse_sql_action(text)
    if parsed.kind != "invalid":
        return parsed.kind
    stripped = (text or "").strip()
    # JSON structured probe
    if stripped.lstrip().startswith("{") and ('"sql"' in stripped or '"submit"' in stripped):
        return "json_structured"
    return "other"


def _schema_str(task) -> str:
    schema = task.get("db_tables", {})
    return "\n".join(
        f"Table {tname} columns: {', '.join(cols)}" for tname, cols in schema.items()
    )


def _build_tasks(n_tasks, seed):
    data = json.loads(Path(SPIDER_DEV).read_text(encoding="utf-8"))
    selected = proportional_stratified_indices(
        data,
        n_samples=n_tasks,
        seed=seed,
        field="hardness",
    )
    return [data[i] for i in selected], selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/intercode-sql-gate"))
    args = parser.parse_args()

    tasks, sel_idx = _build_tasks(args.n_tasks, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    storage = StorageLayout.from_env()
    model = HFModelExecutor(NativeGenerationConfig(**CLOUD_CFG), storage)

    records = []
    per_variant = {}
    for vname, system in PROMPT_VARIANTS.items():
        counts = {}
        samples = []
        for task in tasks:
            db = task["db"]
            query = task["query"]
            schema_str = _schema_str(task)
            user = (
                f"Database: {db}\nSchema:\n{schema_str}\n\n"
                f"Question: {query}\n\n"
                "Write a SQL query to answer the question. Explore if needed, then submit."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            result = model.generate(messages)
            label = _classify(result.text)
            counts[label] = counts.get(label, 0) + 1
            records.append(
                {
                    "variant": vname,
                    "task_id": task.get("id", f"{db}-{query}"),
                    "label": label,
                    "output_tokens": result.output_tokens,
                    "response_hash": result.response_hash,
                    "sample_output": result.text[:300],
                }
            )
            if len(samples) < 2 and label == "other":
                samples.append(result.text[:400])
        per_variant[vname] = {
            "n": len(tasks),
            "action_emission_rate": round(
                sum(v for k, v in counts.items() if k != "other") / len(tasks), 4
            ),
            "label_counts": counts,
            "sample_other_outputs": samples,
        }
        print(f"[{vname}] emission_rate={per_variant[vname]['action_emission_rate']} counts={counts}")

    report = {
        "scope": "cloud_arm_protocol_adherence_definitive_diagnostic",
        "arm": "cloud_google_gemma-4-12b-it_bnb4bit",
        "levers_varied": ["system_prompt_text", "fewshot_demo", "format_enforcement", "json_schema"],
        "levers_frozen": [
            "model", "revision", "dtype=bfloat16", "quantization=bnb_4bit",
            "architecture=multimodal_lm", "enable_thinking=False",
            "max_new_tokens", "do_sample=False", "seed",
        ],
        "n_tasks": len(tasks),
        "task_indices": sel_idx,
        "variants": per_variant,
        "records": records,
        "provenance": {
            "dataset": "InterCode-Spider-dev",
            "official_test_sealed": True,
            "model": CLOUD_CFG["model_id"],
            "revision": CLOUD_CFG["revision"],
        },
    }
    report["report_hash"] = sha256_json(report)
    (out_dir / "intercode-sql-cloud-protocol-diag.json").write_text(
        canonical_json(report) + "\n", encoding="utf-8"
    )
    print("wrote", out_dir / "intercode-sql-cloud-protocol-diag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
