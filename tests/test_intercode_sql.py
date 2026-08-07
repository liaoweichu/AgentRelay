"""Local regression tests for the InterCode-SQL action and gate contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentrelay.intercode_sql import (
    paired_reward_summary,
    parse_sql_action,
    proportional_stratified_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_gate_module():
    path = PROJECT_ROOT / "scripts" / "run_intersql_model_gate.py"
    spec = importlib.util.spec_from_file_location("intercode_sql_gate_test_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load gate module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0

    def generate(self, _messages):
        text = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return SimpleNamespace(
            text=text,
            prompt_hash=f"prompt-{self.index}",
            response_hash=f"response-{self.index}",
            prompt_tokens=10,
            output_tokens=5,
            latency_ms=1.5,
            peak_cuda_memory_bytes=1024,
        )


class _FakeExecutor:
    def run(self, _db, sql):
        if sql == "SELECT 1":
            return [(1,)], None
        return None, "unsupported SQL"

    def gold_rows(self, _db, _gold_sql):
        return [(1,)]


def _task() -> dict:
    return {
        "db": "fixture",
        "query": "Return one.",
        "gold": "SELECT 1",
        "db_tables": {},
        "id": "fixture-1",
    }


def test_parse_sql_submit_shapes() -> None:
    sql = parse_sql_action("```sql\nSELECT 1;\n```")
    submit = parse_sql_action("submit")
    combined = parse_sql_action("```sql\nSELECT 1;\n```\nsubmit")

    assert (sql.kind, sql.sql) == ("sql", "SELECT 1")
    assert (submit.kind, submit.sql) == ("submit", None)
    assert (combined.kind, combined.sql) == ("sql_submit", "SELECT 1")


def test_parse_sql_does_not_confuse_submit_text_with_termination() -> None:
    quoted = parse_sql_action("```sql\nSELECT 'submit';\n```")
    prose = parse_sql_action("```sql\nSELECT 1;\n```\nplease submit this")

    assert (quoted.kind, quoted.sql) == ("sql", "SELECT 'submit'")
    assert prose.kind == "invalid"


def test_proportional_stratified_indices_are_seeded_and_proportional() -> None:
    records = [{"hardness": "easy"} for _ in range(6)] + [
        {"hardness": "hard"} for _ in range(4)
    ]
    first = proportional_stratified_indices(records, n_samples=5, seed=7, field="hardness")
    second = proportional_stratified_indices(records, n_samples=5, seed=7, field="hardness")

    assert first == second
    assert len(first) == len(set(first)) == 5
    assert sum(index < 6 for index in first) == 3
    assert sum(index >= 6 for index in first) == 2
    with pytest.raises(ValueError):
        proportional_stratified_indices(records, n_samples=11, seed=7, field="hardness")


def test_paired_reward_summary_uses_oracle_over_best_fixed() -> None:
    summary = paired_reward_summary([1.0, 0.0], [0.0, 1.0])

    assert summary["edge"] == 0.5
    assert summary["cloud"] == 0.5
    assert summary["best_fixed"] == 0.5
    assert summary["oracle"] == 1.0
    assert summary["oracle_gap"] == 0.5


def test_combined_sql_submit_executes_and_terminates_in_one_turn() -> None:
    gate = _load_gate_module()
    model = _FakeModel(["```sql\nSELECT 1;\n```\nsubmit"])

    episode = gate._run_episode(_FakeExecutor(), model, _task(), max_steps=3)

    assert episode["termination"] == "combined_sql_submit"
    assert episode["reward"] == 1.0
    assert episode["success"] == 1.0
    assert episode["n_steps"] == 1
    assert episode["actions"][0]["kind"] == "sql_submit"


def test_max_steps_without_submit_has_zero_official_reward() -> None:
    gate = _load_gate_module()
    model = _FakeModel(["```sql\nSELECT 1;\n```"])

    episode = gate._run_episode(_FakeExecutor(), model, _task(), max_steps=2)

    assert episode["termination"] == "max_steps_without_submit"
    assert episode["reward"] == 0.0
    assert episode["last_query_reward"] == 1.0
    assert episode["success"] == 0.0
    assert episode["n_steps"] == 2
