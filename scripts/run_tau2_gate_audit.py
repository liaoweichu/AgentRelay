#!/usr/bin/env python3
"""Audit the tau2 train/dev router-gate results and emit three diagnostics.

Produces:
  tau2-gate-audit.json            - pairing/leak/per-domain/per-arm summary
  tau2-task-diagnostics.csv       - one row per task (edge vs cloud rewards, etc.)
  tau2-failure-breakdown.json     - parser/tool/termination failure analysis

This is a read-only, offline audit.  It does not fit any model and does not
consult the held-out test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.router_data import pair_endpoint_episodes  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402

# Parser recovery categories that indicate the model's output was NOT turned
# into an executable structured tool call.
_NON_STRUCTURED_RECOVERIES = {
    "plain_text",
    "non_object_as_text",
    "unknown_object_as_text",
    "structured_message",
    "invalid_tool_as_text",
    "empty_response_fallback",
}
_JSON_ATTEMPT_RE = re.compile(r"\{[^{}]{0,400}\"?(tool|name|message)\"?")


def _read_episodes(path: str) -> tuple[dict[str, Any], ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        rows = value.get("rows", value.get("episodes", ()))
    else:
        rows = value
    return tuple(rows)


def _success(episode: Mapping[str, Any]) -> int:
    return int(float(episode.get("success", 0.0)) > 0.0)


def _reward(episode: Mapping[str, Any]) -> float:
    return float(episode.get("reward", 0.0))


def _json_attempt(text: str) -> bool:
    return bool(_JSON_ATTEMPT_RE.search(text or ""))


def _step_breakdown(episode: Mapping[str, Any]) -> dict[str, Any]:
    steps = tuple(episode.get("steps", ()))
    n = len(steps)
    recovery = {}
    malformed = 0
    invalid_tool = 0
    structured_tool = 0
    for step in steps:
        rec = str(step.get("action_recovery", ""))
        recovery[rec] = recovery.get(rec, 0) + 1
        if rec == "structured_tool":
            structured_tool += 1
        elif rec == "invalid_tool_as_text":
            invalid_tool += 1
            if _json_attempt(str(step.get("action_text", ""))):
                malformed += 1
        elif rec in _NON_STRUCTURED_RECOVERIES:
            if _json_attempt(str(step.get("action_text", ""))):
                malformed += 1
    return {
        "step_count": n,
        "recovery": recovery,
        "structured_tool_count": structured_tool,
        "invalid_tool_count": invalid_tool,
        "malformed_json_count": malformed,
        "parser_recovery_count": n - structured_tool,
        "parser_recovery_fraction": (n - structured_tool) / n if n else 0.0,
        "tool_error_count": int(episode.get("effect_failures", 0)),
        "termination_reason": str(episode.get("termination_reason", "")),
    }


def _first_input_identity(episode: Mapping[str, Any]) -> dict[str, Any]:
    steps = tuple(episode.get("steps", ()))
    if not steps:
        return {"prompt_hash": "", "features_snapshot": {}}
    first = steps[0]
    return {
        "prompt_hash": str(first.get("prompt_hash", "")),
        "features_snapshot": {
            str(k): float(v) for k, v in (first.get("router_features") or {}).items()
        },
    }


def audit(train_rows: tuple, dev_rows: tuple) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": "tau2_router_gate_audit",
        "independent_tasks": {"train": 0, "dev": 0},
        "pairing": {},
        "leak": {},
        "per_domain": {},
        "per_arm": {},
        "exclusive_success": {},
        "first_input_consistency": {"train": {}, "dev": {}},
        "train_dev_overlap": [],
    }

    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        pairs = pair_endpoint_episodes(rows)
        result["independent_tasks"][split] = len(pairs)
        # Per arm / per domain
        domains = sorted({str(e["domain"]) for e in rows})
        per_domain = {}
        for domain in domains:
            d_edge = [e for e in rows if e["domain"] == domain and e["role"] == "edge"]
            d_cloud = [e for e in rows if e["domain"] == domain and e["role"] == "cloud"]
            er = [_reward(e) for e in d_edge]
            cr = [_reward(e) for e in d_cloud]
            es = [_success(e) for e in d_edge]
            cs = [_success(e) for e in d_cloud]
            oracle = [max(a, b) for a, b in zip(er, cr)]
            per_domain[domain] = {
                "n_tasks": len(d_edge),
                "edge": {"n": len(es), "success_rate": float(np.mean(es)), "avg_reward": float(np.mean(er))},
                "cloud": {"n": len(cs), "success_rate": float(np.mean(cs)), "avg_reward": float(np.mean(cr))},
                "oracle_success_rate": float(np.mean(oracle)),
                "edge_exclusive": int(sum(a and not b for a, b in zip(es, cs))),
                "cloud_exclusive": int(sum(b and not a for a, b in zip(es, cs))),
                "both_success": int(sum(a and b for a, b in zip(es, cs))),
                "neither_success": int(sum(not a and not b for a, b in zip(es, cs))),
            }
        result["per_domain"][split] = per_domain

        # Per arm aggregate
        arms = {}
        for role in ("edge", "cloud"):
            arm_rows = [e for e in rows if e["role"] == role]
            succ = [_success(e) for e in arm_rows]
            rew = [_reward(e) for e in arm_rows]
            arms[role] = {
                "n": len(arm_rows),
                "success_rate": float(np.mean(succ)),
                "total_success": int(sum(succ)),
                "avg_reward": float(np.mean(rew)),
            }
        result["per_arm"][split] = arms

        # Exclusive success over pairs
        es = [_success(e) for e, _ in pairs]
        cs = [_success(c) for _, c in pairs]
        result["exclusive_success"][split] = {
            "edge_exclusive": int(sum(a and not b for a, b in zip(es, cs))),
            "cloud_exclusive": int(sum(b and not a for a, b in zip(es, cs))),
            "both_success": int(sum(a and b for a, b in zip(es, cs))),
            "neither_success": int(sum(not a and not b for a, b in zip(es, cs))),
        }

        # First-visible-input identity across the two arms
        mismatch = 0
        checked = 0
        feats_mismatch = 0
        for edge, cloud in pairs:
            ei = _first_input_identity(edge)
            ci = _first_input_identity(cloud)
            checked += 1
            if ei["prompt_hash"] != ci["prompt_hash"]:
                mismatch += 1
            if ei["features_snapshot"] != ci["features_snapshot"]:
                feats_mismatch += 1
        result["first_input_consistency"][split] = {
            "checked_tasks": checked,
            "prompt_hash_mismatch_tasks": mismatch,
            "features_mismatch_tasks": feats_mismatch,
        }

    # Train/dev overlap (leak) check on (benchmark, revision, sample_id)
    def task_keys(rows):
        return {
            (str(e["benchmark"]), str(e["dataset_revision"]), str(e.get("sample_id", e.get("task_id"))))
            for e in rows
        }

    train_keys = task_keys(train_rows)
    dev_keys = task_keys(dev_rows)
    overlap = sorted(train_keys & dev_keys)
    train_domains = {(str(e["benchmark"]), str(e["dataset_revision"])) for e in train_rows}
    dev_domains = {(str(e["benchmark"]), str(e["dataset_revision"])) for e in dev_rows}
    result["leak"] = {
        "train_dev_task_overlap_count": len(overlap),
        "train_dev_task_overlap": overlap[:50],
        "domain_revision_match": train_domains == dev_domains,
        "train_domains": sorted(train_domains),
        "dev_domains": sorted(dev_domains),
    }
    result["train_dev_overlap"] = overlap

    # Pairing completeness
    result["pairing"] = {
        "train_rows": len(train_rows),
        "train_tasks": result["independent_tasks"]["train"],
        "dev_rows": len(dev_rows),
        "dev_tasks": result["independent_tasks"]["dev"],
        "train_rows_per_task": len(train_rows) / result["independent_tasks"]["train"],
        "dev_rows_per_task": len(dev_rows) / result["independent_tasks"]["dev"],
    }
    return result


def task_diagnostics(train_rows: tuple, dev_rows: tuple) -> list[dict[str, Any]]:
    rows_out = []
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        for edge, cloud in pair_endpoint_episodes(rows):
            er = _reward(edge)
            cr = _reward(cloud)
            es = _success(edge)
            cs = _success(cloud)
            eb = _step_breakdown(edge)
            cb = _step_breakdown(cloud)
            ei = _first_input_identity(edge)
            ci = _first_input_identity(cloud)
            rows_out.append(
                {
                    "split": split,
                    "domain": str(edge["domain"]),
                    "task_id": str(edge.get("task_id", "")),
                    "sample_id": str(edge.get("sample_id", "")),
                    "edge_reward": er,
                    "cloud_reward": cr,
                    "delta_reward_cloud_minus_edge": round(cr - er, 6),
                    "edge_success": es,
                    "cloud_success": cs,
                    "edge_exclusive": int(es and not cs),
                    "cloud_exclusive": int(cs and not es),
                    "both_success": int(es and cs),
                    "oracle_reward": max(er, cr),
                    "edge_steps": eb["step_count"],
                    "cloud_steps": cb["step_count"],
                    "edge_termination": eb["termination_reason"],
                    "cloud_termination": cb["termination_reason"],
                    "edge_parser_recovery_count": eb["parser_recovery_count"],
                    "cloud_parser_recovery_count": cb["parser_recovery_count"],
                    "edge_parser_recovery_fraction": round(eb["parser_recovery_fraction"], 6),
                    "cloud_parser_recovery_fraction": round(cb["parser_recovery_fraction"], 6),
                    "edge_malformed_json": eb["malformed_json_count"],
                    "cloud_malformed_json": cb["malformed_json_count"],
                    "edge_invalid_tool": eb["invalid_tool_count"],
                    "cloud_invalid_tool": cb["invalid_tool_count"],
                    "edge_tool_error": eb["tool_error_count"],
                    "cloud_tool_error": cb["tool_error_count"],
                    "first_input_identical": int(ei["features_snapshot"] == ci["features_snapshot"]),
                }
            )
    return rows_out


def failure_breakdown(train_rows: tuple, dev_rows: tuple) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        steps_total = 0
        recovery_total: dict[str, int] = {}
        malformed_total = 0
        invalid_tool_total = 0
        tool_error_total = 0
        termination_total: dict[str, int] = {}
        for e in rows:
            b = _step_breakdown(e)
            steps_total += b["step_count"]
            for k, v in b["recovery"].items():
                recovery_total[k] = recovery_total.get(k, 0) + v
            malformed_total += b["malformed_json_count"]
            invalid_tool_total += b["invalid_tool_count"]
            tool_error_total += b["tool_error_count"]
            termination_total[b["termination_reason"]] = (
                termination_total.get(b["termination_reason"], 0) + 1
            )
        out[split] = {
            "episodes": len(rows),
            "steps": steps_total,
            "recovery": {k: v for k, v in sorted(recovery_total.items())},
            "recovery_fraction": {
                k: round(v / steps_total, 6) for k, v in sorted(recovery_total.items())
            } if steps_total else {},
            "malformed_json_out_of_steps": malformed_total,
            "malformed_json_fraction": round(malformed_total / steps_total, 6) if steps_total else 0.0,
            "invalid_tool_out_of_steps": invalid_tool_total,
            "invalid_tool_fraction": round(invalid_tool_total / steps_total, 6) if steps_total else 0.0,
            "tool_error_total": tool_error_total,
            "tool_error_per_episode": round(tool_error_total / len(rows), 6) if rows else 0.0,
            "termination": {k: v for k, v in sorted(termination_total.items())},
            "termination_fraction": {
                k: round(v / len(rows), 6) for k, v in sorted(termination_total.items())
            } if rows else {},
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_episodes")
    parser.add_argument("dev_episodes")
    parser.add_argument("output_prefix")
    args = parser.parse_args()

    train_rows = _read_episodes(args.train_episodes)
    dev_rows = _read_episodes(args.dev_episodes)

    audit_result = audit(train_rows, dev_rows)
    diagnostics = task_diagnostics(train_rows, dev_rows)
    failures = failure_breakdown(train_rows, dev_rows)

    audit_result["failure_breakdown"] = failures
    audit_result["inputs"] = {
        "train_episodes_hash": sha256_json(train_rows),
        "dev_episodes_hash": sha256_json(dev_rows),
    }
    audit_result["audit_hash"] = sha256_json(audit_result)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_name("tau2-gate-audit.json")).write_text(
        canonical_json(audit_result) + "\n", encoding="utf-8"
    )
    fail_path = prefix.with_name("tau2-failure-breakdown.json")
    fail_path.write_text(canonical_json(failures) + "\n", encoding="utf-8")
    csv_path = prefix.with_name("tau2-task-diagnostics.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        for row in diagnostics:
            writer.writerow(row)

    print(
        f"train_tasks={audit_result['independent_tasks']['train']} "
        f"dev_tasks={audit_result['independent_tasks']['dev']} "
        f"overlap={audit_result['leak']['train_dev_task_overlap_count']}"
    )
    print(f"wrote {prefix.parent / 'tau2-gate-audit.json'}, tau2-task-diagnostics.csv, tau2-failure-breakdown.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())