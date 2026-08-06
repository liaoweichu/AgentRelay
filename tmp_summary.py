#!/usr/bin/env python3
"""Validate an official matrix run and tabulate the per-method baseline matrix.

Checks:
  - every expected (method, task) episode exists and parses
  - result_hash present and matches the payload
  - paper_evidence is true and labels_accessed_by_router is false (no leak)
  - success/reward are finite and in range
Prints a per-method success-rate / reward table and a JSON summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("output", help="summary JSON path")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--expected-per-method", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"run_id={manifest.get('run_id')}")

    ordered = [m for m in args.methods if (run_dir / "webshop" / "test" / m).exists()]
    missing_dir = [m for m in args.methods if m not in ordered]
    if missing_dir:
        print(f"FATAL missing method dirs: {missing_dir}")
        return 2

    summary: dict[str, dict] = {}
    total_ok = 0
    problems: list[str] = []
    for method in ordered:
        method_dir = run_dir / "webshop" / "test" / method
        files = sorted(method_dir.glob("*.json"))
        if len(files) != args.expected_per_method:
            problems.append(f"{method}: expected {args.expected_per_method} got {len(files)}")
        success = 0.0
        rewards: list[float] = []
        for f in files:
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{method}/{f.name}: unparseable {exc}")
                continue
            # hash self-check
            payload = {k: v for k, v in rec.items() if k != "result_hash"}
            from agentrelay.schema import sha256_json
            if "result_hash" not in rec or sha256_json(payload) != rec["result_hash"]:
                problems.append(f"{method}/{f.name}: result_hash mismatch")
            if rec.get("paper_evidence") is not True:
                problems.append(f"{method}/{f.name}: paper_evidence not true")
            if rec.get("labels_accessed_by_router") is not False:
                problems.append(f"{method}/{f.name}: router label leak")
            if rec.get("benchmark") not in ("webshop", "princeton-nlp/WebShop"):
                problems.append(f"{method}/{f.name}: wrong benchmark {rec.get('benchmark')}")
            success_float = float(rec.get("success", 0.0))
            reward_float = float(rec.get("reward", 0.0))
            if not (0.0 <= reward_float <= 1.0):
                problems.append(f"{method}/{f.name}: reward out of range {reward_float}")
            success += success_float
            rewards.append(reward_float)
            total_ok += 1
        n = len(files)
        mean_reward = sum(rewards) / n if n else 0.0
        summary[method] = {
            "episodes": n,
            "success_rate": round(success / n, 4) if n else 0.0,
            "mean_reward": round(mean_reward, 4) if n else 0.0,
        }
        print(f"{method:>20}  n={n:4d}  success={summary[method]['success_rate']:.4f}  "
              f"reward={summary[method]['mean_reward']:.4f}")

    expected_total = args.expected_per_method * len(args.methods)
    summary["_meta"] = {
        "run_id": manifest.get("run_id"),
        "valid_episodes": total_ok,
        "expected_episodes": expected_total,
        "problems": problems,
    }
    print(f"\nvalid_episodes={total_ok}/{expected_total}  problems={len(problems)}")
    for p in problems[:20]:
        print("  PROBLEM:", p)
    if problems:
        sys.stderr.write(f"WARNING {len(problems)} problems found\n")
    summary_path = Path(args.output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary output={summary_path}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())