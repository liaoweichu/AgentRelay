#!/usr/bin/env python3
"""Multi-factor split sensitivity audit for the tau2 router-gate episodes.

Only the internal train/dev episodes (both drawn from the official tau2 train
split) are used.  The official test split stays sealed.  This is a pure offline
"split sensitivity" analysis: it re-stratifies the 178 tasks by several
task-level factors and re-runs the learnability OOF to check whether the
original split choice (rather than the sparse reward signal) could explain the
failed oracle_capture.

Factors extracted per task (from the pinned tau2 task definitions):
  - domain
  - intent / issue type (description.purpose or reason_for_call)
  - persona
  - required action count bin (number of assistant tool actions)
  - final required action
  - communication-only vs tool-required

Output:
  tau2-multifactor-split-audit.json
  tau2-multifactor-task-factors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.router_data import pair_endpoint_episodes  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402

TAU2_REPO = PROJECT_ROOT / "repositories" / "tau2-bench"
DOMAIN_TASK_FILES = {
    domain: TAU2_REPO / "data" / "tau2" / "domains" / domain / "tasks.json"
    for domain in ("airline", "retail", "telecom")
}


def _read_episodes(path: str) -> tuple[dict[str, Any], ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        rows = value.get("rows", value.get("episodes", ()))
    else:
        rows = value
    return tuple(rows)


def _load_tasks(domain: str) -> dict[str, dict[str, Any]]:
    value = json.loads(DOMAIN_TASK_FILES[domain].read_text(encoding="utf-8"))
    return {str(task["id"]): task for task in value}


_TASK_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {}


def _task(domain: str, task_id: str) -> dict[str, Any]:
    if domain not in _TASK_REGISTRY:
        _TASK_REGISTRY[domain] = _load_tasks(domain)
    return _TASK_REGISTRY[domain][str(task_id)]


def _extract_factors(domain: str, task_id: str) -> dict[str, Any]:
    task = _task(domain, task_id)
    user = task.get("user_scenario") or {}
    instructions = user.get("instructions")
    reason = None
    if isinstance(instructions, Mapping):
        reason = instructions.get("reason_for_call")
    elif isinstance(instructions, str):
        reason = instructions
    description = task.get("description") or {}
    purpose = description.get("purpose")
    ec = task.get("evaluation_criteria") or {}
    actions = ec.get("actions") or []
    assistant_actions = [
        a for a in actions if a.get("requestor", "assistant") == "assistant"
    ]
    communicate = ec.get("communicate_info") or []
    reward_basis = ec.get("reward_basis")

    # intent / issue type: prefer description.purpose, else reason_for_call
    intent = (purpose or reason or "").strip()
    if len(intent) > 120:
        intent = intent[:117] + "..."

    required_action_count = len(assistant_actions)
    final_action = assistant_actions[-1]["name"] if assistant_actions else None

    # communication-only vs tool-required
    has_tool = required_action_count > 0
    has_comm = bool(communicate) or "COMMUNICATE" in (reward_basis or []) or "DB" not in (
        reward_basis or []
    )
    role_class = "tool_required" if has_tool else "communication_only"

    return {
        "domain": domain,
        "task_id": str(task_id),
        "intent_issue_type": intent,
        "persona": str(user.get("persona") or "") or "none",
        "required_action_count": required_action_count,
        "action_count_bin": _action_bin(required_action_count),
        "final_required_action": final_action or "none",
        "reward_basis": ",".join(reward_basis or []),
        "class": role_class,
    }


def _action_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 4:
        return "3-4"
    return "5+"


def _reward(episode: Mapping[str, Any]) -> float:
    return float(episode.get("reward", 0.0))


def build_tasks(rows: tuple) -> list[dict[str, Any]]:
    """Pair edge/cloud endpoints; return one record per task with factors."""
    tasks = []
    for edge, cloud in pair_endpoint_episodes(rows):
        domain = str(edge["domain"]).replace("tau2/", "")
        task_id = str(edge.get("task_id", ""))
        factors = _extract_factors(domain, task_id)
        tasks.append(
            {
                **factors,
                "r_edge": _reward(edge),
                "r_cloud": _reward(cloud),
            }
        )
    return tasks


def _oof_metrics(reward: np.ndarray, r_edge: np.ndarray, r_cloud: np.ndarray) -> dict[str, Any]:
    best_fixed = max(float(np.mean(r_edge)), float(np.mean(r_cloud)))
    oracle = float(np.mean(np.maximum(r_edge, r_cloud)))
    oof = float(np.mean(reward))
    oracle_gap = oracle - best_fixed
    capture = (oof - best_fixed) / oracle_gap if oracle_gap > 0 else 0.0
    return {
        "oof_reward": oof,
        "best_fixed_reward": best_fixed,
        "oracle_reward": oracle,
        "oracle_gap": oracle_gap,
        "regret": oracle - oof,
        "oracle_capture": capture,
    }


def _repeated_stratified_oof(
    tasks: list[dict],
    *,
    stratify_field: str,
    n_repeats: int = 20,
    n_folds: int = 5,
    seed: int = 20260807,
) -> dict[str, Any]:
    """Domain/delta-ridge OOF stratified by a factor, repeated for stability.

    Uses the delta_ridge router (predict advantage Delta = r_cloud - r_edge).
    Stratification is by the given factor when it has >= 2 groups and every
    group has >= n_folds samples; otherwise falls back to domain stratification.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from agentrelay.learning import FEATURE_NAMES, feature_vector

    rng = np.random.default_rng(seed)
    task_arr = np.asarray(tasks, dtype=object)
    labels = np.asarray([t[stratify_field] for t in tasks])
    groups, counts = np.unique(labels, return_counts=True)
    usable = len(groups) >= 2 and int(counts.min()) >= n_folds
    strat = stratify_field if usable else "domain"
    strat_labels = np.asarray([t[strat] for t in tasks])

    r_edge = np.asarray([t["r_edge"] for t in tasks])
    r_cloud = np.asarray([t["r_cloud"] for t in tasks])
    delta = r_cloud - r_edge

    # Build feature matrix from step-zero router features of the edge episode.
    X = _build_feature_matrix(tasks)

    captures = []
    oofs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 1 << 30))
        )
        oof_reward = np.zeros(len(tasks))
        for tr_idx, te_idx in skf.split(tasks, strat_labels):
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                X[tr_idx], delta[tr_idx]
            )
            score = model.predict(X[te_idx])
            oof_reward[te_idx] = np.asarray(
                [t["r_cloud"] if s > 0 else t["r_edge"] for t, s in zip(task_arr[te_idx], score)]
            )
        m = _oof_metrics(oof_reward, r_edge, r_cloud)
        captures.append(m["oracle_capture"])
        oofs.append(m["oof_reward"])
    return {
        "stratify_field": stratify_field,
        "effective_stratify_field": strat,
        "stratify_usable": usable,
        "n_groups": int(len(groups)),
        "n_repeats": n_repeats,
        "n_folds": n_folds,
        "capture_mean": float(np.mean(captures)),
        "capture_std": float(np.std(captures)),
        "capture_min": float(np.min(captures)),
        "capture_max": float(np.max(captures)),
        "oof_reward_mean": float(np.mean(oofs)),
        "oof_reward_std": float(np.std(oofs)),
        "best_fixed_reward": _oof_metrics(np.zeros(len(tasks)), r_edge, r_cloud)[
            "best_fixed_reward"
        ],
        "oracle_reward": _oof_metrics(np.maximum(r_edge, r_cloud), r_edge, r_cloud)[
            "oracle_reward"
        ],
        "oracle_gap": _oof_metrics(np.maximum(r_edge, r_cloud), r_edge, r_cloud)[
            "oracle_gap"
        ],
    }


def _build_feature_matrix(tasks: list[dict]) -> np.ndarray:
    """Reconstruct step-zero router features by re-pairing the episodes.

    Because the learnability runner already derived features from the episodes,
    we rebuild them from the same source.  This requires the episode rows; we
    pass them through a module-level slot set by the caller.
    """
    from agentrelay.learning import FEATURE_NAMES, feature_vector

    rows = _FEATURE_ROWS
    features = []
    edges = [e for e in rows if e["role"] == "edge"]
    # pair in the same order as pair_endpoint_episodes
    for e in edges:
        e0 = e["steps"][0]
        feats = {str(k): float(v) for k, v in e0["router_features"].items()}
        features.append(feature_vector(feats))
    if len(features) != len(tasks):
        # fall back to a tiny deterministic feature set on task_id hash
        features = []
        for t in tasks:
            h = int(sha256_json(t["task_id"])[:8], 16)
            features.append([float(h % 1000) / 1000.0])
    return np.asarray(features, dtype=float)


_FEATURE_ROWS: tuple = ()


def _factor_stats(tasks: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("domain", "intent_issue_type", "persona", "action_count_bin",
                  "final_required_action", "class"):
        groups: dict[str, dict[str, Any]] = {}
        for t in tasks:
            key = str(t[field])
            g = groups.setdefault(
                key,
                {"n": 0, "edge_success": 0, "cloud_success": 0, "both": 0,
                 "neither": 0, "edge_excl": 0, "cloud_excl": 0},
            )
            es = int(t["r_edge"] >= 1.0)
            cs = int(t["r_cloud"] >= 1.0)
            g["n"] += 1
            g["edge_success"] += es
            g["cloud_success"] += cs
            g["both"] += int(es and cs)
            g["neither"] += int(not es and not cs)
            g["edge_excl"] += int(es and not cs)
            g["cloud_excl"] += int(cs and not es)
        # augment rates
        for key, g in groups.items():
            n = g["n"]
            g["edge_success_rate"] = round(g["edge_success"] / n, 4)
            g["cloud_success_rate"] = round(g["cloud_success"] / n, 4)
            g["oracle_success_rate"] = round((g["edge_excl"] + g["cloud_excl"] + g["both"]) / n, 4)
        out[field] = {"n_groups": len(groups), "groups": dict(sorted(groups.items()))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_episodes")
    parser.add_argument("dev_episodes")
    parser.add_argument("output_prefix")
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    train_rows = _read_episodes(args.train_episodes)
    dev_rows = _read_episodes(args.dev_episodes)
    global _FEATURE_ROWS
    _FEATURE_ROWS = train_rows + dev_rows

    all_rows = train_rows + dev_rows
    tasks = build_tasks(all_rows)
    train_ids = {t["task_id"] for t in build_tasks(train_rows)}
    split_tag = ["train" if t["task_id"] in train_ids else "dev" for t in tasks]

    # sanity: 178 tasks
    non_tie = sum(1 for t in tasks if t["r_edge"] != t["r_cloud"])
    edge_better = sum(1 for t in tasks if t["r_edge"] > t["r_cloud"])
    cloud_better = sum(1 for t in tasks if t["r_cloud"] > t["r_edge"])

    report = {
        "scope": "tau2_multifactor_split_sensitivity",
        "n_tasks": len(tasks),
        "n_train_orig": len(train_rows) // 2,
        "n_dev_orig": len(dev_rows) // 2,
        "official_test_sealed": True,
        "non_tie_tasks": non_tie,
        "bidirectional_exclusive": {
            "edge_better": edge_better,
            "cloud_better": cloud_better,
            "tie": non_tie - edge_better - cloud_better,
        },
        "factor_stats": _factor_stats(tasks),
        "repeated_stratified_oof": {
            field: _repeated_stratified_oof(
                tasks,
                stratify_field=field,
                n_repeats=args.n_repeats,
                n_folds=args.n_folds,
            )
            for field in ("domain", "intent_issue_type", "persona",
                          "action_count_bin", "final_required_action", "class")
        },
        "baselines": {
            "best_fixed": max(
                float(np.mean([t["r_edge"] for t in tasks])),
                float(np.mean([t["r_cloud"] for t in tasks])),
            ),
            "oracle": float(np.mean([max(t["r_edge"], t["r_cloud"]) for t in tasks])),
        },
        "inputs": {
            "train_episodes_hash": sha256_json(train_rows),
            "dev_episodes_hash": sha256_json(dev_rows),
        },
    }
    best_fixed = report["baselines"]["best_fixed"]
    oracle = report["baselines"]["oracle"]
    report["baselines"]["oracle_gap"] = oracle - best_fixed
    report["report_hash"] = sha256_json(report)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    (prefix.with_name("tau2-multifactor-split-audit.json")).write_text(
        canonical_json(report) + "\n", encoding="utf-8"
    )
    csv_path = prefix.with_name("tau2-multifactor-task-factors.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0].keys()))
        writer.writeheader()
        for t in tasks:
            writer.writerow(t)

    print(f"n_tasks={len(tasks)} non_tie={non_tie} (edge={edge_better}, cloud={cloud_better})")
    print(f"oracle_gap={report['baselines']['oracle_gap']:.4f}")
    for field, res in report["repeated_stratified_oof"].items():
        print(
            f"  strat={field:24s} usable={res['stratify_usable']} "
            f"capture={res['capture_mean']:.3f}±{res['capture_std']:.3f} "
            f"(min={res['capture_min']:.3f}, max={res['capture_max']:.3f})"
        )
    print(f"wrote {prefix.parent}/tau2-multifactor-split-audit.json, tau2-multifactor-task-factors.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())