#!/usr/bin/env python3
"""Offline learnability diagnosis for the tau2 task-level reward router.

Fits several reward-aware routers on the 125 pairwise train tasks and reports
domain-stratified 5-fold out-of-fold (OOF) statistics.  Only the official train
split is used; the dev split is never touched here.

Routers compared:
  1. double_ridge                 - Ridge reward for each arm, pick argmax (current design)
  2. domain_only                  - pick the arm with higher per-domain mean reward
  3. delta_ridge                  - Ridge on advantage Delta = r_cloud - r_edge, pick sign
  4. class_weighted_edge_tie_cloud- balanced 3-class edge/tie/cloud classifier
  5. domain_onehot_delta_ridge    - delta ridge with explicit domain one-hots

Outputs:
  tau2-oof-learnability.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.learning import FEATURE_NAMES, feature_vector  # noqa: E402
from agentrelay.router_data import pair_endpoint_episodes  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402


def _read_episodes(path: str) -> tuple[dict[str, Any], ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        rows = value.get("rows", value.get("episodes", ()))
    else:
        rows = value
    return tuple(rows)


def _reward(episode: Mapping[str, Any]) -> float:
    return float(episode.get("reward", 0.0))


def build_tasks(rows: tuple) -> list[dict[str, Any]]:
    """Pair edge/cloud endpoints and return one record per task."""
    tasks = []
    for edge, cloud in pair_endpoint_episodes(rows):
        e0 = edge["steps"][0]
        feats = {
            str(k): float(v) for k, v in e0["router_features"].items()
        }
        feature_vector(feats)  # schema check
        tasks.append(
            {
                "domain": str(edge["domain"]),
                "task_id": str(edge.get("task_id", "")),
                "sample_id": str(edge.get("sample_id", "")),
                "x": np.asarray(feature_vector(feats), dtype=float),
                "r_edge": _reward(edge),
                "r_cloud": _reward(cloud),
            }
        )
    return tasks


def _domain_onehot(domains: list[str]) -> dict[str, np.ndarray]:
    unique = sorted(set(domains))
    out = {}
    for dom in unique:
        vec = np.zeros(len(unique), dtype=float)
        vec[unique.index(dom)] = 1.0
        out[dom] = vec
    return {dom: out[dom] for dom in domains}


def fit_predict_router(
    name: str,
    tasks_tr: list[dict],
    tasks_te: list[dict],
    domain_onehot_tr: dict,
    domain_onehot_te: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted chosen reward per test task, score vector)."""
    rng = np.random.default_rng(0)
    X_tr = np.stack([t["x"] for t in tasks_tr])
    X_te = np.stack([t["x"] for t in tasks_te])
    re_tr = np.asarray([t["r_edge"] for t in tasks_tr])
    rc_tr = np.asarray([t["r_cloud"] for t in tasks_tr])
    delta_tr = rc_tr - re_tr

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "double_ridge":
        m_edge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, re_tr)
        m_cloud = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, rc_tr)
        pe = m_edge.predict(X_te)
        pc = m_cloud.predict(X_te)
        score = pc - pe
        reward = np.asarray(
            [t["r_cloud"] if s > 0 else t["r_edge"] for t, s in zip(tasks_te, score)],
            dtype=float,
        )
        return reward, score

    if name == "domain_only":
        dom_mean = {}
        for dom in sorted({t["domain"] for t in tasks_tr}):
            re = [t["r_edge"] for t in tasks_tr if t["domain"] == dom]
            rc = [t["r_cloud"] for t in tasks_tr if t["domain"] == dom]
            dom_mean[dom] = "cloud" if np.mean(rc) > np.mean(re) else "edge"
        reward = np.asarray(
            [
                t["r_cloud"] if dom_mean[t["domain"]] == "cloud" else t["r_edge"]
                for t in tasks_te
            ],
            dtype=float,
        )
        score = np.asarray(
            [1.0 if dom_mean[t["domain"]] == "cloud" else 0.0 for t in tasks_te],
            dtype=float,
        )
        return reward, score

    if name == "delta_ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, delta_tr)
        score = model.predict(X_te)
        reward = np.asarray(
            [t["r_cloud"] if s > 0 else t["r_edge"] for t, s in zip(tasks_te, score)],
            dtype=float,
        )
        return reward, score

    if name == "domain_onehot_delta_ridge":
        XO_tr = np.hstack([X_tr, np.asarray([domain_onehot_tr[t["domain"]] for t in tasks_tr])])
        XO_te = np.hstack([X_te, np.asarray([domain_onehot_te[t["domain"]] for t in tasks_te])])
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(XO_tr, delta_tr)
        score = model.predict(XO_te)
        reward = np.asarray(
            [t["r_cloud"] if s > 0 else t["r_edge"] for t, s in zip(tasks_te, score)],
            dtype=float,
        )
        return reward, score

    if name == "class_weighted_edge_tie_cloud":
        labels = np.asarray(
            [
                "cloud" if t["r_cloud"] > t["r_edge"]
                else ("edge" if t["r_edge"] > t["r_cloud"] else "tie")
                for t in tasks_tr
            ]
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, solver="lbfgs", class_weight="balanced"
            ),
        ).fit(X_tr, labels)
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)
        classes = list(model.classes_)
        # fold-wise best-fixed arm as tie default
        best_fixed = "cloud" if np.mean(rc_tr) > np.mean(re_tr) else "edge"
        reward = np.asarray(
            [
                t["r_cloud"] if p == "cloud" else (t["r_edge"] if p == "edge"
                                                   else (t["r_cloud"] if best_fixed == "cloud" else t["r_edge"]))
                for t, p in zip(tasks_te, pred)
            ],
            dtype=float,
        )
        cloud_idx = classes.index("cloud") if "cloud" in classes else None
        score = prob[:, cloud_idx] if cloud_idx is not None else np.zeros(len(tasks_te))
        return reward, score

    raise ValueError(f"unknown router {name}")


def _metrics(reward: np.ndarray, r_edge: np.ndarray, r_cloud: np.ndarray) -> dict[str, Any]:
    best_fixed = max(float(np.mean(r_edge)), float(np.mean(r_cloud)))
    oracle = float(np.mean(np.maximum(r_edge, r_cloud)))
    oof = float(np.mean(reward))
    oracle_gap = oracle - best_fixed
    capture = (oof - best_fixed) / oracle_gap if oracle_gap > 0 else 0.0
    regret = oracle - oof
    return {
        "oof_reward": oof,
        "best_fixed_reward": best_fixed,
        "oracle_reward": oracle,
        "oracle_gap": oracle_gap,
        "regret": regret,
        "oracle_capture": capture,
    }


def _binary_metrics(score: np.ndarray, cloud_better: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import auc, roc_curve
    from sklearn.metrics import balanced_accuracy_score

    pred = (score > 0).astype(int)
    bal_acc = balanced_accuracy_score(cloud_better, pred)
    auc_val = None
    if len(np.unique(score)) > 1 and len(np.unique(cloud_better)) > 1:
        fpr, tpr, _ = roc_curve(cloud_better, score)
        auc_val = float(auc(fpr, tpr))
    return {"cloud_better_balanced_accuracy": bal_acc, "cloud_better_auc": auc_val}


def run_oof(tasks: list[dict], *, n_folds: int = 5, seed: int = 20260806) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    domains = np.asarray([t["domain"] for t in tasks])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    r_edge = np.asarray([t["r_edge"] for t in tasks])
    r_cloud = np.asarray([t["r_cloud"] for t in tasks])
    cloud_better = (r_cloud > r_edge).astype(int)
    domain_onehot_map = _domain_onehot([t["domain"] for t in tasks])

    routers = [
        "double_ridge",
        "domain_only",
        "delta_ridge",
        "class_weighted_edge_tie_cloud",
        "domain_onehot_delta_ridge",
    ]
    results = {}
    for name in routers:
        oof_reward = np.zeros(len(tasks), dtype=float)
        oof_score = np.zeros(len(tasks), dtype=float)
        seen = np.zeros(len(tasks), dtype=bool)
        for tr_idx, te_idx in skf.split(tasks, domains):
            tasks_tr = [tasks[i] for i in tr_idx]
            tasks_te = [tasks[i] for i in te_idx]
            ohot_tr = {t["domain"]: domain_onehot_map[t["domain"]] for t in tasks_tr}
            ohot_te = {t["domain"]: domain_onehot_map[t["domain"]] for t in tasks_te}
            rw, sc = fit_predict_router(name, tasks_tr, tasks_te, ohot_tr, ohot_te)
            oof_reward[te_idx] = rw
            oof_score[te_idx] = sc
            seen[te_idx] = True
        assert seen.all()
        entry = _metrics(oof_reward, r_edge, r_cloud)
        entry.update(_binary_metrics(oof_score, cloud_better))
        entry["cloud_selection_fraction"] = float(np.mean(oof_score > 0))
        results[name] = entry
    return results


def permutation_baseline(tasks: list[dict], *, n_folds: int = 5, n_perm: int = 50, seed: int = 20260806) -> dict[str, Any]:
    """Permute feature-to-task assignment within each training fold and refit."""
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(seed)
    domains = np.asarray([t["domain"] for t in tasks])
    r_edge = np.asarray([t["r_edge"] for t in tasks])
    r_cloud = np.asarray([t["r_cloud"] for t in tasks])
    best_fixed = max(float(np.mean(r_edge)), float(np.mean(r_cloud)))
    oracle = float(np.mean(np.maximum(r_edge, r_cloud)))
    oracle_gap = oracle - best_fixed

    captures = []
    oofs = []
    for pm in range(n_perm):
        perm = rng.permutation(len(tasks))  # shuffle rows (features travel with permuted task)
        oof_reward = np.zeros(len(tasks))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + pm)
        for tr_idx, te_idx in skf.split(tasks, domains):
            tasks_tr = [tasks[i] for i in tr_idx]
            tasks_te = [tasks[i] for i in te_idx]
            X_tr = np.stack([t["x"] for t in tasks_tr])
            re_tr = np.asarray([t["r_edge"] for t in tasks_tr])
            rc_tr = np.asarray([t["r_cloud"] for t in tasks_tr])
            delta_tr = rc_tr - re_tr
            # permute features relative to labels within train fold
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            X_tr_perm = X_tr[rng.permutation(len(X_tr))]
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr_perm, delta_tr)
            X_te = np.stack([t["x"] for t in tasks_te])
            score = model.predict(X_te)
            oof_reward[te_idx] = np.asarray(
                [te["r_cloud"] if s > 0 else te["r_edge"] for te, s in zip(tasks_te, score)]
            )
        oofs.append(float(np.mean(oof_reward)))
        cap = (float(np.mean(oof_reward)) - best_fixed) / oracle_gap if oracle_gap > 0 else 0.0
        captures.append(cap)
    return {
        "n_permutations": n_perm,
        "oof_reward_mean": float(np.mean(oofs)),
        "oof_reward_std": float(np.std(oofs)),
        "capture_mean": float(np.mean(captures)),
        "capture_std": float(np.std(captures)),
        "best_fixed_reward": best_fixed,
        "oracle_reward": oracle,
        "oracle_gap": oracle_gap,
        "max_capture": float(np.max(captures)),
    }


def per_domain_oracle_gap(tasks: list[dict]) -> dict[str, Any]:
    out = {}
    for dom in sorted({t["domain"] for t in tasks}):
        dt = [t for t in tasks if t["domain"] == dom]
        re = np.asarray([t["r_edge"] for t in dt])
        rc = np.asarray([t["r_cloud"] for t in dt])
        best_fixed = max(float(np.mean(re)), float(np.mean(rc)))
        oracle = float(np.mean(np.maximum(re, rc)))
        out[dom] = {
            "n_tasks": len(dt),
            "best_fixed_reward": best_fixed,
            "oracle_reward": oracle,
            "oracle_gap": oracle - best_fixed,
            "best_fixed_role": "cloud" if np.mean(rc) > np.mean(re) else "edge",
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_episodes")
    parser.add_argument("output")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=50)
    args = parser.parse_args()

    rows = _read_episodes(args.train_episodes)
    tasks = build_tasks(rows)
    r_edge = np.asarray([t["r_edge"] for t in tasks])
    r_cloud = np.asarray([t["r_cloud"] for t in tasks])

    non_tie = int(np.sum(r_edge != r_cloud))
    edge_better = int(np.sum(r_edge > r_cloud))
    cloud_better = int(np.sum(r_cloud > r_edge))
    tie = int(np.sum(r_edge == r_cloud))

    report = {
        "scope": "tau2_offline_learnability_oof",
        "n_tasks": len(tasks),
        "non_tie_tasks": non_tie,
        "bidirectional_exclusive": {
            "edge_better": edge_better,
            "cloud_better": cloud_better,
            "tie": tie,
        },
        "per_domain_oracle_gap": per_domain_oracle_gap(tasks),
        "oof_metric_definition": {
            "best_fixed": "max(mean r_edge, mean r_cloud)",
            "oracle": "mean(max(r_edge, r_cloud))",
            "oracle_capture": "(router - best_fixed) / (oracle - best_fixed)",
            "regret": "oracle - router",
        },
        "per_router": run_oof(tasks, n_folds=args.n_folds),
        "permutation_baseline_delta_ridge": permutation_baseline(
            tasks, n_folds=args.n_folds, n_perm=args.n_perm
        ),
        "train_episodes_hash": sha256_json(rows),
    }
    report["report_hash"] = sha256_json(report)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(canonical_json(report) + "\n", encoding="utf-8")
    tmp.replace(target)

    print(f"tasks={len(tasks)} non_tie={non_tie} (edge_better={edge_better}, cloud_better={cloud_better}, tie={tie})")
    for name, entry in report["per_router"].items():
        print(
            f"  {name:34s} oof={entry['oof_reward']:.4f} capture={entry['oracle_capture']:.3f} "
            f"regret={entry['regret']:.4f} bal_acc={entry['cloud_better_balanced_accuracy']:.3f} "
            f"auc={entry['cloud_better_auc']}"
        )
    pb = report["permutation_baseline_delta_ridge"]
    print(f"  permutation oof={pb['oof_reward_mean']:.4f}±{pb['oof_reward_std']:.4f} capture={pb['capture_mean']:.3f}±{pb['capture_std']:.3f}")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())