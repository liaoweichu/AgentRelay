#!/usr/bin/env python3
"""G14c: fit an InterCode-SQL edge/cloud router and run the learnability gate.

Consumes the four G14b fixed-endpoint episode sets (edge/cloud x train/dev)
produced by run_intersql_matrix_endpoint.py and the corresponding selected
library manifests. It fits a task-level reward router on the train endpoints
and evaluates oracle capture on the held-out dev endpoints.

The gate only uses pre-action task features (hardness, question surface
statistics, schema size, library identity hash). Router labels never leak into
the dev split, and the official dev split is disjoint from train by library.

Outputs into <out-dir>/g14c-router/:
  router.joblib              fitted router artifact
  router.joblib.metadata.json router training provenance
  learnability-gate.json      gate result (gate_pass + full metrics)
  freeze.json                 frozen protocol/manifest/metrics/router record
  receipt.json                run summary
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.intercode_sql import intercode_task_features, paired_reward_summary  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402

RESULTS = PROJECT_ROOT / "results/intercode-sql-g14-matrix"
MATRIX_DIR = RESULTS

FEATURE_NAMES = (
    "hardness_ordinal",
    "query_char_count",
    "query_token_count",
    "query_numeric_count",
    "n_tables",
    "n_columns",
    "db_hash_00",
    "db_hash_01",
    "db_hash_02",
    "db_hash_03",
    "db_hash_04",
    "db_hash_05",
    "db_hash_06",
    "db_hash_07",
)


def _load_episodes(matrix_dir: Path, role: str, split: str) -> dict[str, dict]:
    path = matrix_dir / f"intercode-sql-{role}-{split}-episodes.json"
    episodes = __import__("json").loads(path.read_text(encoding="utf-8"))
    return {str(episode["task_id"]): episode for episode in episodes}


def _load_manifest(matrix_dir: Path, split: str) -> dict[str, dict]:
    path = matrix_dir / f"ic_spider_{split}_subset.json"
    records = __import__("json").loads(path.read_text(encoding="utf-8"))
    return {str(record["task_id"]): record for record in records}


def _feature_row(manifest: dict[str, dict], task_id: str, episode: dict) -> dict[str, float]:
    record = manifest[task_id]
    return intercode_task_features(
        db=str(episode["db"]),
        hardness=str(episode.get("hardness", record.get("hardness", "unknown"))),
        query=str(episode.get("query", record.get("query", ""))),
        db_tables=record.get("db_tables"),
    )


class _IntersqlRewardRouter:
    """Per-executor Ridge reward model; pick the higher predicted reward."""

    def __init__(self, target: str = "reward") -> None:
        self.target = target
        self.executor_models: dict[str, object] = {}

    def fit(self, rows: list[tuple[dict[str, float], float, str]]) -> "_IntersqlRewardRouter":
        import numpy as np

        by_executor: dict[str, list] = {}
        for features, value, executor in rows:
            by_executor.setdefault(executor, ([], []))
            by_executor[executor][0].append([features[name] for name in FEATURE_NAMES])
            by_executor[executor][1].append(value)
        self.executor_models = {}
        for executor, (x, y) in by_executor.items():
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if self.target == "success":
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler

                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
                )
                model.fit(x, y)
            else:
                from sklearn.linear_model import Ridge
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler

                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(x, y)
            self.executor_models[executor] = model
        return self

    def predicted_reward_by_executor(self, features: dict[str, float]) -> dict[str, float]:
        import numpy as np

        x = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
        result: dict[str, float] = {}
        for executor, model in self.executor_models.items():
            if self.target == "success":
                value = float(model.predict_proba(x)[0, 1])
            else:
                value = float(model.predict(x)[0])
            result[executor] = min(1.0, max(0.0, value))
        return result

    def metadata(self) -> dict:
        return {
            "kind": "intercode_sql_reward_router",
            "target": self.target,
            "model": "per_executor_logistic_success"
            if self.target == "success"
            else "per_executor_ridge_reward",
            "feature_names": list(FEATURE_NAMES),
            "executors": sorted(self.executor_models),
        }


def _paired(train_manifest, edge_eps, cloud_eps) -> list[tuple[dict, dict, dict[str, float]]]:
    pairs = []
    for task_id in edge_eps:
        if task_id not in cloud_eps:
            continue
        pairs.append(
            (
                edge_eps[task_id],
                cloud_eps[task_id],
                _feature_row(train_manifest, task_id, edge_eps[task_id]),
            )
        )
    pairs.sort(key=lambda item: item[0]["task_id"])
    return pairs


def _evaluate_gate(router, dev_pairs, *, target, minimum_paired_tasks, minimum_oracle_capture,
                   minimum_cloud_fraction, maximum_cloud_fraction):
    from statistics import mean

    def value(episode):
        return float(episode["success"]) if target == "success" else float(episode["reward"])

    edge_values = [value(e) for e, _, _ in dev_pairs]
    cloud_values = [value(c) for _, c, _ in dev_pairs]
    router_values = []
    selected = []
    for edge, cloud, features in dev_pairs:
        predicted = router.predicted_reward_by_executor(features)
        role = "cloud" if predicted["cloud"] > predicted["edge"] else "edge"
        selected.append(role)
        router_values.append(value(cloud) if role == "cloud" else value(edge))

    edge_mean = mean(edge_values)
    cloud_mean = mean(cloud_values)
    best_role = "edge" if edge_mean >= cloud_mean else "cloud"
    best_fixed = max(edge_mean, cloud_mean)
    oracle = mean(max(e, c) for e, c in zip(edge_values, cloud_values))
    router_mean = mean(router_values)
    oracle_gap = oracle - best_fixed
    capture = (router_mean - best_fixed) / oracle_gap if oracle_gap > 0 else 0.0
    cloud_fraction = mean(r == "cloud" for r in selected)

    n_tasks = len(dev_pairs)
    non_tie = sum(1 for e, c, _ in dev_pairs if value(e) != value(c))
    both = sum(1 for e, c, _ in dev_pairs if float(e["success"]) and float(c["success"]))
    edge_only = sum(1 for e, c, _ in dev_pairs if float(e["success"]) and not float(c["success"]))
    cloud_only = sum(1 for e, c, _ in dev_pairs if float(c["success"]) and not float(e["success"]))

    checks = {
        "enough_dev_tasks": n_tasks >= minimum_paired_tasks,
        "positive_oracle_gap": oracle_gap > 0,
        "router_not_below_best_fixed": router_mean >= best_fixed,
        "oracle_capture": capture >= minimum_oracle_capture,
        "nondegenerate_cloud_fraction": (
            minimum_cloud_fraction <= cloud_fraction <= maximum_cloud_fraction
        ),
    }
    return {
        "target": target,
        "gate_pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_paired_tasks": minimum_paired_tasks,
            "minimum_oracle_capture": minimum_oracle_capture,
            "minimum_cloud_fraction": minimum_cloud_fraction,
            "maximum_cloud_fraction": maximum_cloud_fraction,
        },
        "paired_dev_tasks": n_tasks,
        "edge_avg_reward": edge_mean,
        "cloud_avg_reward": cloud_mean,
        "best_fixed_role": best_role,
        "best_fixed_avg_reward": best_fixed,
        "oracle_avg_reward": oracle,
        "oracle_gap": oracle_gap,
        "router_avg_reward": router_mean,
        "oracle_gap_capture": capture,
        "cloud_selection_fraction": cloud_fraction,
        "reward_non_tie": non_tie,
        "success_both": both,
        "success_edge_only": edge_only,
        "success_cloud_only": cloud_only,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", default=str(MATRIX_DIR))
    parser.add_argument("--out-dir", default=str(MATRIX_DIR / "g14c-router"))
    parser.add_argument("--minimum-paired-tasks", type=int, default=50)
    parser.add_argument("--minimum-oracle-capture", type=float, default=0.30)
    parser.add_argument("--minimum-cloud-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-cloud-fraction", type=float, default=0.90)
    parser.add_argument(
        "--target",
        choices=("reward", "success"),
        default="success",
        help="router learning and gate evaluation metric",
    )
    args = parser.parse_args()

    matrix_dir = Path(args.matrix_dir)

    edge_train = _load_episodes(matrix_dir, "edge", "train")
    cloud_train = _load_episodes(matrix_dir, "cloud", "train")
    edge_dev = _load_episodes(matrix_dir, "edge", "dev")
    cloud_dev = _load_episodes(matrix_dir, "cloud", "dev")
    train_manifest = _load_manifest(matrix_dir, "train")
    dev_manifest = _load_manifest(matrix_dir, "dev")

    train_pairs = _paired(train_manifest, edge_train, cloud_train)
    dev_pairs = _paired(dev_manifest, edge_dev, cloud_dev)
    if not train_pairs or not dev_pairs:
        raise ValueError("need at least one paired train and one paired dev task")

    train_rows = []
    for edge, cloud, features in train_pairs:
        target = float(edge["success"]) if args.target == "success" else float(edge["reward"])
        train_rows.append((features, target, "edge"))
        target = float(cloud["success"]) if args.target == "success" else float(cloud["reward"])
        train_rows.append((features, target, "cloud"))
    router = _IntersqlRewardRouter(target=args.target).fit(train_rows)

    train_summary = paired_reward_summary(
        [float(e["reward"]) for e, _, _ in train_pairs],
        [float(c["reward"]) for _, c, _ in train_pairs],
    )
    dev_summary = paired_reward_summary(
        [float(e["reward"]) for e, _, _ in dev_pairs],
        [float(c["reward"]) for _, c, _ in dev_pairs],
    )
    gate = _evaluate_gate(
        router,
        dev_pairs,
        target=args.target,
        minimum_paired_tasks=args.minimum_paired_tasks,
        minimum_oracle_capture=args.minimum_oracle_capture,
        minimum_cloud_fraction=args.minimum_cloud_fraction,
        maximum_cloud_fraction=args.maximum_cloud_fraction,
    )
    gate["gate_pass"] = all(gate["checks"].values())
    gate["paper_evidence"] = False
    gate["scope"] = "offline_intersql_task_router_train_dev_gate"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    router_path = out_dir / "router.joblib"
    joblib.dump(router, router_path)
    (out_dir / "router.joblib.metadata.json").write_text(
        canonical_json(router.metadata()) + "\n", encoding="utf-8"
    )

    freeze = {
        "paper_evidence": False,
        "protocol_hash": sha256_json(
            (PROJECT_ROOT / "src/agentrelay/intercode_sql.py").read_bytes().decode("utf-8")
        ),
        "runner_hash": sha256_json(
            (PROJECT_ROOT / "scripts/run_intersql_matrix_endpoint.py")
            .read_bytes()
            .decode("utf-8")
        ),
        "train_manifest_hash": sha256_json(
            (matrix_dir / "ic_spider_train_subset.json").read_bytes().decode("utf-8")
        ),
        "dev_manifest_hash": sha256_json(
            (matrix_dir / "ic_spider_dev_subset.json").read_bytes().decode("utf-8")
        ),
        "episode_hashes": {
            "edge_train": sha256_json(edge_train),
            "cloud_train": sha256_json(cloud_train),
            "edge_dev": sha256_json(edge_dev),
            "cloud_dev": sha256_json(cloud_dev),
        },
        "router_metadata": router.metadata(),
        "feature_names": list(FEATURE_NAMES),
        "train_tasks": len(train_pairs),
        "dev_tasks": len(dev_pairs),
    }
    freeze["freeze_hash"] = sha256_json(freeze)
    (out_dir / "freeze.json").write_text(canonical_json(freeze) + "\n", encoding="utf-8")

    gate["freeze_hash"] = freeze["freeze_hash"]
    gate["gate_hash"] = sha256_json(gate)
    (out_dir / "learnability-gate.json").write_text(
        canonical_json(gate) + "\n", encoding="utf-8"
    )

    receipt = {
        "paper_evidence": False,
        "scope": "g14c_intersql_router_learnability_gate",
        "train_paired_tasks": len(train_pairs),
        "dev_paired_tasks": len(dev_pairs),
        "train_summary": train_summary,
        "dev_summary": dev_summary,
        "gate": gate,
        "freeze_hash": freeze["freeze_hash"],
    }
    (out_dir / "receipt.json").write_text(
        canonical_json(receipt) + "\n", encoding="utf-8"
    )

    print(
        f"train_pairs={len(train_pairs)} dev_pairs={len(dev_pairs)} "
        f"dev_oracle_gap={gate['oracle_gap']:.4f} "
        f"dev_oracle_capture={gate['oracle_gap_capture']:.4f} "
        f"cloud_fraction={gate['cloud_selection_fraction']:.3f} "
        f"reward_non_tie={gate['reward_non_tie']} "
        f"pass={gate['gate_pass']} output={out_dir}"
    )
    return 0 if gate["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
