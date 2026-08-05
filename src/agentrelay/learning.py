"""Trainable estimator for joint executor, payload, and commit actions.

Training rows must be produced by native rollouts over authorized official
training splits.  The module deliberately refuses evaluation/test-purpose rows.
Heavy dependencies are imported only when fitting or loading a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cost import HandoffMeasurement
from .inference import require_immutable_revision
from .policy import CandidateAction, CandidateEstimate
from .schema import CommitMode, Executor, TransferMode, canonical_json, sha256_json


BASE_FEATURE_NAMES = (
    "step_index",
    "remaining_steps",
    "input_tokens",
    "delta_bytes",
    "local_confidence",
    "invariant_count",
    "unresolved_obligation_count",
    "previous_patch_rate",
    "edge_warm_inference_ms",
    "cloud_warm_inference_ms",
    "measured_bandwidth_mbps",
    "previous_handoff_failed",
    "effect_read_only",
    "effect_reversible",
    "effect_irreversible_or_unknown",
    "closure_node_count",
    "dependency_depth",
    "estimated_patch_probability",
    "effect_frontier_blocked",
    "consecutive_steps",
    "dwell_remaining",
)
GOAL_HASH_DIMENSIONS = 32
GOAL_HASH_FEATURE_NAMES = tuple(
    f"goal_hash_{index:02d}" for index in range(GOAL_HASH_DIMENSIONS)
)
FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    "goal_char_count",
    "goal_token_count",
    "goal_numeric_count",
    "goal_constraint_count",
    "visible_action_count",
    *GOAL_HASH_FEATURE_NAMES,
)


def action_key(action: CandidateAction) -> str:
    return ":".join(
        (action.executor.value, action.transfer_mode.value, action.commit_mode.value)
    )


def action_from_key(value: str) -> CandidateAction:
    executor, transfer_mode, commit_mode = value.split(":", maxsplit=2)
    return CandidateAction(
        Executor(executor),
        TransferMode(transfer_mode),
        CommitMode(commit_mode),
    )


def feature_vector(features: Mapping[str, float]) -> list[float]:
    missing = [name for name in FEATURE_NAMES if name not in features]
    extra = [name for name in features if name not in FEATURE_NAMES]
    if missing or extra:
        raise ValueError(f"feature schema mismatch; missing={missing}, extra={extra}")
    return [float(features[name]) for name in FEATURE_NAMES]


@dataclass(frozen=True)
class RouterTrainingRow:
    dataset_id: str
    dataset_revision: str
    split: str
    sample_id: str
    step_index: int
    purpose: str
    features: Mapping[str, float]
    action: CandidateAction
    success: int
    fidelity_pass: int
    inference_ms: float
    controller_ms: float
    handoff: HandoffMeasurement
    # Continuous episode reward (e.g. WebShop reward in [0, 1]).  Kept alongside
    # the binary success flag so the router can be trained on a reward-optimal
    # objective rather than discarding the graded signal to a hard 0/1.
    reward: float = 0.0

    def validate(self, authorized_train_splits: set[str]) -> None:
        require_immutable_revision(self.dataset_revision, subject=self.dataset_id)
        if self.purpose != "train":
            raise ValueError("router fitting accepts train-purpose rollouts only")
        if self.split not in authorized_train_splits:
            raise ValueError(f"split {self.split!r} is not authorized for router fitting")
        if self.success not in {0, 1} or self.fidelity_pass not in {0, 1}:
            raise ValueError("success and fidelity_pass must be binary")
        if not (0.0 <= self.reward <= 1.0):
            raise ValueError("reward must lie in [0, 1]")
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        feature_vector(self.features)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RouterTrainingRow":
        action = value["action"]
        handoff = value.get("handoff", {})
        return cls(
            dataset_id=str(value["dataset_id"]),
            dataset_revision=str(value["dataset_revision"]),
            split=str(value["split"]),
            sample_id=str(value["sample_id"]),
            step_index=int(value["step_index"]),
            purpose=str(value["purpose"]),
            features={str(key): float(item) for key, item in value["features"].items()},
            action=CandidateAction(
                Executor(action["executor"]),
                TransferMode(action["transfer_mode"]),
                CommitMode(action["commit_mode"]),
            ),
            success=int(value["success"]),
            fidelity_pass=int(value["fidelity_pass"]),
            inference_ms=float(value["inference_ms"]),
            controller_ms=float(value.get("controller_ms", 0.0)),
            handoff=HandoffMeasurement(**handoff),
            reward=float(value.get("reward", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = {
            "executor": self.action.executor.value,
            "transfer_mode": self.action.transfer_mode.value,
            "commit_mode": self.action.commit_mode.value,
        }
        return value


@dataclass
class _ConstantProbability:
    probability: float

    def predict_proba(self, features: Any) -> Any:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("AgentRelay[ml] is required for router inference") from exc
        probability = min(1.0, max(0.0, float(self.probability)))
        return np.asarray([[1.0 - probability, probability] for _ in features])


@dataclass
class _ActionPredictor:
    success_model: Any
    fidelity_model: Any
    reward_model: Any
    regressors: Mapping[str, Any]


class JointRouterEstimator:
    def __init__(self) -> None:
        self.predictors: dict[str, _ActionPredictor] = {}
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def _fit_probability(features: Any, labels: Any) -> Any:
        unique = set(int(item) for item in labels)
        if len(unique) == 1:
            return _ConstantProbability(float(next(iter(unique))))
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
        )
        model.fit(features, labels)
        return model

    @staticmethod
    def _fit_regressor(features: Any, values: Any) -> Any:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(features, values)
        return model

    def fit(
        self,
        rows: Iterable[RouterTrainingRow],
        *,
        authorized_train_splits: Iterable[str],
    ) -> "JointRouterEstimator":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("install AgentRelay with the 'ml' extra") from exc
        rows = tuple(rows)
        if not rows:
            raise ValueError("cannot fit the router on zero native rollout rows")
        authorized = set(authorized_train_splits)
        for row in rows:
            row.validate(authorized)
        grouped: dict[str, list[RouterTrainingRow]] = {}
        for row in rows:
            grouped.setdefault(action_key(row.action), []).append(row)

        targets = (
            "inference_ms",
            "controller_ms",
            "encode_ms",
            "communication_ms",
            "rehydration_ms",
            "verification_ms",
            "patch_ms",
            "effect_sync_ms",
            "effect_wait_ms",
            "reconciliation_ms",
            "payload_bytes",
            "target_tokens",
            "fidelity_risk",
            "effect_risk",
        )
        self.predictors = {}
        for key, action_rows in grouped.items():
            if len(action_rows) < 2:
                raise ValueError(f"action {key} needs at least two native rollout rows")
            x = np.asarray([feature_vector(row.features) for row in action_rows], dtype=float)
            success = np.asarray([row.success for row in action_rows], dtype=int)
            fidelity = np.asarray([row.fidelity_pass for row in action_rows], dtype=int)
            reward = np.asarray([row.reward for row in action_rows], dtype=float)
            regressors = {}
            for target in targets:
                if hasattr(action_rows[0], target):
                    values = [getattr(row, target) for row in action_rows]
                else:
                    values = [getattr(row.handoff, target) for row in action_rows]
                regressors[target] = self._fit_regressor(x, np.asarray(values, dtype=float))
            self.predictors[key] = _ActionPredictor(
                success_model=self._fit_probability(x, success),
                fidelity_model=self._fit_probability(x, fidelity),
                reward_model=self._fit_regressor(x, reward),
                regressors=regressors,
            )

        datasets = sorted({(row.dataset_id, row.dataset_revision, row.split) for row in rows})
        self.metadata = {
            "schema_version": "2.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_names": list(FEATURE_NAMES),
            "row_count": len(rows),
            "actions": sorted(self.predictors),
            "authorized_train_splits": sorted(authorized),
            "datasets": [list(item) for item in datasets],
            "training_rows_hash": sha256_json([row.to_dict() for row in rows]),
            "quality_target": "continuous_episode_reward",
            "independent_task_count": len(
                {(row.dataset_id, row.dataset_revision, row.split, row.sample_id) for row in rows}
            ),
        }
        return self

    def candidates(self, features: Mapping[str, float]) -> tuple[CandidateEstimate, ...]:
        if not self.predictors:
            raise RuntimeError("router estimator has not been fitted")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("install AgentRelay with the 'ml' extra") from exc
        x = np.asarray([feature_vector(features)], dtype=float)
        estimates = []
        for key in sorted(self.predictors):
            predictor = self.predictors[key]
            predicted = {
                name: max(0.0, float(model.predict(x)[0]))
                for name, model in predictor.regressors.items()
            }
            estimates.append(
                CandidateEstimate(
                    action=action_from_key(key),
                    predicted_success=float(predictor.success_model.predict_proba(x)[0, 1]),
                    predicted_fidelity=float(predictor.fidelity_model.predict_proba(x)[0, 1]),
                    inference_ms=predicted["inference_ms"],
                    predicted_reward=min(
                        1.0, max(0.0, float(predictor.reward_model.predict(x)[0]))
                    ),
                    controller_ms=predicted["controller_ms"],
                    handoff=HandoffMeasurement(
                        encode_ms=predicted["encode_ms"],
                        communication_ms=predicted["communication_ms"],
                        rehydration_ms=predicted["rehydration_ms"],
                        verification_ms=predicted["verification_ms"],
                        patch_ms=predicted["patch_ms"],
                        effect_sync_ms=predicted["effect_sync_ms"],
                        effect_wait_ms=predicted["effect_wait_ms"],
                        reconciliation_ms=predicted["reconciliation_ms"],
                        payload_bytes=int(round(predicted["payload_bytes"])),
                        target_tokens=int(round(predicted["target_tokens"])),
                        fidelity_risk=min(1.0, predicted["fidelity_risk"]),
                        effect_risk=min(1.0, predicted["effect_risk"]),
                    ),
                    features=dict(features),
                )
            )
        return tuple(estimates)

    def predicted_reward_by_executor(
        self, features: Mapping[str, float]
    ) -> dict[Executor, float]:
        """Return the best learned endpoint reward for each executor."""

        result: dict[Executor, float] = {}
        for estimate in self.candidates(features):
            value = estimate.quality_score
            executor = estimate.action.executor
            result[executor] = max(result.get(executor, 0.0), value)
        return result

    def save(self, path: str | Path) -> None:
        if not self.predictors:
            raise RuntimeError("cannot save an unfitted router")
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("install AgentRelay with the 'ml' extra") from exc
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        joblib.dump(self, temporary)
        temporary.replace(target)
        metadata_path = target.with_suffix(target.suffix + ".metadata.json")
        metadata_path.write_text(canonical_json(self.metadata) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "JointRouterEstimator":
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("install AgentRelay with the 'ml' extra") from exc
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise ValueError("router artifact has an unexpected type")
        if value.metadata.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("router feature schema does not match this runtime")
        return value


def read_training_rows(path: str | Path) -> tuple[RouterTrainingRow, ...]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(RouterTrainingRow.from_dict(json.loads(line)))
    return tuple(rows)
