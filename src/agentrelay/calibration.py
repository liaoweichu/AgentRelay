"""Train/dev-only one-sided calibration for conservative routing bounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .inference import require_immutable_revision
from .policy import CandidateAction, CandidateEstimate
from .schema import CommitMode, Executor, TransferMode, canonical_json, sha256_json


def _action_key(action: CandidateAction) -> str:
    return ":".join(
        (action.executor.value, action.transfer_mode.value, action.commit_mode.value)
    )


def _finite_sample_quantile(values: Iterable[float], alpha: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("calibration scores cannot be empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rank = min(len(ordered), max(1, math.ceil((len(ordered) + 1) * (1 - alpha))))
    return ordered[rank - 1]


@dataclass(frozen=True)
class CalibrationRow:
    dataset_id: str
    dataset_revision: str
    split: str
    sample_id: str
    purpose: str
    action: CandidateAction
    predicted_success: float
    predicted_fidelity: float
    predicted_effect_risk: float
    observed_success: int
    observed_fidelity_pass: int
    observed_effect_failure: int

    def validate(self, authorized_validation_splits: set[str]) -> None:
        require_immutable_revision(self.dataset_revision, subject=self.dataset_id)
        if self.purpose != "tune":
            raise ValueError("risk calibration accepts tune-purpose rows only")
        if self.split not in authorized_validation_splits:
            raise ValueError(f"split {self.split!r} is not authorized for calibration")
        for value in (
            self.predicted_success,
            self.predicted_fidelity,
            self.predicted_effect_risk,
        ):
            if not 0 <= value <= 1:
                raise ValueError("predicted probabilities must be in [0, 1]")
        for value in (
            self.observed_success,
            self.observed_fidelity_pass,
            self.observed_effect_failure,
        ):
            if value not in {0, 1}:
                raise ValueError("observed calibration outcomes must be binary")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = {
            "executor": self.action.executor.value,
            "transfer_mode": self.action.transfer_mode.value,
            "commit_mode": self.action.commit_mode.value,
        }
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationRow":
        action = value["action"]
        return cls(
            dataset_id=str(value["dataset_id"]),
            dataset_revision=str(value["dataset_revision"]),
            split=str(value["split"]),
            sample_id=str(value["sample_id"]),
            purpose=str(value["purpose"]),
            action=CandidateAction(
                Executor(action["executor"]),
                TransferMode(action["transfer_mode"]),
                CommitMode(action["commit_mode"]),
            ),
            predicted_success=float(value["predicted_success"]),
            predicted_fidelity=float(value["predicted_fidelity"]),
            predicted_effect_risk=float(value["predicted_effect_risk"]),
            observed_success=int(value["observed_success"]),
            observed_fidelity_pass=int(value["observed_fidelity_pass"]),
            observed_effect_failure=int(value["observed_effect_failure"]),
        )


@dataclass(frozen=True)
class CalibrationOffsets:
    success_shortfall: float
    fidelity_shortfall: float
    effect_underestimate: float
    count: int


class ConformalRiskCalibrator:
    """Operational one-sided finite-sample calibration.

    This module records empirical conservative bounds. It deliberately does not
    claim distribution-free safety for sequential agent trajectories.
    """

    def __init__(self, *, alpha: float = 0.1, minimum_group_size: int = 20) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if minimum_group_size < 2:
            raise ValueError("minimum_group_size must be at least two")
        self.alpha = float(alpha)
        self.minimum_group_size = int(minimum_group_size)
        self.global_offsets: CalibrationOffsets | None = None
        self.action_offsets: dict[str, CalibrationOffsets] = {}
        self.metadata: dict[str, Any] = {}

    def _offsets(self, rows: Iterable[CalibrationRow]) -> CalibrationOffsets:
        rows = tuple(rows)
        return CalibrationOffsets(
            success_shortfall=max(
                0.0,
                _finite_sample_quantile(
                    (row.predicted_success - row.observed_success for row in rows),
                    self.alpha,
                ),
            ),
            fidelity_shortfall=max(
                0.0,
                _finite_sample_quantile(
                    (row.predicted_fidelity - row.observed_fidelity_pass for row in rows),
                    self.alpha,
                ),
            ),
            effect_underestimate=max(
                0.0,
                _finite_sample_quantile(
                    (row.observed_effect_failure - row.predicted_effect_risk for row in rows),
                    self.alpha,
                ),
            ),
            count=len(rows),
        )

    def fit(
        self,
        rows: Iterable[CalibrationRow],
        *,
        authorized_validation_splits: Iterable[str],
    ) -> "ConformalRiskCalibrator":
        rows = tuple(rows)
        if len(rows) < 2:
            raise ValueError("risk calibration requires at least two validation rows")
        authorized = set(authorized_validation_splits)
        for row in rows:
            row.validate(authorized)
        self.global_offsets = self._offsets(rows)
        grouped: dict[str, list[CalibrationRow]] = {}
        for row in rows:
            grouped.setdefault(_action_key(row.action), []).append(row)
        self.action_offsets = {
            key: self._offsets(group)
            for key, group in grouped.items()
            if len(group) >= self.minimum_group_size
        }
        self.metadata = {
            "schema_version": "1.0",
            "fit_at": datetime.now(timezone.utc).isoformat(),
            "alpha": self.alpha,
            "minimum_group_size": self.minimum_group_size,
            "row_count": len(rows),
            "authorized_validation_splits": sorted(authorized),
            "dataset_revisions": sorted(
                {f"{row.dataset_id}@{row.dataset_revision}:{row.split}" for row in rows}
            ),
            "rows_hash": sha256_json([row.to_dict() for row in rows]),
            "guarantee_scope": "empirical_validation_only",
        }
        return self

    def calibrate(self, estimate: CandidateEstimate) -> CandidateEstimate:
        if self.global_offsets is None:
            raise RuntimeError("risk calibrator has not been fitted")
        key = _action_key(estimate.action)
        offsets = self.action_offsets.get(key, self.global_offsets)
        source = "action" if key in self.action_offsets else "global"
        return replace(
            estimate,
            success_lower_bound=max(
                0.0, estimate.predicted_success - offsets.success_shortfall
            ),
            fidelity_lower_bound=max(
                0.0, estimate.predicted_fidelity - offsets.fidelity_shortfall
            ),
            effect_risk_upper_bound=min(
                1.0, estimate.handoff.effect_risk + offsets.effect_underestimate
            ),
            calibration_source=f"{source}:n={offsets.count}:alpha={self.alpha}",
        )

    def calibrate_all(
        self, estimates: Iterable[CandidateEstimate]
    ) -> tuple[CandidateEstimate, ...]:
        return tuple(self.calibrate(estimate) for estimate in estimates)

    def save(self, path: str | Path) -> None:
        if self.global_offsets is None:
            raise RuntimeError("cannot save an unfitted calibrator")
        payload = {
            "alpha": self.alpha,
            "minimum_group_size": self.minimum_group_size,
            "global_offsets": asdict(self.global_offsets),
            "action_offsets": {
                key: asdict(value) for key, value in sorted(self.action_offsets.items())
            },
            "metadata": self.metadata,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "ConformalRiskCalibrator":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(
            alpha=float(value["alpha"]),
            minimum_group_size=int(value["minimum_group_size"]),
        )
        result.global_offsets = CalibrationOffsets(**value["global_offsets"])
        result.action_offsets = {
            str(key): CalibrationOffsets(**offsets)
            for key, offsets in value.get("action_offsets", {}).items()
        }
        result.metadata = dict(value.get("metadata", {}))
        return result


def read_calibration_rows(path: str | Path) -> tuple[CalibrationRow, ...]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(CalibrationRow.from_dict(json.loads(line)))
    return tuple(rows)

