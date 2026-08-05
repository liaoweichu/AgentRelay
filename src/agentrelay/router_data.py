"""Leakage-safe task-level router rows from paired fixed-endpoint episodes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .cost import HandoffMeasurement
from .learning import RouterTrainingRow, feature_vector
from .policy import CandidateAction
from .schema import CommitMode, Executor, TransferMode
from .webshop_protocol import (
    WEBSHOP_TOTAL_HUMAN_GOALS,
    canonical_webshop_split,
    official_webshop_indices,
)


def read_episode_records(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        rows = value.get("rows", value.get("episodes", ()))
    else:
        rows = value
    if not isinstance(rows, (list, tuple)):
        raise ValueError("episode artifact must be a list or contain rows/episodes")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("every episode record must be an object")
    return tuple(rows)


def endpoint_role(episode: Mapping[str, Any]) -> Executor:
    direct = str(episode.get("role", "")).strip().lower()
    if direct in {Executor.EDGE.value, Executor.CLOUD.value}:
        return Executor(direct)
    method = str(episode.get("method", "")).strip().lower()
    if method == "edge_only":
        return Executor.EDGE
    if method == "cloud_only":
        return Executor.CLOUD
    selected = {
        str(step.get("selected_executor", "")).strip().lower()
        for step in episode.get("steps", ())
        if isinstance(step, Mapping)
    }
    selected.discard("")
    if len(selected) == 1 and next(iter(selected)) in {"edge", "cloud"}:
        return Executor(next(iter(selected)))
    raise ValueError("episode is not a fixed edge/cloud endpoint rollout")


def episode_key(episode: Mapping[str, Any]) -> tuple[str, str, str, str]:
    sample_id = episode.get("sample_id", episode.get("task_id"))
    if sample_id is None:
        raise ValueError("episode has no sample_id/task_id")
    return (
        str(episode.get("benchmark", "")),
        str(episode.get("dataset_revision", "")),
        str(episode.get("split", "")),
        str(sample_id),
    )


def validate_endpoint_episode_scope(
    episode: Mapping[str, Any],
    *,
    authorized_splits: Iterable[str],
) -> None:
    """Reject split aliases, held-out inputs, and invalid WebShop session ids."""

    allowed = {str(split) for split in authorized_splits}
    split = str(episode.get("split", ""))
    if split not in allowed:
        raise ValueError(
            f"endpoint episode split {split!r} is not authorized; expected {sorted(allowed)}"
        )
    if episode.get("labels_accessed_by_router") is not False:
        raise ValueError("endpoint episode violates the router label boundary")
    benchmark = str(episode.get("benchmark", "")).strip().lower()
    if "webshop" not in benchmark:
        return
    canonical = canonical_webshop_split(split)
    if canonical != split:
        raise ValueError(
            f"WebShop endpoint episodes must use canonical split {canonical!r}"
        )
    sample_id = episode.get("sample_id", episode.get("task_id"))
    try:
        session_id = int(str(sample_id))
    except (TypeError, ValueError) as exc:
        raise ValueError("WebShop endpoint sample_id must be an official session id") from exc
    if session_id not in set(
        official_webshop_indices(WEBSHOP_TOTAL_HUMAN_GOALS, split)
    ):
        raise ValueError(
            f"WebShop session {session_id} is outside the official {split} split"
        )


def pair_endpoint_episodes(
    episodes: Iterable[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    grouped: dict[tuple[str, str, str, str], dict[Executor, Mapping[str, Any]]] = {}
    for episode in episodes:
        key = episode_key(episode)
        role = endpoint_role(episode)
        by_role = grouped.setdefault(key, {})
        if role in by_role:
            raise ValueError(f"duplicate {role.value} endpoint for task {key}")
        by_role[role] = episode
    missing = [key for key, value in grouped.items() if set(value) != set(Executor)]
    if missing:
        raise ValueError(f"paired endpoint artifact is incomplete for tasks: {missing[:5]}")
    return tuple(
        (value[Executor.EDGE], value[Executor.CLOUD])
        for _, value in sorted(grouped.items())
    )


def _handoff_from_step(step: Mapping[str, Any]) -> HandoffMeasurement:
    return HandoffMeasurement(
        encode_ms=float(step.get("handoff_encode_ms", 0.0)),
        communication_ms=float(step.get("handoff_communication_ms", 0.0)),
        rehydration_ms=float(step.get("handoff_rehydration_ms", 0.0)),
        verification_ms=float(step.get("handoff_verify_ms", 0.0)),
        patch_ms=float(step.get("handoff_patch_ms", 0.0)),
        effect_wait_ms=float(step.get("handoff_effect_wait_ms", 0.0)),
        reconciliation_ms=float(step.get("handoff_reconciliation_ms", 0.0)),
        payload_bytes=int(step.get("handoff_bytes", 0)),
        target_tokens=int(step.get("target_tokens", 0)),
        fidelity_risk=float(step.get("fidelity_risk", 0.0)),
        effect_risk=float(step.get("effect_risk", 0.0)),
    )


def task_router_training_rows(
    episodes: Iterable[Mapping[str, Any]],
    *,
    authorized_train_splits: Iterable[str],
) -> tuple[RouterTrainingRow, ...]:
    """Create two independent endpoint rows per task from step-zero features."""

    allowed = set(authorized_train_splits)
    rows: list[RouterTrainingRow] = []
    for edge_episode, cloud_episode in pair_endpoint_episodes(episodes):
        first_features: dict[Executor, Mapping[str, float]] = {}
        for role, episode in (
            (Executor.EDGE, edge_episode),
            (Executor.CLOUD, cloud_episode),
        ):
            validate_endpoint_episode_scope(
                episode,
                authorized_splits=allowed,
            )
            steps = tuple(episode.get("steps", ()))
            if not steps:
                raise ValueError("task router training episode has no recorded steps")
            if any(str(step.get("selected_executor")) != role.value for step in steps):
                raise ValueError("task router rows require fixed-endpoint trajectories")
            first = steps[0]
            if int(first.get("step_index", -1)) != 0:
                raise ValueError("task router row must use the pre-action step-zero context")
            features = {
                str(name): float(value)
                for name, value in first.get("router_features", {}).items()
            }
            feature_vector(features)
            first_features[role] = features
            action = CandidateAction(
                role,
                TransferMode(str(first["transfer_mode"])),
                CommitMode(str(first["commit_mode"])),
            )
            row = RouterTrainingRow(
                dataset_id=str(episode["benchmark"]),
                dataset_revision=str(episode["dataset_revision"]),
                split=str(episode["split"]),
                sample_id=str(episode.get("sample_id", episode.get("task_id"))),
                step_index=0,
                purpose="train",
                features=features,
                action=action,
                success=int(float(episode.get("success", 0.0)) > 0.0),
                reward=min(1.0, max(0.0, float(episode.get("reward", 0.0)))),
                fidelity_pass=int(int(episode.get("effect_failures", 0)) == 0),
                inference_ms=float(first.get("inference_ms", 0.0)),
                controller_ms=float(first.get("controller_ms", 0.0)),
                handoff=_handoff_from_step(first),
            )
            row.validate(allowed)
            rows.append(row)
        if dict(first_features[Executor.EDGE]) != dict(first_features[Executor.CLOUD]):
            raise ValueError(
                "paired task endpoints must expose identical pre-action router features"
            )
    if not rows:
        raise ValueError("no paired task-level router rows were produced")
    return tuple(rows)
