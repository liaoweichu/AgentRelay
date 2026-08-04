"""Runnable baseline registry and deterministic selection controllers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Iterable

from .policy import (
    CandidateEstimate,
    ConstrainedUtilityPolicy,
    RoutingContext,
    RoutingDecision,
)
from .schema import Executor, TransferMode, sha256_json


class BaselineName(str, Enum):
    EDGE_ONLY = "edge_only"
    CLOUD_ONLY = "cloud_only"
    RANDOM_03 = "random_0.3"
    RANDOM_05 = "random_0.5"
    ROUTELLM_TASK = "routellm_task"
    FRUGALGPT_CASCADE = "frugalgpt_cascade"
    MODEL_ONLY_STEP = "model_only_step"
    AGENTROUTER_STYLE = "agentrouter_style"
    HERA_AGREEMENT = "hera_agreement"
    FULL_REPLAY_STEP = "full_replay_step"
    UNCALIBRATED_JOINT = "uncalibrated_joint"
    AGENTRELAY = "agentrelay"


@dataclass(frozen=True)
class BaselineSpec:
    name: BaselineName
    category: str
    requires_router: bool
    requires_both_model_proposals: bool
    transfer_policy: str
    implementation_status: str = "implemented"


BASELINE_REGISTRY: dict[BaselineName, BaselineSpec] = {
    BaselineName.EDGE_ONLY: BaselineSpec(
        BaselineName.EDGE_ONLY, "sanity", False, False, "reuse_or_full_replay"
    ),
    BaselineName.CLOUD_ONLY: BaselineSpec(
        BaselineName.CLOUD_ONLY, "sanity", False, False, "reuse_or_full_replay"
    ),
    BaselineName.RANDOM_03: BaselineSpec(
        BaselineName.RANDOM_03, "sanity", False, False, "reuse_or_full_replay"
    ),
    BaselineName.RANDOM_05: BaselineSpec(
        BaselineName.RANDOM_05, "sanity", False, False, "reuse_or_full_replay"
    ),
    BaselineName.ROUTELLM_TASK: BaselineSpec(
        BaselineName.ROUTELLM_TASK, "task_router", True, False, "fixed_for_trajectory"
    ),
    BaselineName.FRUGALGPT_CASCADE: BaselineSpec(
        BaselineName.FRUGALGPT_CASCADE, "cascade", True, False, "full_replay_on_escalation"
    ),
    BaselineName.MODEL_ONLY_STEP: BaselineSpec(
        BaselineName.MODEL_ONLY_STEP, "step_router", True, False, "full_replay"
    ),
    BaselineName.AGENTROUTER_STYLE: BaselineSpec(
        BaselineName.AGENTROUTER_STYLE, "closest_router", True, False, "full_replay"
    ),
    BaselineName.HERA_AGREEMENT: BaselineSpec(
        BaselineName.HERA_AGREEMENT, "closest_router", False, True, "full_replay"
    ),
    BaselineName.FULL_REPLAY_STEP: BaselineSpec(
        BaselineName.FULL_REPLAY_STEP, "state_baseline", True, False, "full_replay"
    ),
    BaselineName.UNCALIBRATED_JOINT: BaselineSpec(
        BaselineName.UNCALIBRATED_JOINT, "ablation", True, False, "joint"
    ),
    BaselineName.AGENTRELAY: BaselineSpec(
        BaselineName.AGENTRELAY, "method", True, False, "joint_calibrated"
    ),
}


@dataclass(frozen=True)
class BaselineInput:
    estimates: tuple[CandidateEstimate, ...]
    context: RoutingContext
    sample_id: str
    step_index: int
    edge_confidence: float = 0.0
    task_confidence: float = 0.0
    edge_proposal: str = ""
    cloud_proposal: str = ""


def _normalize_action(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _preferred_candidate(
    estimates: Iterable[CandidateEstimate],
    *,
    executor: Executor,
    current_executor: Executor,
    full_replay_on_switch: bool = True,
) -> CandidateEstimate:
    estimates = tuple(estimates)
    switching = executor is not current_executor
    preferred_mode = TransferMode.FULL_REPLAY if switching and full_replay_on_switch else (
        TransferMode.REUSE if not switching else None
    )
    matching = [
        estimate
        for estimate in estimates
        if estimate.action.executor is executor
        and (preferred_mode is None or estimate.action.transfer_mode is preferred_mode)
    ]
    if not matching:
        matching = [estimate for estimate in estimates if estimate.action.executor is executor]
    if not matching:
        raise ValueError(f"no candidate for executor {executor.value}")
    return min(
        matching,
        key=lambda item: (
            item.total_latency_ms,
            -item.predicted_success,
            -item.predicted_fidelity,
            item.action.transfer_mode.value,
        ),
    )


class BaselineController:
    def __init__(
        self,
        name: BaselineName | str,
        *,
        policy: ConstrainedUtilityPolicy | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.name = BaselineName(name)
        self.spec = BASELINE_REGISTRY[self.name]
        self.policy = policy or ConstrainedUtilityPolicy()
        self.threshold = float(threshold)
        self._task_executor: dict[str, Executor] = {}

    @property
    def requires_both_model_proposals(self) -> bool:
        return self.spec.requires_both_model_proposals

    def _decision(self, estimate: CandidateEstimate, reason: str) -> RoutingDecision:
        return RoutingDecision(
            selected=estimate,
            feasible_count=1,
            objective=estimate.total_latency_ms,
            objective_name=f"baseline:{self.name.value}",
            reason=reason,
            dwell_decision="baseline_switch",
        )

    def select(self, value: BaselineInput) -> RoutingDecision:
        context = value.context
        if self.name is BaselineName.EDGE_ONLY:
            return self._decision(
                _preferred_candidate(
                    value.estimates,
                    executor=Executor.EDGE,
                    current_executor=context.current_executor,
                ),
                "fixed edge-only baseline",
            )
        if self.name is BaselineName.CLOUD_ONLY:
            return self._decision(
                _preferred_candidate(
                    value.estimates,
                    executor=Executor.CLOUD,
                    current_executor=context.current_executor,
                ),
                "fixed cloud-only baseline",
            )
        if self.name in {BaselineName.RANDOM_03, BaselineName.RANDOM_05}:
            probability = 0.3 if self.name is BaselineName.RANDOM_03 else 0.5
            draw = int(
                sha256_json(
                    {
                        "sample_id": value.sample_id,
                        "step_index": value.step_index,
                        "baseline": self.name.value,
                    }
                )[:13],
                16,
            ) / float(16**13 - 1)
            executor = Executor.CLOUD if draw < probability else Executor.EDGE
            return self._decision(
                _preferred_candidate(
                    value.estimates,
                    executor=executor,
                    current_executor=context.current_executor,
                ),
                f"deterministic random baseline with cloud_probability={probability}",
            )
        if self.name is BaselineName.ROUTELLM_TASK:
            executor = self._task_executor.setdefault(
                value.sample_id,
                Executor.EDGE if value.task_confidence >= self.threshold else Executor.CLOUD,
            )
            return self._decision(
                _preferred_candidate(
                    value.estimates,
                    executor=executor,
                    current_executor=context.current_executor,
                ),
                "task-level confidence router fixed for the full trajectory",
            )
        if self.name is BaselineName.FRUGALGPT_CASCADE:
            executor = (
                Executor.EDGE if value.edge_confidence >= self.threshold else Executor.CLOUD
            )
            return self._decision(
                _preferred_candidate(
                    value.estimates,
                    executor=executor,
                    current_executor=context.current_executor,
                ),
                "edge-first threshold cascade with full replay on escalation",
            )
        if self.name is BaselineName.HERA_AGREEMENT:
            if not value.edge_proposal or not value.cloud_proposal:
                raise ValueError("Hera-style agreement requires both native model proposals")
            executor = (
                Executor.EDGE
                if _normalize_action(value.edge_proposal) == _normalize_action(value.cloud_proposal)
                else Executor.CLOUD
            )
            return self._decision(
                _preferred_candidate(
                    value.estimates,
                    executor=executor,
                    current_executor=context.current_executor,
                ),
                "Hera-style action agreement reproduction",
            )
        if self.name in {
            BaselineName.MODEL_ONLY_STEP,
            BaselineName.AGENTROUTER_STYLE,
            BaselineName.FULL_REPLAY_STEP,
        }:
            restricted = [
                estimate
                for estimate in value.estimates
                if (
                    estimate.action.transfer_mode is TransferMode.REUSE
                    if estimate.action.executor is context.current_executor
                    else estimate.action.transfer_mode is TransferMode.FULL_REPLAY
                )
            ]
            if not restricted:
                raise ValueError("model-only baseline has no reuse/full-replay candidates")
            if self.name is BaselineName.AGENTROUTER_STYLE:
                selected = max(
                    restricted,
                    key=lambda item: (
                        item.predicted_success - 0.001 * item.inference_ms,
                        item.predicted_fidelity,
                    ),
                )
                reason = "AgentRouter-style trajectory classifier objective"
            else:
                selected = max(
                    restricted,
                    key=lambda item: (
                        item.predicted_success,
                        item.predicted_fidelity,
                        -item.inference_ms,
                    ),
                )
                reason = "model-only step routing with fixed full replay"
            return self._decision(selected, reason)
        if self.name is BaselineName.UNCALIBRATED_JOINT:
            uncalibrated = tuple(
                replace(
                    estimate,
                    success_lower_bound=None,
                    fidelity_lower_bound=None,
                    effect_risk_upper_bound=None,
                    calibration_source="",
                )
                for estimate in value.estimates
            )
            return self.policy.select(
                uncalibrated,
                replace(context, require_calibrated_bounds=False),
            )
        if self.name is BaselineName.AGENTRELAY:
            return self.policy.select(
                value.estimates,
                replace(context, require_calibrated_bounds=True),
            )
        raise AssertionError(f"unhandled baseline {self.name}")


def baseline_manifest() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": spec.name.value,
            "category": spec.category,
            "requires_router": spec.requires_router,
            "requires_both_model_proposals": spec.requires_both_model_proposals,
            "transfer_policy": spec.transfer_policy,
            "implementation_status": spec.implementation_status,
        }
        for spec in BASELINE_REGISTRY.values()
    )
