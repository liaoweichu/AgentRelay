"""Lightweight calibrated, risk-bounded semi-Markov handoff policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .cost import CostWeights, HandoffMeasurement
from .schema import (
    CommitMode,
    EffectClass,
    EffectFrontierSnapshot,
    Executor,
    TransferMode,
)


@dataclass(frozen=True)
class CandidateAction:
    executor: Executor
    transfer_mode: TransferMode
    commit_mode: CommitMode


@dataclass(frozen=True)
class RoutingContext:
    current_executor: Executor
    effect_class: EffectClass = EffectClass.READ_ONLY
    minimum_success: float = 0.0
    minimum_fidelity: float = 0.0
    maximum_effect_risk: float = 1.0
    consecutive_steps: int = 0
    minimum_dwell_steps: int = 0
    effect_frontier: EffectFrontierSnapshot | None = None
    require_calibrated_bounds: bool = False


@dataclass(frozen=True)
class CandidateEstimate:
    action: CandidateAction
    predicted_success: float
    predicted_fidelity: float
    inference_ms: float
    inference_tokens: int = 0
    controller_ms: float = 0.0
    handoff: HandoffMeasurement = HandoffMeasurement()
    features: Mapping[str, float] | None = None
    success_lower_bound: float | None = None
    fidelity_lower_bound: float | None = None
    effect_risk_upper_bound: float | None = None
    calibration_source: str = ""

    @property
    def total_latency_ms(self) -> float:
        return self.inference_ms + self.controller_ms + self.handoff.total_ms

    @property
    def conservative_success(self) -> float:
        return (
            self.success_lower_bound
            if self.success_lower_bound is not None
            else self.predicted_success
        )

    @property
    def conservative_fidelity(self) -> float:
        return (
            self.fidelity_lower_bound
            if self.fidelity_lower_bound is not None
            else self.predicted_fidelity
        )

    @property
    def conservative_effect_risk(self) -> float:
        return (
            self.effect_risk_upper_bound
            if self.effect_risk_upper_bound is not None
            else self.handoff.effect_risk
        )

    @property
    def calibrated(self) -> bool:
        return all(
            value is not None
            for value in (
                self.success_lower_bound,
                self.fidelity_lower_bound,
                self.effect_risk_upper_bound,
            )
        )


@dataclass(frozen=True)
class RoutingDecision:
    selected: CandidateEstimate
    feasible_count: int
    objective: float
    objective_name: str
    reason: str
    abstained: bool = False
    dwell_decision: str = "continue"
    constraint_violations: tuple[str, ...] = ()


class ConstrainedUtilityPolicy:
    def __init__(
        self,
        *,
        cost_weights: CostWeights | None = None,
        success_weight: float = 1000.0,
        switch_hysteresis_ms: float = 0.0,
        selection_mode: str = "constrained_cost",
    ) -> None:
        if selection_mode not in {"constrained_cost", "scalarized_utility"}:
            raise ValueError(f"unsupported selection mode: {selection_mode}")
        self.cost_weights = cost_weights or CostWeights()
        self.success_weight = success_weight
        self.switch_hysteresis_ms = switch_hysteresis_ms
        self.selection_mode = selection_mode

    @staticmethod
    def _legal(estimate: CandidateEstimate, context: RoutingContext) -> bool:
        switching = estimate.action.executor is not context.current_executor
        if switching and estimate.action.transfer_mode is TransferMode.REUSE:
            return False
        if not switching and estimate.action.transfer_mode is not TransferMode.REUSE:
            return False
        if switching and context.consecutive_steps < context.minimum_dwell_steps:
            return False
        frontier = context.effect_frontier
        if switching and frontier is not None:
            if not frontier.migration_allowed:
                return False
            required = frontier.required_commit_mode
            if required is CommitMode.BARRIER and estimate.action.commit_mode is not CommitMode.BARRIER:
                return False
            if required is CommitMode.COMPENSATING and estimate.action.commit_mode not in {
                CommitMode.BARRIER,
                CommitMode.COMPENSATING,
            }:
                return False
            if required is CommitMode.RECONCILE:
                return False
        if context.effect_class is EffectClass.REVERSIBLE:
            if estimate.action.commit_mode not in {CommitMode.BARRIER, CommitMode.COMPENSATING}:
                return False
        elif context.effect_class in {EffectClass.IRREVERSIBLE, EffectClass.UNKNOWN}:
            if estimate.action.commit_mode is not CommitMode.BARRIER:
                return False
        elif (
            context.effect_class is EffectClass.READ_ONLY
            and estimate.action.commit_mode in {CommitMode.COMPENSATING, CommitMode.RECONCILE}
        ):
            return False
        return True

    @staticmethod
    def _violations(estimate: CandidateEstimate, context: RoutingContext) -> tuple[str, ...]:
        violations = []
        if context.require_calibrated_bounds and not estimate.calibrated:
            violations.append("uncalibrated")
        if estimate.conservative_success < context.minimum_success:
            violations.append("success")
        if estimate.conservative_fidelity < context.minimum_fidelity:
            violations.append("fidelity")
        if estimate.conservative_effect_risk > context.maximum_effect_risk:
            violations.append("effect_risk")
        return tuple(violations)

    def _cost(self, estimate: CandidateEstimate, context: RoutingContext) -> float:
        switch_penalty = (
            self.switch_hysteresis_ms
            if estimate.action.executor is not context.current_executor
            else 0.0
        )
        return (
            self.cost_weights.latency * (estimate.inference_ms + estimate.controller_ms)
            + self.cost_weights.score(estimate.handoff)
            + switch_penalty
        )

    def _utility(self, estimate: CandidateEstimate, context: RoutingContext) -> float:
        return self.success_weight * estimate.conservative_success - self._cost(estimate, context)

    def select(
        self,
        estimates: Iterable[CandidateEstimate],
        context: RoutingContext,
    ) -> RoutingDecision:
        candidates = [estimate for estimate in estimates if self._legal(estimate, context)]
        feasible = [
            estimate for estimate in candidates if not self._violations(estimate, context)
        ]
        if feasible:
            if self.selection_mode == "constrained_cost":
                selected = min(
                    feasible,
                    key=lambda item: (
                        self._cost(item, context),
                        -item.conservative_success,
                        -item.conservative_fidelity,
                        item.action.executor.value,
                        item.action.transfer_mode.value,
                        item.action.commit_mode.value,
                    ),
                )
                objective = self._cost(selected, context)
                objective_name = "measured_cost"
                reason = "minimum measured cost among calibrated constraint-satisfying actions"
            else:
                selected = max(
                    feasible,
                    key=lambda item: (
                        self._utility(item, context),
                        item.conservative_success,
                        item.conservative_fidelity,
                        item.action.executor.value,
                        item.action.transfer_mode.value,
                        item.action.commit_mode.value,
                    ),
                )
                objective = self._utility(selected, context)
                objective_name = "scalarized_utility"
                reason = "maximum conservative utility among constraint-satisfying actions"
            switching = selected.action.executor is not context.current_executor
            return RoutingDecision(
                selected=selected,
                feasible_count=len(feasible),
                objective=objective,
                objective_name=objective_name,
                reason=reason,
                dwell_decision="switch" if switching else "continue",
            )
        if not candidates:
            raise ValueError("no legal routing candidate")

        # Conservative abstention: rank by constraint violations first, then
        # prefer staying/full replay and stronger lower bounds. This decision is
        # explicit in artifacts and cannot be silently counted as calibrated.
        selected = min(
            candidates,
            key=lambda item: (
                len(self._violations(item, context)),
                0 if item.action.executor is context.current_executor else 1,
                0 if item.action.transfer_mode is TransferMode.FULL_REPLAY else 1,
                -item.conservative_success,
                -item.conservative_fidelity,
                item.conservative_effect_risk,
                item.total_latency_ms,
            ),
        )
        violations = self._violations(selected, context)
        return RoutingDecision(
            selected=selected,
            feasible_count=0,
            objective=self._cost(selected, context),
            objective_name="abstention_fallback_cost",
            reason="risk-bound abstention: conservative legal fallback",
            abstained=True,
            dwell_decision=(
                "forced_stay"
                if selected.action.executor is context.current_executor
                else "fallback_switch"
            ),
            constraint_violations=violations,
        )

    def select_without_continuation_tax(
        self,
        estimates: Iterable[CandidateEstimate],
        context: RoutingContext,
    ) -> RoutingDecision:
        zeroed = [
            replace(estimate, handoff=estimate.handoff.zero_latency_tax())
            for estimate in estimates
        ]
        return self.select(zeroed, context)

    def select_without_switch_tax(
        self,
        estimates: Iterable[CandidateEstimate],
        context: RoutingContext,
    ) -> RoutingDecision:
        """Backward-compatible alias."""

        return self.select_without_continuation_tax(estimates, context)


def routing_reversal(
    policy: ConstrainedUtilityPolicy,
    estimates: Iterable[CandidateEstimate],
    context: RoutingContext,
) -> tuple[bool, RoutingDecision, RoutingDecision]:
    estimates = tuple(estimates)
    measured = policy.select(estimates, context)
    zero_tax = policy.select_without_continuation_tax(estimates, context)
    return measured.selected.action != zero_tax.selected.action, measured, zero_tax

