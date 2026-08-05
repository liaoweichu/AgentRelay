"""Native baseline/method episode runner over official benchmark adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
import time
from typing import Any, Callable, Mapping, Protocol

from .baselines import BaselineController, BaselineInput
from .benchmark import BenchmarkObservation, PublicBenchmarkAdapter
from .calibration import ConformalRiskCalibrator
from .continuation import closure_depth, render_semantic_continuation
from .cost import HandoffMeasurement
from .effects import EffectLedger, build_effect_frontier
from .inference import NativeGenerationResult
from .learning import FEATURE_NAMES, GOAL_HASH_DIMENSIONS, JointRouterEstimator
from .policy import CandidateAction, CandidateEstimate, RoutingContext
from .runtime import HandoffCoordinator, HandoffTransaction
from .schema import (
    CommitMode,
    EffectClass,
    Executor,
    RelayStatePacket,
    SemanticNodeType,
    TransferMode,
    sha256_text,
)
from .validation import PacketValidator


class NativeExecutor(Protocol):
    def generate(self, messages: Any) -> NativeGenerationResult:
        ...


@dataclass(frozen=True)
class ServiceProfile:
    inference_ms: float
    predicted_success: float
    predicted_fidelity: float
    controller_ms: float = 0.1
    rehydration_ms: float = 0.0


@dataclass(frozen=True)
class EpisodeStepRecord:
    step_index: int
    selected_executor: str
    transfer_mode: str
    commit_mode: str
    routing_reason: str
    abstained: bool
    calibration_source: str
    router_features: Mapping[str, float]
    predicted_success: float
    predicted_reward: float
    predicted_fidelity: float
    predicted_effect_risk: float
    action_hash: str
    action_text: str
    prompt_hash: str
    response_hash: str
    response_text: str
    generation_attempts: int
    action_recovery: str
    inference_ms: float
    controller_ms: float
    handoff_bytes: int
    handoff_encode_ms: float
    handoff_verify_ms: float
    handoff_patch_ms: float
    handoff_communication_ms: float
    handoff_rehydration_ms: float
    handoff_effect_wait_ms: float
    handoff_reconciliation_ms: float
    target_tokens: int
    fidelity_risk: float
    effect_risk: float
    closure_nodes: int
    lazy_nodes: int
    patch_nodes: int
    effect_frontier_hash: str
    migration_allowed: bool
    reward: float
    done: bool


@dataclass(frozen=True)
class EpisodeResult:
    benchmark: str
    dataset_revision: str
    split: str
    sample_id: str
    method: str
    success: float
    reward: float
    official_metrics: Mapping[str, float]
    steps: tuple[EpisodeStepRecord, ...]
    end_to_end_ms: float
    paper_evidence: bool
    labels_accessed_by_router: bool = False
    effect_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EstimateProvider(Protocol):
    def candidates(
        self,
        packet: RelayStatePacket,
        context: RoutingContext,
        *,
        step_index: int,
        remaining_steps: int,
    ) -> tuple[CandidateEstimate, ...]:
        ...


def _feature_mapping(
    packet: RelayStatePacket,
    context: RoutingContext,
    *,
    step_index: int,
    remaining_steps: int,
    edge_ms: float,
    cloud_ms: float,
    bandwidth_mbps: float,
) -> dict[str, float]:
    depth = closure_depth(packet.semantic_nodes, packet.dependency_edges, packet.obligation_ids)
    frontier = packet.effect_frontier or build_effect_frontier(packet.effects)
    goal = ""
    for node in packet.semantic_nodes:
        if node.node_type is SemanticNodeType.GOAL_CONSTRAINT:
            if isinstance(node.value, Mapping):
                goal = str(node.value.get("instruction", node.value))
            else:
                goal = str(node.value)
            break
    goal_tokens = re.findall(r"[a-z0-9]+", goal.lower())
    ngrams = list(goal_tokens)
    ngrams.extend(
        f"{left}_{right}" for left, right in zip(goal_tokens, goal_tokens[1:])
    )
    hashed = [0.0] * GOAL_HASH_DIMENSIONS
    for token in ngrams:
        digest = sha256_text(token)
        index = int(digest[:8], 16) % GOAL_HASH_DIMENSIONS
        sign = 1.0 if int(digest[8:10], 16) % 2 else -1.0
        hashed[index] += sign
    scale = max(1.0, float(len(ngrams)) ** 0.5)
    base = {
        "step_index": float(step_index),
        "remaining_steps": float(remaining_steps),
        "input_tokens": float(max(1, len(packet.to_json()) // 4)),
        "delta_bytes": float(len(packet.to_json().encode("utf-8"))),
        "local_confidence": 0.5,
        "invariant_count": float(len(packet.invariants.hard_constraints)),
        "unresolved_obligation_count": float(len(packet.obligation_ids)),
        "previous_patch_rate": 0.0,
        "edge_warm_inference_ms": float(edge_ms),
        "cloud_warm_inference_ms": float(cloud_ms),
        "measured_bandwidth_mbps": float(bandwidth_mbps),
        "previous_handoff_failed": 0.0,
        "effect_read_only": float(context.effect_class is EffectClass.READ_ONLY),
        "effect_reversible": float(context.effect_class is EffectClass.REVERSIBLE),
        "effect_irreversible_or_unknown": float(
            context.effect_class in {EffectClass.IRREVERSIBLE, EffectClass.UNKNOWN}
        ),
        "closure_node_count": float(len(packet.semantic_nodes)),
        "dependency_depth": float(depth),
        "estimated_patch_probability": 0.0,
        "effect_frontier_blocked": float(not frontier.migration_allowed),
        "consecutive_steps": float(context.consecutive_steps),
        "dwell_remaining": float(
            max(0, context.minimum_dwell_steps - context.consecutive_steps)
        ),
        "goal_char_count": float(len(goal)),
        "goal_token_count": float(len(goal_tokens)),
        "goal_numeric_count": float(sum(token.isdigit() for token in goal_tokens)),
        "goal_constraint_count": float(
            len(
                re.findall(
                    r"\b(?:and|under|below|less|maximum|max|size|color|pack|inch|ounce)\b",
                    goal.lower(),
                )
            )
        ),
        "visible_action_count": float(len(packet.world.resources.get("valid_actions", ()))),
    }
    base.update(
        {f"goal_hash_{index:02d}": value / scale for index, value in enumerate(hashed)}
    )
    return {name: float(base[name]) for name in FEATURE_NAMES}


class ProfileEstimateProvider:
    """Measured-profile provider for fixed baselines and pre-router collection."""

    def __init__(
        self,
        profiles: Mapping[Executor, ServiceProfile],
        *,
        bandwidth_mbps: float = 100.0,
        fidelity_by_mode: Mapping[TransferMode, float] | None = None,
    ) -> None:
        self.profiles = dict(profiles)
        self.bandwidth_mbps = float(bandwidth_mbps)
        self.fidelity_by_mode = dict(
            fidelity_by_mode
            or {
                TransferMode.REUSE: 1.0,
                TransferMode.CLOSED_DELTA: 0.90,
                TransferMode.CLOSED_DELTA_PATCHABLE: 0.97,
                TransferMode.FULL_REPLAY: 0.995,
            }
        )

    def candidates(
        self,
        packet: RelayStatePacket,
        context: RoutingContext,
        *,
        step_index: int,
        remaining_steps: int,
    ) -> tuple[CandidateEstimate, ...]:
        packet_bytes = len(packet.to_json().encode("utf-8"))
        features = _feature_mapping(
            packet,
            context,
            step_index=step_index,
            remaining_steps=remaining_steps,
            edge_ms=self.profiles[Executor.EDGE].inference_ms,
            cloud_ms=self.profiles[Executor.CLOUD].inference_ms,
            bandwidth_mbps=self.bandwidth_mbps,
        )
        candidates = []
        for executor, profile in self.profiles.items():
            switching = executor is not context.current_executor
            modes = (
                (TransferMode.CLOSED_DELTA, TransferMode.CLOSED_DELTA_PATCHABLE, TransferMode.FULL_REPLAY)
                if switching
                else (TransferMode.REUSE,)
            )
            for mode in modes:
                if mode is TransferMode.REUSE:
                    payload_bytes = 0
                elif mode is TransferMode.CLOSED_DELTA:
                    payload_bytes = max(1, int(packet_bytes * 0.45))
                elif mode is TransferMode.CLOSED_DELTA_PATCHABLE:
                    payload_bytes = max(1, int(packet_bytes * 0.55))
                else:
                    payload_bytes = packet_bytes
                communication_ms = (
                    payload_bytes * 8 / (self.bandwidth_mbps * 1_000_000) * 1000
                    if payload_bytes
                    else 0.0
                )
                frontier = context.effect_frontier
                if switching and frontier is not None and not frontier.migration_allowed:
                    commit_mode = CommitMode.RECONCILE
                elif context.effect_class in {EffectClass.IRREVERSIBLE, EffectClass.UNKNOWN}:
                    commit_mode = CommitMode.BARRIER
                elif (
                    switching
                    and frontier is not None
                    and frontier.required_commit_mode is CommitMode.BARRIER
                ):
                    commit_mode = CommitMode.BARRIER
                elif context.effect_class is EffectClass.REVERSIBLE or (
                    switching
                    and frontier is not None
                    and frontier.required_commit_mode is CommitMode.COMPENSATING
                ):
                    commit_mode = CommitMode.COMPENSATING
                else:
                    commit_mode = CommitMode.IMMEDIATE
                fidelity = self.fidelity_by_mode[mode]
                candidates.append(
                    CandidateEstimate(
                        action=CandidateAction(executor, mode, commit_mode),
                        predicted_success=profile.predicted_success,
                        predicted_fidelity=fidelity,
                        inference_ms=profile.inference_ms,
                        controller_ms=profile.controller_ms,
                        handoff=HandoffMeasurement(
                            encode_ms=0.2 if switching else 0.0,
                            communication_ms=communication_ms,
                            rehydration_ms=profile.rehydration_ms if switching else 0.0,
                            verification_ms=0.2 if switching else 0.0,
                            patch_ms=0.1 if mode is TransferMode.CLOSED_DELTA_PATCHABLE else 0.0,
                            payload_bytes=payload_bytes,
                            target_tokens=max(0, payload_bytes // 4),
                            fidelity_risk=1.0 - fidelity,
                            effect_risk=(
                                1.0 if frontier is not None and not frontier.migration_allowed else 0.0
                            ),
                        ),
                        features=features,
                    )
                )
        return tuple(candidates)


class LearnedEstimateProvider:
    def __init__(
        self,
        estimator: JointRouterEstimator,
        *,
        edge_warm_ms: float,
        cloud_warm_ms: float,
        bandwidth_mbps: float,
        calibrator: ConformalRiskCalibrator | None = None,
    ) -> None:
        self.estimator = estimator
        self.edge_warm_ms = float(edge_warm_ms)
        self.cloud_warm_ms = float(cloud_warm_ms)
        self.bandwidth_mbps = float(bandwidth_mbps)
        self.calibrator = calibrator

    def candidates(
        self,
        packet: RelayStatePacket,
        context: RoutingContext,
        *,
        step_index: int,
        remaining_steps: int,
    ) -> tuple[CandidateEstimate, ...]:
        features = _feature_mapping(
            packet,
            context,
            step_index=step_index,
            remaining_steps=remaining_steps,
            edge_ms=self.edge_warm_ms,
            cloud_ms=self.cloud_warm_ms,
            bandwidth_mbps=self.bandwidth_mbps,
        )
        estimates = self.estimator.candidates(features)
        # Safety invariant: an irreversible/unknown effect must never commit
        # through a non-barrier route, regardless of what the learned router
        # predicts for the commit mode.  This mirrors ProfileEstimateProvider
        # and is enforced ahead of the final execution-time barrier check.
        if context.effect_class in {EffectClass.IRREVERSIBLE, EffectClass.UNKNOWN}:
            estimates = tuple(
                replace(estimate, action=replace(estimate.action, commit_mode=CommitMode.BARRIER))
                for estimate in estimates
            )
        return self.calibrator.calibrate_all(estimates) if self.calibrator else estimates


class EpisodeRunner:
    def __init__(
        self,
        *,
        adapter: PublicBenchmarkAdapter,
        executors: Mapping[Executor, NativeExecutor],
        controller: BaselineController,
        estimate_provider: EstimateProvider,
        max_steps: int,
        minimum_dwell_steps: int = 2,
        max_action_retries: int = 1,
        paper_evidence: bool = False,
        lazy_node_selector: Callable[[RelayStatePacket], tuple[str, ...]] | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_action_retries < 0:
            raise ValueError("max_action_retries cannot be negative")
        self.adapter = adapter
        self.executors = dict(executors)
        self.controller = controller
        self.estimate_provider = estimate_provider
        self.max_steps = max_steps
        self.minimum_dwell_steps = minimum_dwell_steps
        self.max_action_retries = max_action_retries
        self.paper_evidence = paper_evidence
        self.lazy_node_selector = lazy_node_selector or (lambda packet: ())

    def _generate(
        self,
        executor: Executor,
        observation: BenchmarkObservation,
        packet: RelayStatePacket,
    ) -> tuple[str, NativeGenerationResult, int, str]:
        continuation = render_semantic_continuation(packet, style="concise_text")
        messages = list(self.adapter.format_model_messages(observation, continuation))
        generations: list[NativeGenerationResult] = []
        rejected: list[str] = []
        for attempt in range(self.max_action_retries + 1):
            generation = self.executors[executor].generate(messages)
            generations.append(generation)
            action = self.adapter.parse_model_output(generation.text)
            validation = self.adapter.validate_model_action(action, observation)
            if validation.accepted:
                combined = replace(
                    generation,
                    prompt_hash=sha256_text("|".join(item.prompt_hash for item in generations)),
                    prompt_tokens=sum(item.prompt_tokens for item in generations),
                    output_tokens=sum(item.output_tokens for item in generations),
                    latency_ms=sum(item.latency_ms for item in generations),
                    peak_cuda_memory_bytes=max(
                        item.peak_cuda_memory_bytes for item in generations
                    ),
                )
                return (
                    validation.action,
                    combined,
                    len(generations),
                    "none" if attempt == 0 else "model_retry",
                )
            rejected.append(validation.action)
            if attempt < self.max_action_retries:
                messages.extend(
                    (
                        {"role": "assistant", "content": generation.text},
                        {
                            "role": "user",
                            "content": validation.feedback + "\nCorrected next action:",
                        },
                    )
                )

        fallback = self.adapter.fallback_model_action(observation, tuple(rejected))
        if fallback:
            validation = self.adapter.validate_model_action(fallback, observation)
            if validation.accepted:
                generation = generations[-1]
                combined = replace(
                    generation,
                    prompt_hash=sha256_text("|".join(item.prompt_hash for item in generations)),
                    prompt_tokens=sum(item.prompt_tokens for item in generations),
                    output_tokens=sum(item.output_tokens for item in generations),
                    latency_ms=sum(item.latency_ms for item in generations),
                    peak_cuda_memory_bytes=max(
                        item.peak_cuda_memory_bytes for item in generations
                    ),
                )
                return validation.action, combined, len(generations), "deterministic_fallback"
        raise ValueError(
            f"model failed to produce a valid {self.adapter.descriptor.dataset_id} action "
            f"after {len(generations)} attempt(s)"
        )

    def run(self) -> EpisodeResult:
        started = time.perf_counter()
        observation = self.adapter.reset()
        packet = self.adapter.build_packet(None)
        trace_store = getattr(self.adapter, "trace_store", None)
        if trace_store is None:
            raise RuntimeError("official adapter must expose its content-addressed trace store")
        validator = PacketValidator(require_v2_graph=True)
        initial = validator.validate(packet, trace_store=trace_store)
        if not initial.valid:
            raise RuntimeError(
                "initial official packet failed validation: "
                + ",".join(issue.code for issue in initial.issues)
            )
        current_executor = Executor.EDGE
        consecutive_steps = 0
        acknowledged = packet
        records: list[EpisodeStepRecord] = []

        coordinator = HandoffCoordinator(
            policy=self.controller.policy,
            validator=validator,
            source_trace_store=trace_store,
        )

        for step_index in range(self.max_steps):
            effect_class = self.adapter.pending_effect_class(observation)
            frontier = packet.effect_frontier or build_effect_frontier(packet.effects)
            context = RoutingContext(
                current_executor=current_executor,
                effect_class=effect_class,
                minimum_success=0.0,
                minimum_fidelity=0.0,
                maximum_effect_risk=1.0,
                consecutive_steps=consecutive_steps,
                minimum_dwell_steps=self.minimum_dwell_steps,
                effect_frontier=frontier,
            )
            estimates = self.estimate_provider.candidates(
                packet,
                context,
                step_index=step_index,
                remaining_steps=self.max_steps - step_index,
            )
            edge_action = cloud_action = ""
            proposals: dict[
                Executor, tuple[str, NativeGenerationResult, int, str]
            ] = {}
            if self.controller.requires_both_model_proposals:
                for executor in (Executor.EDGE, Executor.CLOUD):
                    proposals[executor] = self._generate(executor, observation, packet)
                edge_action = proposals[Executor.EDGE][0]
                cloud_action = proposals[Executor.CLOUD][0]

            edge_confidence = max(
                (
                    estimate.predicted_success
                    for estimate in estimates
                    if estimate.action.executor is Executor.EDGE
                ),
                default=0.0,
            )
            route_started = time.perf_counter()
            decision = self.controller.select(
                BaselineInput(
                    estimates=estimates,
                    context=context,
                    sample_id=self.adapter.descriptor.sample_id,
                    step_index=step_index,
                    edge_confidence=edge_confidence,
                    task_confidence=edge_confidence,
                    edge_proposal=edge_action,
                    cloud_proposal=cloud_action,
                )
            )
            controller_ms = (time.perf_counter() - route_started) * 1000.0
            selected_executor = decision.selected.action.executor
            transaction: HandoffTransaction | None = None
            working_packet = packet
            if selected_executor is not current_executor:
                transaction = coordinator.execute(
                    acknowledged=acknowledged,
                    source_current=packet,
                    estimates=(decision.selected,),
                    context=context,
                    lazy_node_ids=self.lazy_node_selector(packet),
                    decision_override=decision,
                )
                working_packet = transaction.reconstructed_packet

            action, generation, generation_attempts, action_recovery = (
                proposals.get(selected_executor)
                or self._generate(selected_executor, observation, working_packet)
            )
            effect_metadata = self.adapter.effect_metadata(action)
            actual_effect_class = EffectClass(
                effect_metadata.get("effect_class", EffectClass.UNKNOWN.value)
            )
            if (
                actual_effect_class in {EffectClass.IRREVERSIBLE, EffectClass.UNKNOWN}
                and decision.selected.action.commit_mode is not CommitMode.BARRIER
            ):
                raise RuntimeError(
                    "unsafe action blocked before execution: irreversible/unknown "
                    "effect requires a barrier-selected route "
                    f"[benchmark={self.adapter.descriptor.dataset_id} "
                    f"method={self.controller.name.value} action={action!r} "
                    f"valid_actions={getattr(observation, 'valid_actions', ())} "
                    f"pending={effect_class.value} actual={actual_effect_class.value} "
                    f"commit={decision.selected.action.commit_mode.value}]"
                )
            if (
                actual_effect_class is EffectClass.REVERSIBLE
                and decision.selected.action.commit_mode
                not in {CommitMode.BARRIER, CommitMode.COMPENSATING}
            ):
                raise RuntimeError(
                    "unsafe action blocked before execution: reversible effect "
                    "requires a barrier or compensating route"
                )
            ledger: EffectLedger = getattr(self.adapter, "effect_ledger")
            prepared_key = ""
            if actual_effect_class is not EffectClass.READ_ONLY:
                prepared = ledger.prepare(
                    task_id=self.adapter.descriptor.sample_id,
                    tool_name=str(effect_metadata.get("tool_name", "environment.action")),
                    arguments=dict(effect_metadata.get("arguments", {"action_hash": sha256_text(action)})),
                    environment_version=observation.observation_version,
                    effect_class=(
                        EffectClass.IRREVERSIBLE
                        if actual_effect_class is EffectClass.UNKNOWN
                        else actual_effect_class
                    ),
                    scope_key=str(effect_metadata.get("scope_key", "")),
                    compensation=effect_metadata.get("compensation"),
                    recovery_ref=str(effect_metadata.get("recovery_ref", "")),
                )
                prepared_key = prepared.effect_key
                ledger.mark_sent(prepared_key)
            try:
                step_result = self.adapter.step(action)
            except Exception:
                if prepared_key:
                    ledger.mark_indeterminate(prepared_key)
                raise
            if prepared_key:
                result_hash = sha256_text(step_result.observation.text)
                ledger.acknowledge(prepared_key, result_hash=result_hash)
                ledger.commit(prepared_key, result_hash=result_hash)

            next_packet = self.adapter.build_packet(working_packet)
            next_validation = validator.validate(
                next_packet,
                previous=working_packet,
                trace_store=trace_store,
            )
            if not next_validation.valid:
                raise RuntimeError(
                    "post-step packet failed validation: "
                    + ",".join(issue.code for issue in next_validation.issues)
                )
            next_frontier = next_packet.effect_frontier or build_effect_frontier(next_packet.effects)
            patch_nodes = (
                len(transaction.patch_bundle.semantic_nodes)
                if transaction is not None and transaction.patch_bundle is not None
                else 0
            )
            records.append(
                EpisodeStepRecord(
                    step_index=step_index,
                    selected_executor=selected_executor.value,
                    transfer_mode=decision.selected.action.transfer_mode.value,
                    commit_mode=decision.selected.action.commit_mode.value,
                    routing_reason=decision.reason,
                    abstained=decision.abstained,
                    calibration_source=decision.selected.calibration_source,
                    router_features=dict(decision.selected.features or {}),
                    predicted_success=decision.selected.predicted_success,
                    predicted_reward=decision.selected.quality_score,
                    predicted_fidelity=decision.selected.predicted_fidelity,
                    predicted_effect_risk=decision.selected.handoff.effect_risk,
                    action_hash=sha256_text(action),
                    action_text=action,
                    prompt_hash=generation.prompt_hash,
                    response_hash=generation.response_hash,
                    response_text=generation.text,
                    generation_attempts=generation_attempts,
                    action_recovery=action_recovery,
                    inference_ms=generation.latency_ms,
                    controller_ms=controller_ms,
                    handoff_bytes=transaction.transmitted_bytes if transaction else 0,
                    handoff_encode_ms=transaction.encode_ms if transaction else 0.0,
                    handoff_verify_ms=transaction.verify_ms if transaction else 0.0,
                    handoff_patch_ms=transaction.patch_ms if transaction else 0.0,
                    handoff_communication_ms=decision.selected.handoff.communication_ms,
                    handoff_rehydration_ms=decision.selected.handoff.rehydration_ms,
                    handoff_effect_wait_ms=decision.selected.handoff.effect_wait_ms,
                    handoff_reconciliation_ms=decision.selected.handoff.reconciliation_ms,
                    target_tokens=decision.selected.handoff.target_tokens,
                    fidelity_risk=decision.selected.handoff.fidelity_risk,
                    effect_risk=decision.selected.handoff.effect_risk,
                    closure_nodes=transaction.closure_node_count if transaction else len(packet.semantic_nodes),
                    lazy_nodes=transaction.lazy_node_count if transaction else 0,
                    patch_nodes=patch_nodes,
                    effect_frontier_hash=next_frontier.frontier_hash,
                    migration_allowed=next_frontier.migration_allowed,
                    reward=step_result.reward,
                    done=step_result.observation.done,
                )
            )
            consecutive_steps = (
                consecutive_steps + 1 if selected_executor is current_executor else 1
            )
            current_executor = selected_executor
            acknowledged = next_packet
            packet = next_packet
            observation = step_result.observation
            if observation.done:
                break

        evaluation = self.adapter.evaluate()
        return EpisodeResult(
            benchmark=self.adapter.descriptor.dataset_id,
            dataset_revision=self.adapter.descriptor.dataset_revision,
            split=self.adapter.descriptor.split,
            sample_id=self.adapter.descriptor.sample_id,
            method=self.controller.name.value,
            success=evaluation.success,
            reward=evaluation.reward,
            official_metrics=dict(evaluation.official_metrics),
            steps=tuple(records),
            end_to_end_ms=(time.perf_counter() - started) * 1000.0,
            paper_evidence=self.paper_evidence,
            labels_accessed_by_router=False,
        )
