"""Handoff orchestration for obligation-closed semantic continuations."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

from .continuation import ContinuationCut, build_obligation_closed_packet
from .policy import CandidateEstimate, ConstrainedUtilityPolicy, RoutingContext, RoutingDecision
from .schema import RelayStatePacket, TransferMode
from .state import (
    PatchBundle,
    PatchRequest,
    StateDelta,
    TraceStore,
    apply_delta,
    apply_patch_bundle,
    build_patch_bundle,
    compute_delta,
)
from .validation import PacketValidator, ValidationReport


@dataclass(frozen=True)
class HandoffTransaction:
    decision: RoutingDecision
    transfer_mode: TransferMode
    reconstructed_packet: RelayStatePacket
    delta: StateDelta | None
    continuation_cut: ContinuationCut | None
    first_validation: ValidationReport
    final_validation: ValidationReport
    patch_request: PatchRequest | None
    patch_bundle: PatchBundle | None
    transmitted_bytes: int
    encode_ms: float
    verify_ms: float
    patch_ms: float
    closure_node_count: int = 0
    materialized_node_count: int = 0
    lazy_node_count: int = 0


class HandoffValidationError(RuntimeError):
    """Raised when a receiver cannot establish a valid continuation state."""

    def __init__(self, transaction: HandoffTransaction) -> None:
        self.transaction = transaction
        codes = ",".join(issue.code for issue in transaction.final_validation.issues)
        super().__init__(f"handoff aborted after validation failure: {codes}")


class HandoffCoordinator:
    def __init__(
        self,
        *,
        policy: ConstrainedUtilityPolicy,
        validator: PacketValidator,
        source_trace_store: TraceStore | None = None,
        target_trace_store: TraceStore | None = None,
    ) -> None:
        self.policy = policy
        self.validator = validator
        self.source_trace_store = source_trace_store or TraceStore()
        self.target_trace_store = target_trace_store or TraceStore()

    def _transfer_trace_refs(self, packet: RelayStatePacket) -> int:
        transmitted = 0
        for span_id in packet.trace_refs:
            if self.source_trace_store.contains(span_id):
                content = self.source_trace_store.get(span_id)
                self.target_trace_store.put_verified(span_id, content)
                transmitted += len(content.encode("utf-8"))
        return transmitted

    def execute(
        self,
        *,
        acknowledged: RelayStatePacket,
        source_current: RelayStatePacket,
        estimates: Iterable[CandidateEstimate],
        context: RoutingContext,
        lazy_node_ids: Iterable[str] = (),
        decision_override: RoutingDecision | None = None,
    ) -> HandoffTransaction:
        decision = decision_override or self.policy.select(tuple(estimates), context)
        mode = decision.selected.action.transfer_mode
        encode_start = time.perf_counter()
        delta: StateDelta | None = None
        continuation_cut: ContinuationCut | None = None
        transmitted_bytes = 0

        if mode is TransferMode.REUSE:
            reconstructed = source_current
        elif mode is TransferMode.FULL_REPLAY:
            reconstructed = source_current
            transmitted_bytes = len(source_current.to_json().encode("utf-8"))
            transmitted_bytes += self._transfer_trace_refs(source_current)
        elif source_current.semantic_nodes:
            requested_lazy = tuple(lazy_node_ids) if mode is TransferMode.CLOSED_DELTA_PATCHABLE else ()
            continuation_cut = build_obligation_closed_packet(
                source_current,
                lazy_node_ids=requested_lazy,
                source_executor=context.current_executor,
                target_executor=decision.selected.action.executor,
                acknowledged_version=acknowledged.packet_hash,
            )
            reconstructed = continuation_cut.packet
            transmitted_bytes = continuation_cut.encoded_bytes
            transmitted_bytes += self._transfer_trace_refs(reconstructed)
        else:
            # Legacy v1 packets retain the original structural-delta path.
            delta = compute_delta(acknowledged, source_current)
            reconstructed = apply_delta(acknowledged, delta)
            transmitted_bytes = delta.encoded_bytes
        encode_ms = (time.perf_counter() - encode_start) * 1000.0

        verify_start = time.perf_counter()
        previous = acknowledged if reconstructed.packet_hash != acknowledged.packet_hash else None
        validation_trace_store = (
            self.source_trace_store if mode is TransferMode.REUSE else self.target_trace_store
        )
        first = self.validator.validate(
            reconstructed,
            previous=previous,
            trace_store=validation_trace_store,
        )
        verify_ms = (time.perf_counter() - verify_start) * 1000.0
        patch_request: PatchRequest | None = None
        patch_bundle: PatchBundle | None = None
        patch_ms = 0.0
        final = first

        if not first.valid and mode is TransferMode.CLOSED_DELTA_PATCHABLE:
            patch_start = time.perf_counter()
            patch_request = first.patch_request()
            patch_bundle = build_patch_bundle(
                source_current,
                patch_request,
                self.source_trace_store,
            )
            reconstructed = apply_patch_bundle(
                reconstructed,
                patch_bundle,
                self.target_trace_store,
            )
            transmitted_bytes += patch_bundle.encoded_bytes
            final = self.validator.validate(
                reconstructed,
                previous=acknowledged,
                trace_store=self.target_trace_store,
            )
            patch_ms = (time.perf_counter() - patch_start) * 1000.0

        transaction = HandoffTransaction(
            decision=decision,
            transfer_mode=mode,
            reconstructed_packet=reconstructed,
            delta=delta,
            continuation_cut=continuation_cut,
            first_validation=first,
            final_validation=final,
            patch_request=patch_request,
            patch_bundle=patch_bundle,
            transmitted_bytes=transmitted_bytes,
            encode_ms=encode_ms,
            verify_ms=verify_ms,
            patch_ms=patch_ms,
            closure_node_count=(
                len(continuation_cut.closure_node_ids) if continuation_cut else 0
            ),
            materialized_node_count=(
                len(continuation_cut.materialized_node_ids) if continuation_cut else 0
            ),
            lazy_node_count=(len(continuation_cut.lazy_node_ids) if continuation_cut else 0),
        )
        if not transaction.final_validation.valid:
            raise HandoffValidationError(transaction)
        return transaction
