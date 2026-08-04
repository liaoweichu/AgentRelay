"""State-transfer baselines used by forced-handoff fidelity experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable

from .continuation import build_obligation_closed_packet, render_semantic_continuation
from .schema import RelayStatePacket, canonical_json
from .state import TraceStore


class ContinuationCodecName(str, Enum):
    FULL_REPLAY = "full_replay"
    TRUNCATED_HISTORY = "truncated_history"
    NARRATIVE_SUMMARY = "narrative_summary"
    STRUCTURED_SNAPSHOT = "structured_snapshot"
    TYPED_DELTA_NO_EDGES = "typed_delta_no_edges"
    CLOSED_NO_PATCH = "closed_no_patch"
    CLOSED_PATCHABLE = "closed_patchable"


@dataclass(frozen=True)
class EncodedContinuation:
    codec: ContinuationCodecName
    payload: str
    encoded_bytes: int
    packet: RelayStatePacket | None
    retained_trace_ids: tuple[str, ...] = ()
    lazy_node_ids: tuple[str, ...] = ()
    deterministic: bool = True


SummaryFunction = Callable[[str, int], str]


def _trace_text(packet: RelayStatePacket, trace_store: TraceStore, trace_ids: Iterable[str]) -> str:
    values = []
    for trace_id in trace_ids:
        if trace_store.contains(trace_id):
            values.append(trace_store.get(trace_id))
    return "\n\n".join(values)


def encode_continuation(
    codec: ContinuationCodecName | str,
    packet: RelayStatePacket,
    trace_store: TraceStore,
    *,
    max_trace_spans: int = 5,
    summary_budget_chars: int = 2048,
    summary_function: SummaryFunction | None = None,
    lazy_node_ids: Iterable[str] = (),
) -> EncodedContinuation:
    name = ContinuationCodecName(codec)
    if max_trace_spans <= 0:
        raise ValueError("max_trace_spans must be positive")
    if summary_budget_chars <= 0:
        raise ValueError("summary_budget_chars must be positive")

    if name is ContinuationCodecName.FULL_REPLAY:
        trace_ids = tuple(packet.trace_refs)
        payload = canonical_json(
            {
                "packet": packet.to_dict(),
                "raw_trace_spans": {
                    trace_id: trace_store.get(trace_id)
                    for trace_id in trace_ids
                    if trace_store.contains(trace_id)
                },
            }
        )
        return EncodedContinuation(name, payload, len(payload.encode("utf-8")), packet, trace_ids)

    if name is ContinuationCodecName.TRUNCATED_HISTORY:
        trace_ids = tuple(packet.trace_refs[-max_trace_spans:])
        payload = _trace_text(packet, trace_store, trace_ids)
        return EncodedContinuation(name, payload, len(payload.encode("utf-8")), None, trace_ids)

    if name is ContinuationCodecName.NARRATIVE_SUMMARY:
        if summary_function is None:
            raise ValueError("narrative_summary requires a frozen native summary_function")
        source = _trace_text(packet, trace_store, packet.trace_refs)
        payload = summary_function(source, summary_budget_chars)
        if not payload.strip():
            raise ValueError("native summary function returned an empty payload")
        return EncodedContinuation(
            name,
            payload,
            len(payload.encode("utf-8")),
            None,
            tuple(packet.trace_refs),
            deterministic=False,
        )

    if name is ContinuationCodecName.STRUCTURED_SNAPSHOT:
        snapshot = packet.payload_dict()
        payload = canonical_json(
            {
                key: snapshot[key]
                for key in ("task", "invariants", "world", "evidence", "plan", "effects")
            }
        )
        return EncodedContinuation(name, payload, len(payload.encode("utf-8")), None)

    if name is ContinuationCodecName.TYPED_DELTA_NO_EDGES:
        edge_free = replace(
            packet,
            dependency_edges=(),
            patchable_predecessor_ids=(),
            packet_hash="",
        ).seal()
        payload = render_semantic_continuation(edge_free, style="json")
        return EncodedContinuation(name, payload, len(payload.encode("utf-8")), edge_free)

    if name is ContinuationCodecName.CLOSED_NO_PATCH:
        cut = build_obligation_closed_packet(packet)
        payload = render_semantic_continuation(cut.packet, style="json")
        return EncodedContinuation(
            name,
            payload,
            len(payload.encode("utf-8")),
            cut.packet,
            cut.packet.trace_refs,
        )

    cut = build_obligation_closed_packet(packet, lazy_node_ids=lazy_node_ids)
    payload = render_semantic_continuation(cut.packet, style="json")
    return EncodedContinuation(
        name,
        payload,
        len(payload.encode("utf-8")),
        cut.packet,
        cut.packet.trace_refs,
        cut.lazy_node_ids,
    )


def codec_manifest() -> tuple[dict[str, object], ...]:
    return (
        {"name": "full_replay", "uses_native_summary": False, "dependency_edges": True},
        {"name": "truncated_history", "uses_native_summary": False, "dependency_edges": False},
        {"name": "narrative_summary", "uses_native_summary": True, "dependency_edges": False},
        {"name": "structured_snapshot", "uses_native_summary": False, "dependency_edges": False},
        {"name": "typed_delta_no_edges", "uses_native_summary": False, "dependency_edges": False},
        {"name": "closed_no_patch", "uses_native_summary": False, "dependency_edges": True},
        {"name": "closed_patchable", "uses_native_summary": False, "dependency_edges": True},
    )
