"""Obligation-closed semantic continuation construction and rendering."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from .schema import (
    DependencyEdge,
    Executor,
    RelayStatePacket,
    SemanticNode,
    SemanticNodeType,
    canonical_json,
)


class DependencyGraphError(ValueError):
    """Raised when a source graph cannot define a valid continuation."""


@dataclass(frozen=True)
class ContinuationCut:
    packet: RelayStatePacket
    closure_node_ids: tuple[str, ...]
    materialized_node_ids: tuple[str, ...]
    lazy_node_ids: tuple[str, ...]
    encoded_bytes: int


def node_index(nodes: Iterable[SemanticNode]) -> dict[str, SemanticNode]:
    result: dict[str, SemanticNode] = {}
    for node in nodes:
        if node.node_id in result:
            raise DependencyGraphError(f"duplicate semantic node {node.node_id}")
        result[node.node_id] = node
    return result


def predecessor_index(edges: Iterable[DependencyEdge]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for edge in edges:
        result.setdefault(edge.successor_id, set()).add(edge.predecessor_id)
    return result


def successor_index(edges: Iterable[DependencyEdge]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for edge in edges:
        result.setdefault(edge.predecessor_id, set()).add(edge.successor_id)
    return result


def obligation_closure(
    nodes: Iterable[SemanticNode],
    edges: Iterable[DependencyEdge],
    obligation_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return a deterministic transitive predecessor closure.

    Every node and predecessor must exist in the source graph. A cycle is
    rejected because it makes targeted predecessor repair ambiguous.
    """

    indexed = node_index(nodes)
    predecessors = predecessor_index(edges)
    obligations = tuple(dict.fromkeys(str(item) for item in obligation_ids))
    if not obligations:
        raise DependencyGraphError("at least one next-step obligation is required")
    for node_id in obligations:
        if node_id not in indexed:
            raise DependencyGraphError(f"obligation node is absent: {node_id}")

    visited: set[str] = set()
    active: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in active:
            raise DependencyGraphError(f"dependency cycle includes {node_id}")
        if node_id in visited:
            return
        if node_id not in indexed:
            raise DependencyGraphError(f"dependency predecessor is absent: {node_id}")
        active.add(node_id)
        for predecessor_id in sorted(predecessors.get(node_id, ())):
            visit(predecessor_id)
        active.remove(node_id)
        visited.add(node_id)

    for obligation_id in obligations:
        visit(obligation_id)
    return tuple(sorted(visited))


def missing_predecessor_ids(packet: RelayStatePacket) -> tuple[str, ...]:
    """Find predecessor IDs missing from the materialized packet.

    Only edges that can reach an obligation are considered. This prevents
    unrelated graph fragments from creating false patch requests.
    """

    present = {node.node_id for node in packet.semantic_nodes}
    predecessors = predecessor_index(packet.dependency_edges)
    missing: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        for predecessor_id in predecessors.get(node_id, ()):
            if predecessor_id not in present:
                missing.add(predecessor_id)
            else:
                visit(predecessor_id)

    for obligation_id in packet.obligation_ids:
        if obligation_id in present:
            visit(obligation_id)
    return tuple(sorted(missing))


def build_obligation_closed_packet(
    source: RelayStatePacket,
    *,
    obligation_ids: Iterable[str] | None = None,
    lazy_node_ids: Iterable[str] = (),
    source_executor: Executor | None = None,
    target_executor: Executor | None = None,
    acknowledged_version: str | None = None,
) -> ContinuationCut:
    """Build an obligation closure, optionally lazily materializing named nodes.

    Lazy nodes remain named by dependency edges and are therefore detectable by
    the target. Obligations themselves can never be lazy.
    """

    obligations = tuple(obligation_ids or source.obligation_ids)
    closure = obligation_closure(source.semantic_nodes, source.dependency_edges, obligations)
    closure_set = set(closure)
    lazy = set(str(item) for item in lazy_node_ids)
    if not lazy <= closure_set:
        unknown = sorted(lazy - closure_set)
        raise DependencyGraphError(f"lazy nodes are outside the obligation closure: {unknown}")
    if lazy & set(obligations):
        raise DependencyGraphError("an obligation node cannot be lazily materialized")

    indexed = node_index(source.semantic_nodes)
    materialized = closure_set - lazy
    selected_nodes = tuple(indexed[node_id] for node_id in sorted(materialized))
    selected_edges = tuple(
        edge
        for edge in source.dependency_edges
        if edge.successor_id in closure_set and edge.predecessor_id in closure_set
    )
    selected_trace_refs = tuple(
        sorted(
            {
                node.trace_ref
                for node in selected_nodes
                if node.trace_ref
            }
            | {
                evidence.source_span_id
                for evidence in source.evidence
                if evidence.source_span_id
                and any(node.trace_ref == evidence.source_span_id for node in selected_nodes)
            }
        )
    )
    packet = replace(
        source,
        semantic_nodes=selected_nodes,
        dependency_edges=selected_edges,
        obligation_ids=obligations,
        patchable_predecessor_ids=tuple(sorted(lazy)),
        trace_refs=selected_trace_refs,
        source_executor=source_executor if source_executor is not None else source.source_executor,
        target_executor=target_executor if target_executor is not None else source.target_executor,
        acknowledged_version=(
            acknowledged_version
            if acknowledged_version is not None
            else source.acknowledged_version
        ),
        parent_packet_hash=(
            acknowledged_version
            if acknowledged_version is not None
            else source.parent_packet_hash
        ),
        packet_hash="",
    ).seal()
    return ContinuationCut(
        packet=packet,
        closure_node_ids=closure,
        materialized_node_ids=tuple(sorted(materialized)),
        lazy_node_ids=tuple(sorted(lazy)),
        encoded_bytes=len(packet.to_json().encode("utf-8")),
    )


def closure_depth(
    nodes: Iterable[SemanticNode],
    edges: Iterable[DependencyEdge],
    obligation_ids: Iterable[str],
) -> int:
    indexed = node_index(nodes)
    predecessors = predecessor_index(edges)
    memo: dict[str, int] = {}
    active: set[str] = set()

    def depth(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in active:
            raise DependencyGraphError("dependency cycle detected")
        if node_id not in indexed:
            raise DependencyGraphError(f"dependency node is absent: {node_id}")
        active.add(node_id)
        value = 1 + max((depth(item) for item in predecessors.get(node_id, ())), default=0)
        active.remove(node_id)
        memo[node_id] = value
        return value

    obligations = tuple(obligation_ids)
    return max((depth(node_id) for node_id in obligations), default=0)


def build_standard_semantic_graph(
    *,
    goal: Any,
    observation: Any,
    obligation: Any,
    world_version: str,
    goal_trace_ref: str = "",
    observation_trace_ref: str = "",
    goal_provenance_hash: str = "",
    observation_provenance_hash: str = "",
    extra_nodes: Iterable[SemanticNode] = (),
) -> tuple[tuple[SemanticNode, ...], tuple[DependencyEdge, ...], tuple[str, ...]]:
    """Create the common goal/observation -> next-obligation subgraph."""

    goal_node = SemanticNode.create(
        SemanticNodeType.GOAL_CONSTRAINT,
        goal,
        world_version=world_version,
        trace_ref=goal_trace_ref,
        provenance_hash=goal_provenance_hash,
    )
    observation_node = SemanticNode.create(
        SemanticNodeType.WORLD_STATE,
        observation,
        world_version=world_version,
        trace_ref=observation_trace_ref,
        provenance_hash=observation_provenance_hash,
    )
    obligation_node = SemanticNode.create(
        SemanticNodeType.PLAN_OBLIGATION,
        obligation,
        world_version=world_version,
    )
    extras = tuple(extra_nodes)
    nodes = (goal_node, observation_node, *extras, obligation_node)
    edges = [
        DependencyEdge(goal_node.node_id, obligation_node.node_id, "goal_requires"),
        DependencyEdge(observation_node.node_id, obligation_node.node_id, "world_requires"),
    ]
    for node in extras:
        edges.append(DependencyEdge(node.node_id, obligation_node.node_id, "context_requires"))
    return tuple(nodes), tuple(edges), (obligation_node.node_id,)


def render_semantic_continuation(packet: RelayStatePacket, *, style: str = "json") -> str:
    """Render exactly the materialized continuation nodes for model input."""

    if style not in {"json", "concise_text"}:
        raise ValueError(f"unsupported continuation rendering: {style}")
    nodes = sorted(packet.semantic_nodes, key=lambda node: (node.node_type.value, node.node_id))
    if style == "json":
        return canonical_json(
            {
                "obligations": list(packet.obligation_ids),
                "nodes": [
                    {
                        "id": node.node_id,
                        "type": node.node_type.value,
                        "value": node.value,
                        "world_version": node.world_version,
                        "provenance_hash": node.provenance_hash,
                    }
                    for node in nodes
                ],
                "dependencies": [
                    {
                        "predecessor": edge.predecessor_id,
                        "successor": edge.successor_id,
                        "relation": edge.relation,
                    }
                    for edge in packet.dependency_edges
                ],
                "effect_frontier": (
                    packet.effect_frontier.payload_dict() if packet.effect_frontier else None
                ),
            }
        )
    lines = ["Next-step obligations:"]
    lines.extend(f"- {item}" for item in packet.obligation_ids)
    lines.append("Materialized continuation state:")
    for node in nodes:
        lines.append(f"- [{node.node_type.value}] {canonical_json(node.value)}")
    if packet.patchable_predecessor_ids:
        lines.append(
            "Lazy predecessors pending validation: "
            + ", ".join(packet.patchable_predecessor_ids)
        )
    return "\n".join(lines)
