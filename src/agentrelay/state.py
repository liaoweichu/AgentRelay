"""Content-addressed trace storage and continuation delta/patch operations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Mapping, MutableMapping

from .continuation import node_index, obligation_closure
from .schema import DependencyEdge, RelayStatePacket, SemanticNode, canonical_json, sha256_text


Path = tuple[str, ...]


@dataclass(frozen=True)
class PatchOp:
    op: str
    path: Path
    value: Any = None

    def __post_init__(self) -> None:
        if self.op not in {"set", "remove"}:
            raise ValueError(f"unsupported patch operation: {self.op}")
        if not self.path:
            raise ValueError("patch path cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        result = {"op": self.op, "path": list(self.path)}
        if self.op == "set":
            result["value"] = self.value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PatchOp":
        return cls(
            op=str(value["op"]),
            path=tuple(str(item) for item in value["path"]),
            value=value.get("value"),
        )


@dataclass(frozen=True)
class StateDelta:
    base_packet_hash: str
    target_packet_hash: str
    operations: tuple[PatchOp, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_packet_hash": self.base_packet_hash,
            "target_packet_hash": self.target_packet_hash,
            "operations": [operation.to_dict() for operation in self.operations],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def encoded_bytes(self) -> int:
        return len(self.to_json().encode("utf-8"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateDelta":
        return cls(
            base_packet_hash=str(value["base_packet_hash"]),
            target_packet_hash=str(value["target_packet_hash"]),
            operations=tuple(PatchOp.from_dict(item) for item in value.get("operations", ())),
        )


@dataclass(frozen=True)
class PatchRequest:
    missing_paths: tuple[Path, ...] = ()
    trace_span_ids: tuple[str, ...] = ()
    missing_node_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatchBundle:
    operations: tuple[PatchOp, ...] = ()
    trace_spans: Mapping[str, str] = field(default_factory=dict)
    semantic_nodes: tuple[SemanticNode, ...] = ()
    dependency_edges: tuple[DependencyEdge, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operations": [operation.to_dict() for operation in self.operations],
            "trace_spans": dict(self.trace_spans),
            "semantic_nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "value": node.value,
                    "world_version": node.world_version,
                    "trace_ref": node.trace_ref,
                    "provenance_hash": node.provenance_hash,
                }
                for node in self.semantic_nodes
            ],
            "dependency_edges": [
                {
                    "predecessor_id": edge.predecessor_id,
                    "successor_id": edge.successor_id,
                    "relation": edge.relation,
                }
                for edge in self.dependency_edges
            ],
        }

    @property
    def encoded_bytes(self) -> int:
        return len(canonical_json(self.to_dict()).encode("utf-8"))


class TraceStore:
    """In-memory content-addressed trace store used by the runtime core."""

    def __init__(self) -> None:
        self._spans: dict[str, str] = {}

    def add(self, content: str) -> str:
        span_id = sha256_text(content)
        self._spans[span_id] = content
        return span_id

    def put_verified(self, span_id: str, content: str) -> None:
        if sha256_text(content) != span_id:
            raise ValueError("trace span content does not match its identifier")
        self._spans[span_id] = content

    def get(self, span_id: str) -> str:
        return self._spans[span_id]

    def contains(self, span_id: str) -> bool:
        return span_id in self._spans

    def subset(self, span_ids: Iterable[str]) -> dict[str, str]:
        return {span_id: self.get(span_id) for span_id in span_ids}


def _diff(old: Any, new: Any, path: Path, operations: list[PatchOp]) -> None:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        old_keys = set(old)
        new_keys = set(new)
        for key in sorted(old_keys - new_keys, key=str):
            operations.append(PatchOp("remove", path + (str(key),)))
        for key in sorted(new_keys - old_keys, key=str):
            operations.append(PatchOp("set", path + (str(key),), deepcopy(new[key])))
        for key in sorted(old_keys & new_keys, key=str):
            _diff(old[key], new[key], path + (str(key),), operations)
        return
    if old != new:
        operations.append(PatchOp("set", path, deepcopy(new)))


def compute_delta(base: RelayStatePacket, target: RelayStatePacket) -> StateDelta:
    if not base.verify_hash() or not target.verify_hash():
        raise ValueError("both packets must be sealed before delta computation")
    operations: list[PatchOp] = []
    _diff(base.payload_dict(), target.payload_dict(), (), operations)
    return StateDelta(
        base_packet_hash=base.packet_hash,
        target_packet_hash=target.packet_hash,
        operations=tuple(operations),
    )


def _resolve_parent(root: MutableMapping[str, Any], path: Path, create: bool) -> tuple[Any, str]:
    current: Any = root
    for part in path[:-1]:
        if not isinstance(current, MutableMapping):
            raise ValueError(f"patch path crosses a non-object at {part}")
        if part not in current:
            if not create:
                raise KeyError("/".join(path))
            current[part] = {}
        current = current[part]
    return current, path[-1]


def apply_operations(payload: Mapping[str, Any], operations: Iterable[PatchOp]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    for operation in operations:
        parent, leaf = _resolve_parent(result, operation.path, create=operation.op == "set")
        if not isinstance(parent, MutableMapping):
            raise ValueError("patch parent must be a mapping")
        if operation.op == "set":
            parent[leaf] = deepcopy(operation.value)
        else:
            parent.pop(leaf, None)
    return result


def apply_delta(base: RelayStatePacket, delta: StateDelta) -> RelayStatePacket:
    if base.packet_hash != delta.base_packet_hash or not base.verify_hash():
        raise ValueError("delta base hash does not match the sealed packet")
    payload = apply_operations(base.payload_dict(), delta.operations)
    payload["packet_hash"] = delta.target_packet_hash
    result = RelayStatePacket.from_dict(payload)
    if not result.verify_hash():
        raise ValueError("applied delta does not reconstruct the target packet")
    return result


def get_path(value: Mapping[str, Any], path: Path) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError("/".join(path))
        current = current[part]
    return current


def _node_patch_closure(
    source_packet: RelayStatePacket,
    requested_node_ids: Iterable[str],
) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(requested_node_ids))
    if not requested:
        return ()
    return obligation_closure(
        source_packet.semantic_nodes,
        source_packet.dependency_edges,
        requested,
    )


def build_patch_bundle(
    source_packet: RelayStatePacket,
    request: PatchRequest,
    trace_store: TraceStore,
) -> PatchBundle:
    source = source_packet.payload_dict()
    operations = tuple(
        PatchOp("set", path, deepcopy(get_path(source, path))) for path in request.missing_paths
    )
    node_ids = _node_patch_closure(source_packet, request.missing_node_ids)
    indexed = node_index(source_packet.semantic_nodes)
    semantic_nodes = tuple(indexed[node_id] for node_id in node_ids)
    node_id_set = set(node_ids)
    dependency_edges = tuple(
        edge
        for edge in source_packet.dependency_edges
        if edge.predecessor_id in node_id_set or edge.successor_id in node_id_set
    )
    trace_ids = set(request.trace_span_ids)
    trace_ids.update(node.trace_ref for node in semantic_nodes if node.trace_ref)
    return PatchBundle(
        operations=operations,
        trace_spans=trace_store.subset(sorted(trace_ids)),
        semantic_nodes=semantic_nodes,
        dependency_edges=dependency_edges,
    )


def apply_patch_bundle(
    packet: RelayStatePacket,
    bundle: PatchBundle,
    trace_store: TraceStore,
) -> RelayStatePacket:
    for span_id, content in bundle.trace_spans.items():
        trace_store.put_verified(span_id, content)
    payload = apply_operations(packet.payload_dict(), bundle.operations)
    result = RelayStatePacket.from_dict(payload)
    existing_nodes = {node.node_id: node for node in result.semantic_nodes}
    for node in bundle.semantic_nodes:
        existing = existing_nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"semantic node collision for {node.node_id}")
        existing_nodes[node.node_id] = node
    edge_keys = {
        (edge.predecessor_id, edge.successor_id, edge.relation): edge
        for edge in result.dependency_edges
    }
    for edge in bundle.dependency_edges:
        edge_keys[(edge.predecessor_id, edge.successor_id, edge.relation)] = edge
    patched_ids = {node.node_id for node in bundle.semantic_nodes}
    result = RelayStatePacket(
        **{
            **result.__dict__,
            "semantic_nodes": tuple(existing_nodes[key] for key in sorted(existing_nodes)),
            "dependency_edges": tuple(edge_keys[key] for key in sorted(edge_keys)),
            "patchable_predecessor_ids": tuple(
                item for item in result.patchable_predecessor_ids if item not in patched_ids
            ),
            "packet_hash": "",
        }
    ).seal()
    return result


def json_size_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))

