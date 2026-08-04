"""Versioned, model-neutral semantic-continuation packet definitions.

The schema uses only the standard library so integrity and closure checks can
run before heavyweight inference dependencies are installed. Version 2.0 adds
typed dependency nodes, explicit obligations, and an effect-frontier snapshot.
Version 1.0 remains readable so previously certified diagnostic artifacts stay
independently auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


SCHEMA_VERSION = "2.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", SCHEMA_VERSION})


class Executor(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


class TransferMode(str, Enum):
    REUSE = "reuse"
    CLOSED_DELTA = "closed_delta"
    CLOSED_DELTA_PATCHABLE = "closed_delta_patchable"
    FULL_REPLAY = "full_replay"

    # Source-compatible aliases for the v1 implementation and old test code.
    TYPED_DELTA = "closed_delta"
    TYPED_DELTA_PATCHABLE = "closed_delta_patchable"


class CommitMode(str, Enum):
    IMMEDIATE = "immediate"
    BARRIER = "barrier"
    RECONCILE = "reconcile"
    COMPENSATING = "compensating"


class EffectClass(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class EffectStatus(str, Enum):
    INTENT = "intent"
    PREPARED = "prepared"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    INDETERMINATE = "indeterminate"
    COMMITTED = "committed"
    COMPENSATED = "compensated"


class SemanticNodeType(str, Enum):
    GOAL_CONSTRAINT = "goal_constraint"
    EVIDENCE = "evidence"
    WORLD_STATE = "world_state"
    PLAN_OBLIGATION = "plan_obligation"
    TRACE_SPAN = "trace_span"
    EFFECT_RECORD = "effect_record"


def canonical_json(value: Any) -> str:
    """Return a stable UTF-8 JSON representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class TaskIdentity:
    dataset_id: str
    dataset_revision: str
    split: str
    sample_id: str
    goal_hash: str
    success_criteria: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskIdentity":
        return cls(
            dataset_id=str(value["dataset_id"]),
            dataset_revision=str(value["dataset_revision"]),
            split=str(value["split"]),
            sample_id=str(value["sample_id"]),
            goal_hash=str(value["goal_hash"]),
            success_criteria=tuple(str(item) for item in value.get("success_criteria", ())),
        )


@dataclass(frozen=True)
class InvariantState:
    hard_constraints: Mapping[str, Any] = field(default_factory=dict)
    permissions: tuple[str, ...] = ()
    unresolved_obligations: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvariantState":
        return cls(
            hard_constraints=dict(value.get("hard_constraints", {})),
            permissions=tuple(str(item) for item in value.get("permissions", ())),
            unresolved_obligations=tuple(
                str(item) for item in value.get("unresolved_obligations", ())
            ),
        )


@dataclass(frozen=True)
class WorldState:
    observation_version: str
    environment_digest: str
    resources: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorldState":
        return cls(
            observation_version=str(value["observation_version"]),
            environment_digest=str(value["environment_digest"]),
            resources=dict(value.get("resources", {})),
        )


@dataclass(frozen=True)
class EvidenceItem:
    fact_id: str
    value: Any
    source_span_id: str
    provenance_hash: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceItem":
        return cls(
            fact_id=str(value["fact_id"]),
            value=value.get("value"),
            source_span_id=str(value["source_span_id"]),
            provenance_hash=str(value["provenance_hash"]),
        )


@dataclass(frozen=True)
class PlanState:
    active_subgoal: str = ""
    completed_subgoals: tuple[str, ...] = ()
    pending_actions: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanState":
        return cls(
            active_subgoal=str(value.get("active_subgoal", "")),
            completed_subgoals=tuple(str(item) for item in value.get("completed_subgoals", ())),
            pending_actions=tuple(str(item) for item in value.get("pending_actions", ())),
        )


@dataclass(frozen=True)
class EffectRecord:
    effect_key: str
    tool_name: str
    canonical_arguments: Mapping[str, Any]
    environment_version: str
    effect_class: EffectClass
    status: EffectStatus
    scope_key: str = ""
    compensation: Mapping[str, Any] | None = None
    attempt: int = 1
    result_hash: str = ""
    result_lineage_hash: str = ""
    recovery_ref: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EffectRecord":
        compensation = value.get("compensation")
        return cls(
            effect_key=str(value["effect_key"]),
            tool_name=str(value["tool_name"]),
            canonical_arguments=dict(value.get("canonical_arguments", {})),
            environment_version=str(value["environment_version"]),
            effect_class=EffectClass(value["effect_class"]),
            status=EffectStatus(value["status"]),
            scope_key=str(value.get("scope_key", "")),
            compensation=dict(compensation) if compensation is not None else None,
            attempt=int(value.get("attempt", 1)),
            result_hash=str(value.get("result_hash", "")),
            result_lineage_hash=str(value.get("result_lineage_hash", "")),
            recovery_ref=str(value.get("recovery_ref", "")),
        )


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    node_type: SemanticNodeType
    value: Any
    world_version: str = ""
    trace_ref: str = ""
    provenance_hash: str = ""

    def identity_payload(self) -> dict[str, Any]:
        return {
            "node_type": self.node_type.value,
            "value": _jsonable(self.value),
            "world_version": self.world_version,
            "trace_ref": self.trace_ref,
            "provenance_hash": self.provenance_hash,
        }

    def compute_id(self) -> str:
        return sha256_json(self.identity_payload())

    def verify_id(self) -> bool:
        return bool(self.node_id) and self.node_id == self.compute_id()

    @classmethod
    def create(
        cls,
        node_type: SemanticNodeType,
        value: Any,
        *,
        world_version: str = "",
        trace_ref: str = "",
        provenance_hash: str = "",
    ) -> "SemanticNode":
        node = cls(
            node_id="",
            node_type=node_type,
            value=value,
            world_version=world_version,
            trace_ref=trace_ref,
            provenance_hash=provenance_hash,
        )
        return replace(node, node_id=node.compute_id())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticNode":
        return cls(
            node_id=str(value["node_id"]),
            node_type=SemanticNodeType(value["node_type"]),
            value=value.get("value"),
            world_version=str(value.get("world_version", "")),
            trace_ref=str(value.get("trace_ref", "")),
            provenance_hash=str(value.get("provenance_hash", "")),
        )


@dataclass(frozen=True)
class DependencyEdge:
    predecessor_id: str
    successor_id: str
    relation: str = "requires"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DependencyEdge":
        return cls(
            predecessor_id=str(value["predecessor_id"]),
            successor_id=str(value["successor_id"]),
            relation=str(value.get("relation", "requires")),
        )


@dataclass(frozen=True)
class EffectFrontierSnapshot:
    migration_allowed: bool = True
    required_commit_mode: CommitMode = CommitMode.IMMEDIATE
    blocking_effect_keys: tuple[str, ...] = ()
    frontier_hash: str = ""

    def payload_dict(self) -> dict[str, Any]:
        return {
            "migration_allowed": self.migration_allowed,
            "required_commit_mode": self.required_commit_mode.value,
            "blocking_effect_keys": list(self.blocking_effect_keys),
        }

    def compute_hash(self) -> str:
        return sha256_json(self.payload_dict())

    def seal(self) -> "EffectFrontierSnapshot":
        return replace(self, frontier_hash=self.compute_hash())

    def verify_hash(self) -> bool:
        return bool(self.frontier_hash) and self.frontier_hash == self.compute_hash()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EffectFrontierSnapshot":
        return cls(
            migration_allowed=bool(value.get("migration_allowed", True)),
            required_commit_mode=CommitMode(value.get("required_commit_mode", "immediate")),
            blocking_effect_keys=tuple(
                str(item) for item in value.get("blocking_effect_keys", ())
            ),
            frontier_hash=str(value.get("frontier_hash", "")),
        )


@dataclass(frozen=True)
class RelayStatePacket:
    task: TaskIdentity
    invariants: InvariantState
    world: WorldState
    evidence: tuple[EvidenceItem, ...] = ()
    plan: PlanState = field(default_factory=PlanState)
    effects: tuple[EffectRecord, ...] = ()
    trace_refs: tuple[str, ...] = ()
    parent_packet_hash: str = ""
    source_executor: Executor | None = None
    target_executor: Executor | None = None
    acknowledged_version: str = ""
    obligation_ids: tuple[str, ...] = ()
    semantic_nodes: tuple[SemanticNode, ...] = ()
    dependency_edges: tuple[DependencyEdge, ...] = ()
    patchable_predecessor_ids: tuple[str, ...] = ()
    effect_frontier: EffectFrontierSnapshot | None = None
    schema_version: str = SCHEMA_VERSION
    packet_hash: str = ""

    def payload_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw.pop("packet_hash", None)
        return _jsonable(raw)

    def compute_hash(self) -> str:
        return sha256_json(self.payload_dict())

    def seal(self) -> "RelayStatePacket":
        return replace(self, packet_hash=self.compute_hash())

    def verify_hash(self) -> bool:
        return bool(self.packet_hash) and self.packet_hash == self.compute_hash()

    def to_dict(self) -> dict[str, Any]:
        value = self.payload_dict()
        value["packet_hash"] = self.packet_hash
        return value

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RelayStatePacket":
        source_executor = value.get("source_executor")
        target_executor = value.get("target_executor")
        effect_frontier = value.get("effect_frontier")
        return cls(
            schema_version=str(value.get("schema_version", "1.0")),
            task=TaskIdentity.from_dict(value["task"]),
            invariants=InvariantState.from_dict(value.get("invariants", {})),
            world=WorldState.from_dict(value["world"]),
            evidence=tuple(EvidenceItem.from_dict(item) for item in value.get("evidence", ())),
            plan=PlanState.from_dict(value.get("plan", {})),
            effects=tuple(EffectRecord.from_dict(item) for item in value.get("effects", ())),
            trace_refs=tuple(str(item) for item in value.get("trace_refs", ())),
            parent_packet_hash=str(value.get("parent_packet_hash", "")),
            source_executor=Executor(source_executor) if source_executor else None,
            target_executor=Executor(target_executor) if target_executor else None,
            acknowledged_version=str(value.get("acknowledged_version", "")),
            obligation_ids=tuple(str(item) for item in value.get("obligation_ids", ())),
            semantic_nodes=tuple(
                SemanticNode.from_dict(item) for item in value.get("semantic_nodes", ())
            ),
            dependency_edges=tuple(
                DependencyEdge.from_dict(item) for item in value.get("dependency_edges", ())
            ),
            patchable_predecessor_ids=tuple(
                str(item) for item in value.get("patchable_predecessor_ids", ())
            ),
            effect_frontier=(
                EffectFrontierSnapshot.from_dict(effect_frontier)
                if isinstance(effect_frontier, Mapping)
                else None
            ),
            packet_hash=str(value.get("packet_hash", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> "RelayStatePacket":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("packet JSON must contain an object")
        return cls.from_dict(value)


def goal_digest(goal: str) -> str:
    return sha256_text(goal.strip())

