"""Packet integrity, dependency closure, provenance, and transition validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .continuation import DependencyGraphError, closure_depth, missing_predecessor_ids
from .effects import build_effect_frontier, canonical_effect_key
from .schema import (
    EffectClass,
    EffectStatus,
    RelayStatePacket,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    sha256_text,
)
from .state import PatchRequest, Path, TraceStore, get_path


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: Path
    message: str
    severity: str = "error"
    trace_span_id: str = ""
    node_id: str = ""


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def patch_request(self) -> PatchRequest:
        paths: list[Path] = []
        spans: list[str] = []
        nodes: list[str] = []
        reasons: list[str] = []
        for issue in self.issues:
            if issue.severity != "error":
                continue
            if issue.path and issue.code in {"missing_required", "empty_required"}:
                paths.append(issue.path)
            if issue.trace_span_id:
                spans.append(issue.trace_span_id)
            if issue.node_id and issue.code == "missing_predecessor":
                nodes.append(issue.node_id)
            reasons.append(f"{issue.code}:{issue.message}")
        return PatchRequest(
            missing_paths=tuple(dict.fromkeys(paths)),
            trace_span_ids=tuple(dict.fromkeys(spans)),
            missing_node_ids=tuple(dict.fromkeys(nodes)),
            reasons=tuple(reasons),
        )


InvariantRule = Callable[[RelayStatePacket], ValidationIssue | None]


class PacketValidator:
    def __init__(
        self,
        required_paths: Iterable[Path] = (),
        rules: Iterable[InvariantRule] = (),
        *,
        require_v2_graph: bool = False,
    ) -> None:
        self.required_paths = tuple(required_paths)
        self.rules = tuple(rules)
        self.require_v2_graph = require_v2_graph

    def validate(
        self,
        packet: RelayStatePacket,
        *,
        previous: RelayStatePacket | None = None,
        trace_store: TraceStore | None = None,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if packet.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            issues.append(
                ValidationIssue(
                    "schema_version",
                    ("schema_version",),
                    f"unsupported schema version {packet.schema_version}",
                )
            )
        if self.require_v2_graph and packet.schema_version != SCHEMA_VERSION:
            issues.append(
                ValidationIssue(
                    "v2_graph_required",
                    ("schema_version",),
                    "this path requires a version 2 semantic continuation",
                )
            )
        if not packet.verify_hash():
            issues.append(ValidationIssue("packet_hash", ("packet_hash",), "packet hash mismatch"))

        for name, value in (
            ("dataset_id", packet.task.dataset_id),
            ("dataset_revision", packet.task.dataset_revision),
            ("split", packet.task.split),
            ("sample_id", packet.task.sample_id),
            ("goal_hash", packet.task.goal_hash),
            ("observation_version", packet.world.observation_version),
            ("environment_digest", packet.world.environment_digest),
        ):
            if not value:
                issues.append(ValidationIssue("empty_required", (name,), f"{name} is empty"))

        payload = packet.payload_dict()
        for path in self.required_paths:
            try:
                value = get_path(payload, path)
            except KeyError:
                issues.append(ValidationIssue("missing_required", path, "required state is missing"))
                continue
            if value is None or value == "":
                issues.append(ValidationIssue("empty_required", path, "required state is empty"))

        self._validate_evidence(packet, trace_store, issues)
        self._validate_graph(packet, trace_store, issues)
        self._validate_effects(packet, previous, issues)
        if previous is not None:
            self._validate_transition(previous, packet, issues)

        for rule in self.rules:
            issue = rule(packet)
            if issue is not None:
                issues.append(issue)
        return ValidationReport(tuple(issues))

    @staticmethod
    def _validate_evidence(
        packet: RelayStatePacket,
        trace_store: TraceStore | None,
        issues: list[ValidationIssue],
    ) -> None:
        fact_ids: set[str] = set()
        for index, evidence in enumerate(packet.evidence):
            if evidence.fact_id in fact_ids:
                issues.append(
                    ValidationIssue(
                        "duplicate_fact",
                        ("evidence", str(index), "fact_id"),
                        f"duplicate fact id {evidence.fact_id}",
                    )
                )
            fact_ids.add(evidence.fact_id)
            if trace_store is None:
                continue
            if not trace_store.contains(evidence.source_span_id):
                issues.append(
                    ValidationIssue(
                        "missing_trace",
                        ("evidence", str(index), "source_span_id"),
                        "provenance trace is unavailable",
                        trace_span_id=evidence.source_span_id,
                    )
                )
            else:
                content = trace_store.get(evidence.source_span_id)
                if sha256_text(content) != evidence.provenance_hash:
                    issues.append(
                        ValidationIssue(
                            "provenance_hash",
                            ("evidence", str(index), "provenance_hash"),
                            "evidence provenance hash mismatch",
                            trace_span_id=evidence.source_span_id,
                        )
                    )

    @staticmethod
    def _validate_graph(
        packet: RelayStatePacket,
        trace_store: TraceStore | None,
        issues: list[ValidationIssue],
    ) -> None:
        if packet.schema_version == "1.0" and not packet.semantic_nodes:
            return
        if not packet.semantic_nodes:
            issues.append(
                ValidationIssue(
                    "empty_semantic_graph",
                    ("semantic_nodes",),
                    "version 2 packet requires semantic nodes",
                )
            )
            return
        nodes: dict[str, object] = {}
        for index, node in enumerate(packet.semantic_nodes):
            if node.node_id in nodes:
                issues.append(
                    ValidationIssue(
                        "duplicate_node",
                        ("semantic_nodes", str(index), "node_id"),
                        f"duplicate semantic node {node.node_id}",
                        node_id=node.node_id,
                    )
                )
            nodes[node.node_id] = node
            if not node.verify_id():
                issues.append(
                    ValidationIssue(
                        "node_hash",
                        ("semantic_nodes", str(index), "node_id"),
                        "semantic node identifier does not match its content",
                        node_id=node.node_id,
                    )
                )
            if node.world_version and node.world_version != packet.world.observation_version:
                issues.append(
                    ValidationIssue(
                        "stale_node_world",
                        ("semantic_nodes", str(index), "world_version"),
                        "semantic node world version does not match the packet",
                        node_id=node.node_id,
                    )
                )
            if node.trace_ref and trace_store is not None:
                if not trace_store.contains(node.trace_ref):
                    issues.append(
                        ValidationIssue(
                            "missing_trace",
                            ("semantic_nodes", str(index), "trace_ref"),
                            "semantic-node trace is unavailable",
                            trace_span_id=node.trace_ref,
                            node_id=node.node_id,
                        )
                    )
                elif node.provenance_hash and sha256_text(trace_store.get(node.trace_ref)) != node.provenance_hash:
                    issues.append(
                        ValidationIssue(
                            "node_provenance_hash",
                            ("semantic_nodes", str(index), "provenance_hash"),
                            "semantic-node provenance hash mismatch",
                            trace_span_id=node.trace_ref,
                            node_id=node.node_id,
                        )
                    )

        present = set(nodes)
        patchable = set(packet.patchable_predecessor_ids)
        for obligation_id in packet.obligation_ids:
            if obligation_id not in present:
                issues.append(
                    ValidationIssue(
                        "missing_obligation",
                        ("obligation_ids",),
                        f"obligation node is not materialized: {obligation_id}",
                        node_id=obligation_id,
                    )
                )
        for node_id in missing_predecessor_ids(packet):
            if node_id in patchable:
                issues.append(
                    ValidationIssue(
                        "missing_predecessor",
                        ("semantic_nodes",),
                        f"dependency predecessor requires a named patch: {node_id}",
                        node_id=node_id,
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        "unpatchable_predecessor",
                        ("dependency_edges",),
                        f"dependency predecessor is absent and not patchable: {node_id}",
                        node_id=node_id,
                    )
                )
        try:
            # Detect cycles among materialized nodes. Missing lazy predecessors
            # are handled above and intentionally excluded from this check.
            materialized_edges = tuple(
                edge
                for edge in packet.dependency_edges
                if edge.predecessor_id in present and edge.successor_id in present
            )
            closure_depth(packet.semantic_nodes, materialized_edges, packet.obligation_ids)
        except DependencyGraphError as exc:
            issues.append(ValidationIssue("dependency_graph", ("dependency_edges",), str(exc)))

    @staticmethod
    def _validate_effects(
        packet: RelayStatePacket,
        previous: RelayStatePacket | None,
        issues: list[ValidationIssue],
    ) -> None:
        effect_keys: set[str] = set()
        for index, effect in enumerate(packet.effects):
            if effect.effect_key in effect_keys:
                issues.append(
                    ValidationIssue(
                        "duplicate_effect_record",
                        ("effects", str(index), "effect_key"),
                        f"duplicate effect key {effect.effect_key}",
                    )
                )
            effect_keys.add(effect.effect_key)
            expected_key = canonical_effect_key(
                packet.task.sample_id,
                effect.tool_name,
                effect.canonical_arguments,
                effect.environment_version,
            )
            if effect.effect_key != expected_key:
                issues.append(
                    ValidationIssue(
                        "effect_key",
                        ("effects", str(index), "effect_key"),
                        "effect key does not match its immutable intent",
                    )
                )
            if effect.attempt < 1:
                issues.append(
                    ValidationIssue(
                        "effect_attempt",
                        ("effects", str(index), "attempt"),
                        "effect attempt must be positive",
                    )
                )
            if effect.effect_class is EffectClass.REVERSIBLE and effect.compensation is None:
                issues.append(
                    ValidationIssue(
                        "missing_compensation",
                        ("effects", str(index), "compensation"),
                        "reversible effect requires a compensation descriptor",
                    )
                )
            if effect.status in {EffectStatus.ACKNOWLEDGED, EffectStatus.COMMITTED} and not effect.result_hash:
                # Legacy v1 direct commits did not always record a result hash.
                severity = "warning" if packet.schema_version == "1.0" else "error"
                issues.append(
                    ValidationIssue(
                        "missing_effect_result",
                        ("effects", str(index), "result_hash"),
                        "acknowledged/committed effect requires a result hash",
                        severity=severity,
                    )
                )
        if packet.schema_version == SCHEMA_VERSION:
            if packet.effect_frontier is None:
                issues.append(
                    ValidationIssue(
                        "missing_effect_frontier",
                        ("effect_frontier",),
                        "version 2 packet requires an effect frontier",
                    )
                )
            else:
                expected = build_effect_frontier(packet.effects)
                if not packet.effect_frontier.verify_hash() or packet.effect_frontier != expected:
                    issues.append(
                        ValidationIssue(
                            "effect_frontier",
                            ("effect_frontier",),
                            "effect frontier does not match effect records",
                        )
                    )

    @staticmethod
    def _validate_transition(
        previous: RelayStatePacket,
        packet: RelayStatePacket,
        issues: list[ValidationIssue],
    ) -> None:
        if packet.parent_packet_hash != previous.packet_hash:
            issues.append(
                ValidationIssue(
                    "parent_hash",
                    ("parent_packet_hash",),
                    "packet is not based on the acknowledged previous packet",
                )
            )
        if packet.task != previous.task:
            issues.append(
                ValidationIssue("task_identity", ("task",), "task identity changed across handoff")
            )
        current_by_key = {effect.effect_key: effect for effect in packet.effects}
        legal_transitions = {
            EffectStatus.INTENT: {EffectStatus.INTENT, EffectStatus.PREPARED},
            EffectStatus.PREPARED: {
                EffectStatus.PREPARED,
                EffectStatus.SENT,
                EffectStatus.INDETERMINATE,
                EffectStatus.COMMITTED,
            },
            EffectStatus.SENT: {
                EffectStatus.SENT,
                EffectStatus.ACKNOWLEDGED,
                EffectStatus.INDETERMINATE,
            },
            EffectStatus.ACKNOWLEDGED: {
                EffectStatus.ACKNOWLEDGED,
                EffectStatus.COMMITTED,
            },
            EffectStatus.INDETERMINATE: {
                EffectStatus.INDETERMINATE,
                EffectStatus.PREPARED,
                EffectStatus.ACKNOWLEDGED,
                EffectStatus.COMMITTED,
            },
            EffectStatus.COMMITTED: {
                EffectStatus.COMMITTED,
                EffectStatus.COMPENSATED,
            },
            EffectStatus.COMPENSATED: {EffectStatus.COMPENSATED},
        }
        for old in previous.effects:
            new = current_by_key.get(old.effect_key)
            if new is None:
                issues.append(
                    ValidationIssue(
                        "effect_record_lost",
                        ("effects",),
                        f"effect {old.effect_key} disappeared",
                    )
                )
                continue
            immutable_old = (
                old.tool_name,
                old.canonical_arguments,
                old.environment_version,
                old.effect_class,
                old.scope_key,
                old.compensation,
                old.recovery_ref,
            )
            immutable_new = (
                new.tool_name,
                new.canonical_arguments,
                new.environment_version,
                new.effect_class,
                new.scope_key,
                new.compensation,
                new.recovery_ref,
            )
            if immutable_old != immutable_new:
                issues.append(
                    ValidationIssue(
                        "effect_intent_changed",
                        ("effects", old.effect_key),
                        "effect intent changed while retaining the same key",
                    )
                )
            if new.status not in legal_transitions[old.status]:
                issues.append(
                    ValidationIssue(
                        "illegal_effect_transition",
                        ("effects", old.effect_key, "status"),
                        f"illegal transition {old.status.value} -> {new.status.value}",
                    )
                )
            if new.attempt < old.attempt:
                issues.append(
                    ValidationIssue(
                        "effect_attempt_regression",
                        ("effects", old.effect_key, "attempt"),
                        "effect attempt regressed",
                    )
                )
            if old.result_hash and new.result_hash != old.result_hash:
                issues.append(
                    ValidationIssue(
                        "effect_result_changed",
                        ("effects", old.effect_key, "result_hash"),
                        "recorded effect result changed",
                    )
                )

