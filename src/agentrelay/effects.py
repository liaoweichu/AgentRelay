"""Effect-frontier tracking and migration legality for handoff boundaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from .schema import (
    CommitMode,
    EffectClass,
    EffectFrontierSnapshot,
    EffectRecord,
    EffectStatus,
    sha256_json,
)


@dataclass(frozen=True)
class EffectDecision:
    allowed: bool
    effect_key: str
    reason: str
    existing: EffectRecord | None = None


@dataclass(frozen=True)
class MigrationLegality:
    allowed: bool
    required_commit_mode: CommitMode
    blocking_effect_keys: tuple[str, ...]
    reasons: tuple[str, ...]


def canonical_effect_key(
    task_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    environment_version: str,
) -> str:
    return sha256_json(
        {
            "task_id": task_id,
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "environment_version": environment_version,
        }
    )


def assess_migration(records: Iterable[EffectRecord]) -> MigrationLegality:
    """Derive the conservative migration guard from observable effect progress."""

    records = tuple(records)
    unresolved = sorted(
        record.effect_key
        for record in records
        if record.status in {EffectStatus.SENT, EffectStatus.INDETERMINATE}
    )
    if unresolved:
        return MigrationLegality(
            False,
            CommitMode.RECONCILE,
            tuple(unresolved),
            ("effect response is not durably resolved",),
        )
    barrier = sorted(
        record.effect_key
        for record in records
        if record.status in {
            EffectStatus.INTENT,
            EffectStatus.PREPARED,
            EffectStatus.ACKNOWLEDGED,
        }
        and record.effect_class in {EffectClass.IRREVERSIBLE, EffectClass.UNKNOWN}
    )
    if barrier:
        return MigrationLegality(
            True,
            CommitMode.BARRIER,
            tuple(barrier),
            ("irreversible or unknown effect requires a commit barrier",),
        )
    compensating = sorted(
        record.effect_key
        for record in records
        if record.status in {EffectStatus.PREPARED, EffectStatus.ACKNOWLEDGED}
        and record.effect_class is EffectClass.REVERSIBLE
    )
    if compensating:
        return MigrationLegality(
            True,
            CommitMode.COMPENSATING,
            tuple(compensating),
            ("reversible effect requires compensation metadata",),
        )
    return MigrationLegality(True, CommitMode.IMMEDIATE, (), ())


def build_effect_frontier(records: Iterable[EffectRecord]) -> EffectFrontierSnapshot:
    legality = assess_migration(records)
    return EffectFrontierSnapshot(
        migration_allowed=legality.allowed,
        required_commit_mode=legality.required_commit_mode,
        blocking_effect_keys=legality.blocking_effect_keys,
    ).seal()


class EffectLedger:
    """Durable effect state machine.

    The ledger does not promise universal exactly-once behavior. It prevents an
    automatic retry while an official sandbox effect is unresolved and exposes
    that progress to the routing layer.
    """

    def __init__(self, records: Iterable[EffectRecord] = ()) -> None:
        self._records = {record.effect_key: record for record in records}

    def authorize(
        self,
        *,
        task_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        environment_version: str,
        effect_class: EffectClass,
        scope_key: str = "",
    ) -> EffectDecision:
        key = canonical_effect_key(task_id, tool_name, arguments, environment_version)
        existing = self._records.get(key)
        if existing is not None:
            reason_by_status = {
                EffectStatus.INTENT: "already_intended",
                EffectStatus.PREPARED: "already_prepared",
                EffectStatus.SENT: "sent_requires_reconciliation",
                EffectStatus.ACKNOWLEDGED: "already_acknowledged",
                EffectStatus.INDETERMINATE: "indeterminate_requires_reconciliation",
                EffectStatus.COMMITTED: "already_committed",
                EffectStatus.COMPENSATED: "already_compensated",
            }
            return EffectDecision(False, key, reason_by_status[existing.status], existing)
        if scope_key:
            for record in self._records.values():
                if (
                    record.scope_key == scope_key
                    and record.effect_key != key
                    and record.status
                    not in {EffectStatus.COMPENSATED}
                ):
                    return EffectDecision(
                        False,
                        key,
                        f"conflicting_{record.status.value}_effect",
                        record,
                    )
        if effect_class is EffectClass.UNKNOWN:
            return EffectDecision(False, key, "unknown_effect_requires_barrier")
        return EffectDecision(True, key, "authorized")

    def prepare(
        self,
        *,
        task_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        environment_version: str,
        effect_class: EffectClass,
        scope_key: str = "",
        compensation: Mapping[str, Any] | None = None,
        recovery_ref: str = "",
    ) -> EffectRecord:
        decision = self.authorize(
            task_id=task_id,
            tool_name=tool_name,
            arguments=arguments,
            environment_version=environment_version,
            effect_class=effect_class,
            scope_key=scope_key,
        )
        if not decision.allowed:
            raise ValueError(decision.reason)
        if effect_class is EffectClass.REVERSIBLE and compensation is None:
            raise ValueError("reversible effect requires compensation")
        record = EffectRecord(
            effect_key=decision.effect_key,
            tool_name=tool_name,
            canonical_arguments=dict(arguments),
            environment_version=environment_version,
            effect_class=effect_class,
            status=EffectStatus.PREPARED,
            scope_key=scope_key,
            compensation=dict(compensation) if compensation is not None else None,
            recovery_ref=recovery_ref,
        )
        self._records[record.effect_key] = record
        return record

    def mark_sent(self, effect_key: str) -> EffectRecord:
        record = self._records[effect_key]
        if record.status is not EffectStatus.PREPARED:
            raise ValueError("only_prepared_effect_can_be_sent")
        sent = replace(record, status=EffectStatus.SENT)
        self._records[effect_key] = sent
        return sent

    def acknowledge(
        self,
        effect_key: str,
        *,
        result_hash: str,
        result_lineage_hash: str = "",
    ) -> EffectRecord:
        record = self._records[effect_key]
        if record.status not in {EffectStatus.SENT, EffectStatus.PREPARED}:
            raise ValueError("effect_is_not_acknowledgeable")
        if not result_hash:
            raise ValueError("acknowledgement requires a result hash")
        acknowledged = replace(
            record,
            status=EffectStatus.ACKNOWLEDGED,
            result_hash=result_hash,
            result_lineage_hash=result_lineage_hash,
        )
        self._records[effect_key] = acknowledged
        return acknowledged

    def commit(self, effect_key: str, *, result_hash: str = "") -> EffectRecord:
        record = self._records[effect_key]
        if record.status is EffectStatus.COMMITTED:
            raise ValueError("already_committed")
        if record.status is EffectStatus.COMPENSATED:
            raise ValueError("compensated_effect_cannot_be_committed")
        # PREPARED/INDETERMINATE support preserves the v1 API. New formal code
        # uses SENT -> ACKNOWLEDGED -> COMMITTED or reconciliation.
        if record.status not in {
            EffectStatus.PREPARED,
            EffectStatus.ACKNOWLEDGED,
            EffectStatus.INDETERMINATE,
        }:
            raise ValueError("effect_is_not_committable")
        committed = replace(
            record,
            status=EffectStatus.COMMITTED,
            result_hash=result_hash or record.result_hash,
        )
        self._records[effect_key] = committed
        return committed

    def mark_indeterminate(self, effect_key: str) -> EffectRecord:
        record = self._records[effect_key]
        if record.status not in {EffectStatus.PREPARED, EffectStatus.SENT}:
            raise ValueError("only_prepared_or_sent_effect_can_become_indeterminate")
        indeterminate = replace(record, status=EffectStatus.INDETERMINATE)
        self._records[effect_key] = indeterminate
        return indeterminate

    def reconcile(
        self,
        effect_key: str,
        *,
        occurred: bool,
        result_hash: str = "",
        result_lineage_hash: str = "",
    ) -> EffectRecord:
        record = self._records[effect_key]
        if record.status not in {EffectStatus.SENT, EffectStatus.INDETERMINATE}:
            raise ValueError("only_unresolved_effect_can_be_reconciled")
        if occurred:
            if not result_hash:
                raise ValueError("occurred reconciliation requires result_hash")
            reconciled = replace(
                record,
                status=EffectStatus.ACKNOWLEDGED,
                result_hash=result_hash,
                result_lineage_hash=result_lineage_hash,
            )
        else:
            reconciled = replace(
                record,
                status=EffectStatus.PREPARED,
                attempt=record.attempt + 1,
                result_hash="",
                result_lineage_hash="",
            )
        self._records[effect_key] = reconciled
        return reconciled

    def retry_decision(self, effect_key: str) -> EffectDecision:
        record = self._records.get(effect_key)
        if record is None:
            return EffectDecision(False, effect_key, "unknown_effect_key")
        reason_by_status = {
            EffectStatus.INTENT: "prepare_before_send",
            EffectStatus.PREPARED: "prepared_effect_not_retryable",
            EffectStatus.SENT: "reconcile_before_retry",
            EffectStatus.ACKNOWLEDGED: "commit_acknowledged_result",
            EffectStatus.INDETERMINATE: "reconcile_before_retry",
            EffectStatus.COMMITTED: "return_committed_result",
            EffectStatus.COMPENSATED: "already_compensated",
        }
        return EffectDecision(False, effect_key, reason_by_status[record.status], record)

    def compensate(self, effect_key: str) -> EffectRecord:
        record = self._records[effect_key]
        if record.effect_class is not EffectClass.REVERSIBLE:
            raise ValueError("only reversible effects can be compensated")
        if record.compensation is None:
            raise ValueError("compensation descriptor is missing")
        if record.status is not EffectStatus.COMMITTED:
            raise ValueError("only_committed_effect_can_be_compensated")
        compensated = replace(record, status=EffectStatus.COMPENSATED)
        self._records[effect_key] = compensated
        return compensated

    def migration_legality(self) -> MigrationLegality:
        return assess_migration(self.snapshot())

    def frontier(self) -> EffectFrontierSnapshot:
        return build_effect_frontier(self.snapshot())

    def snapshot(self) -> tuple[EffectRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

