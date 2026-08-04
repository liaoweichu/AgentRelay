"""Software-level correctness tests.

The records below are minimal program fixtures, not benchmark tasks or paper
measurements.  Formal evidence must come from the pinned public datasets named
in ``experiments/experiment-plan.md``.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from agentrelay.config import load_json_config, validate_experiment_config
from agentrelay.cost import CostWeights, HandoffMeasurement, RecordedBandwidthTrace
from agentrelay.effects import EffectLedger
from agentrelay.inference import NativeGenerationConfig
from agentrelay.local_smoke import (
    build_diagnostic_messages,
    build_diagnostic_packet,
    process_example_from_record,
)
from agentrelay.metrics import TrajectoryMetrics, aggregate_trajectories
from agentrelay.policy import (
    CandidateAction,
    CandidateEstimate,
    ConstrainedUtilityPolicy,
    RoutingContext,
    routing_reversal,
)
from agentrelay.provenance import (
    SplitPolicy,
    collect_package_versions,
    source_tree_revision,
)
from agentrelay.runtime import HandoffCoordinator, HandoffValidationError
from agentrelay.schema import (
    CommitMode,
    EffectClass,
    EffectStatus,
    EvidenceItem,
    Executor,
    InvariantState,
    PlanState,
    RelayStatePacket,
    TaskIdentity,
    TransferMode,
    WorldState,
    goal_digest,
)
from agentrelay.state import TraceStore, apply_delta, compute_delta
from agentrelay.statistics import holm_adjust, mcnemar_exact, paired_bootstrap_difference
from agentrelay.validation import PacketValidator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fixture_packet(
    *,
    observation_version: str,
    resources: dict[str, object],
    parent_packet_hash: str = "",
    evidence: tuple[EvidenceItem, ...] = (),
) -> RelayStatePacket:
    return RelayStatePacket(
        task=TaskIdentity(
            dataset_id="public/test-only-schema-fixture",
            dataset_revision="0123456789abcdef",
            split="fixture",
            sample_id="fixture-0",
            goal_hash=goal_digest("verify software semantics"),
            success_criteria=("validator accepts the reconstructed packet",),
        ),
        invariants=InvariantState(
            hard_constraints={"network_write": False},
            permissions=("read_fixture",),
        ),
        world=WorldState(
            observation_version=observation_version,
            environment_digest="fixture-environment-v1",
            resources=resources,
        ),
        evidence=evidence,
        plan=PlanState(active_subgoal="validate", pending_actions=("inspect",)),
        parent_packet_hash=parent_packet_hash,
        schema_version="1.0",
    ).seal()


def estimate(
    *,
    executor: Executor,
    mode: TransferMode,
    success: float,
    inference_ms: float,
    handoff_ms: float = 0.0,
) -> CandidateEstimate:
    return CandidateEstimate(
        action=CandidateAction(executor, mode, CommitMode.BARRIER),
        predicted_success=success,
        predicted_fidelity=1.0,
        inference_ms=inference_ms,
        handoff=HandoffMeasurement(communication_ms=handoff_ms),
    )


class PacketTests(unittest.TestCase):
    def test_packet_delta_round_trip_is_hash_exact(self) -> None:
        base = fixture_packet(observation_version="1", resources={"room": "hall"})
        target = fixture_packet(
            observation_version="2",
            resources={"room": "kitchen", "inventory": ["key"]},
            parent_packet_hash=base.packet_hash,
        )
        delta = compute_delta(base, target)
        reconstructed = apply_delta(base, delta)
        self.assertEqual(reconstructed, target)
        self.assertTrue(reconstructed.verify_hash())
        self.assertLess(delta.encoded_bytes, len(target.to_json().encode("utf-8")))

    def test_patchable_handoff_fetches_only_missing_trace(self) -> None:
        source_traces = TraceStore()
        span_id = source_traces.add("public-observation-span")
        evidence = (
            EvidenceItem(
                fact_id="fact-0",
                value="door is closed",
                source_span_id=span_id,
                provenance_hash=span_id,
            ),
        )
        base = fixture_packet(observation_version="1", resources={"door": "unknown"})
        target = fixture_packet(
            observation_version="2",
            resources={"door": "closed"},
            parent_packet_hash=base.packet_hash,
            evidence=evidence,
        )
        edge = estimate(
            executor=Executor.EDGE,
            mode=TransferMode.REUSE,
            success=0.10,
            inference_ms=10,
        )
        cloud = estimate(
            executor=Executor.CLOUD,
            mode=TransferMode.TYPED_DELTA_PATCHABLE,
            success=0.95,
            inference_ms=10,
        )
        coordinator = HandoffCoordinator(
            policy=ConstrainedUtilityPolicy(),
            validator=PacketValidator(),
            source_trace_store=source_traces,
            target_trace_store=TraceStore(),
        )
        transaction = coordinator.execute(
            acknowledged=base,
            source_current=target,
            estimates=(edge, cloud),
            context=RoutingContext(
                current_executor=Executor.EDGE,
                minimum_success=0.5,
            ),
        )
        self.assertFalse(transaction.first_validation.valid)
        self.assertTrue(transaction.final_validation.valid)
        self.assertIsNotNone(transaction.patch_request)
        self.assertEqual(transaction.patch_request.trace_span_ids, (span_id,))
        self.assertEqual(transaction.reconstructed_packet.packet_hash, target.packet_hash)

    def test_invalid_nonpatchable_handoff_aborts(self) -> None:
        source_traces = TraceStore()
        span_id = source_traces.add("required-public-trace")
        evidence = (
            EvidenceItem(
                fact_id="fact-0",
                value="switch is on",
                source_span_id=span_id,
                provenance_hash=span_id,
            ),
        )
        base = fixture_packet(observation_version="1", resources={"switch": "unknown"})
        target = fixture_packet(
            observation_version="2",
            resources={"switch": "on"},
            parent_packet_hash=base.packet_hash,
            evidence=evidence,
        )
        coordinator = HandoffCoordinator(
            policy=ConstrainedUtilityPolicy(),
            validator=PacketValidator(),
            source_trace_store=source_traces,
            target_trace_store=TraceStore(),
        )
        edge = estimate(
            executor=Executor.EDGE,
            mode=TransferMode.REUSE,
            success=0.1,
            inference_ms=10,
        )
        cloud = estimate(
            executor=Executor.CLOUD,
            mode=TransferMode.TYPED_DELTA,
            success=0.9,
            inference_ms=10,
        )
        with self.assertRaises(HandoffValidationError) as raised:
            coordinator.execute(
                acknowledged=base,
                source_current=target,
                estimates=(edge, cloud),
                context=RoutingContext(
                    current_executor=Executor.EDGE,
                    minimum_success=0.5,
                ),
            )
        codes = {issue.code for issue in raised.exception.transaction.final_validation.issues}
        self.assertIn("missing_trace", codes)

    def test_tampering_is_detected(self) -> None:
        packet = fixture_packet(observation_version="1", resources={"door": "closed"})
        tampered = replace(packet, world=replace(packet.world, observation_version="2"))
        report = PacketValidator().validate(tampered)
        self.assertFalse(report.valid)
        self.assertIn("packet_hash", {issue.code for issue in report.issues})


class EffectLedgerTests(unittest.TestCase):
    def test_committed_effect_is_exactly_once(self) -> None:
        ledger = EffectLedger()
        record = ledger.prepare(
            task_id="fixture-0",
            tool_name="write_record",
            arguments={"record_id": "r0", "value": 1},
            environment_version="v1",
            effect_class=EffectClass.IRREVERSIBLE,
            scope_key="record:r0",
        )
        committed = ledger.commit(record.effect_key)
        self.assertIs(committed.status, EffectStatus.COMMITTED)
        retry = ledger.authorize(
            task_id="fixture-0",
            tool_name="write_record",
            arguments={"value": 1, "record_id": "r0"},
            environment_version="v1",
            effect_class=EffectClass.IRREVERSIBLE,
            scope_key="record:r0",
        )
        self.assertFalse(retry.allowed)
        self.assertEqual(retry.reason, "already_committed")

    def test_conflicting_scope_is_rejected(self) -> None:
        ledger = EffectLedger()
        record = ledger.prepare(
            task_id="fixture-0",
            tool_name="write_record",
            arguments={"record_id": "r0", "value": 1},
            environment_version="v1",
            effect_class=EffectClass.IRREVERSIBLE,
            scope_key="record:r0",
        )
        ledger.commit(record.effect_key)
        conflict = ledger.authorize(
            task_id="fixture-0",
            tool_name="write_record",
            arguments={"record_id": "r0", "value": 2},
            environment_version="v1",
            effect_class=EffectClass.IRREVERSIBLE,
            scope_key="record:r0",
        )
        self.assertFalse(conflict.allowed)
        self.assertEqual(conflict.reason, "conflicting_committed_effect")

    def test_direct_duplicate_commit_and_precommit_compensation_are_rejected(self) -> None:
        ledger = EffectLedger()
        record = ledger.prepare(
            task_id="fixture-0",
            tool_name="write_record",
            arguments={"record_id": "r0", "value": 1},
            environment_version="v1",
            effect_class=EffectClass.REVERSIBLE,
            scope_key="record:r0",
            compensation={"tool": "restore_record", "arguments": {"record_id": "r0"}},
        )
        with self.assertRaisesRegex(ValueError, "only_committed"):
            ledger.compensate(record.effect_key)
        ledger.commit(record.effect_key, result_hash="result-v1")
        with self.assertRaisesRegex(ValueError, "already_committed"):
            ledger.commit(record.effect_key)

    def test_indeterminate_effect_requires_reconciliation(self) -> None:
        ledger = EffectLedger()
        record = ledger.prepare(
            task_id="fixture-0",
            tool_name="write_record",
            arguments={"record_id": "r0", "value": 1},
            environment_version="v1",
            effect_class=EffectClass.IRREVERSIBLE,
            scope_key="record:r0",
        )
        uncertain = ledger.mark_indeterminate(record.effect_key)
        self.assertIs(uncertain.status, EffectStatus.INDETERMINATE)
        retry = ledger.retry_decision(record.effect_key)
        self.assertFalse(retry.allowed)
        self.assertEqual(retry.reason, "reconcile_before_retry")
        committed = ledger.commit(record.effect_key, result_hash="confirmed-result")
        self.assertIs(committed.status, EffectStatus.COMMITTED)

    def test_validator_rejects_illegal_effect_transition(self) -> None:
        ledger = EffectLedger()
        prepared = ledger.prepare(
            task_id="fixture-0",
            tool_name="write_record",
            arguments={"record_id": "r0", "value": 1},
            environment_version="v1",
            effect_class=EffectClass.REVERSIBLE,
            scope_key="record:r0",
            compensation={"tool": "restore_record", "arguments": {"record_id": "r0"}},
        )
        base = replace(
            fixture_packet(observation_version="1", resources={"record": 0}),
            effects=(prepared,),
        ).seal()
        illegal = replace(prepared, status=EffectStatus.COMPENSATED)
        target = replace(
            base,
            world=replace(base.world, observation_version="2", resources={"record": 0}),
            effects=(illegal,),
            parent_packet_hash=base.packet_hash,
            packet_hash="",
        ).seal()
        report = PacketValidator().validate(target, previous=base)
        self.assertFalse(report.valid)
        self.assertIn("illegal_effect_transition", {issue.code for issue in report.issues})


class RoutingAndCostTests(unittest.TestCase):
    def test_measured_switch_tax_can_reverse_route(self) -> None:
        policy = ConstrainedUtilityPolicy(
            success_weight=1000,
            cost_weights=CostWeights(latency=1.0),
        )
        stay = estimate(
            executor=Executor.EDGE,
            mode=TransferMode.REUSE,
            success=0.70,
            inference_ms=100,
        )
        switch = estimate(
            executor=Executor.CLOUD,
            mode=TransferMode.TYPED_DELTA,
            success=0.80,
            inference_ms=80,
            handoff_ms=150,
        )
        reversed_route, measured, zero_tax = routing_reversal(
            policy,
            (stay, switch),
            RoutingContext(current_executor=Executor.EDGE),
        )
        self.assertTrue(reversed_route)
        self.assertIs(measured.selected.action.executor, Executor.EDGE)
        self.assertIs(zero_tax.selected.action.executor, Executor.CLOUD)

    def test_constrained_policy_minimizes_cost_after_quality_gate(self) -> None:
        policy = ConstrainedUtilityPolicy(cost_weights=CostWeights(latency=1.0))
        cheap = estimate(
            executor=Executor.EDGE,
            mode=TransferMode.REUSE,
            success=0.81,
            inference_ms=50,
        )
        costly = estimate(
            executor=Executor.CLOUD,
            mode=TransferMode.TYPED_DELTA,
            success=0.99,
            inference_ms=200,
        )
        decision = policy.select(
            (cheap, costly),
            RoutingContext(
                current_executor=Executor.EDGE,
                minimum_success=0.8,
            ),
        )
        self.assertIs(decision.selected.action.executor, Executor.EDGE)
        self.assertEqual(decision.objective_name, "measured_cost")

    def test_recorded_trace_is_consumed_without_looping(self) -> None:
        trace = RecordedBandwidthTrace((1.0, 2.0), 100.0, "public-trace-fixture")
        self.assertAlmostEqual(trace.transfer_time_ms(12_500), 100.0)
        self.assertTrue(math.isinf(trace.transfer_time_ms(100_000)))


class ProvenanceAndMetricsTests(unittest.TestCase):
    def test_exported_source_tree_has_a_stable_revision(self) -> None:
        revision = source_tree_revision(PROJECT_ROOT)
        self.assertRegex(revision, r"^tree-sha256:[0-9a-f]{64}$")
        missing = collect_package_versions(("agentrelay-package-that-does-not-exist",))
        self.assertEqual(
            missing["agentrelay-package-that-does-not-exist"],
            "not-installed",
        )

    def test_split_policy_blocks_train_on_test_and_label_access(self) -> None:
        policy = SplitPolicy(("train",), ("validation",), ("test",))
        policy.validate(split="train", purpose="train", labels_accessed=True)
        with self.assertRaises(ValueError):
            policy.validate(split="test", purpose="train", labels_accessed=False)
        with self.assertRaises(ValueError):
            policy.validate(split="test", purpose="evaluate", labels_accessed=True)

    def test_metrics_aggregate_exactly(self) -> None:
        rows = (
            TrajectoryMetrics(
                run_id="r0",
                benchmark="fixture",
                split="fixture",
                sample_id="0",
                method="relay",
                success=1.0,
                reward=1.0,
                end_to_end_ms=100.0,
                step_latencies_ms=(40.0, 60.0),
                cloud_steps=1,
                total_steps=2,
                switches=1,
                transfer_bytes=100,
                transfer_tokens=20,
                relay_tax_ms=10.0,
                invariant_checks=2,
                invariant_passes=2,
                committed_effects=1,
            ),
            TrajectoryMetrics(
                run_id="r1",
                benchmark="fixture",
                split="fixture",
                sample_id="1",
                method="relay",
                success=0.0,
                reward=0.0,
                end_to_end_ms=300.0,
                step_latencies_ms=(100.0,),
                cloud_steps=0,
                total_steps=1,
                switches=0,
                transfer_bytes=0,
                transfer_tokens=0,
                relay_tax_ms=0.0,
            ),
        )
        result = aggregate_trajectories(rows)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["success_mean"], 0.5)
        self.assertEqual(result["cloud_step_ratio"], 1 / 3)
        self.assertEqual(result["relay_tax_share"], 0.025)


class ConfigurationTests(unittest.TestCase):
    def test_unlocked_template_is_non_executable(self) -> None:
        config = load_json_config(PROJECT_ROOT / "configs" / "local-smoke.template.json")
        validate_experiment_config(config, allow_unlocked=True)
        with self.assertRaises(ValueError):
            validate_experiment_config(config)

    def test_formal_config_cannot_limit_samples(self) -> None:
        config = load_json_config(PROJECT_ROOT / "configs" / "formal-autodl-4090d.template.json")
        config = json.loads(json.dumps(config))
        config["limits"]["sample_limit"] = 10
        with self.assertRaises(ValueError):
            validate_experiment_config(config, allow_unlocked=True)

    def test_native_model_config_ignores_requested_revision_metadata(self) -> None:
        config = NativeGenerationConfig.from_dict(
            {
                "model_id": "public/model",
                "requested_revision": "main",
                "revision": "0" * 40,
            }
        )
        self.assertEqual(config.revision, "0" * 40)


class LocalSmokeUnitTests(unittest.TestCase):
    def test_prompt_excludes_labels_and_future_assistant_messages(self) -> None:
        record = {
            "id": "public-fixture-0",
            "data_source": "public-fixture",
            "question": "Choose the declared read-only tool.",
            "tools": [
                {
                    "name": "inspect",
                    "parameters": {"properties": {"target": {"type": "string"}}},
                }
            ],
            "messages": [
                {
                    "role": "system",
                    "content": "Use the declared tools.",
                    "metadata": {"final_label": 1},
                },
                {"role": "user", "content": "Inspect the record."},
                {"role": "assistant", "content": "SECRET_FUTURE_ACTION"},
            ],
            "step_labels": {"2": 1},
            "final_label": 1,
        }
        example = process_example_from_record(record, index=0)
        messages = build_diagnostic_messages(example)
        serialized = json.dumps(messages)
        self.assertNotIn("SECRET_FUTURE_ACTION", serialized)
        self.assertNotIn("step_labels", serialized)
        self.assertNotIn("final_label", serialized)
        self.assertNotIn("metadata", serialized)
        self.assertIn("target", serialized)

        traces = TraceStore()
        packet = build_diagnostic_packet(
            example,
            dataset_id="public/dataset",
            dataset_revision="0" * 40,
            split="train",
            trace_store=traces,
        )
        report = PacketValidator().validate(packet, trace_store=traces)
        self.assertTrue(report.valid)
        self.assertFalse(packet.invariants.hard_constraints["labels_visible_to_model"])


class StatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_is_seed_deterministic(self) -> None:
        first = (1.0, 2.0, 3.0, 4.0)
        second = (0.0, 1.0, 2.0, 3.0)
        one = paired_bootstrap_difference(first, second, resamples=100, seed=7)
        two = paired_bootstrap_difference(first, second, resamples=100, seed=7)
        self.assertEqual(one, two)
        self.assertEqual(one.mean_difference, 1.0)
        self.assertEqual(one.confidence_low, 1.0)
        self.assertEqual(one.confidence_high, 1.0)

    def test_exact_mcnemar_and_holm(self) -> None:
        self.assertEqual(mcnemar_exact((1, 1, 1, 1), (0, 0, 0, 0)), 0.125)
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.2})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.08)
        self.assertAlmostEqual(adjusted["c"], 0.2)


if __name__ == "__main__":
    unittest.main()
