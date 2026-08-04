"""Software fixtures for the v2 continuation protocol; no benchmark evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from agentrelay.baselines import BaselineController, BaselineInput, BaselineName
from agentrelay.calibration import CalibrationRow, ConformalRiskCalibrator
from agentrelay.codecs import ContinuationCodecName, encode_continuation
from agentrelay.continuation import (
    build_obligation_closed_packet,
    build_standard_semantic_graph,
    missing_predecessor_ids,
    obligation_closure,
)
from agentrelay.cost import HandoffMeasurement
from agentrelay.effects import EffectLedger, assess_migration, build_effect_frontier
from agentrelay.experiment_runtime import ProfileEstimateProvider, ServiceProfile
from agentrelay.official_adapters import classify_appworld_code
from agentrelay.policy import (
    CandidateAction,
    CandidateEstimate,
    ConstrainedUtilityPolicy,
    RoutingContext,
)
from agentrelay.runtime import HandoffCoordinator
from agentrelay.schema import (
    CommitMode,
    EffectClass,
    EffectStatus,
    Executor,
    InvariantState,
    RelayStatePacket,
    TaskIdentity,
    TransferMode,
    WorldState,
    goal_digest,
    sha256_text,
)
from agentrelay.state import TraceStore
from agentrelay.validation import PacketValidator


REVISION = "2" * 40


def v2_packet() -> tuple[RelayStatePacket, TraceStore]:
    traces = TraceStore()
    goal = "software-only goal fixture"
    observation = "software-only observation fixture"
    goal_span = traces.add(goal)
    observation_span = traces.add(observation)
    nodes, edges, obligations = build_standard_semantic_graph(
        goal={"goal": goal},
        observation={"observation": observation},
        obligation={"next_action": True},
        world_version="fixture-0",
        goal_trace_ref=goal_span,
        observation_trace_ref=observation_span,
        goal_provenance_hash=sha256_text(goal),
        observation_provenance_hash=sha256_text(observation),
    )
    packet = RelayStatePacket(
        task=TaskIdentity(
            dataset_id="public/software-v2-fixture",
            dataset_revision=REVISION,
            split="fixture",
            sample_id="v2-0",
            goal_hash=goal_digest(goal),
        ),
        invariants=InvariantState(unresolved_obligations=("next_action",)),
        world=WorldState("fixture-0", sha256_text(observation)),
        trace_refs=(goal_span, observation_span),
        source_executor=Executor.EDGE,
        target_executor=Executor.EDGE,
        obligation_ids=obligations,
        semantic_nodes=nodes,
        dependency_edges=edges,
        effect_frontier=build_effect_frontier(()),
    ).seal()
    return packet, traces


def estimate(executor: Executor, mode: TransferMode, *, calibrated: bool = False) -> CandidateEstimate:
    return CandidateEstimate(
        action=CandidateAction(executor, mode, CommitMode.IMMEDIATE),
        predicted_success=0.9,
        predicted_fidelity=0.9,
        inference_ms=5.0 if executor is Executor.CLOUD else 10.0,
        handoff=HandoffMeasurement(communication_ms=2.0),
        success_lower_bound=0.8 if calibrated else None,
        fidelity_lower_bound=0.8 if calibrated else None,
        effect_risk_upper_bound=0.01 if calibrated else None,
    )


class ContinuationProtocolTests(unittest.TestCase):
    def test_obligation_closure_contains_all_predecessors(self) -> None:
        packet, _ = v2_packet()
        closure = obligation_closure(
            packet.semantic_nodes,
            packet.dependency_edges,
            packet.obligation_ids,
        )
        self.assertEqual(set(closure), {node.node_id for node in packet.semantic_nodes})

    def test_lazy_predecessor_is_named_and_selectively_patched(self) -> None:
        packet, traces = v2_packet()
        lazy = packet.semantic_nodes[1].node_id
        context = RoutingContext(
            current_executor=Executor.EDGE,
            effect_frontier=packet.effect_frontier,
        )
        coordinator = HandoffCoordinator(
            policy=ConstrainedUtilityPolicy(),
            validator=PacketValidator(require_v2_graph=True),
            source_trace_store=traces,
            target_trace_store=TraceStore(),
        )
        transaction = coordinator.execute(
            acknowledged=packet,
            source_current=packet,
            estimates=(estimate(Executor.CLOUD, TransferMode.CLOSED_DELTA_PATCHABLE),),
            context=context,
            lazy_node_ids=(lazy,),
        )
        self.assertFalse(transaction.first_validation.valid)
        self.assertTrue(transaction.final_validation.valid)
        self.assertEqual(transaction.patch_request.missing_node_ids, (lazy,))
        self.assertIn(lazy, {node.node_id for node in transaction.patch_bundle.semantic_nodes})

    def test_unpatchable_predecessor_is_rejected(self) -> None:
        packet, traces = v2_packet()
        lazy = packet.semantic_nodes[1].node_id
        cut = build_obligation_closed_packet(packet, lazy_node_ids=(lazy,)).packet
        cut = replace(cut, patchable_predecessor_ids=(), packet_hash="").seal()
        report = PacketValidator(require_v2_graph=True).validate(cut, trace_store=traces)
        self.assertIn("unpatchable_predecessor", {issue.code for issue in report.issues})

    def test_semantic_node_tamper_is_detected(self) -> None:
        packet, traces = v2_packet()
        tampered_node = replace(packet.semantic_nodes[0], value={"goal": "tampered"})
        tampered = replace(
            packet,
            semantic_nodes=(tampered_node, *packet.semantic_nodes[1:]),
            packet_hash="",
        ).seal()
        report = PacketValidator(require_v2_graph=True).validate(tampered, trace_store=traces)
        self.assertIn("node_hash", {issue.code for issue in report.issues})

    def test_typed_delta_without_edges_is_a_distinct_baseline(self) -> None:
        packet, traces = v2_packet()
        encoded = encode_continuation(
            ContinuationCodecName.TYPED_DELTA_NO_EDGES,
            packet,
            traces,
        )
        self.assertEqual(encoded.packet.dependency_edges, ())
        self.assertGreater(encoded.encoded_bytes, 0)


class EffectFrontierTests(unittest.TestCase):
    def test_sent_effect_blocks_migration_until_reconciliation(self) -> None:
        ledger = EffectLedger()
        prepared = ledger.prepare(
            task_id="v2-0",
            tool_name="sandbox.mutate",
            arguments={"value": 1},
            environment_version="fixture-0",
            effect_class=EffectClass.IRREVERSIBLE,
        )
        ledger.mark_sent(prepared.effect_key)
        blocked = assess_migration(ledger.snapshot())
        self.assertFalse(blocked.allowed)
        self.assertIs(blocked.required_commit_mode, CommitMode.RECONCILE)
        ledger.reconcile(prepared.effect_key, occurred=False)
        self.assertTrue(ledger.migration_legality().allowed)

    def test_acknowledge_then_commit_preserves_result(self) -> None:
        ledger = EffectLedger()
        prepared = ledger.prepare(
            task_id="v2-0",
            tool_name="sandbox.mutate",
            arguments={"value": 1},
            environment_version="fixture-0",
            effect_class=EffectClass.IRREVERSIBLE,
        )
        ledger.mark_sent(prepared.effect_key)
        ledger.acknowledge(prepared.effect_key, result_hash="result")
        committed = ledger.commit(prepared.effect_key)
        self.assertIs(committed.status, EffectStatus.COMMITTED)
        self.assertEqual(committed.result_hash, "result")


class RiskAndBaselineTests(unittest.TestCase):
    def test_minimum_dwell_masks_a_switch(self) -> None:
        policy = ConstrainedUtilityPolicy()
        context = RoutingContext(
            current_executor=Executor.EDGE,
            consecutive_steps=1,
            minimum_dwell_steps=2,
        )
        decision = policy.select(
            (
                estimate(Executor.EDGE, TransferMode.REUSE),
                estimate(Executor.CLOUD, TransferMode.FULL_REPLAY),
            ),
            context,
        )
        self.assertIs(decision.selected.action.executor, Executor.EDGE)

    def test_calibrator_uses_only_authorized_validation_rows(self) -> None:
        action = CandidateAction(Executor.CLOUD, TransferMode.FULL_REPLAY, CommitMode.IMMEDIATE)
        rows = tuple(
            CalibrationRow(
                dataset_id="public/software-v2-fixture",
                dataset_revision=REVISION,
                split="dev",
                sample_id=str(index),
                purpose="tune",
                action=action,
                predicted_success=0.9,
                predicted_fidelity=0.9,
                predicted_effect_risk=0.1,
                observed_success=index % 2,
                observed_fidelity_pass=1,
                observed_effect_failure=0,
            )
            for index in range(4)
        )
        calibrator = ConformalRiskCalibrator(alpha=0.2, minimum_group_size=2).fit(
            rows,
            authorized_validation_splits=("dev",),
        )
        calibrated = calibrator.calibrate(estimate(Executor.CLOUD, TransferMode.FULL_REPLAY))
        self.assertTrue(calibrated.calibrated)
        self.assertLessEqual(calibrated.success_lower_bound, calibrated.predicted_success)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibrator.save(path)
            self.assertTrue(path.exists())

    def test_agentrelay_requires_calibrated_candidates(self) -> None:
        packet, _ = v2_packet()
        context = RoutingContext(
            current_executor=Executor.EDGE,
            minimum_success=0.5,
            minimum_fidelity=0.5,
            maximum_effect_risk=0.2,
            effect_frontier=packet.effect_frontier,
        )
        controller = BaselineController(BaselineName.AGENTRELAY)
        decision = controller.select(
            BaselineInput(
                estimates=(estimate(Executor.EDGE, TransferMode.REUSE),),
                context=context,
                sample_id="v2-0",
                step_index=0,
            )
        )
        self.assertTrue(decision.abstained)
        self.assertIn("uncalibrated", decision.constraint_violations)

    def test_random_baseline_is_reproducible(self) -> None:
        packet, _ = v2_packet()
        value = BaselineInput(
            estimates=(
                estimate(Executor.EDGE, TransferMode.REUSE),
                estimate(Executor.CLOUD, TransferMode.FULL_REPLAY),
            ),
            context=RoutingContext(
                current_executor=Executor.EDGE,
                effect_frontier=packet.effect_frontier,
            ),
            sample_id="v2-0",
            step_index=7,
        )
        first = BaselineController(BaselineName.RANDOM_05).select(value)
        second = BaselineController(BaselineName.RANDOM_05).select(value)
        self.assertEqual(first.selected.action, second.selected.action)

    def test_appworld_effect_classifier_is_conservative(self) -> None:
        self.assertIs(
            classify_appworld_code("apis.spotify.show_song(song_id='x')"),
            EffectClass.READ_ONLY,
        )
        self.assertIs(
            classify_appworld_code("apis.supervisor.complete_task()"),
            EffectClass.IRREVERSIBLE,
        )
        self.assertIs(
            classify_appworld_code("apis.venmo.send_money(to='x', amount=1)"),
            EffectClass.UNKNOWN,
        )

    def test_pending_unknown_effect_forces_barrier_candidates(self) -> None:
        packet, _ = v2_packet()
        provider = ProfileEstimateProvider(
            {
                Executor.EDGE: ServiceProfile(10.0, 0.5, 0.9),
                Executor.CLOUD: ServiceProfile(20.0, 0.7, 0.9),
            }
        )
        candidates = provider.candidates(
            packet,
            RoutingContext(
                current_executor=Executor.EDGE,
                effect_class=EffectClass.UNKNOWN,
                effect_frontier=packet.effect_frontier,
            ),
            step_index=0,
            remaining_steps=1,
        )
        self.assertTrue(candidates)
        self.assertTrue(
            all(candidate.action.commit_mode is CommitMode.BARRIER for candidate in candidates)
        )


if __name__ == "__main__":
    unittest.main()
