"""Run-artifact audit tests use software fixtures, never benchmark evidence."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agentrelay.provenance import source_tree_revision
from agentrelay.run_audit import audit_run
from agentrelay.schema import (
    EvidenceItem,
    InvariantState,
    PlanState,
    RelayStatePacket,
    TaskIdentity,
    WorldState,
    canonical_json,
    goal_digest,
    sha256_json,
    sha256_text,
)
from agentrelay.state import TraceStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RunAuditTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        run_id = "software-audit-fixture"
        dataset_revision = "0" * 40
        model_revision = "1" * 40
        traces = TraceStore()
        trace_text = canonical_json({"question": "software schema fixture"})
        span_id = traces.add(trace_text)
        packet = RelayStatePacket(
            task=TaskIdentity(
                dataset_id="public/software-fixture",
                dataset_revision=dataset_revision,
                split="fixture",
                sample_id="fixture-0",
                goal_hash=goal_digest("audit software artifact"),
                success_criteria=("independent audit passes",),
            ),
            invariants=InvariantState(
                hard_constraints={
                    "diagnostic_only": True,
                    "labels_visible_to_model": False,
                    "paper_evidence": False,
                }
            ),
            world=WorldState(
                observation_version="fixture-0",
                environment_digest=span_id,
            ),
            evidence=(
                EvidenceItem(
                    fact_id="fixture",
                    value={"trace": span_id},
                    source_span_id=span_id,
                    provenance_hash=span_id,
                ),
            ),
            plan=PlanState(active_subgoal="audit"),
            trace_refs=(span_id,),
            schema_version="1.0",
        ).seal()
        response = "software fixture response"
        row = {
            "run_id": run_id,
            "dataset_revision": dataset_revision,
            "model_revision": model_revision,
            "sample_id": "fixture-0",
            "packet_hash": packet.packet_hash,
            "packet_bytes": len(packet.to_json().encode("utf-8")),
            "packet_valid": True,
            "packet_json": packet.to_json(),
            "trace_spans": traces.subset(packet.trace_refs),
            "response_text": response,
            "response_hash": sha256_text(response),
            "labels_accessed": False,
            "paper_evidence": False,
        }
        config = {
            "schema_version": "1.0",
            "run_mode": "local_smoke",
            "paper_evidence": False,
            "data_root": "artifacts/local-data",
            "models": {
                "edge": {"model_id": "public/software-model", "revision": model_revision}
            },
            "datasets": [
                {
                    "name": "software-fixture",
                    "hf_id": "public/software-fixture",
                    "revision": dataset_revision,
                    "splits": ["fixture"],
                }
            ],
            "limits": {"sample_limit": 1},
            "integrity": {
                "allow_synthetic_tasks": False,
                "allow_test_label_access": False,
                "native_inference_only": True,
                "allow_prompt_answer_injection": False,
            },
        }
        manifest = {
            "experiment_id": run_id,
            "dataset_revision": dataset_revision,
            "config": config,
            "code_revision": source_tree_revision(PROJECT_ROOT),
            "hardware": {
                "cuda_available": True,
                "gpu_name": "NVIDIA GeForce RTX 4080 Laptop GPU",
                "packages": {"torch": "fixture", "transformers": "fixture", "datasets": "fixture"},
            },
            "purpose": "local_diagnostic_only",
            "labels_accessed": False,
        }
        manifest["manifest_hash"] = sha256_json(manifest)
        summary = {
            "run_id": run_id,
            "manifest_hash": manifest["manifest_hash"],
            "output_hash": sha256_json([row]),
            "completed_records": 1,
            "diagnostic_only": True,
            "paper_evidence": False,
            "labels_accessed": False,
        }
        (root / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        (root / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
        (root / "outputs.jsonl").write_text(canonical_json(row) + "\n", encoding="utf-8")

    def test_audit_passes_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            report = audit_run(
                root,
                required_gpu_name="4080",
                require_diagnostic_only=True,
                project_root=PROJECT_ROOT,
            )
            self.assertTrue(report["valid"])

            output_path = root / "outputs.jsonl"
            row = json.loads(output_path.read_text(encoding="utf-8"))
            row["response_text"] = "tampered"
            output_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
            tampered = audit_run(
                root,
                required_gpu_name="4080",
                require_diagnostic_only=True,
                project_root=PROJECT_ROOT,
            )
            self.assertFalse(tampered["valid"])
            self.assertFalse(tampered["checks"]["output_hash"])
            self.assertFalse(tampered["checks"]["record_integrity"])


if __name__ == "__main__":
    unittest.main()
