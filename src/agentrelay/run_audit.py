"""Independent integrity checks for immutable AgentRelay run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from .config import FULL_COMMIT_RE, validate_experiment_config
from .provenance import source_tree_revision, validate_manifest_hash
from .schema import RelayStatePacket, sha256_json, sha256_text
from .state import TraceStore
from .validation import PacketValidator


CODE_REVISION_RE = re.compile(
    r"^(?:[0-9a-f]{40,64}|tree-sha256:[0-9a-f]{64})$",
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def audit_run(
    run_directory: str | Path,
    *,
    required_gpu_name: str = "",
    require_diagnostic_only: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_directory).resolve()
    manifest = _read_json(root / "manifest.json")
    summary = _read_json(root / "summary.json")
    outputs = _read_jsonl(root / "outputs.jsonl")
    checks: dict[str, bool] = {}
    issues: list[dict[str, str]] = []

    def check(code: str, condition: bool, message: str) -> None:
        checks[code] = bool(condition)
        if not condition:
            issues.append({"code": code, "message": message})

    check("manifest_hash", validate_manifest_hash(manifest), "manifest self-hash mismatch")
    check(
        "summary_manifest_hash",
        summary.get("manifest_hash") == manifest.get("manifest_hash"),
        "summary and manifest hashes differ",
    )
    check(
        "output_hash",
        summary.get("output_hash") == sha256_json(outputs),
        "outputs.jsonl content hash mismatch",
    )
    check(
        "record_count",
        summary.get("completed_records") == len(outputs) and bool(outputs),
        "summary record count does not match nonempty outputs",
    )
    check(
        "code_revision",
        bool(CODE_REVISION_RE.fullmatch(str(manifest.get("code_revision", "")))),
        "code revision is missing or not immutable",
    )
    recorded_code_revision = str(manifest.get("code_revision", ""))
    if project_root is not None and recorded_code_revision.startswith("tree-sha256:"):
        check(
            "source_tree_match",
            source_tree_revision(project_root) == recorded_code_revision,
            "current runtime source tree does not match the run manifest",
        )
    config = manifest.get("config")
    config_valid = isinstance(config, Mapping)
    if config_valid:
        try:
            validate_experiment_config(config)
        except (KeyError, TypeError, ValueError):
            config_valid = False
    check("locked_config", config_valid, "embedded experiment config is invalid or unlocked")

    hardware = manifest.get("hardware", {})
    check(
        "cuda_available",
        isinstance(hardware, Mapping) and hardware.get("cuda_available") is True,
        "manifest does not attest a CUDA run",
    )
    gpu_name = str(hardware.get("gpu_name", "")) if isinstance(hardware, Mapping) else ""
    check(
        "required_gpu",
        not required_gpu_name or required_gpu_name.lower() in gpu_name.lower(),
        f"GPU name {gpu_name!r} does not contain {required_gpu_name!r}",
    )
    packages = hardware.get("packages", {}) if isinstance(hardware, Mapping) else {}
    check(
        "package_versions",
        isinstance(packages, Mapping)
        and all(str(packages.get(name, "")) not in {"", "not-installed"} for name in ("torch", "transformers", "datasets")),
        "core package versions are absent from the manifest",
    )

    run_id = str(summary.get("run_id", ""))
    manifest_run_id = str(manifest.get("experiment_id", ""))
    check("run_identity", bool(run_id) and run_id == manifest_run_id, "run IDs differ")
    manifest_dataset_revision = str(manifest.get("dataset_revision", ""))
    check(
        "dataset_revision",
        bool(FULL_COMMIT_RE.fullmatch(manifest_dataset_revision)),
        "dataset revision is not a full commit SHA",
    )

    packet_checks_ok = True
    record_checks_ok = True
    semantic_protocol_ok = True
    for index, row in enumerate(outputs):
        try:
            trace_store = TraceStore()
            trace_spans = row.get("trace_spans", {})
            if not isinstance(trace_spans, Mapping):
                raise ValueError("trace_spans is not an object")
            for span_id, content in trace_spans.items():
                trace_store.put_verified(str(span_id), str(content))
            packet_text = str(row["packet_json"])
            packet = RelayStatePacket.from_json(packet_text)
            report = PacketValidator().validate(packet, trace_store=trace_store)
            packet_ok = (
                report.valid
                and packet.packet_hash == row.get("packet_hash")
                and len(packet_text.encode("utf-8")) == row.get("packet_bytes")
                and packet.task.dataset_revision == row.get("dataset_revision")
                and packet.task.sample_id == str(row.get("sample_id"))
            )
            if packet.schema_version == "2.0":
                frontier_hash = (
                    packet.effect_frontier.frontier_hash if packet.effect_frontier else ""
                )
                semantic_ok = (
                    row.get("semantic_schema_version") == "2.0"
                    and row.get("semantic_node_count") == len(packet.semantic_nodes)
                    and row.get("obligation_count") == len(packet.obligation_ids)
                    and row.get("effect_frontier_hash") == frontier_hash
                    and row.get("protocol_valid") is True
                    and bool(row.get("protocol_patch_node_ids"))
                    and int(row.get("protocol_transmitted_bytes", 0)) > 0
                    and row.get("routing_reversal_fixture") is True
                    and bool(row.get("baseline_manifest_hash"))
                )
                semantic_protocol_ok = semantic_protocol_ok and semantic_ok
        except (KeyError, TypeError, ValueError):
            packet_ok = False
            semantic_protocol_ok = False
        packet_checks_ok = packet_checks_ok and packet_ok
        row_ok = (
            row.get("run_id") == run_id
            and row.get("packet_valid") is True
            and row.get("labels_accessed") is False
            and sha256_text(str(row.get("response_text", ""))) == row.get("response_hash")
            and bool(FULL_COMMIT_RE.fullmatch(str(row.get("model_revision", ""))))
            and bool(FULL_COMMIT_RE.fullmatch(str(row.get("dataset_revision", ""))))
        )
        if require_diagnostic_only:
            row_ok = row_ok and row.get("paper_evidence") is False
        if not row_ok:
            issues.append(
                {"code": "record_integrity", "message": f"record {index} failed integrity checks"}
            )
        record_checks_ok = record_checks_ok and row_ok
    checks["packet_integrity"] = packet_checks_ok
    if not packet_checks_ok:
        issues.append({"code": "packet_integrity", "message": "one or more packets failed independent validation"})
    checks["record_integrity"] = record_checks_ok
    checks["semantic_protocol"] = semantic_protocol_ok
    if not semantic_protocol_ok:
        issues.append(
            {
                "code": "semantic_protocol",
                "message": "one or more v2 continuation protocol probes failed integrity checks",
            }
        )

    if require_diagnostic_only:
        diagnostic_ok = (
            summary.get("diagnostic_only") is True
            and summary.get("paper_evidence") is False
            and summary.get("labels_accessed") is False
            and manifest.get("purpose") == "local_diagnostic_only"
            and manifest.get("labels_accessed") is False
        )
        if summary.get("semantic_schema_version") == "2.0":
            diagnostic_ok = (
                diagnostic_ok
                and summary.get("all_protocol_probes_valid") is True
                and summary.get("routing_reversal_fixture_passed") is True
                and int(summary.get("implemented_baseline_count", 0)) > 0
            )
        check(
            "diagnostic_boundary",
            diagnostic_ok,
            "run is not consistently marked diagnostic-only/non-evidence",
        )

    return {
        "valid": not issues and all(checks.values()),
        "run_directory": str(root),
        "run_id": run_id,
        "records": len(outputs),
        "checks": checks,
        "issues": issues,
    }
