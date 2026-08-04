"""Diagnostic-only native inference over an immutable public trajectory record.

The runner deliberately does not compute a paper metric. It proves that a
locked public record can travel through dataset loading, prompt construction,
native GPU generation, relay-packet sealing, validation, and immutable output
recording on the local verification machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from .config import StorageLayout, load_json_config, validate_experiment_config
from .baselines import baseline_manifest
from .continuation import closure_depth, render_semantic_continuation, build_standard_semantic_graph
from .cost import HandoffMeasurement
from .datasets import DatasetSpec, load_public_dataset, sample_identifier
from .effects import build_effect_frontier
from .inference import HFModelExecutor, NativeGenerationConfig
from .policy import (
    CandidateAction,
    CandidateEstimate,
    ConstrainedUtilityPolicy,
    RoutingContext,
    routing_reversal,
)
from .provenance import (
    RunManifest,
    collect_hardware_metadata,
    collect_package_versions,
)
from .schema import (
    EvidenceItem,
    CommitMode,
    Executor,
    InvariantState,
    PlanState,
    RelayStatePacket,
    SCHEMA_VERSION,
    TaskIdentity,
    TransferMode,
    WorldState,
    canonical_json,
    goal_digest,
    sha256_json,
    sha256_text,
)
from .runtime import HandoffCoordinator
from .state import TraceStore
from .validation import PacketValidator


FORBIDDEN_PROMPT_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "annotation",
        "annotations",
        "final_label",
        "ground_truth",
        "label",
        "labels",
        "reference",
        "reference_answer",
        "reference_output",
        "step_labels",
        "target",
        "targets",
    }
)

SAFE_MESSAGE_FIELDS = frozenset(
    {
        "role",
        "content",
        "name",
        "tool_calls",
        "tool_call_id",
    }
)

SAFE_DATASET_FIELDS = frozenset(
    {
        "total_index",
        "query_index",
        "sample_index",
        "id",
        "task_id",
        "sample_id",
        "source_path",
        "index",
        "data_source",
        "dataset",
        "subset",
        "question",
        "task_description",
        "tools",
        "messages",
    }
)

DIAGNOSTIC_SYSTEM_PROMPT = (
    "You are running a diagnostic continuation over one immutable public "
    "tool-agent trajectory prefix. Use only the visible prefix and public tool "
    "definitions. Do not assume unseen tool results or a reference answer. "
    "Return only the next assistant step. If a tool call is needed, express the "
    "tool name and arguments explicitly."
)


@dataclass(frozen=True)
class ProcessDiagnosticExample:
    sample_id: str
    source_subset: str
    visible_payload: Mapping[str, Any]

    @property
    def visible_payload_hash(self) -> str:
        return sha256_json(self.visible_payload)


@dataclass(frozen=True)
class LocalSmokeRecord:
    run_id: str
    dataset_id: str
    dataset_revision: str
    dataset_config_name: str | None
    split: str
    sample_id: str
    source_subset: str
    visible_payload_hash: str
    packet_hash: str
    packet_bytes: int
    packet_valid: bool
    packet_json: str
    trace_spans: Mapping[str, str]
    semantic_schema_version: str
    semantic_node_count: int
    obligation_count: int
    dependency_depth: int
    effect_frontier_hash: str
    protocol_patch_node_ids: tuple[str, ...]
    protocol_transmitted_bytes: int
    protocol_valid: bool
    routing_reversal_fixture: bool
    baseline_manifest_hash: str
    prompt_hash: str
    response_hash: str
    response_text: str
    prompt_tokens: int
    output_tokens: int
    latency_ms: float
    peak_cuda_memory_bytes: int
    model_id: str
    model_revision: str
    seed: int
    paper_evidence: bool = False
    labels_accessed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalSmokeResult:
    run_directory: Path
    manifest_path: Path
    outputs_path: Path
    summary_path: Path
    records: tuple[LocalSmokeRecord, ...]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


def _safe_message_prefix(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    prefix: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        role = str(raw.get("role", ""))
        if role == "assistant":
            break
        if role not in {"system", "user", "tool"}:
            continue
        message = {
            str(key): _jsonable(item)
            for key, item in raw.items()
            if str(key).lower() in SAFE_MESSAGE_FIELDS
        }
        prefix.append(message)
    return prefix


def process_example_from_record(
    record: Mapping[str, Any],
    *,
    index: int,
) -> ProcessDiagnosticExample:
    """Expose only pre-action public context; human labels never enter the prompt."""

    source_subset = str(
        record.get("data_source")
        or record.get("dataset")
        or record.get("subset")
        or "unknown"
    )
    visible_payload: dict[str, Any] = {
        "source_subset": source_subset,
        "question": _jsonable(record.get("question")),
        "task_description": _jsonable(record.get("task_description")),
        "tools": _jsonable(record.get("tools")),
        "messages_before_first_assistant_step": _safe_message_prefix(record.get("messages")),
    }
    if not any(
        value not in (None, "", [], {})
        for key, value in visible_payload.items()
        if key != "source_subset"
    ):
        raise ValueError("public record has no visible task context")
    return ProcessDiagnosticExample(
        sample_id=sample_identifier(record, index),
        source_subset=source_subset,
        visible_payload=visible_payload,
    )


def build_diagnostic_messages(
    example: ProcessDiagnosticExample,
    *,
    continuation: str = "",
) -> tuple[dict[str, str], ...]:
    user_payload = (
        "Public trajectory prefix follows as canonical JSON.\n"
        + canonical_json(example.visible_payload)
        + ("\nVerified semantic continuation:\n" + continuation if continuation else "")
        + "\nProduce the next assistant step now."
    )
    return (
        {"role": "system", "content": DIAGNOSTIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    )


def build_diagnostic_packet(
    example: ProcessDiagnosticExample,
    *,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    trace_store: TraceStore,
) -> RelayStatePacket:
    visible_text = canonical_json(example.visible_payload)
    span_id = trace_store.add(visible_text)
    goal_text = str(
        example.visible_payload.get("question")
        or example.visible_payload.get("task_description")
        or example.visible_payload_hash
    )
    messages = example.visible_payload.get("messages_before_first_assistant_step") or []
    tools = example.visible_payload.get("tools") or []
    semantic_nodes, dependency_edges, obligation_ids = build_standard_semantic_graph(
        goal={"goal": goal_text},
        observation={"visible_payload_hash": example.visible_payload_hash},
        obligation={"next_action": "produce_next_assistant_step"},
        world_version="prefix-0",
        goal_trace_ref=span_id,
        observation_trace_ref=span_id,
        goal_provenance_hash=sha256_text(visible_text),
        observation_provenance_hash=sha256_text(visible_text),
    )
    return RelayStatePacket(
        task=TaskIdentity(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            split=split,
            sample_id=example.sample_id,
            goal_hash=goal_digest(goal_text),
            success_criteria=(
                "native generation completes",
                "relay packet passes deterministic validation",
            ),
        ),
        invariants=InvariantState(
            hard_constraints={
                "diagnostic_only": True,
                "labels_visible_to_model": False,
                "paper_evidence": False,
            },
            permissions=("read_public_trajectory_prefix",),
            unresolved_obligations=("produce_next_assistant_step",),
        ),
        world=WorldState(
            observation_version="prefix-0",
            environment_digest=example.visible_payload_hash,
            resources={
                "source_subset": example.source_subset,
                "visible_message_count": len(messages) if isinstance(messages, Sequence) else 0,
                "tool_count": len(tools) if isinstance(tools, Sequence) else 0,
            },
        ),
        evidence=(
            EvidenceItem(
                fact_id="public-trajectory-prefix",
                value={"visible_payload_hash": example.visible_payload_hash},
                source_span_id=span_id,
                provenance_hash=sha256_text(visible_text),
            ),
        ),
        plan=PlanState(
            active_subgoal="produce the first unseen assistant step",
            pending_actions=("native_model_generate",),
        ),
        trace_refs=(span_id,),
        source_executor=Executor.EDGE,
        target_executor=Executor.EDGE,
        obligation_ids=obligation_ids,
        semantic_nodes=semantic_nodes,
        dependency_edges=dependency_edges,
        effect_frontier=build_effect_frontier(()),
    ).seal()


def _select_examples(
    dataset: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    subset: str | None,
) -> tuple[ProcessDiagnosticExample, ...]:
    selected: list[ProcessDiagnosticExample] = []
    for index, record in enumerate(dataset):
        if not isinstance(record, Mapping):
            continue
        example = process_example_from_record(record, index=index)
        if subset and subset.lower() not in example.source_subset.lower():
            continue
        selected.append(example)
        if len(selected) >= limit:
            break
    if not selected:
        suffix = f" for subset {subset!r}" if subset else ""
        raise ValueError(f"no usable public diagnostic records found{suffix}")
    return tuple(selected)


def _write_jsonl(path: Path, records: Iterable[LocalSmokeRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record.to_dict()) + "\n")
    temporary.replace(path)


def _run_protocol_probe(
    packet: RelayStatePacket,
    source_trace_store: TraceStore,
) -> tuple[Any, bool]:
    """Exercise closure validation, named patching, and tax-aware routing.

    Candidate estimates are an explicitly labeled software fixture. They are
    never treated as benchmark measurements or paper evidence.
    """

    if len(packet.semantic_nodes) < 2:
        raise ValueError("diagnostic semantic graph is unexpectedly small")
    lazy_node_id = packet.semantic_nodes[1].node_id
    policy = ConstrainedUtilityPolicy()
    switch_estimate = CandidateEstimate(
        action=CandidateAction(
            Executor.CLOUD,
            TransferMode.CLOSED_DELTA_PATCHABLE,
            CommitMode.IMMEDIATE,
        ),
        predicted_success=0.95,
        predicted_fidelity=0.98,
        inference_ms=5.0,
        handoff=HandoffMeasurement(
            encode_ms=1.0,
            communication_ms=50.0,
            rehydration_ms=5.0,
            verification_ms=2.0,
            patch_ms=2.0,
            fidelity_risk=0.02,
        ),
    )
    stay_estimate = CandidateEstimate(
        action=CandidateAction(Executor.EDGE, TransferMode.REUSE, CommitMode.IMMEDIATE),
        predicted_success=0.85,
        predicted_fidelity=1.0,
        inference_ms=30.0,
    )
    context = RoutingContext(
        current_executor=Executor.EDGE,
        minimum_success=0.8,
        minimum_fidelity=0.8,
        effect_frontier=packet.effect_frontier,
    )
    reversal, _, _ = routing_reversal(policy, (stay_estimate, switch_estimate), context)
    coordinator = HandoffCoordinator(
        policy=policy,
        validator=PacketValidator(require_v2_graph=True),
        source_trace_store=source_trace_store,
        target_trace_store=TraceStore(),
    )
    transaction = coordinator.execute(
        acknowledged=packet,
        source_current=packet,
        estimates=(switch_estimate,),
        context=context,
        lazy_node_ids=(lazy_node_id,),
    )
    return transaction, reversal


def run_local_smoke(
    config_path: str | Path,
    *,
    limit: int | None = None,
    subset: str | None = None,
    command: Sequence[str] | None = None,
    required_gpu_name: str = "4080",
) -> LocalSmokeResult:
    config_path = Path(config_path).resolve()
    config = load_json_config(config_path)
    validate_experiment_config(config)
    if config["run_mode"] != "local_smoke" or config["paper_evidence"] is not False:
        raise ValueError("local smoke requires run_mode=local_smoke and paper_evidence=false")

    configured_limit = int(config["limits"]["sample_limit"])
    selected_limit = configured_limit if limit is None else int(limit)
    if selected_limit <= 0 or selected_limit > configured_limit:
        raise ValueError(f"limit must be in [1, {configured_limit}]")

    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = (config_path.parents[1] / data_root).resolve()
    storage = StorageLayout(data_root)
    storage.create()

    dataset_config = config["datasets"][0]
    split = str(dataset_config["splits"][0])
    dataset_spec = DatasetSpec(
        name=str(dataset_config["name"]),
        hf_id=str(dataset_config["hf_id"]),
        revision=str(dataset_config["revision"]),
        allowed_splits=tuple(str(item) for item in dataset_config["splits"]),
        config_name=dataset_config.get("config_name"),
        upstream_url=f"https://huggingface.co/datasets/{dataset_config['hf_id']}",
    )
    dataset = load_public_dataset(dataset_spec, split=split, storage=storage)
    if hasattr(dataset, "column_names") and hasattr(dataset, "select_columns"):
        safe_columns = [
            name for name in dataset.column_names if name in SAFE_DATASET_FIELDS
        ]
        if not safe_columns:
            raise ValueError("public dataset exposes no allowlisted diagnostic columns")
        dataset = dataset.select_columns(safe_columns)
    examples = _select_examples(dataset, limit=selected_limit, subset=subset)

    model_role = "edge"
    model_config = NativeGenerationConfig.from_dict(config["models"][model_role])
    executor = HFModelExecutor(model_config, storage)
    torch = executor.torch
    if not torch.cuda.is_available():
        raise RuntimeError("local native smoke requires CUDA; CPU fallback is not evidence")
    gpu_name = str(torch.cuda.get_device_name(0))
    if required_gpu_name and required_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"expected a GPU name containing {required_gpu_name!r}, found {gpu_name!r}"
        )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"local-smoke-{run_stamp}-{sha256_json(config)[:8]}"
    run_directory = storage.runs / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    outputs_path = run_directory / "outputs.jsonl"
    manifest_path = run_directory / "manifest.json"
    summary_path = run_directory / "summary.json"

    records: list[LocalSmokeRecord] = []
    for example in examples:
        trace_store = TraceStore()
        packet = build_diagnostic_packet(
            example,
            dataset_id=dataset_spec.hf_id,
            dataset_revision=dataset_spec.revision,
            split=split,
            trace_store=trace_store,
        )
        continuation = render_semantic_continuation(packet, style="concise_text")
        messages = build_diagnostic_messages(example, continuation=continuation)
        generation = executor.generate(messages)
        validation = PacketValidator(require_v2_graph=True).validate(
            packet, trace_store=trace_store
        )
        if not validation.valid:
            codes = ",".join(issue.code for issue in validation.issues)
            raise RuntimeError(f"diagnostic packet failed validation: {codes}")
        transaction, reversal = _run_protocol_probe(packet, trace_store)
        patch_node_ids = (
            tuple(node.node_id for node in transaction.patch_bundle.semantic_nodes)
            if transaction.patch_bundle is not None
            else ()
        )
        records.append(
            LocalSmokeRecord(
                run_id=run_id,
                dataset_id=dataset_spec.hf_id,
                dataset_revision=dataset_spec.revision,
                dataset_config_name=dataset_spec.config_name,
                split=split,
                sample_id=example.sample_id,
                source_subset=example.source_subset,
                visible_payload_hash=example.visible_payload_hash,
                packet_hash=packet.packet_hash,
                packet_bytes=len(packet.to_json().encode("utf-8")),
                packet_valid=True,
                packet_json=packet.to_json(),
                trace_spans=trace_store.subset(packet.trace_refs),
                semantic_schema_version=packet.schema_version,
                semantic_node_count=len(packet.semantic_nodes),
                obligation_count=len(packet.obligation_ids),
                dependency_depth=closure_depth(
                    packet.semantic_nodes,
                    packet.dependency_edges,
                    packet.obligation_ids,
                ),
                effect_frontier_hash=(
                    packet.effect_frontier.frontier_hash if packet.effect_frontier else ""
                ),
                protocol_patch_node_ids=patch_node_ids,
                protocol_transmitted_bytes=transaction.transmitted_bytes,
                protocol_valid=transaction.final_validation.valid,
                routing_reversal_fixture=reversal,
                baseline_manifest_hash=sha256_json(baseline_manifest()),
                prompt_hash=generation.prompt_hash,
                response_hash=generation.response_hash,
                response_text=generation.text,
                prompt_tokens=generation.prompt_tokens,
                output_tokens=generation.output_tokens,
                latency_ms=generation.latency_ms,
                peak_cuda_memory_bytes=generation.peak_cuda_memory_bytes,
                model_id=generation.model_id,
                model_revision=generation.model_revision,
                seed=generation.seed,
            )
        )

    _write_jsonl(outputs_path, records)
    hardware = collect_hardware_metadata()
    hardware.update(
        {
            "cuda_available": True,
            "cuda_runtime": str(torch.version.cuda),
            "gpu_name": gpu_name,
            "gpu_count": int(torch.cuda.device_count()),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "packages": collect_package_versions(),
        }
    )
    manifest = RunManifest.create(
        experiment_id=run_id,
        dataset_id=dataset_spec.hf_id,
        dataset_revision=dataset_spec.revision,
        split=split,
        sample_ids=tuple(record.sample_id for record in records),
        model_revisions={model_role: model_config.revision},
        seed=model_config.seed,
        prompt=DIAGNOSTIC_SYSTEM_PROMPT,
        config=config,
        command=tuple(command or sys.argv),
        purpose="local_diagnostic_only",
        labels_accessed=False,
        hardware=hardware,
    )
    manifest.write(manifest_path)
    summary = {
        "run_id": run_id,
        "diagnostic_only": True,
        "paper_evidence": False,
        "labels_accessed": False,
        "completed_records": len(records),
        "all_packets_valid": all(record.packet_valid for record in records),
        "all_protocol_probes_valid": all(record.protocol_valid for record in records),
        "semantic_schema_version": SCHEMA_VERSION,
        "implemented_baseline_count": len(baseline_manifest()),
        "routing_reversal_fixture_passed": all(
            record.routing_reversal_fixture for record in records
        ),
        "nonempty_responses": sum(bool(record.response_text.strip()) for record in records),
        "total_latency_ms": sum(record.latency_ms for record in records),
        "max_peak_cuda_memory_bytes": max(
            record.peak_cuda_memory_bytes for record in records
        ),
        "output_hash": sha256_json([record.to_dict() for record in records]),
        "manifest_hash": manifest.manifest_hash,
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return LocalSmokeResult(
        run_directory=run_directory,
        manifest_path=manifest_path,
        outputs_path=outputs_path,
        summary_path=summary_path,
        records=tuple(records),
    )
