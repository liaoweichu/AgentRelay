#!/usr/bin/env python3
"""Profile pinned native models on an allowlisted public trajectory prefix."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import median
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import StorageLayout, load_json_config, validate_experiment_config  # noqa: E402
from agentrelay.continuation import render_semantic_continuation  # noqa: E402
from agentrelay.cost import RecordedBandwidthTrace  # noqa: E402
from agentrelay.datasets import DatasetSpec, load_public_dataset  # noqa: E402
from agentrelay.inference import HFModelExecutor, NativeGenerationConfig  # noqa: E402
from agentrelay.local_smoke import (  # noqa: E402
    SAFE_DATASET_FIELDS,
    build_diagnostic_messages,
    build_diagnostic_packet,
    process_example_from_record,
)
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402
from agentrelay.state import TraceStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--network-trace")
    parser.add_argument("--rate-column")
    parser.add_argument("--sample-period-ms", type=float)
    parser.add_argument("--trace-source")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    config_path = Path(args.config).resolve()
    config = load_json_config(config_path)
    validate_experiment_config(config)
    storage = StorageLayout(Path(config["data_root"]).resolve())
    dataset_config = config["datasets"][0]
    spec = DatasetSpec(
        name=str(dataset_config["name"]),
        hf_id=str(dataset_config["hf_id"]),
        revision=str(dataset_config["revision"]),
        allowed_splits=tuple(str(item) for item in dataset_config["splits"]),
        config_name=dataset_config.get("config_name"),
    )
    split = str(dataset_config["splits"][0])
    dataset = load_public_dataset(spec, split=split, storage=storage)
    if hasattr(dataset, "column_names") and hasattr(dataset, "select_columns"):
        safe_columns = [name for name in dataset.column_names if name in SAFE_DATASET_FIELDS]
        dataset = dataset.select_columns(safe_columns)
    example = None
    for index, record in enumerate(dataset):
        example = process_example_from_record(record, index=index)
        break
    if example is None:
        raise ValueError("public profile dataset contains no usable row")
    trace_store = TraceStore()
    packet = build_diagnostic_packet(
        example,
        dataset_id=spec.hf_id,
        dataset_revision=spec.revision,
        split=split,
        trace_store=trace_store,
    )
    messages = build_diagnostic_messages(
        example,
        continuation=render_semantic_continuation(packet, style="concise_text"),
    )
    report: dict[str, object] = {}
    for role, model_value in config["models"].items():
        model_config = NativeGenerationConfig.from_dict(model_value)
        load_started = time.perf_counter()
        executor = HFModelExecutor(model_config, storage)
        rehydration_ms = (time.perf_counter() - load_started) * 1000.0
        executor.generate(messages)  # declared warmup, never included in the profile median
        values = [executor.generate(messages) for _ in range(args.repeats)]
        report[str(role)] = {
            "model_id": model_config.model_id,
            "model_revision": model_config.revision,
            "inference_ms": median(value.latency_ms for value in values),
            "rehydration_ms": rehydration_ms,
            "prompt_tokens": values[0].prompt_tokens,
            "output_tokens_median": median(value.output_tokens for value in values),
            "peak_cuda_memory_bytes": max(value.peak_cuda_memory_bytes for value in values),
            "controller_ms": 0.0,
        }
        torch = executor.torch
        del values
        del executor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.network_trace:
        if not args.rate_column or not args.sample_period_ms or not args.trace_source:
            raise ValueError(
                "network trace requires --rate-column, --sample-period-ms, and --trace-source"
            )
        trace = RecordedBandwidthTrace.from_csv(
            args.network_trace,
            rate_column=args.rate_column,
            sample_period_ms=args.sample_period_ms,
            source=args.trace_source,
        )
        report["bandwidth_mbps"] = median(trace.samples_mbps)
        report["network_trace"] = {
            "source": trace.source,
            "path": str(Path(args.network_trace).resolve()),
            "file_hash": sha256_json(Path(args.network_trace).read_text(encoding="utf-8")),
            "sample_count": len(trace.samples_mbps),
            "sample_period_ms": trace.sample_period_ms,
        }
    else:
        report["bandwidth_mbps"] = None
        report["network_trace"] = None
    report.update(
        {
            "diagnostic_only": True,
            "paper_evidence": False,
            "labels_accessed": False,
            "dataset_id": spec.hf_id,
            "dataset_revision": spec.revision,
            "split": split,
            "sample_id": example.sample_id,
            "visible_payload_hash": example.visible_payload_hash,
        }
    )
    report["profile_hash"] = sha256_json(report)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(canonical_json(report) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

