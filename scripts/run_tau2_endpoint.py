#!/usr/bin/env python3
"""Run one Gemma endpoint over pinned tau2 tasks with strict resume/provenance."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import (
    GEMMA4_FORMAL_MODEL_PAIR,
    StorageLayout,
    load_json_config,
    validate_experiment_config,
)
from agentrelay.inference import HFModelExecutor, NativeGenerationConfig
from agentrelay.provenance import source_tree_revision
from agentrelay.schema import canonical_json, sha256_json, sha256_text
from agentrelay.tau2_adapter import (
    TAU2_DOMAINS,
    TAU2_PINNED_REVISION,
    Tau2Adapter,
    Tau2TaskRef,
    Tau2UserSimulatorConfig,
    collect_resumable_episodes,
    load_resumable_episode,
    prepare_run_context,
    read_tau2_manifest,
    save_resumable_episode,
    verify_tau2_repository,
)


def _read_profile(path: str | Path, models: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded_hash = str(value.pop("profile_hash", ""))
    if not recorded_hash or recorded_hash != sha256_json(value):
        raise ValueError("hardware profile hash mismatch")
    bandwidth = value.get("bandwidth_mbps")
    if value.get("network_trace") is None or bandwidth is None or float(bandwidth) <= 0:
        raise ValueError("tau2 formal endpoints require a measured network profile")
    for role in ("edge", "cloud"):
        measured = value.get(role)
        expected = models[role]
        if not isinstance(measured, Mapping):
            raise TypeError(f"hardware profile is missing {role}")
        checks = {
            "model_id": expected["model_id"],
            "model_revision": expected["revision"],
            "model_source": expected["model_source"],
            "dtype": expected["dtype"],
            "quantization": expected["quantization"],
        }
        for field, expected_value in checks.items():
            if str(measured.get(field)) != str(expected_value):
                raise ValueError(
                    f"hardware profile {role}.{field} mismatch: "
                    f"{measured.get(field)!r} != {expected_value!r}"
                )
    return {
        "edge_inference_ms": float(value["edge"]["inference_ms"]),
        "cloud_inference_ms": float(value["cloud"]["inference_ms"]),
        "bandwidth_mbps": float(bandwidth),
        "profile_hash": recorded_hash,
    }


def _limit_per_domain(tasks: tuple[Tau2TaskRef, ...], limit: int | None) -> tuple[Tau2TaskRef, ...]:
    if limit is None:
        return tasks
    if limit <= 0:
        raise ValueError("tasks-per-domain must be positive")
    selected = []
    for domain in TAU2_DOMAINS:
        domain_tasks = [task for task in tasks if task.domain == domain]
        selected.extend(domain_tasks[:limit])
    if {task.domain for task in selected} != set(TAU2_DOMAINS):
        raise ValueError("three-domain tau2 selection is incomplete")
    return tuple(selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("profile")
    parser.add_argument("task_manifest")
    parser.add_argument("tau2_repository")
    parser.add_argument("user_config")
    parser.add_argument("output_dir")
    parser.add_argument("--role", choices=("edge", "cloud"), required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--tasks-per-domain", type=int)
    parser.add_argument("--as-smoke", action="store_true")
    parser.add_argument(
        "--edge-precision",
        choices=("formal", "fp16"),
        default="formal",
        help="fp16 is a diagnostic-only E4B sensitivity arm",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.edge_precision == "fp16" and args.role != "edge":
        raise ValueError("the fp16 sensitivity arm is defined only for E4B")
    if args.edge_precision == "fp16" and (not args.as_smoke or args.tasks_per_domain is None):
        raise ValueError("E4B FP16 is diagnostic-only and requires --as-smoke with a task limit")

    config_path = Path(args.config).resolve()
    config = load_json_config(config_path)
    validate_experiment_config(config)
    model_ids = {role: value["model_id"] for role, value in config["models"].items()}
    if model_ids != GEMMA4_FORMAL_MODEL_PAIR:
        raise ValueError("tau2 endpoint runner requires Gemma 4 E4B/12B")
    verify_tau2_repository(args.tau2_repository)
    user = Tau2UserSimulatorConfig.load(
        args.user_config,
        require_secret=not args.validate_only,
    )
    profile = _read_profile(args.profile, config["models"])
    tasks = _limit_per_domain(
        read_tau2_manifest(args.task_manifest, split=args.split),
        args.tasks_per_domain,
    )
    if args.as_smoke:
        tasks = tuple(replace(task, split="smoke") for task in tasks)
    manifest_document = json.loads(Path(args.task_manifest).read_text(encoding="utf-8"))
    task_manifest_hash = str(manifest_document["manifest_hash"])
    model_config = NativeGenerationConfig.from_dict(config["models"][args.role])
    precision_label = "formal_bnb4bit"
    if args.edge_precision == "fp16":
        model_config = replace(model_config, dtype="float16", quantization="none")
        model_config.validate()
        precision_label = "e4b_fp16_diagnostic"
    context = {
        "schema_version": "1.0",
        "scope": "tau2_fixed_endpoint",
        "code_revision": source_tree_revision(PROJECT_ROOT),
        "config_hash": sha256_json(config),
        "config_path": str(config_path),
        "profile_hash": profile["profile_hash"],
        "task_manifest_hash": task_manifest_hash,
        "task_selection_hash": sha256_json([task.__dict__ for task in tasks]),
        "tau2_revision": TAU2_PINNED_REVISION,
        "user_config_hash": sha256_text(Path(args.user_config).read_text(encoding="utf-8")),
        "role": args.role,
        "split": "smoke" if args.as_smoke else args.split,
        "precision_label": precision_label,
        "model": {
            "model_id": model_config.model_id,
            "revision": model_config.revision,
            "model_source": model_config.model_source,
            "dtype": model_config.dtype,
            "quantization": model_config.quantization,
        },
        "paper_evidence": False,
        "labels_accessed_by_router": False,
    }
    document, context_hash = prepare_run_context(args.output_dir, context)
    if args.validate_only:
        print(
            f"validated role={args.role} split={context['split']} tasks={len(tasks)} "
            f"context={context_hash}"
        )
        return 0

    storage = StorageLayout(Path(config["data_root"]).resolve())
    executor = HFModelExecutor(model_config, storage)
    adapter = Tau2Adapter(args.tau2_repository, user)
    run_id = f"tau2-{context['split']}-{args.role}-{context_hash[:12]}"
    endpoint_provenance = {
        "run_id": run_id,
        "manifest_hash": context_hash,
        "code_revision": context["code_revision"],
        "config_hash": context["config_hash"],
        "profile_hash": context["profile_hash"],
        "model_ids": model_ids,
        "model_revisions": {
            role: value["revision"] for role, value in config["models"].items()
        },
        "model_precisions": {
            role: {
                "model_source": value["model_source"],
                "dtype": value["dtype"],
                "quantization": value["quantization"],
            }
            for role, value in config["models"].items()
        },
        "active_precision": context["model"],
        "task_manifest_hash": task_manifest_hash,
        "tau2_revision": TAU2_PINNED_REVISION,
        "user_config_hash": context["user_config_hash"],
    }
    completed = 0
    for task in tasks:
        existing = load_resumable_episode(
            args.output_dir,
            task,
            context_hash=context_hash,
            role=args.role,
        )
        if existing is not None:
            completed += 1
            print(f"resume {completed}/{len(tasks)} {task.domain}/{task.task_id}")
            continue
        episode = adapter.run(
            task,
            executor=executor,
            role=args.role,
            profile=profile,
            endpoint_provenance=endpoint_provenance,
        )
        save_resumable_episode(args.output_dir, task, episode)
        completed += 1
        print(
            f"run {completed}/{len(tasks)} {task.domain}/{task.task_id} "
            f"reward={episode['reward']:.3f}"
        )
    episodes = collect_resumable_episodes(
        args.output_dir,
        tasks,
        context_hash=context_hash,
        role=args.role,
    )
    artifact = {
        "run_context": document,
        "rows": list(episodes),
        "rows_hash": sha256_json(episodes),
    }
    target = Path(args.output_dir).resolve() / "episodes.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(f"completed={len(episodes)} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
