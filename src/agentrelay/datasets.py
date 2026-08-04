"""Hugging Face dataset loading with pinned provenance and split checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import StorageLayout


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    revision: str
    allowed_splits: tuple[str, ...]
    config_name: str | None = None
    upstream_url: str = ""
    is_mirror: bool = False


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "agentprocessbench": DatasetSpec(
        name="agentprocessbench",
        hf_id="LulaCola/AgentProcessBench",
        revision="main",
        allowed_splits=("test",),
        config_name="bfcl",
        upstream_url="https://huggingface.co/datasets/LulaCola/AgentProcessBench",
    ),
    "gaia": DatasetSpec(
        name="gaia",
        hf_id="gaia-benchmark/GAIA",
        revision="main",
        allowed_splits=("validation", "test"),
        upstream_url="https://huggingface.co/datasets/gaia-benchmark/GAIA",
    ),
    "alfworld_raw_mirror": DatasetSpec(
        name="alfworld_raw_mirror",
        hf_id="awawa-agi/alfworld-raw",
        revision="main",
        allowed_splits=("train", "eval_in_distribution", "eval_out_of_distribution"),
        upstream_url="https://github.com/alfworld/alfworld",
        is_mirror=True,
    ),
}


def get_dataset_spec(name: str, *, revision: str | None = None) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset: {name}")
    spec = DATASET_REGISTRY[name]
    if revision is None:
        return spec
    return DatasetSpec(
        name=spec.name,
        hf_id=spec.hf_id,
        revision=revision,
        allowed_splits=spec.allowed_splits,
        config_name=spec.config_name,
        upstream_url=spec.upstream_url,
        is_mirror=spec.is_mirror,
    )


def load_public_dataset(
    spec: DatasetSpec,
    *,
    split: str,
    storage: StorageLayout,
    trust_remote_code: bool = False,
) -> Any:
    if split not in spec.allowed_splits:
        raise ValueError(f"split {split!r} is not allowed for {spec.name}")
    if not spec.revision or spec.revision == "main":
        raise ValueError(
            "formal runs require an immutable dataset revision; resolve and pass a commit hash"
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install AgentRelay with the 'ml' extra to load datasets") from exc
    storage.create()
    return load_dataset(
        spec.hf_id,
        spec.config_name,
        split=split,
        revision=spec.revision,
        cache_dir=str(storage.datasets),
        trust_remote_code=trust_remote_code,
    )


def dataset_provenance(spec: DatasetSpec, split: str) -> dict[str, Any]:
    return {
        "name": spec.name,
        "hf_id": spec.hf_id,
        "revision": spec.revision,
        "split": split,
        "config_name": spec.config_name,
        "upstream_url": spec.upstream_url,
        "is_mirror": spec.is_mirror,
    }


def sample_identifier(record: Mapping[str, Any], index: int) -> str:
    for key in (
        "id",
        "task_id",
        "sample_id",
        "source_path",
        "total_index",
        "query_index",
        "sample_index",
        "index",
    ):
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    return str(index)
