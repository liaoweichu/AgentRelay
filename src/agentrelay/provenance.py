"""Run provenance and official-split discipline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

from .schema import canonical_json, sha256_json, sha256_text


@dataclass(frozen=True)
class SplitPolicy:
    train_splits: tuple[str, ...]
    validation_splits: tuple[str, ...]
    test_splits: tuple[str, ...]

    def validate(self, *, split: str, purpose: str, labels_accessed: bool) -> None:
        if purpose == "train" and split not in self.train_splits:
            raise ValueError(f"split {split!r} is not authorized for training")
        if purpose == "tune" and split not in self.validation_splits:
            raise ValueError(f"split {split!r} is not authorized for tuning")
        if purpose == "evaluate" and split not in self.test_splits + self.validation_splits:
            raise ValueError(f"split {split!r} is not an evaluation split")
        if split in self.test_splits and labels_accessed:
            raise ValueError("test labels/evaluator internals cannot be accessed by the runtime")


@dataclass(frozen=True)
class RunManifest:
    experiment_id: str
    dataset_id: str
    dataset_revision: str
    split: str
    sample_ids: tuple[str, ...]
    model_ids: Mapping[str, str]
    model_revisions: Mapping[str, str]
    seed: int
    prompt_hash: str
    config: Mapping[str, Any]
    command: tuple[str, ...]
    hardware: Mapping[str, Any]
    code_revision: str
    started_at: str
    purpose: str
    labels_accessed: bool = False
    schema_version: str = "1.0"

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        dataset_id: str,
        dataset_revision: str,
        split: str,
        sample_ids: tuple[str, ...],
        model_ids: Mapping[str, str],
        model_revisions: Mapping[str, str],
        seed: int,
        prompt: str,
        config: Mapping[str, Any],
        command: tuple[str, ...],
        purpose: str,
        labels_accessed: bool = False,
        hardware: Mapping[str, Any] | None = None,
    ) -> "RunManifest":
        return cls(
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            split=split,
            sample_ids=sample_ids,
            model_ids=dict(model_ids),
            model_revisions=dict(model_revisions),
            seed=seed,
            prompt_hash=sha256_text(prompt),
            config=dict(config),
            command=command,
            hardware=dict(hardware or collect_hardware_metadata()),
            code_revision=git_revision(),
            started_at=datetime.now(timezone.utc).isoformat(),
            purpose=purpose,
            labels_accessed=labels_accessed,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def manifest_hash(self) -> str:
        return sha256_json(self.to_dict())

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["manifest_hash"] = self.manifest_hash
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        temporary.replace(target)


def git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        revision = result.stdout.strip()
        return revision if revision else source_tree_revision()
    except (OSError, subprocess.SubprocessError):
        return source_tree_revision()


def source_tree_revision(root: str | Path | None = None) -> str:
    """Hash runtime source when the exported project has no Git metadata."""

    project_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    candidates: set[Path] = set()
    for pattern in ("pyproject.toml", "src/**/*.py", "scripts/**/*.py"):
        candidates.update(path for path in project_root.glob(pattern) if path.is_file())
    if not candidates:
        return "unavailable"
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(project_root).as_posix()):
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"tree-sha256:{digest.hexdigest()}"


def collect_package_versions(
    packages: tuple[str, ...] = (
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "huggingface-hub",
        "safetensors",
    ),
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def collect_hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        metadata["cuda_visible_devices"] = visible
    return metadata


def validate_manifest_hash(value: Mapping[str, Any]) -> bool:
    payload = dict(value)
    expected = str(payload.pop("manifest_hash", ""))
    return bool(expected) and sha256_json(payload) == expected
