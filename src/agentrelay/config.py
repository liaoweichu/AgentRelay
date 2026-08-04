"""Storage and experiment configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


AUTODL_DEFAULT_ROOT_TEXT = "/root/autodl-tmp/AgentRelay"
AUTODL_DEFAULT_ROOT = Path(AUTODL_DEFAULT_ROOT_TEXT)
FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


@dataclass(frozen=True)
class StorageLayout:
    root: Path

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def create(self) -> None:
        for path in (self.root, self.datasets, self.models, self.runs, self.results):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, fallback: str | Path | None = None) -> "StorageLayout":
        configured = os.environ.get("AGENTRELAY_DATA_ROOT")
        if configured:
            return cls(Path(configured).expanduser().resolve())
        if AUTODL_DEFAULT_ROOT.parent.exists():
            return cls(AUTODL_DEFAULT_ROOT)
        if fallback is None:
            fallback = Path.cwd() / "artifacts" / "local-data"
        return cls(Path(fallback).expanduser().resolve())


def load_json_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a JSON object")
    return value


def require_keys(config: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"configuration is missing keys: {', '.join(missing)}")


def validate_experiment_config(
    config: Mapping[str, Any],
    *,
    allow_unlocked: bool = False,
) -> None:
    """Validate execution and integrity constraints before a run starts."""

    require_keys(
        config,
        (
            "schema_version",
            "run_mode",
            "paper_evidence",
            "data_root",
            "models",
            "datasets",
            "limits",
            "integrity",
        ),
    )
    if config["schema_version"] != "1.0":
        raise ValueError("unsupported experiment configuration schema")
    mode = str(config["run_mode"])
    if mode not in {"local_smoke", "formal_autodl"}:
        raise ValueError(f"unsupported run_mode: {mode}")
    integrity = config["integrity"]
    if not isinstance(integrity, Mapping):
        raise ValueError("integrity must be an object")
    required_integrity = {
        "allow_synthetic_tasks": False,
        "allow_test_label_access": False,
        "native_inference_only": True,
        "allow_prompt_answer_injection": False,
    }
    for key, expected in required_integrity.items():
        if integrity.get(key) is not expected:
            raise ValueError(f"integrity.{key} must be {expected!r}")

    models = config["models"]
    if not isinstance(models, Mapping) or not models:
        raise ValueError("models must be a non-empty object")
    for role, model in models.items():
        if not isinstance(model, Mapping):
            raise ValueError(f"models.{role} must be an object")
        require_keys(model, ("model_id", "revision"))
        if not allow_unlocked and not FULL_COMMIT_RE.fullmatch(str(model["revision"])):
            raise ValueError(f"models.{role}.revision is not a full immutable commit hash")

    datasets = config["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty array")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, Mapping):
            raise ValueError(f"datasets[{index}] must be an object")
        require_keys(dataset, ("name", "hf_id", "revision", "splits"))
        if not allow_unlocked and not FULL_COMMIT_RE.fullmatch(str(dataset["revision"])):
            raise ValueError(f"datasets[{index}].revision is not a full immutable commit hash")
        if not dataset["splits"]:
            raise ValueError(f"datasets[{index}].splits cannot be empty")

    repositories = config.get("repositories", [])
    if not isinstance(repositories, list):
        raise ValueError("repositories must be an array")
    for index, repository in enumerate(repositories):
        if not isinstance(repository, Mapping):
            raise ValueError(f"repositories[{index}] must be an object")
        require_keys(repository, ("name", "url", "revision"))
        if not allow_unlocked and not FULL_COMMIT_RE.fullmatch(str(repository["revision"])):
            raise ValueError(f"repositories[{index}].revision is not a full commit hash")

    limits = config["limits"]
    if not isinstance(limits, Mapping):
        raise ValueError("limits must be an object")
    if mode == "formal_autodl":
        if str(config["data_root"]).replace("\\", "/") != AUTODL_DEFAULT_ROOT_TEXT:
            raise ValueError("formal runs must use /root/autodl-tmp/AgentRelay")
        if limits.get("sample_limit") is not None:
            raise ValueError("formal runs cannot set a sample limit")
        if config["paper_evidence"] is not True:
            raise ValueError("formal_autodl must explicitly mark paper_evidence=true")
    else:
        if config["paper_evidence"] is not False:
            raise ValueError("local smoke runs cannot be marked as paper evidence")
