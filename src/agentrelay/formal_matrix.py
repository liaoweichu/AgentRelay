"""Formal AutoDL matrix orchestration over pinned official task manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .baselines import BASELINE_REGISTRY, BaselineController, BaselineName
from .calibration import ConformalRiskCalibrator
from .config import StorageLayout, validate_experiment_config
from .experiment_runtime import (
    EpisodeRunner,
    LearnedEstimateProvider,
    ProfileEstimateProvider,
    ServiceProfile,
)
from .inference import HFModelExecutor, NativeGenerationConfig
from .learning import JointRouterEstimator
from .official_adapters import ALFWorldAdapter, AppWorldAdapter, WebShopAdapter
from .policy import ConstrainedUtilityPolicy
from .provenance import collect_hardware_metadata, source_tree_revision
from .schema import Executor, canonical_json, sha256_json


SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class FormalTask:
    benchmark: str
    split: str
    task_id: str
    purpose: str
    task_index: int | None = None
    train_eval: str = ""
    alfworld_config: str = ""
    webshop_file_path: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalTask":
        return cls(
            benchmark=str(value["benchmark"]),
            split=str(value["split"]),
            task_id=str(value["task_id"]),
            purpose=str(value["purpose"]),
            task_index=(int(value["task_index"]) if value.get("task_index") is not None else None),
            train_eval=str(value.get("train_eval", "")),
            alfworld_config=str(value.get("alfworld_config", "")),
            webshop_file_path=str(value.get("webshop_file_path", "")),
        )


@dataclass(frozen=True)
class OfficialTaskManifest:
    dataset_revision: str
    complete_official_split: bool
    tasks: tuple[FormalTask, ...]
    manifest_hash: str

    @classmethod
    def load(cls, path: str | Path) -> "OfficialTaskManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("task manifest must be a JSON object")
        payload = dict(value)
        expected_hash = str(payload.pop("manifest_hash", ""))
        if not expected_hash or sha256_json(payload) != expected_hash:
            raise ValueError("task manifest hash mismatch")
        tasks = tuple(FormalTask.from_dict(item) for item in value.get("tasks", ()))
        if not tasks:
            raise ValueError("task manifest cannot be empty")
        identities = {(task.benchmark, task.split, task.task_id) for task in tasks}
        if len(identities) != len(tasks):
            raise ValueError("task manifest contains duplicate task identities")
        return cls(
            dataset_revision=str(value["dataset_revision"]),
            complete_official_split=bool(value.get("complete_official_split", False)),
            tasks=tasks,
            manifest_hash=expected_hash,
        )


def write_task_manifest(
    path: str | Path,
    *,
    dataset_revision: str,
    tasks: Iterable[FormalTask],
    complete_official_split: bool,
) -> Path:
    tasks = tuple(tasks)
    payload = {
        "schema_version": "1.0",
        "dataset_revision": dataset_revision,
        "complete_official_split": complete_official_split,
        "tasks": [task.__dict__ for task in tasks],
    }
    payload["manifest_hash"] = sha256_json(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def _profile_provider(
    path: str | Path,
    *,
    expected_models: Mapping[str, Mapping[str, Any]],
) -> ProfileEstimateProvider:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded_hash = str(value.pop("profile_hash", ""))
    if not recorded_hash or recorded_hash != sha256_json(value):
        raise ValueError("hardware profile hash mismatch")
    if value.get("network_trace") is None or value.get("bandwidth_mbps") is None:
        raise ValueError(
            "formal runs require a real recorded network trace; "
            "a null/default bandwidth is not admissible"
        )
    bandwidth_mbps = float(value["bandwidth_mbps"])
    if bandwidth_mbps <= 0:
        raise ValueError("measured bandwidth_mbps must be positive")
    for executor in (Executor.EDGE, Executor.CLOUD):
        role = executor.value
        measured = value.get(role)
        expected = expected_models.get(role)
        if not isinstance(measured, Mapping) or not isinstance(expected, Mapping):
            raise ValueError(f"hardware profile is missing the {role} model")
        for field in ("model_id", "revision"):
            expected_value = str(expected[field])
            measured_value = str(
                measured["model_revision"] if field == "revision" else measured[field]
            )
            if measured_value != expected_value:
                raise ValueError(
                    f"hardware profile {role}.{field} mismatch: "
                    f"{measured_value!r} != {expected_value!r}"
                )
    profiles = {
        executor: ServiceProfile(
            inference_ms=float(value[executor.value]["inference_ms"]),
            predicted_success=0.5,
            predicted_fidelity=0.5,
            controller_ms=float(value[executor.value].get("controller_ms", 0.1)),
            rehydration_ms=float(value[executor.value].get("rehydration_ms", 0.0)),
        )
        for executor in (Executor.EDGE, Executor.CLOUD)
    }
    return ProfileEstimateProvider(
        profiles,
        bandwidth_mbps=bandwidth_mbps,
    )


class FormalMatrixRunner:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        task_manifest: OfficialTaskManifest,
        methods: Iterable[BaselineName | str],
        profile_path: str | Path,
        router_path: str | Path | None = None,
        calibrator_path: str | Path | None = None,
    ) -> None:
        validate_experiment_config(config)
        if config["run_mode"] != "formal_autodl" or config["paper_evidence"] is not True:
            raise ValueError("formal matrix requires the locked formal_autodl configuration")
        self.config = dict(config)
        self.task_manifest = task_manifest
        self.methods = tuple(BaselineName(method) for method in methods)
        if not self.methods:
            raise ValueError("at least one method is required")
        if any(task.purpose == "evaluate" for task in task_manifest.tasks):
            if not task_manifest.complete_official_split:
                raise ValueError("evaluation manifests must attest the complete official split")
        self.profile_provider = _profile_provider(
            profile_path,
            expected_models=config["models"],
        )
        self.estimator = JointRouterEstimator.load(router_path) if router_path else None
        self.calibrator = (
            ConformalRiskCalibrator.load(calibrator_path) if calibrator_path else None
        )
        for method in self.methods:
            spec = BASELINE_REGISTRY[method]
            if spec.requires_router and self.estimator is None:
                raise ValueError(f"method {method.value} requires a fitted train-split router")
            if method is BaselineName.AGENTRELAY and self.calibrator is None:
                raise ValueError("AgentRelay requires a validation-only calibration artifact")
        storage = StorageLayout(Path(config["data_root"]).resolve())
        self.storage = storage
        self.executors = {
            role: HFModelExecutor(NativeGenerationConfig.from_dict(model), storage)
            for role, model in (
                (Executor.EDGE, config["models"]["edge"]),
                (Executor.CLOUD, config["models"]["cloud"]),
            )
        }

    def _adapter(self, task: FormalTask, run_name: str) -> Any:
        revision = self.task_manifest.dataset_revision
        if task.benchmark == "alfworld":
            if task.task_index is None or not task.alfworld_config or not task.train_eval:
                raise ValueError("ALFWorld tasks require task_index, train_eval, and config path")
            return ALFWorldAdapter(
                config_path=task.alfworld_config,
                train_eval=task.train_eval,
                dataset_revision=revision,
                split=task.split,
                task_index=task.task_index,
            )
        if task.benchmark == "webshop":
            if task.task_index is None:
                raise ValueError("WebShop tasks require an official session index")
            return WebShopAdapter(
                session=task.task_index,
                dataset_revision=revision,
                split=task.split,
                file_path=task.webshop_file_path or None,
            )
        if task.benchmark == "appworld":
            return AppWorldAdapter(
                task_id=task.task_id,
                dataset_revision=revision,
                split=task.split,
                experiment_name=run_name,
                allow_official_evaluation=task.purpose == "evaluate",
            )
        raise ValueError(f"unsupported official benchmark {task.benchmark!r}")

    def _provider(self, method: BaselineName) -> Any:
        if not BASELINE_REGISTRY[method].requires_router:
            return self.profile_provider
        assert self.estimator is not None
        profiles = self.profile_provider.profiles
        return LearnedEstimateProvider(
            self.estimator,
            edge_warm_ms=profiles[Executor.EDGE].inference_ms,
            cloud_warm_ms=profiles[Executor.CLOUD].inference_ms,
            bandwidth_mbps=self.profile_provider.bandwidth_mbps,
            calibrator=self.calibrator if method is BaselineName.AGENTRELAY else None,
        )

    def run(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"formal-matrix-{stamp}-{self.task_manifest.manifest_hash[:8]}"
        run_root = self.storage.runs / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        written: list[dict[str, Any]] = []
        for task in self.task_manifest.tasks:
            for method in self.methods:
                run_name = f"{run_id}-{task.benchmark}-{task.split}-{method.value}"
                adapter = self._adapter(task, run_name)
                try:
                    runner = EpisodeRunner(
                        adapter=adapter,
                        executors=self.executors,
                        controller=BaselineController(
                            method,
                            policy=ConstrainedUtilityPolicy(switch_hysteresis_ms=5.0),
                        ),
                        estimate_provider=self._provider(method),
                        max_steps=int(self.config["benchmarks"][task.benchmark]["max_steps"]),
                        minimum_dwell_steps=int(
                            self.config.get("controller", {}).get("minimum_dwell_steps", 2)
                        ),
                        paper_evidence=True,
                    )
                    result = runner.run()
                finally:
                    adapter.close()
                safe_task = SAFE_ID_RE.sub("_", task.task_id)[:120]
                relative = Path(task.benchmark) / task.split / method.value / f"{safe_task}.json"
                target = run_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = result.to_dict()
                payload["result_hash"] = sha256_json(payload)
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
                temporary.replace(target)
                written.append(
                    {
                        "path": relative.as_posix(),
                        "result_hash": payload["result_hash"],
                        "sample_id": result.sample_id,
                        "method": method.value,
                    }
                )
        manifest = {
            "run_id": run_id,
            "paper_evidence": True,
            "task_manifest_hash": self.task_manifest.manifest_hash,
            "code_revision": source_tree_revision(),
            "model_revisions": {
                role: str(value["revision"])
                for role, value in self.config["models"].items()
            },
            "methods": [method.value for method in self.methods],
            "results": written,
            "hardware": collect_hardware_metadata(),
            "labels_accessed_by_router": False,
        }
        manifest["manifest_hash"] = sha256_json(manifest)
        (run_root / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        return run_root
