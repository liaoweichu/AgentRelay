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
from .webshop_protocol import canonical_webshop_split, validate_webshop_manifest_indices


SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def task_artifact_scope(tasks: Iterable[FormalTask]) -> tuple[bool, str]:
    """Classify output evidence without promoting train/dev diagnostics."""

    purposes = {task.purpose for task in tasks}
    if purposes == {"evaluate"}:
        return True, "official_evaluation"
    if purposes <= {"train", "tune"}:
        return False, "train_dev_development"
    raise ValueError(f"a run cannot mix evaluation and development purposes: {purposes}")


def required_model_executors(methods: Iterable[BaselineName | str]) -> tuple[Executor, ...]:
    """Load one model for a single fixed endpoint; all routers require both."""

    names = tuple(BaselineName(method) for method in methods)
    if names and set(names) == {BaselineName.EDGE_ONLY}:
        return (Executor.EDGE,)
    if names and set(names) == {BaselineName.CLOUD_ONLY}:
        return (Executor.CLOUD,)
    return (Executor.EDGE, Executor.CLOUD)


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
        webshop_groups: dict[str, list[FormalTask]] = {}
        for task in tasks:
            if task.benchmark == "webshop":
                split = canonical_webshop_split(task.split)
                if split != task.split:
                    raise ValueError(
                        f"WebShop manifest must use canonical split {split!r}, got {task.split!r}"
                    )
                if task.task_index is None or str(task.task_index) != task.task_id:
                    raise ValueError(
                        "WebShop task_id and task_index must be the same official session id"
                    )
                expected_purpose = {"train": "train", "dev": "tune", "test": "evaluate"}[
                    split
                ]
                if task.purpose != expected_purpose:
                    raise ValueError(
                        f"WebShop {split} tasks require purpose={expected_purpose!r}, "
                        f"got {task.purpose!r}"
                    )
                webshop_groups.setdefault(split, []).append(task)
        for split, split_tasks in webshop_groups.items():
            validate_webshop_manifest_indices(
                split,
                (int(task.task_index) for task in split_tasks if task.task_index is not None),
                complete_official_split=bool(value.get("complete_official_split", False)),
            )
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
    webshop_groups: dict[str, list[FormalTask]] = {}
    for task in tasks:
        if task.benchmark == "webshop":
            webshop_groups.setdefault(canonical_webshop_split(task.split), []).append(task)
    for split, split_tasks in webshop_groups.items():
        validate_webshop_manifest_indices(
            split,
            (int(task.task_index) for task in split_tasks if task.task_index is not None),
            complete_official_split=complete_official_split,
        )
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
) -> tuple[ProfileEstimateProvider, str]:
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
    return (
        ProfileEstimateProvider(
            profiles,
            bandwidth_mbps=bandwidth_mbps,
        ),
        recorded_hash,
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
        self.profile_provider, self.profile_hash = _profile_provider(
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
        resident = required_model_executors(self.methods)
        self.executors = {
            role: HFModelExecutor(NativeGenerationConfig.from_dict(model), storage)
            for role, model in (
                (Executor.EDGE, config["models"]["edge"]),
                (Executor.CLOUD, config["models"]["cloud"]),
            )
            if role in resident
        }
        # Shared official benchmark environments (WebShop full corpus) reused
        # across every (task, method) episode to avoid rebuilding the product
        # corpus and search engine per episode.  Pure performance optimization.
        self._shared_envs: dict[str, Any] = {}

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
            file_path = task.webshop_file_path or None
            key = file_path or "default"
            if key not in self._shared_envs:
                try:
                    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
                except ImportError as exc:
                    raise RuntimeError(
                        "install the pinned official WebShop environment"
                    ) from exc
                env_kwargs: dict[str, Any] = {
                    "observation_mode": "text",
                    "human_goals": 1,
                }
                if file_path is not None:
                    env_kwargs["file_path"] = file_path
                self._shared_envs[key] = WebAgentTextEnv(**env_kwargs)
            return WebShopAdapter(
                session=task.task_index,
                dataset_revision=revision,
                split=task.split,
                file_path=file_path,
                env=self._shared_envs[key],
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

    def run(self, *, resume_key: str | None = None) -> Path:
        if resume_key is not None:
            safe_resume_key = SAFE_ID_RE.sub("_", resume_key)
            if not safe_resume_key or safe_resume_key != resume_key:
                raise ValueError("resume_key must contain only safe id characters")
            run_id = f"formal-matrix-{resume_key}-{self.task_manifest.manifest_hash[:8]}"
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_id = f"formal-matrix-{stamp}-{self.task_manifest.manifest_hash[:8]}"
        run_root = (self.storage.runs / run_id).resolve()
        if run_root.parent != self.storage.runs.resolve():
            raise ValueError("formal run root escaped the configured run store")
        paper_evidence, artifact_scope = task_artifact_scope(self.task_manifest.tasks)
        code_revision = source_tree_revision()
        context = {
            "schema_version": "1.0",
            "run_id": run_id,
            "task_manifest_hash": self.task_manifest.manifest_hash,
            "code_revision": code_revision,
            "config_hash": sha256_json(self.config),
            "profile_hash": self.profile_hash,
            "model_ids": {
                role: str(value["model_id"])
                for role, value in self.config["models"].items()
            },
            "model_revisions": {
                role: str(value["revision"])
                for role, value in self.config["models"].items()
            },
            "methods": [method.value for method in self.methods],
            "resident_executors": [executor.value for executor in self.executors],
            "paper_evidence": paper_evidence,
            "artifact_scope": artifact_scope,
        }
        context["context_hash"] = sha256_json(context)
        context_path = run_root / "run-context.json"
        if run_root.exists():
            if resume_key is None:
                raise FileExistsError(run_root)
            if not context_path.is_file():
                raise ValueError(f"resume run has no immutable context: {run_root}")
            recorded_context = json.loads(context_path.read_text(encoding="utf-8"))
            if recorded_context != context:
                raise ValueError("resume context does not match current code/config/manifest")
        else:
            run_root.mkdir(parents=True, exist_ok=False)
            temporary_context = context_path.with_suffix(".json.tmp")
            temporary_context.write_text(
                canonical_json(context) + "\n",
                encoding="utf-8",
            )
            temporary_context.replace(context_path)
        written: list[dict[str, Any]] = []
        total = len(self.task_manifest.tasks) * len(self.methods)
        try:
            for task in self.task_manifest.tasks:
                for method in self.methods:
                    safe_task = SAFE_ID_RE.sub("_", task.task_id)[:120]
                    relative = (
                        Path(task.benchmark)
                        / task.split
                        / method.value
                        / f"{safe_task}.json"
                    )
                    target = run_root / relative
                    if target.is_file():
                        payload = json.loads(target.read_text(encoding="utf-8"))
                        recorded_hash = str(payload.pop("result_hash", ""))
                        if not recorded_hash or sha256_json(payload) != recorded_hash:
                            raise ValueError(f"resume episode hash mismatch: {target}")
                        expected_identity = {
                            "benchmark": task.benchmark,
                            "dataset_revision": self.task_manifest.dataset_revision,
                            "split": task.split,
                            "sample_id": task.task_id,
                            "method": method.value,
                            "paper_evidence": paper_evidence,
                        }
                        if any(payload.get(key) != value for key, value in expected_identity.items()):
                            raise ValueError(f"resume episode identity mismatch: {target}")
                        written.append(
                            {
                                "path": relative.as_posix(),
                                "result_hash": recorded_hash,
                                "sample_id": task.task_id,
                                "method": method.value,
                            }
                        )
                        print(
                            f"episode_resumed={len(written)}/{total} "
                            f"sample_id={task.task_id} method={method.value}",
                            flush=True,
                        )
                        continue
                    run_name = f"{run_id}-{task.benchmark}-{task.split}-{method.value}"
                    adapter = self._adapter(task, run_name)
                    try:
                        runner = EpisodeRunner(
                            adapter=adapter,
                            executors=self.executors,
                            controller=BaselineController(
                                method,
                                policy=ConstrainedUtilityPolicy(
                                    switch_hysteresis_ms=float(
                                        self.config.get("controller", {}).get(
                                            "switch_hysteresis_ms", 5.0
                                        )
                                    ),
                                    success_weight=float(
                                        self.config.get("controller", {}).get(
                                            "reward_weight_ms", 10_000.0
                                        )
                                    ),
                                    selection_mode=(
                                        "scalarized_utility"
                                        if method
                                        in {
                                            BaselineName.UNCALIBRATED_JOINT,
                                            BaselineName.AGENTRELAY,
                                        }
                                        else "constrained_cost"
                                    ),
                                ),
                            ),
                            estimate_provider=self._provider(method),
                            max_steps=int(
                                self.config["benchmarks"][task.benchmark]["max_steps"]
                            ),
                            minimum_dwell_steps=int(
                                self.config.get("controller", {}).get(
                                    "minimum_dwell_steps", 2
                                )
                            ),
                            max_action_retries=int(
                                self.config.get("controller", {}).get(
                                    "max_action_retries", 1
                                )
                            ),
                            paper_evidence=paper_evidence,
                        )
                        result = runner.run()
                    finally:
                        adapter.close()
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
                    print(
                        f"episode_completed={len(written)}/{total} "
                        f"sample_id={result.sample_id} method={method.value}",
                        flush=True,
                    )
        finally:
            for shared_env in self._shared_envs.values():
                try:
                    shared_env.close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
        manifest = {
            "run_id": run_id,
            "paper_evidence": paper_evidence,
            "artifact_scope": artifact_scope,
            "task_purposes": sorted(
                {task.purpose for task in self.task_manifest.tasks}
            ),
            "task_manifest_hash": self.task_manifest.manifest_hash,
            "code_revision": code_revision,
            "config_hash": context["config_hash"],
            "profile_hash": self.profile_hash,
            "model_ids": {
                role: str(value["model_id"])
                for role, value in self.config["models"].items()
            },
            "model_revisions": {
                role: str(value["revision"])
                for role, value in self.config["models"].items()
            },
            "methods": [method.value for method in self.methods],
            "resident_executors": [executor.value for executor in self.executors],
            "results": written,
            "hardware": collect_hardware_metadata(),
            "labels_accessed_by_router": False,
            "run_context_hash": context["context_hash"],
        }
        manifest["manifest_hash"] = sha256_json(manifest)
        (run_root / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        return run_root
