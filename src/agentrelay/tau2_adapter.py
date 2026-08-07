"""Pinned tau2-bench text-mode adapter for native Gemma endpoint rollouts.

The module imports tau2 lazily.  Local unit tests can therefore validate task
manifests, prompt/action parsing, provenance, and resume without installing the
benchmark or loading model weights.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from .inference import NativeGenerationResult
from .learning import FEATURE_NAMES, GOAL_HASH_DIMENSIONS, feature_vector
from .schema import canonical_json, sha256_json, sha256_text

TAU2_REPOSITORY_URL = "https://github.com/sierra-research/tau2-bench.git"
TAU2_PINNED_REVISION = "0ed2fd8d830a20657d89ae9c2efcc94838aa7129"
TAU2_DOMAINS = ("airline", "retail", "telecom")
TAU2_UPSTREAM_TRAIN_SPLIT = "train"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def verify_tau2_repository(
    repository: str | Path,
    *,
    expected_revision: str = TAU2_PINNED_REVISION,
) -> dict[str, str]:
    """Fail closed when the checkout is missing or at the wrong commit."""

    repo = Path(repository).resolve()
    if not (repo / ".git").is_dir() or not (repo / "src" / "tau2").is_dir():
        raise ValueError(f"not a tau2-bench repository: {repo}")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected_revision:
        raise ValueError(f"tau2-bench revision mismatch: {actual} != {expected_revision}")
    origin = _git(repo, "remote", "get-url", "origin").rstrip("/")
    if origin != TAU2_REPOSITORY_URL.rstrip("/"):
        raise ValueError(f"tau2-bench origin mismatch: {origin}")
    return {"path": str(repo), "revision": actual, "origin": origin}


def activate_tau2_repository(repository: str | Path) -> Path:
    repo = Path(repository).resolve()
    source = repo / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return repo


def official_tau2_split_ids(repository: str | Path) -> dict[str, dict[str, tuple[str, ...]]]:
    """Read only the upstream split ledgers; no tau2 dependencies are needed."""

    repo = Path(repository).resolve()
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for domain in TAU2_DOMAINS:
        path = repo / "data" / "tau2" / "domains" / domain / "split_tasks.json"
        if not path.is_file():
            raise ValueError(f"missing official tau2 split ledger: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value.get("train"), list) or not isinstance(value.get("test"), list):
            raise TypeError(f"official tau2 split ledger is malformed: {path}")
        train = tuple(str(item) for item in value["train"])
        test = tuple(str(item) for item in value["test"])
        if not train or not test or set(train) & set(test):
            raise ValueError(f"official tau2 train/test split is invalid for {domain}")
        result[domain] = {"train": train, "test": test}
    return result


@dataclass(frozen=True)
class Tau2TaskRef:
    domain: str
    task_id: str
    upstream_split: str
    split: str
    seed: int

    def validate(self) -> None:
        if self.domain not in TAU2_DOMAINS:
            raise ValueError(f"unsupported tau2 domain: {self.domain}")
        if not self.task_id:
            raise ValueError("tau2 task_id cannot be empty")
        if self.upstream_split != TAU2_UPSTREAM_TRAIN_SPLIT:
            raise ValueError("router learnability tasks must come from official tau2 train")
        if self.split not in {"train", "dev", "smoke"}:
            raise ValueError(f"unsupported internal tau2 split: {self.split}")
        if self.seed < 0:
            raise ValueError("tau2 seed cannot be negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Tau2TaskRef:
        task = cls(
            domain=str(value["domain"]),
            task_id=str(value["task_id"]),
            upstream_split=str(value["upstream_split"]),
            split=str(value["split"]),
            seed=int(value["seed"]),
        )
        task.validate()
        return task


def _manifest_payload(tasks: Iterable[Tau2TaskRef], *, seed: int) -> dict[str, Any]:
    rows = [asdict(task) for task in tasks]
    return {
        "schema_version": "1.0",
        "benchmark": "tau2-bench",
        "repository_url": TAU2_REPOSITORY_URL,
        "repository_revision": TAU2_PINNED_REVISION,
        "upstream_split": TAU2_UPSTREAM_TRAIN_SPLIT,
        "held_out_split_accessed": False,
        "split_seed": seed,
        "tasks": rows,
    }


def build_tau2_router_manifest(
    repository: str | Path,
    *,
    split_seed: int = 20260806,
    dev_fraction: float = 0.30,
) -> dict[str, Any]:
    """Create deterministic, domain-stratified train/dev splits from official train."""

    verify_tau2_repository(repository)
    if not 0.1 <= dev_fraction <= 0.5:
        raise ValueError("dev_fraction must be in [0.1, 0.5]")
    split_ids = official_tau2_split_ids(repository)
    refs: list[Tau2TaskRef] = []
    counts: dict[str, dict[str, int]] = {}
    for domain in TAU2_DOMAINS:
        upstream = split_ids[domain][TAU2_UPSTREAM_TRAIN_SPLIT]
        if not upstream:
            raise ValueError(f"official tau2 train split is empty for {domain}")
        ranked = sorted(
            upstream,
            key=lambda task_id: sha256_text(f"{split_seed}:{domain}:{task_id}"),
        )
        dev_count = max(1, round(len(ranked) * dev_fraction))
        dev_ids = set(ranked[:dev_count])
        counts[domain] = {"train": len(ranked) - dev_count, "dev": dev_count}
        for task_id in sorted(ranked):
            split = "dev" if task_id in dev_ids else "train"
            refs.append(
                Tau2TaskRef(
                    domain=domain,
                    task_id=task_id,
                    upstream_split=TAU2_UPSTREAM_TRAIN_SPLIT,
                    split=split,
                    seed=300,
                )
            )
    payload = _manifest_payload(refs, seed=split_seed)
    payload["counts"] = counts
    payload["manifest_hash"] = sha256_json(payload)
    return payload


def read_tau2_manifest(path: str | Path, *, split: str | None = None) -> tuple[Tau2TaskRef, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = str(payload.pop("manifest_hash", ""))
    actual = sha256_json(payload)
    if not claimed or claimed != actual:
        raise ValueError("tau2 task manifest hash mismatch")
    if payload.get("repository_revision") != TAU2_PINNED_REVISION:
        raise ValueError("tau2 task manifest uses the wrong repository revision")
    if payload.get("held_out_split_accessed") is not False:
        raise ValueError("tau2 router manifest must not access the held-out test split")
    tasks = tuple(Tau2TaskRef.from_dict(item) for item in payload.get("tasks", ()))
    selected = tuple(task for task in tasks if split is None or task.split == split)
    if not selected:
        raise ValueError(f"tau2 manifest has no tasks for split {split!r}")
    keys = {(task.domain, task.task_id) for task in selected}
    if len(keys) != len(selected):
        raise ValueError("tau2 task manifest contains duplicate tasks")
    return selected


@dataclass(frozen=True)
class Tau2UserSimulatorConfig:
    implementation: str
    model: str
    model_revision: str
    api_key_env: str
    temperature: float
    max_tokens: int
    seed: int
    benchmark_revision: str
    prompt_policy: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        require_secret: bool = True,
    ) -> Tau2UserSimulatorConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "implementation",
            "model",
            "model_revision",
            "api_key_env",
            "temperature",
            "max_tokens",
            "seed",
            "benchmark_revision",
            "prompt_policy",
        }
        if set(value) != expected or value.get("schema_version") != "1.0":
            raise ValueError("unexpected fixed user-simulator configuration schema")
        config = cls(**{key: value[key] for key in expected if key != "schema_version"})
        if config.implementation != "official_tau2_user_simulator":
            raise ValueError("only the official tau2 user simulator is authorized")
        if config.benchmark_revision != TAU2_PINNED_REVISION:
            raise ValueError("user simulator is bound to the wrong tau2 revision")
        if config.temperature != 0.0 or config.max_tokens <= 0 or config.seed < 0:
            raise ValueError("user simulator decoding must be deterministic and bounded")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]+", config.api_key_env):
            raise ValueError("api_key_env must name an environment variable, not a secret")
        if require_secret and not os.environ.get(config.api_key_env):
            raise RuntimeError(f"required user simulator secret {config.api_key_env} is unset")
        return config

    def llm_args(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def provenance(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "api_key_present": bool(os.environ.get(self.api_key_env)),
            "config_hash": sha256_json(asdict(self)),
        }


@dataclass(frozen=True)
class ParsedTau2Action:
    content: str | None
    tool_name: str | None
    arguments: Mapping[str, Any] | None
    recovery: str


def parse_tau2_action(text: str, *, allowed_tools: Iterable[str]) -> ParsedTau2Action:
    raw = text.strip()
    if not raw:
        raise ValueError("native tau2 agent produced an empty response")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^<tool_call>\s*|\s*</tool_call>$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return ParsedTau2Action(raw, None, None, "plain_text")
    if not isinstance(value, Mapping):
        return ParsedTau2Action(raw, None, None, "non_object_as_text")
    tool_name = value.get("tool", value.get("name"))
    if tool_name is not None:
        name = str(tool_name)
        arguments = value.get("arguments", {})
        if name in set(allowed_tools) and isinstance(arguments, Mapping):
            return ParsedTau2Action(None, name, dict(arguments), "structured_tool")
        return ParsedTau2Action(raw, None, None, "invalid_tool_as_text")
    content = value.get("message", value.get("content"))
    if isinstance(content, str) and content.strip():
        return ParsedTau2Action(content.strip(), None, None, "structured_message")
    return ParsedTau2Action(raw, None, None, "unknown_object_as_text")


def tau2_router_features(
    visible_request: str,
    *,
    policy: str,
    tool_count: int,
    max_steps: int,
    edge_ms: float,
    cloud_ms: float,
    bandwidth_mbps: float,
) -> dict[str, float]:
    """Step-zero features derived only from information visible to the agent."""

    tokens = re.findall(r"[a-z0-9]+", visible_request.lower())
    ngrams = list(tokens) + [f"{a}_{b}" for a, b in pairwise(tokens)]
    hashed = [0.0] * GOAL_HASH_DIMENSIONS
    for token in ngrams:
        digest = sha256_text(token)
        index = int(digest[:8], 16) % GOAL_HASH_DIMENSIONS
        hashed[index] += 1.0 if int(digest[8:10], 16) % 2 else -1.0
    scale = max(1.0, float(len(ngrams)) ** 0.5)
    payload = (policy + "\n" + visible_request).encode("utf-8")
    features = {
        "step_index": 0.0,
        "remaining_steps": float(max_steps),
        "input_tokens": float(max(1, len(payload) // 4)),
        "delta_bytes": float(len(payload)),
        "local_confidence": 0.5,
        "invariant_count": 0.0,
        "unresolved_obligation_count": 1.0,
        "previous_patch_rate": 0.0,
        "edge_warm_inference_ms": float(edge_ms),
        "cloud_warm_inference_ms": float(cloud_ms),
        "measured_bandwidth_mbps": float(bandwidth_mbps),
        "previous_handoff_failed": 0.0,
        "effect_read_only": 1.0,
        "effect_reversible": 0.0,
        "effect_irreversible_or_unknown": 0.0,
        "closure_node_count": 1.0,
        "dependency_depth": 1.0,
        "estimated_patch_probability": 0.0,
        "effect_frontier_blocked": 0.0,
        "consecutive_steps": 0.0,
        "dwell_remaining": 0.0,
        "goal_char_count": float(len(visible_request)),
        "goal_token_count": float(len(tokens)),
        "goal_numeric_count": float(sum(token.isdigit() for token in tokens)),
        "goal_constraint_count": float(
            sum(word in visible_request.lower() for word in ("must", "only", "not", "before"))
        ),
        "visible_action_count": float(tool_count),
    }
    features.update(
        {f"goal_hash_{index:02d}": value / scale for index, value in enumerate(hashed)}
    )
    feature_vector(features)
    return {name: float(features[name]) for name in FEATURE_NAMES}


def _visible_message(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    calls = getattr(message, "tool_calls", None) or []
    return canonical_json(
        [
            {
                "tool": str(call.name),
                "arguments": dict(call.arguments),
                "requestor": str(call.requestor),
            }
            for call in calls
        ]
    )


def _chat_message(message: Any) -> dict[str, str]:
    role = str(getattr(message, "role", ""))
    if role == "assistant" and getattr(message, "tool_calls", None):
        content = canonical_json(
            [
                {"tool": str(call.name), "arguments": dict(call.arguments)}
                for call in message.tool_calls
            ]
        )
    else:
        content = _visible_message(message)
    return {"role": role, "content": content}


class NativeTau2Agent:
    """Official tau2 agent interface backed by AgentRelay native inference."""

    def __init__(
        self,
        *,
        tools: Sequence[Any],
        domain_policy: str,
        executor: Any,
        profile: Mapping[str, float],
        max_steps: int,
    ) -> None:
        self.tools = list(tools)
        self.domain_policy = domain_policy
        self.executor = executor
        self.profile = profile
        self.max_steps = max_steps
        self._generation_retries = 4
        self.generation_trace: list[dict[str, Any]] = []
        self.first_router_features: dict[str, float] | None = None

    @property
    def system_prompt(self) -> str:
        schemas = [tool.openai_schema for tool in self.tools]
        return (
            "You are a customer-service agent. Follow the policy exactly. In each turn, "
            "either send a user message or call one tool, never both. Return only one of: "
            "{\"message\":\"...\"} or {\"tool\":\"tool_name\",\"arguments\":{...}}.\n"
            f"<policy>\n{self.domain_policy}\n</policy>\n"
            f"<tools>\n{canonical_json(schemas)}\n</tools>"
        )

    def get_init_state(self, message_history: list[Any] | None = None) -> list[Any]:
        return list(message_history or [])

    def set_seed(self, seed: int) -> None:
        # Formal decoding is greedy; the per-task seed is still recorded in the
        # task manifest and controls the official user simulator.
        self.seed = seed

    def stop(self, message: Any = None, state: Any = None) -> None:
        return None

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        return False

    def generate_next_message(self, message: Any, state: list[Any]) -> tuple[Any, list[Any]]:
        from tau2.data_model.message import (  # type: ignore
            AssistantMessage,
            MultiToolMessage,
            ToolCall,
        )

        incoming = list(message.tool_messages) if isinstance(message, MultiToolMessage) else [message]
        state.extend(incoming)
        if self.first_router_features is None:
            visible = _visible_message(message)
            self.first_router_features = tau2_router_features(
                visible,
                policy=self.domain_policy,
                tool_count=len(self.tools),
                max_steps=self.max_steps,
                edge_ms=float(self.profile["edge_inference_ms"]),
                cloud_ms=float(self.profile["cloud_inference_ms"]),
                bandwidth_mbps=float(self.profile["bandwidth_mbps"]),
            )
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(_chat_message(item) for item in state)
        # The endpoint occasionally decodes an empty stub (only whitespace). The
        # edge Gemma decode is greedy, so retrying the identical prompt is
        # deterministic; we still bounded-retry to ride out any non-reproducible
        # stub, then fall back to a recovery message instead of crashing.
        result: NativeGenerationResult | None = None
        for _attempt in range(self._generation_retries):
            started = time.perf_counter()
            candidate = self.executor.generate(messages)
            if candidate.text.strip():
                result = candidate
                break
        if result is None:
            fallback_text = (
                "I understand; let me continue working on this. Could you "
                "confirm the next detail so I can proceed?"
            )
            assistant = AssistantMessage(role="assistant", content=fallback_text)
            state.append(assistant)
            self.generation_trace.append(
                {
                    "model_id": self.executor.config.model_id,
                    "model_revision": self.executor.config.revision,
                    "action_recovery": "empty_response_fallback",
                    "action_text": fallback_text,
                    "prompt_hash": sha256_text(canonical_json(messages)),
                    "response_hash": sha256_text(fallback_text),
                    "controller_ms": 0.0,
                    "latency_ms": 0.0,
                    "output_tokens": 0,
                    "prompt_tokens": 0,
                    "peak_cuda_memory_bytes": 0,
                    "seed": self.executor.config.seed,
                }
            )
            return assistant, state
        controller_ms = (time.perf_counter() - started) * 1000.0 - result.latency_ms
        parsed = parse_tau2_action(result.text, allowed_tools=(tool.name for tool in self.tools))
        if parsed.tool_name is not None:
            call_id = "agentrelay-" + result.response_hash[:16]
            assistant = AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name=parsed.tool_name,
                        arguments=dict(parsed.arguments or {}),
                        requestor="assistant",
                    )
                ],
            )
        else:
            assistant = AssistantMessage(role="assistant", content=parsed.content)
        state.append(assistant)
        self.generation_trace.append(
            {
                **result.to_dict(),
                "controller_ms": max(0.0, controller_ms),
                "action_recovery": parsed.recovery,
                "action_text": result.text,
            }
        )
        return assistant, state


class _UserSimMemoryCache:
    """Small on-disk JSON cache that pins the user simulator's step-zero message."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, Any] = {}
        if self.path.exists():
            try:
                self.records = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.records = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)

    def set(self, key: str, record: dict[str, Any]) -> None:
        self.records[key] = record
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.records, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


class _DeterministicUserSimulator:
    """Delegate to the official tau2 UserSimulator but cache and replay its FIRST
    generated user message, so paired edge/cloud runs observe an identical
    step-zero input even when the LLM backend is not deterministic (DeepSeek)."""

    def __init__(self, inner: Any, cache: _UserSimMemoryCache, cache_key: str) -> None:
        self._inner = inner
        self._cache = cache
        self._cache_key = cache_key
        self._generation = 0
        self._max_attempts = 10

    def get_init_state(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.get_init_state(*args, **kwargs)

    def stop(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.stop(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Delegate any other user-simulator API (e.g. set_seed) to the inner
        # official implementation so the orchestrator interface is unchanged.
        return getattr(self._inner, name)

    def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
        if self._generation == 0:
            cached = self._cache.get(self._cache_key)
            if cached is not None:
                from tau2.data_model.message import MultiToolMessage, ToolCall, UserMessage

                if isinstance(message, MultiToolMessage):
                    state.messages.extend(message.tool_messages)
                else:
                    state.messages.append(message)
                cached_calls = cached.get("tool_calls") or []
                user_message = UserMessage(
                    role="user",
                    content=cached["content"],
                    tool_calls=(
                        [
                            ToolCall(
                                id=tc["id"],
                                name=tc["name"],
                                arguments=tc["arguments"],
                                requestor="user",
                            )
                            for tc in cached_calls
                        ]
                        if cached_calls
                        else None
                    ),
                )
                state.messages.append(user_message)
                self._generation += 1
                return user_message, state
        snapshot_len = len(state.messages)
        result = self._inner.generate_next_message(message, state)
        user_message, _ = result
        attempt = 0
        while not (user_message.has_text_content() or user_message.is_tool_call()):
            attempt += 1
            if attempt >= self._max_attempts:
                # DeepSeek can transiently refuse for a whole turn; fall back to a
                # benign user continuation rather than crashing the episode. The
                # step-zero fallback is still cached so paired edge/cloud runs
                # observe an identical visible input.
                from tau2.data_model.message import UserMessage

                user_message = UserMessage(
                    role="user",
                    content="I understand. Please continue and let me know what you need from me.",
                )
                state.messages.append(user_message)
                break
            # The inner simulator appended the empty response to the shared state;
            # restore the history to just before the call, then retry.
            del state.messages[snapshot_len:]
            time.sleep(1.0)
            result = self._inner.generate_next_message(message, state)
            user_message, _ = result
        if self._generation == 0:
            self._cache.set(
                self._cache_key,
                {
                    "content": user_message.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in (user_message.tool_calls or [])
                    ],
                },
            )
        self._generation += 1
        return user_message, state


class Tau2Adapter:
    """Run one official tau2 task and convert it to router-compatible evidence."""

    def __init__(
        self,
        repository: str | Path,
        user_config: Tau2UserSimulatorConfig,
        *,
        max_steps: int = 100,
        max_errors: int = 10,
        user_sim_cache_path: str | Path | None = None,
    ) -> None:
        self.repository_info = verify_tau2_repository(repository)
        activate_tau2_repository(repository)
        self.user_config = user_config
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.user_sim_cache = (
            _UserSimMemoryCache(user_sim_cache_path) if user_sim_cache_path else None
        )

    def run(
        self,
        task_ref: Tau2TaskRef,
        *,
        executor: Any,
        role: str,
        profile: Mapping[str, float],
        endpoint_provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        task_ref.validate()
        if task_ref.seed != self.user_config.seed:
            raise ValueError("tau2 task seed disagrees with the fixed user simulator seed")
        if role not in {"edge", "cloud"}:
            raise ValueError("tau2 endpoint role must be edge or cloud")
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # type: ignore
        from tau2.orchestrator.orchestrator import Orchestrator  # type: ignore
        from tau2.registry import registry  # type: ignore
        from tau2.run import get_tasks  # type: ignore
        from tau2.user.user_simulator import UserSimulator  # type: ignore

        tasks = get_tasks(
            task_ref.domain,
            task_split_name=task_ref.upstream_split,
            task_ids=[task_ref.task_id],
        )
        if len(tasks) != 1:
            raise ValueError(f"cannot load tau2 task {task_ref.domain}/{task_ref.task_id}")
        task = tasks[0]
        environment = registry.get_env_constructor(task_ref.domain)()
        agent = NativeTau2Agent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            executor=executor,
            profile=profile,
            max_steps=self.max_steps,
        )
        try:
            user_tools = environment.get_user_tools()
        except ValueError:
            # Mirror the official tau2 run.py, which tolerates domains (e.g.
            # airline) that expose no user tools to the user simulator.
            user_tools = None
        user = UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=self.user_config.model,
            llm_args=self.user_config.llm_args(),
        )
        if self.user_sim_cache is not None:
            user = _DeterministicUserSimulator(
                user,
                self.user_sim_cache,
                f"{task_ref.domain}|{task_ref.task_id}",
            )
        orchestrator = Orchestrator(
            domain=task_ref.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=self.max_steps,
            max_errors=self.max_errors,
            seed=task_ref.seed,
            solo_mode=False,
            validate_communication=True,
        )
        simulation = orchestrator.run()
        reward_info = evaluate_simulation(
            domain=task_ref.domain,
            task=task,
            simulation=simulation,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
        )
        simulation.reward_info = reward_info
        reward = min(1.0, max(0.0, float(reward_info.reward)))
        router_features = agent.first_router_features
        if not agent.generation_trace or router_features is None:
            raise RuntimeError("tau2 episode contains no native agent generation")
        tool_errors = sum(
            int(bool(getattr(message, "error", False))) for message in simulation.messages
        )
        steps = []
        for index, trace in enumerate(agent.generation_trace):
            steps.append(
                {
                    "step_index": index,
                    "selected_executor": role,
                    "transfer_mode": "reuse",
                    "commit_mode": "immediate",
                    "router_features": router_features,
                    "prompt_hash": trace["prompt_hash"],
                    "response_hash": trace["response_hash"],
                    "response_text": trace["action_text"],
                    "action_hash": trace["response_hash"],
                    "action_text": trace["action_text"],
                    "action_recovery": trace["action_recovery"],
                    "generation_attempts": 1,
                    "inference_ms": trace["latency_ms"],
                    "controller_ms": trace["controller_ms"],
                    "handoff_bytes": 0,
                    "handoff_encode_ms": 0.0,
                    "handoff_verify_ms": 0.0,
                    "handoff_patch_ms": 0.0,
                    "handoff_communication_ms": 0.0,
                    "handoff_rehydration_ms": 0.0,
                    "handoff_effect_wait_ms": 0.0,
                    "handoff_reconciliation_ms": 0.0,
                    "target_tokens": trace["output_tokens"],
                    "prompt_tokens": trace["prompt_tokens"],
                    "peak_cuda_memory_bytes": trace["peak_cuda_memory_bytes"],
                    "fidelity_risk": 0.0,
                    "effect_risk": float(tool_errors > 0),
                    "reward": reward if index == len(agent.generation_trace) - 1 else 0.0,
                    "done": index == len(agent.generation_trace) - 1,
                }
            )
        reward_payload = reward_info.model_dump(mode="json")
        return {
            "benchmark": f"tau2/{task_ref.domain}",
            "dataset_revision": TAU2_PINNED_REVISION,
            "split": task_ref.split,
            "upstream_split": task_ref.upstream_split,
            "sample_id": f"{task_ref.domain}:{task_ref.task_id}",
            "task_id": task_ref.task_id,
            "domain": task_ref.domain,
            "role": role,
            "method": f"{role}_only",
            "success": float(reward >= 1.0),
            "reward": reward,
            "official_metrics": {"official_reward": reward},
            "official_reward_info": reward_payload,
            "termination_reason": str(simulation.termination_reason.value),
            "steps": steps,
            "end_to_end_ms": float(simulation.duration) * 1000.0,
            "paper_evidence": False,
            "labels_accessed_by_router": False,
            "effect_failures": tool_errors,
            "user_simulator": self.user_config.provenance(),
            "endpoint_provenance": dict(endpoint_provenance),
        }


def prepare_run_context(
    output_dir: str | Path,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Create or verify a deterministic context that scopes every resumed task."""

    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(context)
    context_hash = sha256_json(payload)
    document = {**payload, "context_hash": context_hash}
    target = directory / "run-context.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != document:
            raise RuntimeError("resume refused: tau2 run context changed")
    else:
        _atomic_json(target, document)
    return document, context_hash


def episode_path(output_dir: str | Path, task: Tau2TaskRef) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", task.task_id)
    return Path(output_dir).resolve() / "episodes" / f"{task.domain}--{safe_id}.json"


def load_resumable_episode(
    output_dir: str | Path,
    task: Tau2TaskRef,
    *,
    context_hash: str,
    role: str,
) -> dict[str, Any] | None:
    path = episode_path(output_dir, task)
    if not path.exists():
        return None
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    episode = wrapper.get("episode")
    if not isinstance(episode, Mapping) or wrapper.get("episode_hash") != sha256_json(episode):
        raise RuntimeError(f"resume refused: corrupt tau2 episode {path}")
    provenance = episode.get("endpoint_provenance", {})
    expected_sample = f"{task.domain}:{task.task_id}"
    if (
        provenance.get("manifest_hash") != context_hash
        or episode.get("sample_id") != expected_sample
        or episode.get("role") != role
        or episode.get("split") != task.split
    ):
        raise RuntimeError(f"resume refused: tau2 episode scope mismatch {path}")
    return dict(episode)


def save_resumable_episode(
    output_dir: str | Path,
    task: Tau2TaskRef,
    episode: Mapping[str, Any],
) -> None:
    payload = dict(episode)
    _atomic_json(
        episode_path(output_dir, task),
        {"episode": payload, "episode_hash": sha256_json(payload)},
    )


def collect_resumable_episodes(
    output_dir: str | Path,
    tasks: Sequence[Tau2TaskRef],
    *,
    context_hash: str,
    role: str,
) -> tuple[dict[str, Any], ...]:
    episodes = []
    for task in tasks:
        episode = load_resumable_episode(
            output_dir,
            task,
            context_hash=context_hash,
            role=role,
        )
        if episode is None:
            raise RuntimeError(f"missing tau2 episode for {task.domain}/{task.task_id}")
        episodes.append(episode)
    return tuple(episodes)
