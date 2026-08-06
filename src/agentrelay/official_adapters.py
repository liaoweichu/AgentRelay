"""Adapters for official ALFWorld, WebShop, and AppWorld environments.

Imports are lazy so the AgentRelay core can be tested without installing all
three benchmark stacks in one local environment. Formal runs must pin each
official repository revision in the experiment configuration.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .benchmark import (
    ActionValidation,
    BenchmarkEvaluation,
    BenchmarkObservation,
    BenchmarkStepResult,
    PublicBenchmarkAdapter,
    PublicTaskDescriptor,
)
from .continuation import build_standard_semantic_graph
from .effects import EffectLedger, build_effect_frontier
from .inference import require_immutable_revision
from .schema import (
    EffectClass,
    EvidenceItem,
    InvariantState,
    PlanState,
    RelayStatePacket,
    SemanticNode,
    SemanticNodeType,
    TaskIdentity,
    WorldState,
    goal_digest,
    sha256_json,
    sha256_text,
)
from .state import TraceStore
from .webshop_protocol import check_webshop_action, webshop_fallback_action


def _first(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    try:
        if hasattr(value, "shape") and len(value):
            return value[0]
    except TypeError:
        pass
    return default if value is None else value


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return _first(value.get(key), default)
    return default


@dataclass
class _ObservableState:
    goal: str = ""
    observation: BenchmarkObservation | None = None
    last_action: str = ""
    steps: int = 0
    reward: float = 0.0
    done: bool = False
    # Readable action -> observation history, in execution order.  Storing the
    # readable action text (not just a hash) lets the continuation carry the
    # full visited-state trajectory so a model cannot loop on stale state.
    history: list[tuple[str, str]] = field(default_factory=list)
    attempted_transitions: dict[str, set[str]] = field(default_factory=dict)

    def record(self, action: str, observation_text: str) -> None:
        self.history.append((action, observation_text))
        self.last_action = action

    def note_attempt(self, observation: BenchmarkObservation, action: str) -> None:
        self.attempted_transitions.setdefault(observation.environment_digest, set()).add(
            action.strip().lower()
        )

    def attempted_actions(self, observation: BenchmarkObservation) -> tuple[str, ...]:
        return tuple(sorted(self.attempted_transitions.get(observation.environment_digest, ())))


class ObservableOfficialAdapter(PublicBenchmarkAdapter):
    """Common packet construction from public runtime state only."""

    def __init__(self, descriptor: PublicTaskDescriptor) -> None:
        descriptor.validate()
        self.descriptor = descriptor
        self.trace_store = TraceStore()
        self.effect_ledger = EffectLedger()
        self.state = _ObservableState()

    def _packet_resources(self) -> Mapping[str, Any]:
        observation = self.state.observation
        return {
            "benchmark": self.descriptor.dataset_id,
            "step_index": self.state.steps,
            "valid_actions": list(observation.valid_actions if observation else ()),
            "last_action": self.state.last_action,
            "last_action_hash": sha256_text(self.state.last_action) if self.state.last_action else "",
        }

    def build_packet(self, previous: RelayStatePacket | None) -> RelayStatePacket:
        observation = self.state.observation
        if observation is None:
            raise RuntimeError("reset the official environment before building a packet")
        goal_span = self.trace_store.add(self.state.goal)
        observation_span = self.trace_store.add(observation.text)
        history_nodes: list[SemanticNode] = []
        history_spans: list[str] = []
        for index, (action, text) in enumerate(self.state.history):
            span = self.trace_store.add(text)
            history_spans.append(span)
            history_nodes.append(
                SemanticNode.create(
                    SemanticNodeType.WORLD_STATE,
                    {
                        "step_index": index,
                        "action": action,
                        "observation": text,
                        "resources": dict(observation.resources),
                    },
                    world_version=observation.observation_version,
                    trace_ref=span,
                    provenance_hash=sha256_text(text),
                )
            )
        nodes, edges, obligations = build_standard_semantic_graph(
            goal={"instruction": self.state.goal},
            observation={
                "text": observation.text,
                "resources": dict(observation.resources),
                "valid_actions": list(observation.valid_actions),
            },
            obligation={"next_action": True, "step_index": self.state.steps},
            world_version=observation.observation_version,
            goal_trace_ref=goal_span,
            observation_trace_ref=observation_span,
            goal_provenance_hash=sha256_text(self.state.goal),
            observation_provenance_hash=sha256_text(observation.text),
            extra_nodes=history_nodes,
        )
        effects = self.effect_ledger.snapshot()
        packet = RelayStatePacket(
            task=TaskIdentity(
                dataset_id=self.descriptor.dataset_id,
                dataset_revision=self.descriptor.dataset_revision,
                split=self.descriptor.split,
                sample_id=self.descriptor.sample_id,
                goal_hash=goal_digest(self.state.goal),
                success_criteria=("official benchmark evaluator",),
            ),
            invariants=InvariantState(
                hard_constraints={
                    "official_environment": True,
                    "hidden_evaluator_visible_to_router": False,
                },
                permissions=("use_public_observation_and_action_schema",),
                unresolved_obligations=("produce_one_next_environment_action",),
            ),
            world=WorldState(
                observation_version=observation.observation_version,
                environment_digest=observation.environment_digest,
                resources=self._packet_resources(),
            ),
            evidence=(
                EvidenceItem(
                    fact_id="public-goal",
                    value={"goal_hash": goal_digest(self.state.goal)},
                    source_span_id=goal_span,
                    provenance_hash=sha256_text(self.state.goal),
                ),
                EvidenceItem(
                    fact_id=f"observation-{self.state.steps}",
                    value={"observation_hash": sha256_text(observation.text)},
                    source_span_id=observation_span,
                    provenance_hash=sha256_text(observation.text),
                ),
            ),
            plan=PlanState(
                active_subgoal="produce the next official environment action",
                completed_subgoals=tuple(f"step-{index}" for index in range(self.state.steps)),
                pending_actions=("native_model_generate", "official_environment_step"),
            ),
            effects=effects,
            trace_refs=(goal_span, observation_span, *history_spans),
            parent_packet_hash=previous.packet_hash if previous is not None else "",
            acknowledged_version=previous.packet_hash if previous is not None else "",
            obligation_ids=obligations,
            semantic_nodes=nodes,
            dependency_edges=edges,
            effect_frontier=build_effect_frontier(effects),
        ).seal()
        return packet


class ALFWorldAdapter(ObservableOfficialAdapter):
    def __init__(
        self,
        *,
        config_path: str | Path,
        train_eval: str,
        dataset_revision: str,
        split: str,
        task_index: int = 0,
    ) -> None:
        require_immutable_revision(dataset_revision, subject="alfworld")
        try:
            import yaml
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise RuntimeError("install the pinned official ALFWorld environment") from exc
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        env_type = config["env"]["type"]
        raw_env = get_environment(env_type)(config, train_eval=train_eval)
        if task_index < 0 or task_index >= len(raw_env.game_files):
            raise IndexError(
                f"ALFWorld task_index {task_index} outside official split of "
                f"{len(raw_env.game_files)} tasks"
            )
        raw_env.game_files = [raw_env.game_files[task_index]]
        raw_env.num_games = 1
        self._env = raw_env.init_env(batch_size=1)
        self._task_index = task_index
        self._info: Mapping[str, Any] = {}
        self._last_score = 0.0
        self._won = False
        super().__init__(
            PublicTaskDescriptor(
                dataset_id="alfworld/official",
                dataset_revision=dataset_revision,
                split=split,
                sample_id="pending-reset",
            )
        )

    def reset(self) -> BenchmarkObservation:
        observations, info = self._env.reset()
        text = str(_first(observations, ""))
        self._info = info if isinstance(info, Mapping) else {}
        gamefile = str(_mapping_value(self._info, "extra.gamefile", "unknown-game"))
        sample_id = Path(gamefile).parent.name or sha256_text(gamefile)[:16]
        self.descriptor = replace(self.descriptor, sample_id=sample_id)
        admissible = tuple(
            str(item)
            for item in (_mapping_value(self._info, "admissible_commands", ()) or ())
        )
        self.state = _ObservableState(
            goal=text.split("\n", 1)[0],
            observation=BenchmarkObservation(
                text=text,
                observation_version="step-0",
                environment_digest=sha256_json({"text": text, "gamefile": gamefile}),
                resources={"gamefile_hash": sha256_text(gamefile)},
                valid_actions=admissible,
            ),
        )
        return self.state.observation

    def step(self, action: str) -> BenchmarkStepResult:
        observations, scores, dones, infos = self._env.step([action])
        text = str(_first(observations, ""))
        reward = float(_first(scores, 0.0))
        done = bool(_first(dones, False))
        self._info = infos if isinstance(infos, Mapping) else {}
        self._last_score = reward
        self._won = bool(_mapping_value(self._info, "won", done and reward > 0))
        admissible = tuple(
            str(item)
            for item in (_mapping_value(self._info, "admissible_commands", ()) or ())
        )
        self.state.steps += 1
        self.state.reward = reward
        self.state.done = done
        self.state.record(action, text)
        self.state.observation = BenchmarkObservation(
            text=text,
            observation_version=f"step-{self.state.steps}",
            environment_digest=sha256_json({"text": text, "step": self.state.steps}),
            resources={"score": reward},
            valid_actions=admissible,
            done=done,
        )
        return BenchmarkStepResult(self.state.observation, reward, dict(self._info))

    def evaluate(self) -> BenchmarkEvaluation:
        return BenchmarkEvaluation(
            success=float(self._won),
            reward=self._last_score,
            official_metrics={"won": float(self._won)},
        )

    def effect_metadata(self, action: str) -> Mapping[str, Any]:
        return {
            "effect_class": EffectClass.READ_ONLY.value,
            "tool_name": "alfworld.text_action",
            "arguments": {"action": action},
            "scope_key": "alfworld-control",
        }

    def pending_effect_class(self, observation: BenchmarkObservation) -> EffectClass:
        # ALFWorld actions mutate only the isolated benchmark simulator and have
        # no external tool side effect requiring a distributed commit barrier.
        return EffectClass.READ_ONLY

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


class WebShopAdapter(ObservableOfficialAdapter):
    def __init__(
        self,
        *,
        session: int,
        dataset_revision: str,
        split: str,
        file_path: str | None = None,
        env: Any = None,
    ) -> None:
        require_immutable_revision(dataset_revision, subject="webshop")
        if env is not None:
            # Reuse a caller-provided shared environment (e.g. the full official
            # corpus constructed once per matrix run) to avoid rebuilding the
            # product/search-engine state on every episode.
            self._env = env
            self._close_env = False
        else:
            try:
                from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
            except ImportError as exc:
                raise RuntimeError("install the pinned official WebShop environment") from exc
            kwargs: dict[str, Any] = {
                "observation_mode": "text",
                "human_goals": 1,
                "session": int(session),
            }
            if file_path is not None:
                kwargs["file_path"] = file_path
            self._env = WebAgentTextEnv(**kwargs)
            self._close_env = True
        self._session = int(session)
        self._final_reward = 0.0
        super().__init__(
            PublicTaskDescriptor(
                dataset_id="princeton-nlp/WebShop",
                dataset_revision=dataset_revision,
                split=split,
                sample_id=str(session),
            )
        )

    def _valid_actions(self) -> tuple[str, ...]:
        available = self._env.get_available_actions()
        values = ["search[<keywords>]"] if available.get("has_search_bar") else []
        values.extend(f"click[{item}]" for item in available.get("clickables", ()))
        return tuple(values)

    def reset(self) -> BenchmarkObservation:
        value = self._env.reset(session=self._session)
        text = str(_first(value, ""))
        goal = str(getattr(self._env, "instruction_text", ""))
        valid_actions = self._valid_actions()
        self.state = _ObservableState(
            goal=goal,
            observation=BenchmarkObservation(
                text=text,
                observation_version="step-0",
                environment_digest=sha256_json(
                    {
                        "session": self._session,
                        "text": text,
                        "valid_actions": list(valid_actions),
                    }
                ),
                resources={"session": self._session, "human_goals": True},
                valid_actions=valid_actions,
            ),
        )
        return self.state.observation

    def step(self, action: str) -> BenchmarkStepResult:
        if self.state.observation is None:
            raise RuntimeError("reset WebShop before stepping")
        self.state.note_attempt(self.state.observation, action)
        text, reward, done, info = self._env.step(action)
        self._final_reward = float(reward)
        self.state.steps += 1
        self.state.reward = float(reward)
        self.state.done = bool(done)
        self.state.record(action, str(text))
        valid_actions = self._valid_actions() if not done else ()
        self.state.observation = BenchmarkObservation(
            text=str(text),
            observation_version=f"step-{self.state.steps}",
            environment_digest=sha256_json(
                {
                    "session": self._session,
                    "text": str(text),
                    "valid_actions": list(valid_actions),
                }
            ),
            resources={"session": self._session, "reward": float(reward)},
            valid_actions=valid_actions,
            done=bool(done),
        )
        return BenchmarkStepResult(
            self.state.observation,
            float(reward),
            info if isinstance(info, Mapping) else {},
        )

    def validate_model_action(
        self,
        action: str,
        observation: BenchmarkObservation,
    ) -> ActionValidation:
        checked = check_webshop_action(
            action,
            observation.valid_actions,
            attempted_actions=self.state.attempted_actions(observation),
        )
        return ActionValidation(checked.action, checked.accepted, checked.feedback)

    def fallback_model_action(
        self,
        observation: BenchmarkObservation,
        rejected_actions: Sequence[str],
    ) -> str | None:
        attempted = set(self.state.attempted_actions(observation))
        attempted.update(str(action).strip().lower() for action in rejected_actions)
        return webshop_fallback_action(
            observation.valid_actions,
            goal=self.state.goal,
            attempted_actions=attempted,
        )

    def evaluate(self) -> BenchmarkEvaluation:
        return BenchmarkEvaluation(
            success=float(self._final_reward >= 1.0),
            reward=self._final_reward,
            official_metrics={"reward": self._final_reward},
        )

    def effect_metadata(self, action: str) -> Mapping[str, Any]:
        # An irreversible effect only applies when the purchase action is actually
        # actionable in the current state.  A premature/out-of-scope "buy" proposal
        # is rejected by the environment as a no-op, so it must not be classified
        # as irreversible (which would otherwise mismatch the routing commit mode
        # derived from pending_effect_class and falsely trip the safety barrier).
        normalized = action.strip().lower()
        available = {item.strip().lower() for item in self._valid_actions()}
        irreversible = normalized == "click[buy now]" and normalized in available
        return {
            "effect_class": (
                EffectClass.IRREVERSIBLE.value if irreversible else EffectClass.READ_ONLY.value
            ),
            "tool_name": "webshop.action",
            "arguments": {"action": action},
            "scope_key": f"webshop-session-{self._session}",
        }

    def pending_effect_class(self, observation: BenchmarkObservation) -> EffectClass:
        # The visible action set is authoritative.  Once purchase is reachable,
        # route conservatively before either executor proposes the concrete click.
        if any(action.strip().lower() == "click[buy now]" for action in observation.valid_actions):
            return EffectClass.IRREVERSIBLE
        return EffectClass.READ_ONLY

    def close(self) -> None:
        if getattr(self, "_close_env", True):
            self._env.close()


def _call_name(node: ast.Call) -> str:
    value: ast.AST = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def classify_appworld_code(code: str) -> EffectClass:
    """Conservative public-syntax classifier; unknown calls require a barrier."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return EffectClass.UNKNOWN
    names = [_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if not names:
        return EffectClass.READ_ONLY
    if any(name.endswith("complete_task") for name in names):
        return EffectClass.IRREVERSIBLE
    read_prefixes = ("get_", "show_", "search_", "list_", "find_", "lookup_")
    if all(name.rsplit(".", 1)[-1].startswith(read_prefixes) for name in names):
        return EffectClass.READ_ONLY
    return EffectClass.UNKNOWN


class AppWorldAdapter(ObservableOfficialAdapter):
    def __init__(
        self,
        *,
        task_id: str,
        dataset_revision: str,
        split: str,
        experiment_name: str,
        allow_official_evaluation: bool = False,
    ) -> None:
        require_immutable_revision(dataset_revision, subject="appworld")
        try:
            from appworld import AppWorld
        except ImportError as exc:
            raise RuntimeError("install the pinned official AppWorld environment") from exc
        self._world = AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            load_ground_truth=allow_official_evaluation,
        )
        self._allow_official_evaluation = allow_official_evaluation
        super().__init__(
            PublicTaskDescriptor(
                dataset_id="StonyBrookNLP/appworld",
                dataset_revision=dataset_revision,
                split=split,
                sample_id=task_id,
            )
        )

    def reset(self) -> BenchmarkObservation:
        goal = str(self._world.task.instruction)
        allowed_apps = tuple(str(item) for item in self._world.task.allowed_apps)
        text = (
            goal
            + "\nUse Python code with the public AppWorld APIs. "
            + "Allowed apps: "
            + ", ".join(allowed_apps)
        )
        self.state = _ObservableState(
            goal=goal,
            observation=BenchmarkObservation(
                text=text,
                observation_version="step-0",
                environment_digest=sha256_json(
                    {"task_id": self.descriptor.sample_id, "instruction": goal}
                ),
                resources={"allowed_apps": list(allowed_apps)},
                valid_actions=("python code using apis.<app>.<method>(...)",),
            ),
        )
        return self.state.observation

    def parse_model_output(self, text: str) -> str:
        value = text.strip()
        fenced = re.search(r"```(?:python)?\s*(.*?)```", value, re.DOTALL | re.IGNORECASE)
        return fenced.group(1).strip() if fenced else value

    def step(self, action: str) -> BenchmarkStepResult:
        output = str(self._world.execute(action))
        self.state.steps += 1
        self.state.done = bool(self._world.task_completed())
        self.state.record(action, output)
        self.state.observation = BenchmarkObservation(
            text=output,
            observation_version=f"step-{self.state.steps}",
            environment_digest=sha256_json(
                {"task_id": self.descriptor.sample_id, "step": self.state.steps, "output": output}
            ),
            resources={"task_completed": self.state.done},
            valid_actions=("python code using apis.<app>.<method>(...)",),
            done=self.state.done,
        )
        return BenchmarkStepResult(self.state.observation, 0.0, {})

    def evaluate(self) -> BenchmarkEvaluation:
        if not self._allow_official_evaluation:
            return BenchmarkEvaluation(
                success=float(self._world.task_completed()),
                reward=0.0,
                official_metrics={"official_evaluation_deferred": 1.0},
            )
        tracker = self._world.evaluate()
        payload = tracker.to_dict() if hasattr(tracker, "to_dict") else {}
        if not isinstance(payload, Mapping):
            payload = {}
        passed = float(payload.get("num_passed", payload.get("passes", 0.0)))
        total = float(payload.get("num_tests", payload.get("total", 0.0)))
        success = float(total > 0 and passed == total)
        metrics = {
            str(key): float(value)
            for key, value in payload.items()
            if isinstance(value, (int, float, bool))
        }
        return BenchmarkEvaluation(success=success, reward=success, official_metrics=metrics)

    def effect_metadata(self, action: str) -> Mapping[str, Any]:
        return {
            "effect_class": classify_appworld_code(action).value,
            "tool_name": "appworld.execute",
            "arguments": {"code_hash": sha256_text(action)},
            "scope_key": f"appworld-task-{self.descriptor.sample_id}",
        }

    def pending_effect_class(self, observation: BenchmarkObservation) -> EffectClass:
        # Arbitrary public API code is proposed only after routing.  Without
        # inspecting a proposal, its side-effect class cannot be proven safe.
        return EffectClass.UNKNOWN

    def close(self) -> None:
        self._world.close()
