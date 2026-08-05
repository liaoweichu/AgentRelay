"""Public-benchmark adapter contract for live, native agent execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .inference import require_immutable_revision
from .schema import EffectClass, RelayStatePacket


@dataclass(frozen=True)
class PublicTaskDescriptor:
    dataset_id: str
    dataset_revision: str
    split: str
    sample_id: str

    def validate(self) -> None:
        if not self.dataset_id or not self.split or not self.sample_id:
            raise ValueError("public task provenance fields cannot be empty")
        require_immutable_revision(self.dataset_revision, subject=self.dataset_id)


@dataclass(frozen=True)
class BenchmarkObservation:
    text: str
    observation_version: str
    environment_digest: str
    resources: Mapping[str, Any] = field(default_factory=dict)
    valid_actions: tuple[str, ...] = ()
    done: bool = False


@dataclass(frozen=True)
class BenchmarkStepResult:
    observation: BenchmarkObservation
    reward: float
    official_info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkEvaluation:
    success: float
    reward: float
    official_metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionValidation:
    action: str
    accepted: bool
    feedback: str = ""


class PublicBenchmarkAdapter(ABC):
    """Minimal boundary that keeps official data and evaluators authoritative."""

    descriptor: PublicTaskDescriptor

    @abstractmethod
    def reset(self) -> BenchmarkObservation:
        """Reset exactly the official task identified by ``descriptor``."""

    @abstractmethod
    def step(self, action: str) -> BenchmarkStepResult:
        """Execute an action through the official environment implementation."""

    @abstractmethod
    def build_packet(self, previous: RelayStatePacket | None) -> RelayStatePacket:
        """Build typed relay state from observable harness state only."""

    @abstractmethod
    def evaluate(self) -> BenchmarkEvaluation:
        """Return official evaluator outputs after the episode has ended."""

    @abstractmethod
    def effect_metadata(self, action: str) -> Mapping[str, Any]:
        """Classify an action from public tool metadata, never hidden test labels."""

    def pending_effect_class(self, observation: BenchmarkObservation) -> EffectClass:
        """Conservatively classify the next public action before model selection.

        Official adapters should override this when the visible action schema is
        sufficient to prove that the next step is read-only.  Unknown is the safe
        default because the joint router runs before either model proposes an
        action.
        """

        return EffectClass.UNKNOWN

    def format_model_messages(
        self,
        observation: BenchmarkObservation,
        continuation: str,
    ) -> Sequence[Mapping[str, str]]:
        """Return a benchmark-neutral action prompt with no evaluator state."""

        valid = "\n".join(f"- {item}" for item in observation.valid_actions)
        return (
            {
                "role": "system",
                "content": (
                    "Act in the public benchmark environment. Use only the visible "
                    "observation, tool/action schema, and semantic continuation. "
                    "Return exactly one executable next action and no analysis."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Observation:\n{observation.text}\n\n"
                    f"Valid action schema/examples:\n{valid or '- benchmark-defined action'}\n\n"
                    f"Semantic continuation:\n{continuation}\n\nNext action:"
                ),
            },
        )

    def parse_model_output(self, text: str) -> str:
        """Convert a native model response into one environment action."""

        value = text.strip()
        if not value:
            raise ValueError("model returned an empty action")
        if value.lower().startswith("action:"):
            value = value.split(":", 1)[1].strip()
        return value.splitlines()[0].strip()

    def validate_model_action(
        self,
        action: str,
        observation: BenchmarkObservation,
    ) -> ActionValidation:
        """Validate an action against public state before environment execution."""

        del observation
        value = str(action).strip()
        return ActionValidation(value, bool(value), "model returned an empty action")

    def fallback_model_action(
        self,
        observation: BenchmarkObservation,
        rejected_actions: Sequence[str],
    ) -> str | None:
        """Optional deterministic public-state fallback after model retries."""

        del observation, rejected_actions
        return None

    def close(self) -> None:
        """Release official-environment resources when supported."""
