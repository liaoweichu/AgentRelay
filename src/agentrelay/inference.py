"""Pinned, native Hugging Face inference used by formal AgentRelay runs.

No hosted API or cached answer path is exposed here.  A model and tokenizer are
loaded from the same immutable repository revision, and generation timing and
token counts are captured directly from the local process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import re
import time
from typing import Any, Mapping, Sequence

from .config import StorageLayout
from .schema import sha256_text


FULL_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


def require_immutable_revision(revision: str, *, subject: str = "artifact") -> None:
    if not FULL_COMMIT_RE.fullmatch(revision):
        raise ValueError(
            f"{subject} revision must be a full immutable commit hash, got {revision!r}"
        )


@dataclass(frozen=True)
class NativeGenerationConfig:
    model_id: str
    revision: str
    dtype: str = "bfloat16"
    quantization: str = "none"
    device_map: str = "auto"
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    trust_remote_code: bool = False
    local_files_only: bool = False

    def validate(self) -> None:
        if not self.model_id or "/" not in self.model_id:
            raise ValueError("model_id must be a Hugging Face repository identifier")
        require_immutable_revision(self.revision, subject=self.model_id)
        if self.dtype not in {"float16", "bfloat16", "float32", "auto"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.quantization not in {"none", "bnb_4bit", "bnb_8bit", "repository"}:
            raise ValueError(f"unsupported quantization: {self.quantization}")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("sampling temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NativeGenerationConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = set(value) - allowed - {"requested_revision"}
        if unknown:
            raise ValueError(f"unknown native generation fields: {sorted(unknown)}")
        config = cls(**{key: item for key, item in value.items() if key in allowed})
        config.validate()
        return config


@dataclass(frozen=True)
class NativeGenerationResult:
    text: str
    model_id: str
    model_revision: str
    prompt_hash: str
    response_hash: str
    prompt_tokens: int
    output_tokens: int
    latency_ms: float
    peak_cuda_memory_bytes: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HFModelExecutor:
    """A lazily imported local Transformers executor.

    Constructing this class performs the real model load.  Unit tests can import
    the module without installing PyTorch or Transformers.
    """

    def __init__(self, config: NativeGenerationConfig, storage: StorageLayout) -> None:
        config.validate()
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError("install AgentRelay with the 'ml' extra for native inference") from exc

        storage.create()
        self.config = config
        self.storage = storage
        self.torch = torch
        common: dict[str, Any] = {
            "revision": config.revision,
            "cache_dir": str(storage.models),
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)

        model_kwargs = dict(common)
        model_kwargs["device_map"] = config.device_map
        if config.dtype != "auto":
            model_kwargs["dtype"] = getattr(torch, config.dtype)
        else:
            model_kwargs["dtype"] = "auto"
        if config.quantization == "bnb_4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif config.quantization == "bnb_8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        # "repository" leaves quantization to the pinned repository config.
        self.model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
        self.model.eval()

    def _input_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.torch.device("cpu")

    def generate(self, messages: Sequence[Mapping[str, str]]) -> NativeGenerationResult:
        if not messages:
            raise ValueError("messages cannot be empty")
        normalized = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"unsupported chat role: {role!r}")
            normalized.append({"role": role, "content": content})

        prompt = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = self._input_device()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])

        torch = self.torch
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        }
        if self.config.do_sample:
            generation_kwargs.update(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        else:
            generation_kwargs.update(temperature=None, top_p=None, top_k=None)

        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**encoded, **generation_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0

        continuation = output[0, prompt_tokens:]
        text = self.tokenizer.decode(continuation, skip_special_tokens=True)
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        return NativeGenerationResult(
            text=text,
            model_id=self.config.model_id,
            model_revision=self.config.revision,
            prompt_hash=sha256_text(prompt),
            response_hash=sha256_text(text),
            prompt_tokens=prompt_tokens,
            output_tokens=int(continuation.shape[-1]),
            latency_ms=latency_ms,
            peak_cuda_memory_bytes=peak_memory,
            seed=self.config.seed,
        )
