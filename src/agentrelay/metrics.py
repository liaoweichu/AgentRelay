"""Immutable trajectory metrics and deterministic aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable


@dataclass(frozen=True)
class TrajectoryMetrics:
    run_id: str
    benchmark: str
    split: str
    sample_id: str
    method: str
    success: float
    reward: float
    end_to_end_ms: float
    step_latencies_ms: tuple[float, ...]
    cloud_steps: int
    total_steps: int
    switches: int
    transfer_bytes: int
    transfer_tokens: int
    relay_tax_ms: float
    invariant_checks: int = 0
    invariant_passes: int = 0
    patch_requests: int = 0
    patch_bytes: int = 0
    committed_effects: int = 0
    duplicate_effects: int = 0
    collateral_failures: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def percentile(values: Iterable[float], percentile_value: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be in [0, 100]")
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def aggregate_trajectories(rows: Iterable[TrajectoryMetrics]) -> dict[str, float | int]:
    rows = tuple(rows)
    if not rows:
        raise ValueError("cannot aggregate an empty result set")
    step_latencies = [value for row in rows for value in row.step_latencies_ms]
    total_steps = sum(row.total_steps for row in rows)
    invariant_checks = sum(row.invariant_checks for row in rows)
    committed_effects = sum(row.committed_effects for row in rows)
    return {
        "count": len(rows),
        "success_mean": mean(row.success for row in rows),
        "reward_mean": mean(row.reward for row in rows),
        "latency_mean_ms": mean(row.end_to_end_ms for row in rows),
        "latency_median_ms": median(row.end_to_end_ms for row in rows),
        "step_latency_p95_ms": percentile(step_latencies, 95),
        "cloud_step_ratio": (
            sum(row.cloud_steps for row in rows) / total_steps if total_steps else float("nan")
        ),
        "switches_mean": mean(row.switches for row in rows),
        "transfer_bytes_mean": mean(row.transfer_bytes for row in rows),
        "transfer_tokens_mean": mean(row.transfer_tokens for row in rows),
        "relay_tax_share": (
            sum(row.relay_tax_ms for row in rows)
            / sum(row.end_to_end_ms for row in rows)
            if sum(row.end_to_end_ms for row in rows) > 0
            else float("nan")
        ),
        "invariant_pass_rate": (
            sum(row.invariant_passes for row in rows) / invariant_checks
            if invariant_checks
            else float("nan")
        ),
        "duplicate_effect_rate": (
            sum(row.duplicate_effects for row in rows) / committed_effects
            if committed_effects
            else 0.0
        ),
        "collateral_failures": sum(row.collateral_failures for row in rows),
    }


def read_jsonl(path: str | Path) -> tuple[TrajectoryMetrics, ...]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            value["step_latencies_ms"] = tuple(value.get("step_latencies_ms", ()))
            rows.append(TrajectoryMetrics(**value))
    return tuple(rows)


def write_jsonl(path: str | Path, rows: Iterable[TrajectoryMetrics]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)

