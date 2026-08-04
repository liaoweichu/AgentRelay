"""Measured handoff-cost primitives and trace-driven transfer timing."""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class HandoffMeasurement:
    encode_ms: float = 0.0
    communication_ms: float = 0.0
    rehydration_ms: float = 0.0
    verification_ms: float = 0.0
    patch_ms: float = 0.0
    effect_sync_ms: float = 0.0
    effect_wait_ms: float = 0.0
    reconciliation_ms: float = 0.0
    payload_bytes: int = 0
    target_tokens: int = 0
    fidelity_risk: float = 0.0
    effect_risk: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.encode_ms
            + self.communication_ms
            + self.rehydration_ms
            + self.verification_ms
            + self.patch_ms
            + self.effect_sync_ms
            + self.effect_wait_ms
            + self.reconciliation_ms
        )

    def zero_latency_tax(self) -> "HandoffMeasurement":
        return replace(
            self,
            encode_ms=0.0,
            communication_ms=0.0,
            rehydration_ms=0.0,
            verification_ms=0.0,
            patch_ms=0.0,
            effect_sync_ms=0.0,
            effect_wait_ms=0.0,
            reconciliation_ms=0.0,
            payload_bytes=0,
            target_tokens=0,
        )


@dataclass(frozen=True)
class CostWeights:
    latency: float = 1.0
    transfer_token: float = 0.0
    payload_byte: float = 0.0
    fidelity_risk: float = 1000.0
    effect_risk: float = 1000.0

    def score(self, measurement: HandoffMeasurement) -> float:
        return (
            self.latency * measurement.total_ms
            + self.transfer_token * measurement.target_tokens
            + self.payload_byte * measurement.payload_bytes
            + self.fidelity_risk * measurement.fidelity_risk
            + self.effect_risk * measurement.effect_risk
        )


def continuation_tax_share(measurement: HandoffMeasurement, end_to_end_ms: float) -> float:
    if end_to_end_ms <= 0:
        raise ValueError("end-to-end latency must be positive")
    return measurement.total_ms / end_to_end_ms


def relay_tax_share(measurement: HandoffMeasurement, end_to_end_ms: float) -> float:
    """Backward-compatible name for :func:`continuation_tax_share`."""

    return continuation_tax_share(measurement, end_to_end_ms)


@dataclass(frozen=True)
class RecordedBandwidthTrace:
    """Ordered real-trace throughput samples.

    Samples are never shuffled or synthesized. If the requested transfer extends
    beyond the available suffix, the method returns infinity rather than looping.
    """

    samples_mbps: tuple[float, ...]
    sample_period_ms: float
    source: str

    def __post_init__(self) -> None:
        if not self.samples_mbps:
            raise ValueError("bandwidth trace cannot be empty")
        if self.sample_period_ms <= 0:
            raise ValueError("sample period must be positive")
        if any(sample < 0 for sample in self.samples_mbps):
            raise ValueError("bandwidth samples cannot be negative")

    def transfer_time_ms(self, payload_bytes: int, start_index: int = 0) -> float:
        if payload_bytes < 0:
            raise ValueError("payload size cannot be negative")
        if payload_bytes == 0:
            return 0.0
        if start_index < 0 or start_index >= len(self.samples_mbps):
            raise IndexError("start index outside trace")
        remaining_bits = float(payload_bytes * 8)
        elapsed_ms = 0.0
        for rate_mbps in self.samples_mbps[start_index:]:
            bits_per_ms = rate_mbps * 1000.0
            capacity = bits_per_ms * self.sample_period_ms
            if capacity >= remaining_bits and bits_per_ms > 0:
                elapsed_ms += remaining_bits / bits_per_ms
                return elapsed_ms
            remaining_bits -= capacity
            elapsed_ms += self.sample_period_ms
        return float("inf")

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        rate_column: str,
        sample_period_ms: float,
        source: str,
    ) -> "RecordedBandwidthTrace":
        values: list[float] = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if rate_column not in (reader.fieldnames or []):
                raise ValueError(f"missing rate column: {rate_column}")
            for row in reader:
                values.append(float(row[rate_column]))
        return cls(tuple(values), sample_period_ms, source)


def sum_handoff_measurements(values: Iterable[HandoffMeasurement]) -> HandoffMeasurement:
    result = HandoffMeasurement()
    for value in values:
        result = HandoffMeasurement(
            encode_ms=result.encode_ms + value.encode_ms,
            communication_ms=result.communication_ms + value.communication_ms,
            rehydration_ms=result.rehydration_ms + value.rehydration_ms,
            verification_ms=result.verification_ms + value.verification_ms,
            patch_ms=result.patch_ms + value.patch_ms,
            effect_sync_ms=result.effect_sync_ms + value.effect_sync_ms,
            effect_wait_ms=result.effect_wait_ms + value.effect_wait_ms,
            reconciliation_ms=result.reconciliation_ms + value.reconciliation_ms,
            payload_bytes=result.payload_bytes + value.payload_bytes,
            target_tokens=result.target_tokens + value.target_tokens,
            fidelity_risk=max(result.fidelity_risk, value.fidelity_risk),
            effect_risk=max(result.effect_risk, value.effect_risk),
        )
    return result
