"""Deterministic paired statistics for recorded benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import random
from statistics import mean
from typing import Iterable, Mapping, Sequence

from .metrics import percentile


@dataclass(frozen=True)
class BootstrapDifference:
    mean_difference: float
    confidence_low: float
    confidence_high: float
    confidence_level: float
    resamples: int
    seed: int


def paired_bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapDifference:
    if len(first) != len(second) or not first:
        raise ValueError("paired samples must be non-empty and have equal length")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    differences = [float(a) - float(b) for a, b in zip(first, second)]
    generator = random.Random(seed)
    bootstrapped = []
    for _ in range(resamples):
        bootstrapped.append(
            mean(differences[generator.randrange(len(differences))] for _ in differences)
        )
    alpha = 1.0 - confidence_level
    return BootstrapDifference(
        mean_difference=mean(differences),
        confidence_low=percentile(bootstrapped, 100 * alpha / 2),
        confidence_high=percentile(bootstrapped, 100 * (1 - alpha / 2)),
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )


def mcnemar_exact(first: Sequence[int], second: Sequence[int]) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""

    if len(first) != len(second) or not first:
        raise ValueError("paired outcomes must be non-empty and have equal length")
    if any(value not in {0, 1} for value in (*first, *second)):
        raise ValueError("McNemar outcomes must be binary")
    first_only = sum(a == 1 and b == 0 for a, b in zip(first, second))
    second_only = sum(a == 0 and b == 1 for a, b in zip(first, second))
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    for name, value in p_values.items():
        if not 0 <= value <= 1:
            raise ValueError(f"p-value for {name} is outside [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def pareto_frontier(
    rows: Iterable[Mapping[str, float]],
    *,
    maximize: tuple[str, ...],
    minimize: tuple[str, ...],
) -> tuple[Mapping[str, float], ...]:
    """Return non-dominated recorded rows without interpolating measurements."""

    rows = tuple(rows)
    frontier = []
    for index, candidate in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            no_worse = all(other[key] >= candidate[key] for key in maximize) and all(
                other[key] <= candidate[key] for key in minimize
            )
            strictly_better = any(other[key] > candidate[key] for key in maximize) or any(
                other[key] < candidate[key] for key in minimize
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)
