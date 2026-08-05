"""Auditable paired-endpoint summaries and train/dev router gates."""

from __future__ import annotations

from statistics import mean
from typing import Any, Iterable, Mapping, Protocol

from .router_data import pair_endpoint_episodes
from .schema import Executor
from .statistics import mcnemar_exact, paired_bootstrap_difference


class RewardRouter(Protocol):
    def predicted_reward_by_executor(
        self, features: Mapping[str, float]
    ) -> Mapping[Executor, float]:
        ...


def _reward(episode: Mapping[str, Any]) -> float:
    return float(episode.get("reward", 0.0))


def _success(episode: Mapping[str, Any]) -> int:
    return int(float(episode.get("success", 0.0)) > 0.0)


def summarize_paired_endpoints(
    episodes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = pair_endpoint_episodes(episodes)
    edge_rewards = [_reward(edge) for edge, _ in pairs]
    cloud_rewards = [_reward(cloud) for _, cloud in pairs]
    edge_success = [_success(edge) for edge, _ in pairs]
    cloud_success = [_success(cloud) for _, cloud in pairs]
    reward_difference = paired_bootstrap_difference(
        cloud_rewards, edge_rewards, seed=20260805
    )
    both = sum(edge and cloud for edge, cloud in zip(edge_success, cloud_success))
    edge_only = sum(edge and not cloud for edge, cloud in zip(edge_success, cloud_success))
    cloud_only = sum(cloud and not edge for edge, cloud in zip(edge_success, cloud_success))
    neither = len(pairs) - both - edge_only - cloud_only
    return {
        "paired_tasks": len(pairs),
        "edge": {
            "avg_reward": mean(edge_rewards),
            "success_rate": mean(edge_success),
            "total_success": sum(edge_success),
            "exclusive_success": edge_only,
        },
        "cloud": {
            "avg_reward": mean(cloud_rewards),
            "success_rate": mean(cloud_success),
            "total_success": sum(cloud_success),
            "exclusive_success": cloud_only,
        },
        "both_success": both,
        "neither_success": neither,
        "success_union_rate": mean(
            max(edge, cloud) for edge, cloud in zip(edge_success, cloud_success)
        ),
        "oracle_avg_reward": mean(
            max(edge, cloud) for edge, cloud in zip(edge_rewards, cloud_rewards)
        ),
        "reward_directions": {
            "edge_better": sum(edge > cloud for edge, cloud in zip(edge_rewards, cloud_rewards)),
            "cloud_better": sum(cloud > edge for edge, cloud in zip(edge_rewards, cloud_rewards)),
            "tie": sum(edge == cloud for edge, cloud in zip(edge_rewards, cloud_rewards)),
        },
        "cloud_minus_edge_reward": {
            "mean": reward_difference.mean_difference,
            "bootstrap95_low": reward_difference.confidence_low,
            "bootstrap95_high": reward_difference.confidence_high,
            "resamples": reward_difference.resamples,
            "seed": reward_difference.seed,
        },
        "success_mcnemar_exact_p": mcnemar_exact(edge_success, cloud_success),
    }


def evaluate_router_learnability(
    router: RewardRouter,
    episodes: Iterable[Mapping[str, Any]],
    *,
    minimum_paired_tasks: int = 100,
    minimum_oracle_capture: float = 0.30,
    minimum_cloud_fraction: float = 0.10,
    maximum_cloud_fraction: float = 0.90,
) -> dict[str, Any]:
    pairs = pair_endpoint_episodes(episodes)
    router_rewards: list[float] = []
    selected_roles: list[Executor] = []
    edge_rewards: list[float] = []
    cloud_rewards: list[float] = []
    for edge, cloud in pairs:
        edge_steps = tuple(edge.get("steps", ()))
        cloud_steps = tuple(cloud.get("steps", ()))
        if not edge_steps or not cloud_steps:
            raise ValueError("learnability gate requires native dev trajectories")
        edge_features = {
            str(name): float(value)
            for name, value in edge_steps[0].get("router_features", {}).items()
        }
        cloud_features = {
            str(name): float(value)
            for name, value in cloud_steps[0].get("router_features", {}).items()
        }
        if edge_features != cloud_features:
            raise ValueError("paired dev endpoints have different pre-action features")
        predicted = router.predicted_reward_by_executor(edge_features)
        if set(predicted) != set(Executor):
            raise ValueError("router must predict both edge and cloud endpoint rewards")
        role = (
            Executor.CLOUD
            if predicted[Executor.CLOUD] > predicted[Executor.EDGE]
            else Executor.EDGE
        )
        selected_roles.append(role)
        edge_reward = _reward(edge)
        cloud_reward = _reward(cloud)
        edge_rewards.append(edge_reward)
        cloud_rewards.append(cloud_reward)
        router_rewards.append(cloud_reward if role is Executor.CLOUD else edge_reward)

    edge_mean = mean(edge_rewards)
    cloud_mean = mean(cloud_rewards)
    best_role = Executor.EDGE if edge_mean >= cloud_mean else Executor.CLOUD
    best_fixed_rewards = edge_rewards if best_role is Executor.EDGE else cloud_rewards
    best_fixed = max(edge_mean, cloud_mean)
    oracle = mean(max(edge, cloud) for edge, cloud in zip(edge_rewards, cloud_rewards))
    router_mean = mean(router_rewards)
    oracle_gap = oracle - best_fixed
    capture = (router_mean - best_fixed) / oracle_gap if oracle_gap > 0 else 0.0
    cloud_fraction = mean(role is Executor.CLOUD for role in selected_roles)
    paired_difference = paired_bootstrap_difference(
        router_rewards, best_fixed_rewards, seed=20260805
    )
    checks = {
        "enough_dev_tasks": len(pairs) >= minimum_paired_tasks,
        "positive_oracle_gap": oracle_gap > 0,
        "router_not_below_best_fixed": router_mean >= best_fixed,
        "oracle_capture": capture >= minimum_oracle_capture,
        "nondegenerate_cloud_fraction": (
            minimum_cloud_fraction <= cloud_fraction <= maximum_cloud_fraction
        ),
    }
    return {
        "gate_pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "minimum_paired_tasks": minimum_paired_tasks,
            "minimum_oracle_capture": minimum_oracle_capture,
            "minimum_cloud_fraction": minimum_cloud_fraction,
            "maximum_cloud_fraction": maximum_cloud_fraction,
        },
        "paired_dev_tasks": len(pairs),
        "edge_avg_reward": edge_mean,
        "cloud_avg_reward": cloud_mean,
        "best_fixed_role": best_role.value,
        "best_fixed_avg_reward": best_fixed,
        "oracle_avg_reward": oracle,
        "oracle_gap": oracle_gap,
        "router_avg_reward": router_mean,
        "oracle_gap_capture": capture,
        "cloud_selection_fraction": cloud_fraction,
        "router_minus_best_fixed": {
            "mean": paired_difference.mean_difference,
            "bootstrap95_low": paired_difference.confidence_low,
            "bootstrap95_high": paired_difference.confidence_high,
            "resamples": paired_difference.resamples,
            "seed": paired_difference.seed,
        },
    }
