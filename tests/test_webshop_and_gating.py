"""Software-only checks for WebShop split, loop, and router gates."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from agentrelay.baselines import BaselineName
from agentrelay.formal_matrix import (
    FormalTask,
    OfficialTaskManifest,
    required_model_executors,
    write_task_manifest,
)
from agentrelay.gating import evaluate_router_learnability, summarize_paired_endpoints
from agentrelay.learning import FEATURE_NAMES, JointRouterEstimator
from agentrelay.official_adapters import WebShopAdapter
from agentrelay.router_data import task_router_training_rows
from agentrelay.schema import Executor, canonical_json
from agentrelay.webshop_protocol import (
    WEBSHOP_TOTAL_HUMAN_GOALS,
    check_webshop_action,
    official_webshop_indices,
    webshop_fallback_action,
)


REVISION = "4" * 40
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def features(signal: float = 0.0) -> dict[str, float]:
    value = {name: 0.0 for name in FEATURE_NAMES}
    value["goal_hash_00"] = signal
    return value


def endpoint_episode(
    task_id: str,
    role: Executor,
    reward: float,
    *,
    split: str,
    signal: float = 0.0,
) -> dict:
    transfer = "reuse" if role is Executor.EDGE else "full_replay"
    steps = [
        {
            "step_index": index,
            "selected_executor": role.value,
            "transfer_mode": transfer,
            "commit_mode": "immediate",
            "router_features": features(signal),
            "inference_ms": 10.0 if role is Executor.EDGE else 20.0,
            "controller_ms": 0.1,
        }
        for index in range(2)
    ]
    return {
        "benchmark": "princeton-nlp/WebShop",
        "dataset_revision": REVISION,
        "split": split,
        "sample_id": task_id,
        "method": f"{role.value}_only",
        "success": float(reward >= 1.0),
        "reward": reward,
        "steps": steps,
        "labels_accessed_by_router": False,
        "effect_failures": 0,
    }


class WebShopSplitTests(unittest.TestCase):
    def test_fixed_endpoint_runs_load_only_one_model(self) -> None:
        self.assertEqual(
            required_model_executors((BaselineName.EDGE_ONLY,)),
            (Executor.EDGE,),
        )
        self.assertEqual(
            required_model_executors((BaselineName.CLOUD_ONLY,)),
            (Executor.CLOUD,),
        )
        self.assertEqual(
            set(required_model_executors((BaselineName.ROUTELLM_TASK,))),
            set(Executor),
        )

    def test_official_ranges_are_exact_and_disjoint(self) -> None:
        test = official_webshop_indices(WEBSHOP_TOTAL_HUMAN_GOALS, "test")
        dev = official_webshop_indices(WEBSHOP_TOTAL_HUMAN_GOALS, "dev")
        train = official_webshop_indices(WEBSHOP_TOTAL_HUMAN_GOALS, "train")
        self.assertEqual((test[0], test[-1], len(test)), (0, 499, 500))
        self.assertEqual((dev[0], dev[-1], len(dev)), (500, 1499, 1000))
        self.assertEqual((train[0], train[-1], len(train)), (1500, 12086, 10587))
        self.assertEqual(len(set(test) | set(dev) | set(train)), 12087)

    def test_complete_manifest_cannot_attest_a_subset(self) -> None:
        tasks = tuple(
            FormalTask("webshop", "test", str(index), "evaluate", task_index=index)
            for index in range(50)
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_task_manifest(
                    Path(directory) / "bad.json",
                    dataset_revision=REVISION,
                    tasks=tasks,
                    complete_official_split=True,
                )

    def test_sampled_train_manifest_round_trips(self) -> None:
        tasks = tuple(
            FormalTask("webshop", "train", str(index), "train", task_index=index)
            for index in (1500, 1600, 1700)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_task_manifest(
                Path(directory) / "train.json",
                dataset_revision=REVISION,
                tasks=tasks,
                complete_official_split=False,
            )
            loaded = OfficialTaskManifest.load(path)
            self.assertEqual(len(loaded.tasks), 3)
            self.assertFalse(loaded.complete_official_split)


class WebShopActionTests(unittest.TestCase):
    def test_adapter_detects_same_page_same_action_transition(self) -> None:
        class StaticPageEnv:
            instruction_text = "red shoes under 50"

            def reset(self, session=None):
                del session
                return "same page", None

            def get_available_actions(self):
                return {"has_search_bar": True, "clickables": ("next >",)}

            def step(self, action):
                del action
                return "same page", 0.0, False, {}

        adapter = WebShopAdapter(
            session=1500,
            dataset_revision=REVISION,
            split="train",
            env=StaticPageEnv(),
        )
        observation = adapter.reset()
        self.assertTrue(
            adapter.validate_model_action("click[next >]", observation).accepted
        )
        repeated_observation = adapter.step("click[next >]").observation
        self.assertEqual(
            observation.environment_digest,
            repeated_observation.environment_digest,
        )
        self.assertFalse(
            adapter.validate_model_action(
                "click[next >]", repeated_observation
            ).accepted
        )

    def test_invalid_and_repeated_actions_request_recovery(self) -> None:
        valid = ("search[<keywords>]", "click[b0123]", "click[next >]")
        invalid = check_webshop_action("thought", valid)
        self.assertFalse(invalid.accepted)
        accepted = check_webshop_action("CLICK[B0123]", valid)
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.action, "click[b0123]")
        repeated = check_webshop_action(
            "click[b0123]", valid, attempted_actions=("click[b0123]",)
        )
        self.assertFalse(repeated.accepted)
        self.assertIn("without progress", repeated.feedback)

    def test_fallback_uses_public_goal_and_avoids_attempts(self) -> None:
        fallback = webshop_fallback_action(
            ("search[<keywords>]", "click[next >]"),
            goal="red shoes under 50",
            attempted_actions=("click[next >]",),
        )
        self.assertEqual(fallback, "search[red shoes under 50]")


class PairedGateTests(unittest.TestCase):
    def test_summary_distinguishes_total_and_exclusive_success(self) -> None:
        episodes = []
        edge_success = {"a", "b", "shared"}
        cloud_success = {"c", "d", "e", "shared"}
        for task_id in ("a", "b", "c", "d", "e", "shared"):
            episodes.append(
                endpoint_episode(
                    task_id,
                    Executor.EDGE,
                    float(task_id in edge_success),
                    split="test",
                )
            )
            episodes.append(
                endpoint_episode(
                    task_id,
                    Executor.CLOUD,
                    float(task_id in cloud_success),
                    split="test",
                )
            )
        summary = summarize_paired_endpoints(episodes)
        self.assertEqual(summary["edge"]["total_success"], 3)
        self.assertEqual(summary["edge"]["exclusive_success"], 2)
        self.assertEqual(summary["cloud"]["total_success"], 4)
        self.assertEqual(summary["cloud"]["exclusive_success"], 3)
        self.assertEqual(summary["both_success"], 1)
        self.assertEqual(summary["success_union_rate"], 1.0)

    def test_task_rows_use_one_pre_action_row_per_endpoint(self) -> None:
        episodes = [
            endpoint_episode("0", Executor.EDGE, 1.0, split="train", signal=1.0),
            endpoint_episode("0", Executor.CLOUD, 0.0, split="train", signal=1.0),
            endpoint_episode("1", Executor.EDGE, 0.0, split="train", signal=-1.0),
            endpoint_episode("1", Executor.CLOUD, 1.0, split="train", signal=-1.0),
        ]
        rows = task_router_training_rows(episodes, authorized_train_splits=("train",))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.step_index == 0 for row in rows))
        self.assertEqual({row.sample_id for row in rows}, {"0", "1"})

    def test_learnability_gate_uses_disjoint_dev_outcomes(self) -> None:
        class FakeRouter:
            def predicted_reward_by_executor(self, row_features):
                if row_features["goal_hash_00"] > 0:
                    return {Executor.EDGE: 1.0, Executor.CLOUD: 0.0}
                return {Executor.EDGE: 0.0, Executor.CLOUD: 1.0}

        episodes = []
        for index, signal in enumerate((1.0, -1.0, 1.0, -1.0)):
            edge_reward = float(signal > 0)
            episodes.extend(
                (
                    endpoint_episode(
                        str(index), Executor.EDGE, edge_reward, split="dev", signal=signal
                    ),
                    endpoint_episode(
                        str(index), Executor.CLOUD, 1.0 - edge_reward, split="dev", signal=signal
                    ),
                )
            )
        gate = evaluate_router_learnability(
            FakeRouter(),
            episodes,
            minimum_paired_tasks=4,
            minimum_oracle_capture=0.3,
        )
        self.assertTrue(gate["gate_pass"])
        self.assertEqual(gate["router_avg_reward"], 1.0)
        self.assertEqual(gate["oracle_gap_capture"], 1.0)
        self.assertEqual(gate["cloud_selection_fraction"], 0.5)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "AgentRelay[ml] not installed")
    def test_joint_router_fits_continuous_task_reward(self) -> None:
        train = []
        for index in range(20):
            signal = 1.0 if index % 2 == 0 else -1.0
            edge_reward = float(signal > 0)
            train.extend(
                (
                    endpoint_episode(
                        str(index), Executor.EDGE, edge_reward, split="train", signal=signal
                    ),
                    endpoint_episode(
                        str(index), Executor.CLOUD, 1.0 - edge_reward, split="train", signal=signal
                    ),
                )
            )
        rows = task_router_training_rows(train, authorized_train_splits=("train",))
        router = JointRouterEstimator().fit(rows, authorized_train_splits=("train",))
        positive = router.predicted_reward_by_executor(features(1.0))
        negative = router.predicted_reward_by_executor(features(-1.0))
        self.assertGreater(positive[Executor.EDGE], positive[Executor.CLOUD])
        self.assertGreater(negative[Executor.CLOUD], negative[Executor.EDGE])
        self.assertEqual(router.metadata["independent_task_count"], 20)
        self.assertEqual(router.metadata["quality_target"], "continuous_episode_reward")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.joblib"
            router.save(path)
            loaded = JointRouterEstimator.load(path)
            self.assertEqual(
                loaded.predicted_reward_by_executor(features(1.0)),
                positive,
            )

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "AgentRelay[ml] not installed")
    def test_train_dev_gate_cli_round_trip(self) -> None:
        train = []
        dev = []
        for index in range(20):
            signal = 1.0 if index % 2 == 0 else -1.0
            edge_reward = float(signal > 0)
            train.extend(
                (
                    endpoint_episode(
                        str(index), Executor.EDGE, edge_reward, split="train", signal=signal
                    ),
                    endpoint_episode(
                        str(index), Executor.CLOUD, 1.0 - edge_reward, split="train", signal=signal
                    ),
                )
            )
        for index in range(4):
            signal = 1.0 if index % 2 == 0 else -1.0
            edge_reward = float(signal > 0)
            dev.extend(
                (
                    endpoint_episode(
                        str(index + 500),
                        Executor.EDGE,
                        edge_reward,
                        split="dev",
                        signal=signal,
                    ),
                    endpoint_episode(
                        str(index + 500),
                        Executor.CLOUD,
                        1.0 - edge_reward,
                        split="dev",
                        signal=signal,
                    ),
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            dev_path = root / "dev.jsonl"
            router_path = root / "router.joblib"
            gate_path = root / "gate.json"
            train_path.write_text(
                "".join(canonical_json(item) + "\n" for item in train),
                encoding="utf-8",
            )
            dev_path.write_text(
                "".join(canonical_json(item) + "\n" for item in dev),
                encoding="utf-8",
            )
            completed = subprocess.run(
                (
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_router_learnability_gate.py"),
                    str(train_path),
                    str(dev_path),
                    str(router_path),
                    str(gate_path),
                    "--minimum-paired-tasks",
                    "4",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertTrue(gate["gate_pass"])
            self.assertEqual(gate["independent_train_tasks"], 20)
            self.assertEqual(gate["paired_dev_tasks"], 4)


if __name__ == "__main__":
    unittest.main()
