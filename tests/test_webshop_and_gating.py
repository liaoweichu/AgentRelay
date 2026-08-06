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
    FormalMatrixRunner,
    FormalTask,
    OfficialTaskManifest,
    required_model_executors,
    task_artifact_scope,
    write_task_manifest,
)
from agentrelay.config import GEMMA4_FORMAL_MODEL_PAIR, StorageLayout
from agentrelay.gating import evaluate_router_learnability, summarize_paired_endpoints
from agentrelay.learning import FEATURE_NAMES, JointRouterEstimator
from agentrelay.official_adapters import WebShopAdapter
from agentrelay.provenance import source_tree_revision
from agentrelay.router_data import task_router_training_rows
from agentrelay.schema import Executor, canonical_json, sha256_json
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
    episode = {
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
    episode["endpoint_provenance"] = {
        "run_id": f"{split}-{role.value}",
        "manifest_hash": sha256_json([split, role.value, "endpoint"]),
        "code_revision": "e" * 40,
        "config_hash": "2" * 64,
        "profile_hash": "3" * 64,
        "model_ids": dict(GEMMA4_FORMAL_MODEL_PAIR),
        "model_revisions": {"edge": "f" * 40, "cloud": "1" * 40},
        "task_manifest_hash": sha256_json([split, "tasks"]),
    }
    return episode


class WebShopSplitTests(unittest.TestCase):
    def test_formal_matrix_resume_validates_and_reuses_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = StorageLayout(Path(directory) / "store")
            storage.create()
            task = FormalTask(
                "webshop",
                "train",
                "1500",
                "train",
                task_index=1500,
            )
            task_manifest = OfficialTaskManifest(
                dataset_revision=REVISION,
                complete_official_split=False,
                tasks=(task,),
                manifest_hash="a" * 64,
            )
            runner = object.__new__(FormalMatrixRunner)
            runner.config = {
                "models": {
                    "edge": {
                        "model_id": GEMMA4_FORMAL_MODEL_PAIR["edge"],
                        "revision": "b" * 40,
                    },
                    "cloud": {
                        "model_id": GEMMA4_FORMAL_MODEL_PAIR["cloud"],
                        "revision": "c" * 40,
                    },
                }
            }
            runner.task_manifest = task_manifest
            runner.methods = (BaselineName.EDGE_ONLY,)
            runner.storage = storage
            runner.executors = {Executor.EDGE: object()}
            runner._shared_envs = {}
            runner.profile_hash = "d" * 64
            resume_key = "unit-resume"
            run_id = f"formal-matrix-{resume_key}-{task_manifest.manifest_hash[:8]}"
            run_root = storage.runs / run_id
            run_root.mkdir()
            context = {
                "schema_version": "1.0",
                "run_id": run_id,
                "task_manifest_hash": task_manifest.manifest_hash,
                "code_revision": source_tree_revision(),
                "config_hash": sha256_json(runner.config),
                "profile_hash": runner.profile_hash,
                "model_ids": dict(GEMMA4_FORMAL_MODEL_PAIR),
                "model_revisions": {"edge": "b" * 40, "cloud": "c" * 40},
                "methods": ["edge_only"],
                "resident_executors": ["edge"],
                "paper_evidence": False,
                "artifact_scope": "train_dev_development",
            }
            context["context_hash"] = sha256_json(context)
            (run_root / "run-context.json").write_text(
                canonical_json(context) + "\n",
                encoding="utf-8",
            )
            relative = Path("webshop/train/edge_only/1500.json")
            target = run_root / relative
            target.parent.mkdir(parents=True)
            episode = endpoint_episode(
                "1500",
                Executor.EDGE,
                1.0,
                split="train",
            )
            episode["benchmark"] = "webshop"
            episode["paper_evidence"] = False
            episode["result_hash"] = sha256_json(episode)
            target.write_text(canonical_json(episode) + "\n", encoding="utf-8")
            completed_root = runner.run(resume_key=resume_key)
            self.assertEqual(completed_root, run_root.resolve())
            manifest = json.loads(
                (completed_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["results"]), 1)
            self.assertFalse(manifest["paper_evidence"])
            target.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resume episode hash mismatch"):
                runner.run(resume_key=resume_key)

    def test_train_dev_artifacts_are_not_promoted_to_paper_evidence(self) -> None:
        train_task = FormalTask("webshop", "train", "1500", "train", task_index=1500)
        dev_task = FormalTask("webshop", "dev", "500", "tune", task_index=500)
        test_task = FormalTask("webshop", "test", "0", "evaluate", task_index=0)
        self.assertEqual(
            task_artifact_scope((train_task,)),
            (False, "train_dev_development"),
        )
        self.assertEqual(
            task_artifact_scope((dev_task,)),
            (False, "train_dev_development"),
        )
        self.assertEqual(
            task_artifact_scope((test_task,)),
            (True, "official_evaluation"),
        )
        with self.assertRaises(ValueError):
            task_artifact_scope((train_task, test_task))

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
            endpoint_episode("1500", Executor.EDGE, 1.0, split="train", signal=1.0),
            endpoint_episode("1500", Executor.CLOUD, 0.0, split="train", signal=1.0),
            endpoint_episode("1501", Executor.EDGE, 0.0, split="train", signal=-1.0),
            endpoint_episode("1501", Executor.CLOUD, 1.0, split="train", signal=-1.0),
        ]
        rows = task_router_training_rows(episodes, authorized_train_splits=("train",))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row.step_index == 0 for row in rows))
        self.assertEqual({row.sample_id for row in rows}, {"1500", "1501"})

    def test_task_rows_reject_webshop_id_outside_train(self) -> None:
        episodes = [
            endpoint_episode("0", Executor.EDGE, 1.0, split="train"),
            endpoint_episode("0", Executor.CLOUD, 0.0, split="train"),
        ]
        with self.assertRaisesRegex(ValueError, "outside the official train split"):
            task_router_training_rows(
                episodes,
                authorized_train_splits=("train",),
            )

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

    def test_learnability_gate_rejects_test_as_dev(self) -> None:
        class FakeRouter:
            def predicted_reward_by_executor(self, row_features):
                del row_features
                return {Executor.EDGE: 0.5, Executor.CLOUD: 0.5}

        episodes = [
            endpoint_episode("0", Executor.EDGE, 1.0, split="test"),
            endpoint_episode("0", Executor.CLOUD, 0.0, split="test"),
        ]
        with self.assertRaisesRegex(ValueError, "held-out test"):
            evaluate_router_learnability(
                FakeRouter(),
                episodes,
                authorized_dev_splits=("test",),
            )

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "AgentRelay[ml] not installed")
    def test_joint_router_fits_continuous_task_reward(self) -> None:
        train = []
        for index in range(20):
            signal = 1.0 if index % 2 == 0 else -1.0
            edge_reward = float(signal > 0)
            train.extend(
                (
                    endpoint_episode(
                        str(index + 1500),
                        Executor.EDGE,
                        edge_reward,
                        split="train",
                        signal=signal,
                    ),
                    endpoint_episode(
                        str(index + 1500),
                        Executor.CLOUD,
                        1.0 - edge_reward,
                        split="train",
                        signal=signal,
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
                        str(index + 1500),
                        Executor.EDGE,
                        edge_reward,
                        split="train",
                        signal=signal,
                    ),
                    endpoint_episode(
                        str(index + 1500),
                        Executor.CLOUD,
                        1.0 - edge_reward,
                        split="train",
                        signal=signal,
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
                    "--minimum-train-tasks",
                    "20",
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
            underpowered_gate_path = root / "underpowered-gate.json"
            underpowered = subprocess.run(
                (
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_router_learnability_gate.py"),
                    str(train_path),
                    str(dev_path),
                    str(root / "underpowered-router.joblib"),
                    str(underpowered_gate_path),
                    "--minimum-paired-tasks",
                    "4",
                    "--minimum-train-tasks",
                    "21",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(underpowered.returncode, 2, underpowered.stderr)
            underpowered_gate = json.loads(
                underpowered_gate_path.read_text(encoding="utf-8")
            )
            self.assertFalse(underpowered_gate["checks"]["enough_train_tasks"])
            self.assertFalse(underpowered_gate["gate_pass"])

    def test_gate_orchestrator_rejects_smaller_than_predeclared_sample(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_webshop_train_dev_gate.py"),
                "--config",
                "missing.json",
                "--profile",
                "missing-profile.json",
                "--webshop-file",
                "missing-items.json",
                "--revision",
                REVISION,
                "--train-count",
                "199",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("at least 200 train tasks", completed.stderr)

    def test_endpoint_collector_validates_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_roots = []
            for role, method in (
                (Executor.EDGE, "edge_only"),
                (Executor.CLOUD, "cloud_only"),
            ):
                run_root = root / method
                relative = Path("webshop") / "train" / method / "1500.json"
                target = run_root / relative
                target.parent.mkdir(parents=True)
                episode = endpoint_episode(
                    "1500",
                    role,
                    1.0 if role is Executor.EDGE else 0.0,
                    split="train",
                )
                episode["paper_evidence"] = False
                episode["result_hash"] = sha256_json(episode)
                target.write_text(canonical_json(episode) + "\n", encoding="utf-8")
                context = {
                    "schema_version": "1.0",
                    "run_id": method,
                    "task_manifest_hash": "a" * 64,
                    "code_revision": "b" * 40,
                    "config_hash": "e" * 64,
                    "profile_hash": "f" * 64,
                    "model_ids": dict(GEMMA4_FORMAL_MODEL_PAIR),
                    "model_revisions": {"edge": "c" * 40, "cloud": "d" * 40},
                    "methods": [method],
                    "resident_executors": [role.value],
                    "paper_evidence": False,
                    "artifact_scope": "train_dev_development",
                }
                context["context_hash"] = sha256_json(context)
                (run_root / "run-context.json").write_text(
                    canonical_json(context) + "\n",
                    encoding="utf-8",
                )
                manifest = {
                    "run_id": method,
                    "paper_evidence": False,
                    "artifact_scope": "train_dev_development",
                    "task_manifest_hash": "a" * 64,
                    "code_revision": "b" * 40,
                    "config_hash": "e" * 64,
                    "profile_hash": "f" * 64,
                    "model_ids": dict(GEMMA4_FORMAL_MODEL_PAIR),
                    "model_revisions": {"edge": "c" * 40, "cloud": "d" * 40},
                    "methods": [method],
                    "resident_executors": [role.value],
                    "results": [
                        {
                            "path": relative.as_posix(),
                            "result_hash": episode["result_hash"],
                            "sample_id": "1500",
                            "method": method,
                        }
                    ],
                    "labels_accessed_by_router": False,
                    "run_context_hash": context["context_hash"],
                }
                manifest["manifest_hash"] = sha256_json(manifest)
                (run_root / "manifest.json").write_text(
                    canonical_json(manifest) + "\n",
                    encoding="utf-8",
                )
                run_roots.append(run_root)
            output = root / "paired.jsonl"
            completed = subprocess.run(
                (
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "collect_endpoint_episodes.py"),
                    str(run_roots[0]),
                    str(run_roots[1]),
                    "--split",
                    "train",
                    "--output",
                    str(output),
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            episodes = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(episodes), 2)
            self.assertTrue(all("endpoint_provenance" in item for item in episodes))


if __name__ == "__main__":
    unittest.main()
