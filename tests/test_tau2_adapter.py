from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_ROOT))

from agentrelay.config import load_json_config, validate_experiment_config
from agentrelay.learning import FEATURE_NAMES
from agentrelay.schema import canonical_json
from agentrelay.tau2_adapter import (
    TAU2_DOMAINS,
    TAU2_PINNED_REVISION,
    Tau2TaskRef,
    Tau2UserSimulatorConfig,
    build_tau2_router_manifest,
    collect_resumable_episodes,
    load_resumable_episode,
    parse_tau2_action,
    prepare_run_context,
    read_tau2_manifest,
    save_resumable_episode,
    tau2_router_features,
)


class Tau2AdapterTests(unittest.TestCase):
    def test_fixed_modelscope_and_tau2_contract(self) -> None:
        config = load_json_config(PROJECT_ROOT / "configs" / "formal-autodl-4090d.locked.json")
        validate_experiment_config(config)
        self.assertEqual(
            {role: model["model_id"] for role, model in config["models"].items()},
            {"edge": "google/gemma-4-E4B-it", "cloud": "google/gemma-4-12b-it"},
        )
        for model in config["models"].values():
            self.assertEqual(model["model_source"], "modelscope")
            self.assertEqual(model["dtype"], "bfloat16")
            self.assertEqual(model["quantization"], "bnb_4bit")
        tau2 = next(item for item in config["repositories"] if item["name"] == "tau2-bench")
        self.assertEqual(tau2["revision"], TAU2_PINNED_REVISION)
        self.assertIn('"modelscope==1.39.1"', (PROJECT_ROOT / "pyproject.toml").read_text())

    def test_formal_pair_rejects_symmetric_precision_drift(self) -> None:
        config = load_json_config(PROJECT_ROOT / "configs" / "formal-autodl-4090d.locked.json")
        for model in config["models"].values():
            model["dtype"] = "float16"
            model["quantization"] = "none"
        with self.assertRaisesRegex(ValueError, "formal value"):
            validate_experiment_config(config)

    def test_parse_structured_tool_and_text(self) -> None:
        tool = parse_tau2_action(
            '```json\n{"tool":"lookup","arguments":{"id":"7"}}\n```',
            allowed_tools=("lookup",),
        )
        self.assertEqual(tool.tool_name, "lookup")
        self.assertEqual(tool.arguments, {"id": "7"})
        message = parse_tau2_action(
            '{"message":"Please confirm."}',
            allowed_tools=("lookup",),
        )
        self.assertEqual(message.content, "Please confirm.")
        invalid = parse_tau2_action(
            '{"tool":"hidden_tool","arguments":{}}',
            allowed_tools=("lookup",),
        )
        self.assertEqual(invalid.recovery, "invalid_tool_as_text")

    def test_router_features_are_complete_and_public_input_sensitive(self) -> None:
        first = tau2_router_features(
            "Please change order 123, but do not cancel it.",
            policy="public policy",
            tool_count=4,
            max_steps=100,
            edge_ms=10.0,
            cloud_ms=20.0,
            bandwidth_mbps=30.0,
        )
        second = tau2_router_features(
            "My phone has no service.",
            policy="public policy",
            tool_count=4,
            max_steps=100,
            edge_ms=10.0,
            cloud_ms=20.0,
            bandwidth_mbps=30.0,
        )
        self.assertEqual(tuple(first), FEATURE_NAMES)
        self.assertEqual(first["goal_numeric_count"], 1.0)
        self.assertNotEqual(
            [first[f"goal_hash_{index:02d}"] for index in range(32)],
            [second[f"goal_hash_{index:02d}"] for index in range(32)],
        )

    def test_manifest_is_three_domain_deterministic_and_never_uses_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for domain, count in (("airline", 10), ("retail", 20), ("telecom", 20)):
                split_path = root / "data" / "tau2" / "domains" / domain / "split_tasks.json"
                split_path.parent.mkdir(parents=True)
                split_path.write_text(
                    json.dumps(
                        {
                            "train": [f"{domain}-train-{index}" for index in range(count)],
                            "test": [f"{domain}-test-{index}" for index in range(4)],
                        }
                    ),
                    encoding="utf-8",
                )
            with patch("agentrelay.tau2_adapter.verify_tau2_repository"):
                first = build_tau2_router_manifest(root, split_seed=42)
                second = build_tau2_router_manifest(root, split_seed=42)
            self.assertEqual(first, second)
            self.assertEqual(
                {item["domain"] for item in first["tasks"]},
                set(TAU2_DOMAINS),
            )
            self.assertFalse(first["held_out_split_accessed"])
            self.assertTrue(
                all("-test-" not in item["task_id"] for item in first["tasks"])
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(canonical_json(first), encoding="utf-8")
            train = read_tau2_manifest(manifest_path, split="train")
            dev = read_tau2_manifest(manifest_path, split="dev")
            self.assertFalse(
                {(task.domain, task.task_id) for task in train}
                & {(task.domain, task.task_id) for task in dev}
            )

    def test_fixed_user_simulator_config_contains_no_secret(self) -> None:
        path = PROJECT_ROOT / "configs" / "tau2-user-simulator.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = Tau2UserSimulatorConfig.load(path, require_secret=False)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.benchmark_revision, TAU2_PINNED_REVISION)
        self.assertEqual(raw["api_key_env"], "OPENAI_API_KEY")
        self.assertNotIn("api_key", raw)

    def test_resume_is_scoped_and_hash_checked(self) -> None:
        task = Tau2TaskRef("airline", "7", "train", "dev", 123)
        with tempfile.TemporaryDirectory() as temporary:
            context, context_hash = prepare_run_context(
                temporary,
                {"scope": "test", "model": "edge"},
            )
            self.assertEqual(context["context_hash"], context_hash)
            episode = {
                "sample_id": "airline:7",
                "role": "edge",
                "split": "dev",
                "endpoint_provenance": {"manifest_hash": context_hash},
            }
            save_resumable_episode(temporary, task, episode)
            self.assertEqual(
                load_resumable_episode(
                    temporary,
                    task,
                    context_hash=context_hash,
                    role="edge",
                ),
                episode,
            )
            self.assertEqual(
                collect_resumable_episodes(
                    temporary,
                    (task,),
                    context_hash=context_hash,
                    role="edge",
                ),
                (episode,),
            )
            with self.assertRaises(RuntimeError):
                prepare_run_context(temporary, {"scope": "changed", "model": "edge"})
            wrapper_path = next((Path(temporary) / "episodes").glob("*.json"))
            wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
            wrapper["episode"]["split"] = "train"
            wrapper_path.write_text(canonical_json(wrapper), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_resumable_episode(
                    temporary,
                    task,
                    context_hash=context_hash,
                    role="edge",
                )

    def test_tau2_gate_rejects_nonformal_active_precision(self) -> None:
        script = PROJECT_ROOT / "scripts" / "run_router_learnability_gate.py"
        spec = importlib.util.spec_from_file_location("tau2_gate_test_module", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_ids = {
            "edge": "google/gemma-4-E4B-it",
            "cloud": "google/gemma-4-12b-it",
        }
        revisions = {"edge": "a" * 40, "cloud": "b" * 40}
        precisions = {
            role: {
                "model_source": "modelscope",
                "dtype": "bfloat16",
                "quantization": "bnb_4bit",
            }
            for role in ("edge", "cloud")
        }

        def episode(role: str) -> dict:
            provenance = {
                "run_id": f"run-{role}",
                "manifest_hash": f"manifest-{role}",
                "code_revision": "c" * 64,
                "config_hash": "d" * 64,
                "profile_hash": "e" * 64,
                "model_ids": model_ids,
                "model_revisions": revisions,
                "model_precisions": precisions,
                "active_precision": {
                    "model_id": model_ids[role],
                    "revision": revisions[role],
                    **precisions[role],
                },
                "task_manifest_hash": "f" * 64,
                "tau2_revision": TAU2_PINNED_REVISION,
                "user_config_hash": "1" * 64,
            }
            return {"benchmark": "tau2/airline", "role": role, "endpoint_provenance": provenance}

        episodes = (episode("edge"), episode("cloud"))
        scope = module._provenance_scope(episodes, label="test")
        self.assertEqual(scope["model_precisions"], precisions)
        episodes[0]["endpoint_provenance"]["active_precision"]["dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "active precision"):
            module._provenance_scope(episodes, label="test")


if __name__ == "__main__":
    unittest.main()
