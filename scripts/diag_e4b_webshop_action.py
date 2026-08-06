#!/usr/bin/env python3
"""Diagnose why E4B fails to produce a valid WebShop action in the gate.

Replicates the exact failure path for a single WebShop task: pinned config ->
shared official env -> WebShopAdapter -> packet -> messages -> E4B generate ->
parse -> validate -> step.  Runs the full episode (up to max_steps) and prints
the raw model text and validation reason at the step where it crashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import StorageLayout, load_json_config  # noqa: E402
from agentrelay.continuation import render_semantic_continuation  # noqa: E402
from agentrelay.inference import HFModelExecutor, NativeGenerationConfig  # noqa: E402
from agentrelay.official_adapters import WebShopAdapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/formal-autodl-4090d.locked.json")
    parser.add_argument("--webshop-file", default="repositories/webshop/data/items_shuffle.json")
    parser.add_argument("--revision", default="64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd")
    parser.add_argument("--session", type=int, default=1500)
    parser.add_argument("--split", default="train")
    parser.add_argument("--role", default="edge")
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    webshop_file = Path(args.webshop_file).resolve()
    config = load_json_config(config_path)
    storage = StorageLayout(Path(config["data_root"]).resolve())

    model_value = config["models"][args.role]
    model_config = NativeGenerationConfig.from_dict(model_value)
    print(f"model   = {model_config.model_id} @ {model_config.revision} "
          f"({model_config.model_source}, {model_config.quantization})")
    print(f"session = {args.session} split={args.split} max_steps={args.max_steps}")

    executor = HFModelExecutor(model_config, storage)

    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env = WebAgentTextEnv(
        observation_mode="text",
        human_goals=1,
        file_path=str(webshop_file),
    )
    adapter = WebShopAdapter(
        session=args.session,
        dataset_revision=args.revision,
        split=args.split,
        file_path=str(webshop_file),
        env=env,
    )

    observation = adapter.reset()
    packet = adapter.build_packet(None)
    done = False
    for step_index in range(args.max_steps):
        continuation = render_semantic_continuation(packet, style="concise_text")
        messages = list(adapter.format_model_messages(observation, continuation))
        print(f"\n===== step {step_index} =====")
        print(f"valid_actions = {list(observation.valid_actions)}")
        print(f"observation   = {observation.text[:200]!r}")

        generations = []
        rejected = []
        action = None
        for attempt in range(2):
            result = executor.generate(messages)
            generations.append(result)
            parsed = adapter.parse_model_output(result.text)
            validation = adapter.validate_model_action(parsed, observation)
            print(f"  attempt {attempt}: raw={result.text!r} parsed={parsed!r} "
                  f"accepted={validation.accepted}")
            if validation.accepted:
                action = validation.action
                break
            print(f"    feedback={validation.feedback!r}")
            rejected.append(validation.action)
            messages.extend(
                (
                    {"role": "assistant", "content": result.text},
                    {"role": "user",
                     "content": validation.feedback + "\nCorrected next action:"},
                )
            )
        if action is None:
            fallback = adapter.fallback_model_action(observation, tuple(rejected))
            print(f"  fallback={fallback!r}")
            if fallback:
                v = adapter.validate_model_action(fallback, observation)
                if v.accepted:
                    action = v.action
            if action is None:
                print(f"  >>> CRASH at step {step_index}: no valid action after 2 attempts "
                      f"+ fallback")
                return 1

        result_step = adapter.step(action)
        observation = result_step.observation
        packet = adapter.build_packet(packet)
        done = observation.done
        print(f"  -> reward={result_step.reward} done={done}")
        if done:
            print(f"  episode finished at step {step_index}")
            return 0

    print("episode reached max_steps without crashing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())