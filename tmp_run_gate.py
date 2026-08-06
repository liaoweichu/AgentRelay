#!/usr/bin/env python3
"""Capability gate: run a fixed model (edge/cloud) on a gate manifest.

Loads Gemma 4 (or Qwen) models directly from a local snapshot dir and runs
edge_only / cloud_only native episodes over the official adapters, reporting
per-task success, reward, and a summary of capability divergence.  This is a
diagnostic tune-purpose gate, not the formal test matrix.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from agentrelay.config import StorageLayout  # noqa: E402
from agentrelay.continuation import render_semantic_continuation  # noqa: E402
from agentrelay.formal_matrix import OfficialTaskManifest  # noqa: E402
from agentrelay.inference import HFModelExecutor, NativeGenerationConfig  # noqa: E402
from agentrelay.official_adapters import ALFWorldAdapter, WebShopAdapter  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402


def build_executor(snapshot: str, *, storage: StorageLayout, max_new_tokens: int) -> HFModelExecutor:
    # Gemma 4 is fetched from a pinned ModelScope snapshot; no upstream git
    # commit hash exists, so derive a stable 40-hex local-snapshot revision from
    # the canonoical snapshot path purely as a provenance cache marker.
    local_revision = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()[:40]
    cfg = NativeGenerationConfig(
        model_id=snapshot,
        revision=local_revision,
        dtype="bfloat16",
        quantization="bnb_4bit",
        device_map="auto",
        max_new_tokens=max_new_tokens,
        do_sample=False,
        seed=0,
        architecture="multimodal_lm",
        enable_thinking=False,
        local_files_only=True,
        trust_remote_code=False,
    )
    return HFModelExecutor(cfg, storage)


def run_episode(adapter, executor, *, max_steps: int) -> dict:
    started = time.perf_counter()
    observation = adapter.reset()
    packet = adapter.build_packet(None)
    steps = []
    for _ in range(max_steps):
        continuation = render_semantic_continuation(packet, style="concise_text")
        messages = adapter.format_model_messages(observation, continuation)
        generation = executor.generate(messages)
        action = adapter.parse_model_output(generation.text)
        result = adapter.step(action)
        steps.append(
            {
                "action": action,
                "response": generation.text,
                "reward": result.reward,
                "done": result.observation.done,
            }
        )
        packet = adapter.build_packet(packet)
        observation = result.observation
        if result.observation.done:
            break
    evaluation = adapter.evaluate()
    return {
        "success": float(evaluation.success),
        "reward": float(evaluation.reward),
        "official_metrics": dict(evaluation.official_metrics),
        "steps": len(steps),
        "end_to_end_ms": (time.perf_counter() - started) * 1000.0,
        "last_actions": [s["action"] for s in steps[-5:]],
    }


def make_adapter(task, *, revision: str, shared_env):
    if task.benchmark == "webshop":
        return WebShopAdapter(
            session=task.task_index,
            dataset_revision=revision,
            split=task.split,
            file_path=task.webshop_file_path,
            env=shared_env,
        )
    if task.benchmark == "alfworld":
        return ALFWorldAdapter(
            config_path=task.alfworld_config,
            train_eval=task.train_eval,
            dataset_revision=revision,
            split=task.split,
            task_index=task.task_index,
        )
    raise ValueError(f"unsupported gate benchmark {task.benchmark!r}")


WS_REV = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
ALF_REV = "aaba6870f86c5be6a08a491f32a50b906227bc3e"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("edge_snapshot")
    ap.add_argument("cloud_snapshot")
    ap.add_argument("output")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    manifest = OfficialTaskManifest.load(args.manifest)
    storage = StorageLayout(Path("/root/autodl-tmp/AgentRelay").resolve())

    shared_env = None
    if any(t.benchmark == "webshop" for t in manifest.tasks):
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        file_path = next(t.webshop_file_path for t in manifest.tasks if t.webshop_file_path)
        shared_env = WebAgentTextEnv(
            observation_mode="text",
            human_goals=1,
            file_path=file_path,
        )

    # Serial model residency: load exactly one model at a time, run that model
    # across every manifest task, then release it before loading the next.  This
    # keeps peak GPU residency to a single 4-bit model instead of both models
    # simultaneously, which otherwise OOMs the 24 GiB card on long episodes.
    rows = []
    for role, snapshot in (("edge", args.edge_snapshot), ("cloud", args.cloud_snapshot)):
        print(f"loading {role} {snapshot}", flush=True)
        executor = build_executor(snapshot, storage=storage, max_new_tokens=args.max_new_tokens)
        for task in manifest.tasks:
            rev = WS_REV if task.benchmark == "webshop" else ALF_REV
            adapter = make_adapter(task, revision=rev, shared_env=shared_env)
            try:
                result = run_episode(adapter, executor, max_steps=args.max_steps)
                result["task_id"] = task.task_id
                result["benchmark"] = task.benchmark
                result["split"] = task.split
                result["role"] = role
                result["dataset_revision"] = rev
                rows.append(result)
                print(
                    f"[{task.benchmark}] task={task.task_id} {role}: "
                    f"success={result['success']} reward={result['reward']:.4f} "
                    f"steps={result['steps']}",
                    flush=True,
                )
            finally:
                adapter.close()
        # Release this model before loading the next one.
        del executor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if shared_env is not None:
        try:
            shared_env.close()
        except Exception:  # noqa: BLE001
            pass

    # per-task edge vs cloud divergence
    by_task = {}
    for r in rows:
        by_task.setdefault((r["benchmark"], r["task_id"]), {})[r["role"]] = r
    diverged = 0
    for key, pair in by_task.items():
        if "edge" in pair and "cloud" in pair:
            if abs(pair["edge"]["reward"] - pair["cloud"]["reward"]) > 1e-6:
                diverged += 1

    def agg(role):
        rs = [r for r in rows if r["role"] == role]
        if not rs:
            return None
        return {
            "n": len(rs),
            "success_rate": sum(r["success"] for r in rs) / len(rs),
            "avg_reward": sum(r["reward"] for r in rs) / len(rs),
            "exclusive_success": sum(1 for r in rs if r["success"] > 0),
        }

    summary = {
        "manifest": str(args.manifest),
        "edge_snapshot": args.edge_snapshot,
        "cloud_snapshot": args.cloud_snapshot,
        "edge": agg("edge"),
        "cloud": agg("cloud"),
        "reward_diverged_tasks": diverged,
        "total_tasks": len(by_task),
        "rows": rows,
    }
    summary["summary_hash"] = sha256_json(
        {k: v for k, v in summary.items() if k != "summary_hash"}
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    tmp.replace(out)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())