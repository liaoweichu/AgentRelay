#!/usr/bin/env python3
"""Enumerate a complete pinned official benchmark split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.formal_matrix import FormalTask, write_task_manifest  # noqa: E402
from agentrelay.inference import require_immutable_revision  # noqa: E402
from agentrelay.webshop_protocol import (  # noqa: E402
    canonical_webshop_split,
    official_webshop_indices,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("alfworld", "webshop", "appworld"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--purpose", choices=("train", "tune", "evaluate"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alfworld-config")
    parser.add_argument("--train-eval")
    parser.add_argument("--webshop-file-path")
    parser.add_argument("--webshop-human-goals", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260805)
    args = parser.parse_args()
    require_immutable_revision(args.revision, subject=args.benchmark)
    tasks: list[FormalTask] = []
    # ALFWorld and AppWorld enumerate the complete official split by
    # construction; WebShop may produce a sampled diagnostic subset instead.
    complete = True

    if args.benchmark == "alfworld":
        if not args.alfworld_config or not args.train_eval:
            raise ValueError("ALFWorld requires --alfworld-config and --train-eval")
        try:
            import yaml
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise RuntimeError("install the pinned official ALFWorld environment") from exc
        config = yaml.safe_load(Path(args.alfworld_config).read_text(encoding="utf-8"))
        raw_env = get_environment(config["env"]["type"])(config, train_eval=args.train_eval)
        for index, gamefile in enumerate(raw_env.game_files):
            tasks.append(
                FormalTask(
                    benchmark="alfworld",
                    split=args.split,
                    task_id=Path(gamefile).parent.name,
                    purpose=args.purpose,
                    task_index=index,
                    train_eval=args.train_eval,
                    alfworld_config=str(Path(args.alfworld_config).resolve()),
                )
            )
    elif args.benchmark == "webshop":
        try:
            from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        except ImportError as exc:
            raise RuntimeError("install the pinned official WebShop environment") from exc
        kwargs = {
            "observation_mode": "text",
            "human_goals": args.webshop_human_goals,
        }
        if args.webshop_file_path:
            kwargs["file_path"] = args.webshop_file_path
        env = WebAgentTextEnv(**kwargs)
        try:
            count = len(env.server.goals)
        finally:
            env.close()
        split = canonical_webshop_split(args.split)
        expected_purpose = {"train": "train", "dev": "tune", "test": "evaluate"}[split]
        if args.purpose != expected_purpose:
            raise ValueError(
                f"official WebShop {split} manifests require purpose={expected_purpose!r}, "
                f"got {args.purpose!r}"
            )
        official_indices = official_webshop_indices(count, split)
        if args.sample_count is not None:
            if not 0 < args.sample_count <= len(official_indices):
                raise ValueError(
                    f"sample_count must be in (0, {len(official_indices)}], "
                    f"got {args.sample_count}"
                )
            indices = sorted(
                random.Random(args.sample_seed).sample(official_indices, args.sample_count)
            )
            # A deterministic subset is a diagnostic sample, never the complete
            # official split, so it must not be attested as complete.
            complete = False
        else:
            indices = official_indices
            complete = True
        for index in indices:
            tasks.append(
                FormalTask(
                    benchmark="webshop",
                    split=split,
                    task_id=str(index),
                    purpose=args.purpose,
                    task_index=index,
                    webshop_file_path=str(args.webshop_file_path or ""),
                )
            )
    else:
        try:
            from appworld import load_task_ids
        except ImportError as exc:
            raise RuntimeError("install the pinned official AppWorld environment") from exc
        tasks.extend(
            FormalTask(
                benchmark="appworld",
                split=args.split,
                task_id=str(task_id),
                purpose=args.purpose,
            )
            for task_id in load_task_ids(args.split)
        )

    target = write_task_manifest(
        args.output,
        dataset_revision=args.revision,
        tasks=tasks,
        complete_official_split=complete,
    )
    print(f"benchmark={args.benchmark} split={args.split} tasks={len(tasks)} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
