#!/usr/bin/env python3
"""Combine separately resident edge/cloud matrix runs into paired episode JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.router_data import pair_endpoint_episodes  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", nargs=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()
    episodes: list[dict] = []
    identities: set[tuple[str, str]] = set()
    manifests: list[dict] = []
    roots_by_method: dict[str, Path] = {}
    for root_value in args.run_root:
        root = Path(root_value)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest_payload = dict(manifest)
        recorded_manifest_hash = str(manifest_payload.pop("manifest_hash", ""))
        if (
            not recorded_manifest_hash
            or sha256_json(manifest_payload) != recorded_manifest_hash
        ):
            raise ValueError(f"run manifest hash mismatch: {root}")
        context_path = root / "run-context.json"
        if not context_path.is_file():
            raise ValueError(f"run has no immutable context: {root}")
        context = json.loads(context_path.read_text(encoding="utf-8"))
        context_payload = dict(context)
        recorded_context_hash = str(context_payload.pop("context_hash", ""))
        if (
            not recorded_context_hash
            or sha256_json(context_payload) != recorded_context_hash
            or recorded_context_hash != str(manifest.get("run_context_hash", ""))
        ):
            raise ValueError(f"run context hash mismatch: {root}")
        for field in (
            "run_id",
            "task_manifest_hash",
            "code_revision",
            "config_hash",
            "profile_hash",
            "model_revisions",
            "methods",
            "resident_executors",
            "paper_evidence",
            "artifact_scope",
        ):
            if context.get(field) != manifest.get(field):
                raise ValueError(f"run context/manifest mismatch for {field}: {root}")
        methods = tuple(str(method) for method in manifest.get("methods", ()))
        if len(methods) != 1 or methods[0] not in {"edge_only", "cloud_only"}:
            raise ValueError(
                f"endpoint collection requires one fixed method per run: {root}"
            )
        method = methods[0]
        if method in roots_by_method:
            raise ValueError(f"duplicate {method} run roots")
        roots_by_method[method] = root
        expected_resident = "edge" if method == "edge_only" else "cloud"
        if tuple(manifest.get("resident_executors", ())) != (expected_resident,):
            raise ValueError(
                f"{method} run did not use single-model residency: {root}"
            )
        if manifest.get("paper_evidence") is not False:
            raise ValueError(
                "train/dev endpoint runs must be marked paper_evidence=false"
            )
        manifests.append(manifest)
        for item in manifest.get("results", ()):
            if str(item.get("method")) != method:
                raise ValueError(f"run manifest contains a foreign endpoint method: {root}")
            path = root / str(item["path"])
            episode = json.loads(path.read_text(encoding="utf-8"))
            recorded_hash = str(episode.pop("result_hash", ""))
            if not recorded_hash or sha256_json(episode) != recorded_hash:
                raise ValueError(f"episode result hash mismatch: {path}")
            if recorded_hash != str(item.get("result_hash", "")):
                raise ValueError(f"manifest/result hash disagreement: {path}")
            if str(episode.get("split")) != args.split:
                raise ValueError(f"endpoint run contains a foreign split: {path}")
            if str(episode.get("method")) != method:
                raise ValueError(f"episode method mismatch: {path}")
            if episode.get("paper_evidence") is not False:
                raise ValueError(f"train/dev episode is mislabeled as paper evidence: {path}")
            episode["endpoint_provenance"] = {
                "run_id": str(manifest.get("run_id", "")),
                "manifest_hash": recorded_manifest_hash,
                "code_revision": str(manifest.get("code_revision", "")),
                "config_hash": str(manifest.get("config_hash", "")),
                "profile_hash": str(manifest.get("profile_hash", "")),
                "model_revisions": dict(manifest.get("model_revisions", {})),
                "task_manifest_hash": str(manifest.get("task_manifest_hash", "")),
            }
            identity = (str(episode.get("sample_id")), str(episode.get("method")))
            if identity in identities:
                raise ValueError(f"duplicate endpoint episode: {identity}")
            identities.add(identity)
            episodes.append(episode)
    if set(roots_by_method) != {"edge_only", "cloud_only"}:
        raise ValueError("endpoint collection requires one edge_only and one cloud_only run")
    common_fields = (
        "task_manifest_hash",
        "code_revision",
        "config_hash",
        "profile_hash",
        "model_revisions",
    )
    for field in common_fields:
        values = [manifest.get(field) for manifest in manifests]
        if values[0] != values[1]:
            raise ValueError(f"edge/cloud run provenance mismatch for {field}")
    pairs = pair_endpoint_episodes(episodes)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        "".join(
            canonical_json(episode) + "\n"
            for pair in pairs
            for episode in pair
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    print(
        f"paired_tasks={len(pairs)} episodes={len(episodes)} "
        f"code_revision={manifests[0]['code_revision']} output={target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
