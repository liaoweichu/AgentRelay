#!/usr/bin/env python3
"""Audit active configs and formal manifests against the frozen Gemma 4 pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.config import (  # noqa: E402
    GEMMA4_FORMAL_MODEL_PAIR,
    load_json_config,
    validate_experiment_config,
    validate_gemma4_model_pair,
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _is_excluded(relative: str, prefixes: tuple[str, ...]) -> bool:
    return any(relative == prefix or relative.startswith(prefix) for prefix in prefixes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default=str(PROJECT_ROOT / "configs" / "evidence-policy.json"),
    )
    args = parser.parse_args()
    policy = load_json_config(args.policy)
    if policy.get("formal_model_pair") != GEMMA4_FORMAL_MODEL_PAIR:
        raise ValueError("evidence policy does not match the code-level Gemma 4 pair")
    exclusions = tuple(
        str(item["path_prefix"])
        for item in policy.get("excluded_artifacts", ())
    )

    checks: list[dict[str, object]] = []
    issues: list[dict[str, str]] = []

    def check(subject: str, condition: bool, message: str) -> None:
        checks.append({"subject": subject, "passed": bool(condition)})
        if not condition:
            issues.append({"subject": subject, "message": message})

    formal_path = PROJECT_ROOT / "configs" / "formal-autodl-4090d.template.json"
    formal = load_json_config(formal_path)
    try:
        validate_experiment_config(formal, allow_unlocked=True)
        formal_ok = True
    except (KeyError, TypeError, ValueError) as exc:
        formal_ok = False
        issues.append({"subject": _relative(formal_path), "message": str(exc)})
    check(_relative(formal_path), formal_ok, "formal template is not Gemma 4 compliant")

    local_path = PROJECT_ROOT / "configs" / "local-smoke.template.json"
    local = load_json_config(local_path)
    try:
        validate_experiment_config(local, allow_unlocked=True)
        validate_gemma4_model_pair(local["models"])
        local_ok = True
    except (KeyError, TypeError, ValueError) as exc:
        local_ok = False
        issues.append({"subject": _relative(local_path), "message": str(exc)})
    check(_relative(local_path), local_ok, "local template is not Gemma 4 compliant")

    manifest_paths = sorted(
        set((PROJECT_ROOT / "cloud_results").rglob("manifest.json"))
        | set((PROJECT_ROOT / "artifacts").rglob("manifest.json"))
    )
    scanned_evidence = 0
    excluded_manifests = 0
    for path in manifest_paths:
        relative = _relative(path)
        if _is_excluded(relative, exclusions):
            excluded_manifests += 1
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("paper_evidence") is not True:
            continue
        scanned_evidence += 1
        check(
            relative,
            value.get("model_ids") == GEMMA4_FORMAL_MODEL_PAIR,
            "unexcluded paper-evidence manifest lacks the frozen Gemma 4 model_ids",
        )

    report = {
        "valid": not issues,
        "formal_model_pair": dict(GEMMA4_FORMAL_MODEL_PAIR),
        "checks": checks,
        "paper_evidence_manifests_checked": scanned_evidence,
        "historical_manifests_excluded": excluded_manifests,
        "issues": issues,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
