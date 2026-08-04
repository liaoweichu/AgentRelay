#!/usr/bin/env python3
"""Convert official validation episodes into empirical risk-calibration rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.calibration import CalibrationRow  # noqa: E402
from agentrelay.policy import CandidateAction  # noqa: E402
from agentrelay.schema import CommitMode, Executor, TransferMode, canonical_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes")
    parser.add_argument("output")
    parser.add_argument("--validation-split", action="append", required=True)
    args = parser.parse_args()
    allowed = set(args.validation_split)
    rows: list[CalibrationRow] = []
    for line_number, line in enumerate(Path(args.episodes).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        episode = json.loads(line)
        if episode.get("split") not in allowed:
            raise ValueError(f"line {line_number} is not from an authorized validation split")
        if episode.get("labels_accessed_by_router") is not False:
            raise ValueError(f"line {line_number} violates the router label boundary")
        success = int(float(episode.get("success", 0.0)) > 0.0)
        effect_failure = int(int(episode.get("effect_failures", 0)) > 0)
        for step in episode.get("steps", ()):
            row = CalibrationRow(
                dataset_id=str(episode["benchmark"]),
                dataset_revision=str(episode["dataset_revision"]),
                split=str(episode["split"]),
                sample_id=str(episode["sample_id"]),
                purpose="tune",
                action=CandidateAction(
                    Executor(step["selected_executor"]),
                    TransferMode(step["transfer_mode"]),
                    CommitMode(step["commit_mode"]),
                ),
                predicted_success=float(step["predicted_success"]),
                predicted_fidelity=float(step["predicted_fidelity"]),
                predicted_effect_risk=float(step["predicted_effect_risk"]),
                observed_success=success,
                observed_fidelity_pass=1,
                observed_effect_failure=effect_failure,
            )
            row.validate(allowed)
            rows.append(row)
    if not rows:
        raise ValueError("no official validation calibration rows were produced")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        "".join(canonical_json(row.to_dict()) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(target)
    print(f"rows={len(rows)} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

