#!/usr/bin/env python3
"""Convert immutable official-train episode artifacts into router rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.cost import HandoffMeasurement  # noqa: E402
from agentrelay.learning import RouterTrainingRow  # noqa: E402
from agentrelay.policy import CandidateAction  # noqa: E402
from agentrelay.schema import CommitMode, Executor, TransferMode, canonical_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", help="JSONL from native official-train episodes")
    parser.add_argument("output")
    parser.add_argument("--train-split", action="append", required=True)
    args = parser.parse_args()
    allowed = set(args.train_split)
    rows: list[RouterTrainingRow] = []
    for line_number, line in enumerate(Path(args.episodes).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        episode = json.loads(line)
        if episode.get("split") not in allowed:
            raise ValueError(f"line {line_number} is not from an authorized train split")
        if episode.get("labels_accessed_by_router") is not False:
            raise ValueError(f"line {line_number} violates the router label boundary")
        success = int(float(episode.get("success", 0.0)) > 0.0)
        reward = min(1.0, max(0.0, float(episode.get("reward", 0.0))))
        for step in episode.get("steps", ()):
            features = step.get("router_features", {})
            if not features:
                raise ValueError(f"line {line_number} has a step without router features")
            row = RouterTrainingRow(
                dataset_id=str(episode["benchmark"]),
                dataset_revision=str(episode["dataset_revision"]),
                split=str(episode["split"]),
                sample_id=str(episode["sample_id"]),
                step_index=int(step["step_index"]),
                purpose="train",
                features={str(key): float(value) for key, value in features.items()},
                action=CandidateAction(
                    Executor(step["selected_executor"]),
                    TransferMode(step["transfer_mode"]),
                    CommitMode(step["commit_mode"]),
                ),
                success=success,
                reward=reward,
                fidelity_pass=1,
                inference_ms=float(step["inference_ms"]),
                controller_ms=float(step["controller_ms"]),
                handoff=HandoffMeasurement(
                    encode_ms=float(step["handoff_encode_ms"]),
                    communication_ms=float(step["handoff_communication_ms"]),
                    rehydration_ms=float(step["handoff_rehydration_ms"]),
                    verification_ms=float(step["handoff_verify_ms"]),
                    patch_ms=float(step["handoff_patch_ms"]),
                    effect_wait_ms=float(step["handoff_effect_wait_ms"]),
                    reconciliation_ms=float(step["handoff_reconciliation_ms"]),
                    payload_bytes=int(step["handoff_bytes"]),
                    target_tokens=int(step["target_tokens"]),
                    fidelity_risk=float(step["fidelity_risk"]),
                    effect_risk=float(step["effect_risk"]),
                ),
            )
            row.validate(allowed)
            rows.append(row)
    if not rows:
        raise ValueError("no native official-train router rows were produced")
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

