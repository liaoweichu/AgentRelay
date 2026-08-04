#!/usr/bin/env python3
"""Export implemented routing and state-transfer methods for matrix auditing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentrelay.baselines import baseline_manifest  # noqa: E402
from agentrelay.codecs import codec_manifest  # noqa: E402
from agentrelay.schema import canonical_json, sha256_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    payload = {
        "routing_methods": baseline_manifest(),
        "state_transfer_codecs": codec_manifest(),
    }
    payload["manifest_hash"] = sha256_json(payload)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(f"routing_methods={len(payload['routing_methods'])} codecs={len(payload['state_transfer_codecs'])} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

