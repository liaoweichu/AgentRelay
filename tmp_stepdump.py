#!/usr/bin/env python3
"""Inspect one agentrelay episode's step-level routing decisions to confirm the
learned router is exercised and whether its predictions are degenerate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode")
    args = parser.parse_args()

    rec = json.loads(Path(args.episode).read_text(encoding="utf-8"))
    print("benchmark=", rec.get("benchmark"), "method=", rec.get("method"),
          "split=", rec.get("split"), "sample_id=", rec.get("sample_id"))
    print("success=", rec.get("success"), "reward=", rec.get("reward"),
          "steps=", len(rec.get("steps", ())), "max_steps=", rec.get("max_steps"))
    print("paper_evidence=", rec.get("paper_evidence"),
          "labels_accessed_by_router=", rec.get("labels_accessed_by_router"))
    print("\n--- step routing trace ---")
    for i, s in enumerate(rec.get("steps", ())[:40]):
        print(f"  step{i:>2} ex={s.get('selected_executor'):>5} "
              f"mode={s.get('transfer_mode'):<22} commit={s.get('commit_mode'):<12} "
              f"reason={s.get('routing_reason','')[:28]} "
              f"ps={s.get('predicted_success'):.3f} pf={s.get('predicted_fidelity'):.3f} "
              f"h={(s.get('predicted_effect_risk') or 0):.2f} "
              f"feat={ {k: round(v,3) for k,v in (s.get('router_features') or {}).items() if k in ('step_index','remaining_steps','input_tokens','effect_irreversible_or_unknown','effect_read_only','dependency_depth','closure_node_count')} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())