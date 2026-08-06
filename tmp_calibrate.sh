#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
set -euo pipefail
export NLTK_DISABLE_IMPORT_SECURITY=1
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/alfworld:/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
MON=$R/results/manifests
cd $R

echo "=== validation rollout on ALFWorld valid_seen subset (uncalibrated_joint) ==="
OUT=$($PY scripts/run_autodl_matrix.py \
  $R/formal-autodl-4090d.locked.json \
  $MON/alfworld-valid-seen-sub8.json \
  $R/results/service-profile.json \
  --router $R/results/router.joblib \
  --method uncalibrated_joint)
echo "$OUT"
RUN_DIR=$(echo "$OUT" | grep -o 'run_directory=.*' | cut -d= -f2)
echo "RUN_DIR=$RUN_DIR"

echo "=== concatenate validation episodes into JSONL ==="
$PY - "$RUN_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
out = Path("/root/autodl-tmp/AgentRelay/results/validation-episodes.jsonl")
records = []
for f in sorted(run_dir.rglob("*.json")):
    if f.name == "manifest.json":
        continue
    records.append(json.loads(f.read_text(encoding="utf-8")))
with out.open("w", encoding="utf-8") as h:
    for rec in records:
        h.write(json.dumps(rec) + "\n")
print(f"episodes={len(records)} output={out}")
print("splits=", sorted({r.get('split') for r in records}))
print("methods=", sorted({r.get('method') for r in records}))
PYEOF

echo "=== build calibration rows ==="
$PY scripts/build_calibration_rows.py \
  $R/results/validation-episodes.jsonl \
  $R/results/calibration-rows.jsonl \
  --validation-split valid_seen

echo "=== calibrate router (validation-only) ==="
$PY scripts/calibrate_router.py \
  $R/results/calibration-rows.jsonl \
  $R/results/calibrator.json \
  --validation-split valid_seen \
  --alpha 0.1 --minimum-group-size 20