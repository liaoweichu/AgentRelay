#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
set -euo pipefail
export NLTK_DISABLE_IMPORT_SECURITY=1
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/alfworld:/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
cd "$R"

CONF=$R/formal-autodl-4090d.locked.json
PROF=$R/results/service-profile.json
MON=$R/results/manifests

echo "=== build ALFWorld valid-seen 20-task gate slice ==="
$PY scripts/slice_task_manifest.py \
  $MON/alfworld-valid-seen.json \
  $MON/alfworld-valid-seen-gate20.json \
  --count 20

echo ""
echo "=== run edge_only + cloud_only on ALFWorld valid-seen gate20 (Qwen) ==="
$PY scripts/run_autodl_matrix.py \
  $CONF $MON/alfworld-valid-seen-gate20.json $PROF \
  --method edge_only --method cloud_only

echo ""
echo "=== DONE ==="