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

echo "=== WebShop 200 x 7 baseline matrix ==="
OUT=$($PY scripts/run_autodl_matrix.py \
  $R/formal-autodl-4090d.locked.json \
  $MON/webshop-test-200.json \
  $R/results/service-profile.json \
  --router $R/results/router.joblib \
  --calibrator $R/results/calibrator.json \
  --method edge_only \
  --method cloud_only \
  --method agentrouter_style \
  --method hera_agreement \
  --method full_replay_step \
  --method uncalibrated_joint \
  --method agentrelay)
echo "$OUT"
RUN_DIR=$(echo "$OUT" | grep -o 'run_directory=.*' | cut -d= -f2)
echo "RUN_DIR=$RUN_DIR"
echo "$RUN_DIR" > /root/autodl-tmp/AgentRelay/results/webshop-matrix-run.txt