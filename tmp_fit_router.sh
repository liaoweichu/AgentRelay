#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
set -euo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
cd $R

echo "=== build router rows from train episodes ==="
$PY scripts/build_router_rows.py \
  $R/results/train-episodes.jsonl \
  $R/results/router-rows.jsonl \
  --train-split train

echo "=== fit router ==="
$PY scripts/fit_router.py \
  $R/results/router-rows.jsonl \
  $R/results/router.joblib \
  --train-split train