#!/usr/bin/env bash
set -e
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
cd /root/autodl-tmp/AgentRelay
$PY scripts/preflight.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json \
  --output /root/autodl-tmp/AgentRelay/results/preflight.json
echo ""
echo "=== preflight.json ==="
cat /root/autodl-tmp/AgentRelay/results/preflight.json