#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay

echo "=== base_config.yaml ==="
cat $R/repositories/alfworld/configs/base_config.yaml 2>&1 | head -40

echo ""
echo "=== configs dir ==="
ls $R/repositories/alfworld/configs/ 2>&1

echo ""
echo "=== ALFWORLD_DATA env / datasets/alfworld ==="
echo "ALFWORLD_DATA=$ALFWORLD_DATA"
ls -la $R/datasets/alfworld 2>&1 | head

echo ""
echo "=== find json task files ==="
find $R/datasets/alfworld -maxdepth 2 -name "*.json" 2>/dev/null | head
find /root/autodl-tmp -maxdepth 4 -type d -name "alfworld" 2>/dev/null | head