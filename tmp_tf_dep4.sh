#!/usr/bin/env bash
set -uo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
echo "=== ALFWorld environment module transformers imports ==="
grep -rl "transformers" /root/autodl-tmp/AgentRelay/repositories/alfworld/alfworld/agents/environment/ 2>/dev/null || echo "NONE in environment/"
echo "=== AppWorld core (appworld/) transformers imports ==="
grep -rl "transformers" /root/autodl-tmp/AgentRelay/repositories/appworld/appworld/ 2>/dev/null || echo "NONE in appworld core"
echo "=== torch / accelerate / protobuf versions ==="
$PY -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.version.cuda)" 2>&1 | tail -3
$PY -c "import accelerate; print('accelerate', accelerate.__version__)" 2>&1 | tail -1
$PY -c "import bitsandbytes; print('bnb', bitsandbytes.__version__)" 2>&1 | tail -1