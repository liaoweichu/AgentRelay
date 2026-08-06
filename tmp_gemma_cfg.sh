#!/usr/bin/env bash
set -uo pipefail
echo "=== download progress ==="
du -sh /root/autodl-tmp/AgentRelay/models/google/* 2>/dev/null
echo "=== E4B config.json ==="
cat /root/autodl-tmp/AgentRelay/models/google/gemma-4-E4B-it/config.json 2>/dev/null | /root/autodl-tmp/AgentRelay/venv/bin/python -c "import sys,json; d=json.load(sys.stdin); print('model_type=',d.get('model_type')); print('architectures=',d.get('architectures')); print('keys=',list(d.keys())[:30])" 2>&1 | head