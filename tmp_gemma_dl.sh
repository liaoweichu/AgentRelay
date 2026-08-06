#!/usr/bin/env bash
set -uo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
MODELS=/root/autodl-tmp/AgentRelay/models
echo "=== downloading E4B-it ==="
$PY - <<'PY'
from modelscope import snapshot_download
p = snapshot_download("google/gemma-4-E4B-it", cache_dir="/root/autodl-tmp/AgentRelay/models")
print("E4B_DONE", p)
PY
echo "=== downloading 12b-it ==="
$PY - <<'PY'
from modelscope import snapshot_download
p = snapshot_download("google/gemma-4-12b-it", cache_dir="/root/autodl-tmp/AgentRelay/models")
print("12B_DONE", p)
PY
echo "=== result ==="
du -sh /root/autodl-tmp/AgentRelay/models/google/* 2>/dev/null