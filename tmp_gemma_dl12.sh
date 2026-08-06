#!/usr/bin/env bash
# Resume the interrupted Gemma 4 12B-it download from ModelScope.
set -uo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
echo "=== resuming 12b-it download ==="
$PY - <<'PY'
from modelscope import snapshot_download
p = snapshot_download(
    "google/gemma-4-12b-it",
    cache_dir="/root/autodl-tmp/AgentRelay/models",
)
print("12B_DONE", p)
PY
echo "=== 12B download finished at $(date) ==="