#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export NLTK_DISABLE_IMPORT_SECURITY=1
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
$PY - <<'EOF'
import sys
sys.path.insert(0, "/root/autodl-tmp/AgentRelay/repositories/appworld/src")
from appworld import load_task_ids
for split in ["train", "dev", "test_normal", "test_challenge"]:
    try:
        ids = load_task_ids(split)
        print(split, "->", len(ids))
    except Exception as e:
        print(split, "FAIL", repr(e))
EOF