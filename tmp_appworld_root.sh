#!/usr/bin/env bash
AW=/root/autodl-tmp/AgentRelay/repositories/appworld
echo "---APPWORLD_ROOT resolution---"
grep -rn "APPWORLD_ROOT\|APPWORLD_DATA\|def.*root\|DATA_DIR\|data/tasks" "$AW/src/appworld/common.py" 2>/dev/null | head -20
echo "---common.py data dir logic---"
grep -rn "APPWORLD_ROOT\|DATA_DIR\|tasks" "$AW/src/appworld/"*.py 2>/dev/null | grep -iE "root|data_dir|task_dir|get_root|env" | head -20