#!/usr/bin/env bash
echo "=== task.py path logic ==="
grep -rn "task_directory\|data_dir\|os.environ\|getenv\|/data\|Path(" /root/autodl-tmp/AgentRelay/repositories/appworld/src/appworld/task.py 2>/dev/null | head -30
echo "=== how /root/data resolved ==="
grep -rn "root\|Path(\"data\|/data\|home\|expanduser" /root/autodl-tmp/AgentRelay/repositories/appworld/src/appworld/common.py /root/autodl-tmp/AgentRelay/repositories/appworld/src/appworld/constants.py 2>/dev/null | head -30