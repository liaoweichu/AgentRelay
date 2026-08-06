#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
set -euo pipefail
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
cd "$R"
DATA=$R/repositories/webshop/data

for f in items_human_ins.json items_ins_v2_1000.json items_ins_v2.json; do
  echo "=== $f ==="
  $PY - "$DATA/$f" <<'PYEOF'
import sys
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
kwargs = {"observation_mode": "text", "human_goals": 1, "file_path": sys.argv[1]}
env = WebAgentTextEnv(**kwargs)
print("goals =", len(env.server.goals))
env.close()
PYEOF
done