#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay

echo "=== installed package versions ==="
$PY - <<'EOF'
import importlib.metadata as md
for p in ["torch","transformers","datasets","bitsandbytes","alfworld","web_agent_site"]:
    try:
        print(f"  {p} = {md.version(p)}")
    except Exception as e:
        print(f"  {p} = NOT INSTALLED ({e})")
EOF

echo ""
echo "=== AppWorld adapter smoke (already verified) ==="

echo ""
echo "=== WebShop adapter reset+step ==="
$PY - <<'EOF'
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
env = WebAgentTextEnv(observation_mode="text", human_goals=1, session=0,
    file_path="/root/autodl-tmp/AgentRelay/repositories/webshop/data/items_shuffle.json")
obs = env.reset(session=0)
print("WS RESET OK:", str(obs)[:80])
obs2, reward, done, info = env.step('search[toaster 2 slice]')
print("WS STEP OK reward=", reward, "done=", done)
EOF

echo ""
echo "=== ALFWorld adapter reset ==="
$PY - <<'EOF'
from agentrelay.official_adapters import ALFWorldAdapter
import glob
import os
cfg = "/root/autodl-tmp/AgentRelay/configs/alfworld_official_config.yaml"
if not os.path.exists(cfg):
    # locate the official config that ships with the repo
    cand = glob.glob("/root/autodl-tmp/AgentRelay/repositories/alfworld/**/base_config.yaml", recursive=True)
    cfg = cand[0] if cand else None
print("alfworld config:", cfg)
EOF