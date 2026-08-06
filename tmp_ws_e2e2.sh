#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export PYTHONPATH="$WS"
export NLTK_DISABLE_IMPORT_SECURITY=1
cd /root
$PY - <<'EOF'
import traceback
try:
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    # full product set
    env = WebAgentTextEnv(observation_mode="text", human_goals=1, session=0,
                          file_path="/root/autodl-tmp/AgentRelay/repositories/webshop/data/items_shuffle.json")
    obs = env.reset(session=0)
    print("RESET OK. instruction:", getattr(env, "instruction_text", "")[:120])
    print("obs len:", len(str(obs)))
    # step a search
    obs2, reward, done, info = env.step('search[toaster 2 slice]')
    print("STEP OK reward=", reward, "done=", done)
    print("available keys:", list(env.get_available_actions().keys()))
    print("clickables count:", len(env.get_available_actions().get('clickables', [])))
except Exception as e:
    print("ENV ERR:", type(e).__name__)
    traceback.print_exc()
EOF