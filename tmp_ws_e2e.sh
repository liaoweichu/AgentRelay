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
    env = WebAgentTextEnv(observation_mode="text", human_goals=1, session=12345)
    obs = env.reset(session=12345)
    print("RESET OK. instruction len:", len(getattr(env, "instruction_text", "")))
    print("OBS (first 300):", str(obs)[:300])
    # step a search
    try:
        obs2, reward, done, info = env.step('search[toaster]')
        print("STEP OK reward=", reward, "done=", done, "obs len=", len(str(obs2)))
        print("available actions (first 5):", list(env.get_available_actions().keys()))
    except Exception as e:
        print("STEP ERR:", type(e).__name__, e)
        traceback.print_exc()
except Exception as e:
    print("ENV ERR:", type(e).__name__)
    traceback.print_exc()
EOF