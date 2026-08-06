#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export PYTHONPATH="$WS"
export PYTHONSAFEPATH=1
cd /root
echo "---web_agent_site __init__---"
cat "$WS/web_agent_site/__init__.py"
echo "---TEST import env with SAFEPATH (no .pth)---"
$PY -c "from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv; print('WebAgentTextEnv OK')" 2>&1 | tail -8