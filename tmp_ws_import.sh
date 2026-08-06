#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export PYTHONSAFEPATH=1
echo "---IMPORT engine with SAFEPATH---"
cd /root/autodl-tmp/AgentRelay/repositories/webshop
$PY -c "import web_agent_site.engine.engine as e; print('engine import OK')" 2>&1 | tail -12
echo "---IMPORT env---"
$PY -c "from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv; print('env import OK')" 2>&1 | tail -12