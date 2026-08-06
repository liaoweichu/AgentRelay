#!/usr/bin/env bash
set -e
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
PIP=/root/autodl-tmp/AgentRelay/venv/bin/pip
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export PYTHONSAFEPATH=1
echo "---INSTALL SELENIUM---"
$PIP install -q selenium 2>&1 | tail -3
echo "---VERIFY ENV IMPORT---"
$PY -c "from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv; print('WebAgentTextEnv import OK')" 2>&1 | tail -6