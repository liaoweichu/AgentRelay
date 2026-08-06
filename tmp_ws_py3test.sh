#!/usr/bin/env bash
SP=/root/autodl-tmp/AgentRelay/venv/lib/python3.12/site-packages/webshop.pth
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
rm -f "$SP"
echo "removed .pth: $(ls -la "$SP" 2>&1)"
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
cd /root
echo "---TEST import xml (no webshop on path)---"
$PY -c "import xml.etree.ElementTree; print('xml OK')" 2>&1 | tail -3
export PYTHONPATH="$WS"
echo "---TEST import xml WITH webshop on PYTHONPATH---"
$PY -c "import xml.etree.ElementTree; print('xml OK with PYTHONPATH')" 2>&1 | tail -3
echo "---TEST import web_agent_site.envs.web_agent_text_env---"
$PY -c "from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv; print('WebAgentTextEnv OK')" 2>&1 | tail -6