#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export PYTHONPATH="$WS"
export NLTK_DISABLE_IMPORT_SECURITY=1
cd /root
echo "---receive/goal idx logic---"
sed -n '500,540p' "$WS/web_agent_site/envs/web_agent_text_env.py"