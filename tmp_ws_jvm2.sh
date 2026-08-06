#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export PATH="/root/autodl-tmp/AgentRelay/venv/bin:$PATH"
JVM_DIR="$JAVA_HOME/lib/server"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "---TEST 1: raw jnius JVM---"
$PY -c "import jnius_config; from jnius import autoclass; System=autoclass('java.lang.System'); print('JVM OK', System.getProperty('java.version'))" 2>&1 | tail -5
echo "---TEST 2: import web_agent_site.engine.engine---"
$PY -c "import web_agent_site.engine.engine as e; print('engine import OK')" 2>&1 | tail -8