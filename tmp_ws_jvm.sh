#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
JVM_DIR=$(dirname $(readlink -f $(which java)))/../lib/server
echo "---JVM LIB---"
ls -la "$JVM_DIR" 2>&1 | head
echo "---MEM---"
free -h
echo "---TEST JNIUS JVM---"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JVM_DIR"
$PY -c "import jnius_config; jnius_config.set_classpath('/root/autodl-tmp/AgentRelay/repositories/webshop'); from jnius import autoclass; System=autoclass('java.lang.System'); print('JVM OK', System.getProperty('java.version'))" 2>&1 | tail -8