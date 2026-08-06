#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
SP=/root/autodl-tmp/AgentRelay/venv/lib/python3.12/site-packages
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export PYTHONSAFEPATH=1
echo "---pyserini pyclass head---"
sed -n '1,60p' "$SP/pyserini/pyclass.py"
echo "---pyserini resources---"
ls -la "$SP/pyserini/resources" 2>/dev/null | head
echo "---anserini jar?---"
find "$SP/pyserini" -name "*.jar" 2>/dev/null | head
echo "---JVM options test---"
$PY -c "import jnius_config; jnius_config.add_options('-Xmx2g'); from jnius import autoclass; print('OK')" 2>&1 | tail -5