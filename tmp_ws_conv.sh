#!/usr/bin/env bash
set -e
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server"
export PYTHONSAFEPATH=1
echo "---EXISTING DOCS---"
for d in resources resources_100 resources_1k resources_100k; do echo "$d: $(wc -l < $WS/search_engine/$d/documents.jsonl 2>/dev/null) docs"; done
echo "---RUN CONVERT (verbose)---"
cd "$WS/search_engine"
$PY convert_product_file_format.py 2>&1 | grep -vE "^$" | tail -20
echo "---AFTER CONVERT DOCS---"
for d in resources resources_100 resources_1k resources_100k; do echo "$d: $(wc -l < $WS/search_engine/$d/documents.jsonl 2>/dev/null) docs"; done