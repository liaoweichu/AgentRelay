#!/usr/bin/env bash
set -e
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
PIP=/root/autodl-tmp/AgentRelay/venv/bin/pip
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
echo "JAVA_HOME=$JAVA_HOME"
echo "---SPACY SM INSTALL---"
$PY -m spacy download en_core_web_sm 2>&1 | tail -8
echo "---SPACY SM VERIFY---"
$PY -c "import en_core_web_sm; print('sm OK')" 2>&1 | tail -1
echo "---CONVERT PRODUCT FORMAT---"
cd "$WS/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k indexes
$PY convert_product_file_format.py 2>&1 | tail -8
echo "---CONVERT DONE, doc counts---"
for d in resources resources_100 resources_1k resources_100k; do echo "$d: $(wc -l < $d/documents.jsonl 2>/dev/null) docs"; done