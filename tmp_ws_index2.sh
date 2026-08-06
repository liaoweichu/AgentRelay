#!/usr/bin/env bash
set -e
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
PIP=/root/autodl-tmp/AgentRelay/venv/bin/pip
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server"
export PATH="/root/autodl-tmp/AgentRelay/venv/bin:$PATH"
export PYTHONSAFEPATH=1
echo "---VERIFY JVM under pyserini---"
$PY -c "from pyserini.pyclass import autoclass; System=autoclass('java.lang.System'); print('pyserini JVM OK', System.getProperty('java.version'))" 2>&1 | tail -6
echo "---CONVERT PRODUCT FORMAT---"
cd "$WS/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k indexes
$PY convert_product_file_format.py 2>&1 | tail -12
echo "---DOC COUNTS---"
for d in resources resources_100 resources_1k resources_100k; do echo "$d: $(wc -l < $d/documents.jsonl 2>/dev/null) docs"; done
echo "---RUN INDEXING (full)---"
$PY -m pyserini.index.lucene --collection JsonCollection --input resources --index indexes --generator DefaultLuceneDocumentGenerator --threads 4 --storePositions --storeDocvectors --storeRaw 2>&1 | tail -20
echo "---INDEX RESULT---"
ls -la indexes 2>&1 | head