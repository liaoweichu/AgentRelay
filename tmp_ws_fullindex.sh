#!/usr/bin/env bash
set -e
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export PATH="/root/autodl-tmp/AgentRelay/venv/bin:$PATH"
export PYTHONSAFEPATH=1
cd "$WS/search_engine"
echo "---INDEXING FULL (1.18M) -> indexes---"
$PY -m pyserini.index.lucene --collection JsonCollection --input resources --index indexes --generator DefaultLuceneDocumentGenerator --threads 8 --storePositions --storeDocvectors --storeRaw 2>&1 | grep -E "Indexing Complete|documents indexed|ERROR|Exception" | tail -10
echo "---INDEX RESULT---"
ls -la indexes 2>&1 | head
du -sh indexes 2>&1