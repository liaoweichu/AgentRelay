#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
PIP=/root/autodl-tmp/AgentRelay/venv/bin/pip
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
echo "---REQUIREMENTS---"
cat "$WS/requirements.txt"
echo "---JAVA---"
java -version 2>&1 | head -3 || echo "no java"
echo "---SPACY---"
$PY -c "import spacy; print('spacy', spacy.__version__)" 2>&1
echo "---SPACY MODEL INSTALLED?---"
$PY -c "import en_core_web_lg; print('en_core_web_lg present')" 2>&1 | tail -1
echo "---SEARCH ENGINE SCRIPTS---"
ls -la "$WS/search_engine"
echo "---INDEXING SH---"
cat "$WS/search_engine/run_indexing.sh"
echo "---FAISS---"
$PY -c "import faiss; print('faiss', faiss.__version__)" 2>&1
echo "---PYSERINI/JAVA---"
$PY -c "import pyserini; print('pyserini ok')" 2>&1 | tail -1