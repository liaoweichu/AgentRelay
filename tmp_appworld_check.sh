#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
AW=/root/autodl-tmp/AgentRelay/repositories/appworld
export PYTHONPATH="$AW/src:$PYTHONPATH"
echo "---APPWORLD DATA---"
echo "version: $(cat $AW/data/version.txt 2>&1)"
echo "datasets fs:"
ls -la "$AW/data/datasets"
echo "tasks count: $(ls "$AW/data/tasks" | wc -l)"
echo "base_dbs: $(ls "$AW/data/base_dbs" | wc -l)"
echo "sample task dir: $(ls "$AW/data/tasks" | head -3)"
echo "---IMPORT AP worlds---"
$PY -c "import appworld; print('appworld import OK', appworld.__file__)" 2>&1 | tail -5
echo "---CHECK official_adapters import chain---"
$PY -c "from agentrelay.official_adapters import AppWorldAdapter; print('AppWorldAdapter import OK')" 2>&1 | tail -8