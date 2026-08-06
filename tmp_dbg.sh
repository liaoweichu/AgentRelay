#!/usr/bin/env bash
echo "LINE1"
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
echo "AFTER_SOURCE"
export NLTK_DISABLE_IMPORT_SECURITY=1
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/alfworld:/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
MON=$R/results/manifests
cd $R
echo "AFTER_CD"
echo "PY_EXISTS=$([ -x $PY ] && echo yes || echo no)"
echo "CONFIG_EXISTS=$([ -f $R/formal-autodl-4090d.locked.json ] && echo yes || echo no)"
echo "PROFILE_EXISTS=$([ -f $R/results/service-profile.json ] && echo yes || echo no)"
echo "MANIFEST_EXISTS=$([ -f $MON/alfworld-train-sub12.json ] && echo yes || echo no)"
$PY -c "import sys; sys.path.insert(0,'src'); import agentrelay; print('IMPORT_AGENTRELAY_OK')"
echo "DONE"