#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
MON=$R/results/manifests
mkdir -p "$MON"
cd $R

AW_REV=aaba6870f86c5be6a08a491f32a50b906227bc3e
WS_REV=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
AP_REV=a072b7a86e7c1d5b1d7175659d750ebb9b79f10a
AW_CFG=$R/repositories/alfworld/configs/base_config.yaml
WS_FILE=$R/repositories/webshop/data/items_shuffle_1000.json

# ALFWorld: train (router), valid_seen (calibration), valid_unseen (eval)
$PY scripts/build_official_task_manifest.py --benchmark alfworld --split train --purpose train --revision $AW_REV --output $MON/alfworld-train.json --alfworld-config $AW_CFG --train-eval train
$PY scripts/build_official_task_manifest.py --benchmark alfworld --split valid_seen --purpose tune --revision $AW_REV --output $MON/alfworld-valid-seen.json --alfworld-config $AW_CFG --train-eval eval
$PY scripts/build_official_task_manifest.py --benchmark alfworld --split valid_unseen --purpose evaluate --revision $AW_REV --output $MON/alfworld-valid-unseen.json --alfworld-config $AW_CFG --train-eval eval

# AppWorld: train (router), dev (calibration), test_normal + test_challenge (eval)
$PY scripts/build_official_task_manifest.py --benchmark appworld --split train --purpose train --revision $AP_REV --output $MON/appworld-train.json
$PY scripts/build_official_task_manifest.py --benchmark appworld --split dev --purpose tune --revision $AP_REV --output $MON/appworld-dev.json
$PY scripts/build_official_task_manifest.py --benchmark appworld --split test_normal --purpose evaluate --revision $AP_REV --output $MON/appworld-test-normal.json
$PY scripts/build_official_task_manifest.py --benchmark appworld --split test_challenge --purpose evaluate --revision $AP_REV --output $MON/appworld-test-challenge.json

# WebShop: test (eval, official 1000-sample)
$PY scripts/build_official_task_manifest.py --benchmark webshop --split test --purpose evaluate --revision $WS_REV --output $MON/webshop-test.json --webshop-file-path $WS_FILE

echo ""
echo "=== manifests ==="
ls -la $MON/*.json
echo ""
for f in $MON/*.json; do
  echo -n "$(basename $f): "
  $PY -c "import json; d=json.load(open('$f')); print(len(d['tasks']), 'tasks, complete=', d['complete_official_split'])"
done