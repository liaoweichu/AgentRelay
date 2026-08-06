#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:/root/autodl-tmp/AgentRelay/repositories/alfworld:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
MON=$R/results/manifests
cd $R

AW_REV=aaba6870f86c5be6a08a491f32a50b906227bc3e
WS_REV=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd

echo "=== slice ALFWorld train -> 12 ==="
$PY scripts/slice_task_manifest.py $MON/alfworld-train.json $MON/alfworld-train-sub12.json --count 12

echo "=== slice ALFWorld valid_seen -> 8 ==="
$PY scripts/slice_task_manifest.py $MON/alfworld-valid-seen.json $MON/alfworld-valid-seen-sub8.json --count 8

echo "=== build WebShop 200-goal manifest (full corpus) ==="
$PY scripts/build_official_task_manifest.py --benchmark webshop --split test --purpose evaluate \
  --revision $WS_REV --output $MON/webshop-test-200.json \
  --webshop-file-path $R/repositories/webshop/data/items_shuffle.json \
  --sample-count 200 --sample-seed 20260805

echo ""
echo "=== verify ==="
for f in alfworld-train-sub12.json alfworld-valid-seen-sub8.json webshop-test-200.json; do
  echo -n "$f: "
  $PY -c "import json;d=json.load(open('$MON/$f'));print(len(d['tasks']),'complete=',d['complete_official_split'])"
done