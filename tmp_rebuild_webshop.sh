#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
set -euo pipefail
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
MON=$R/results/manifests
cd "$R"

WS_REV=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
WS_FILE=$R/repositories/webshop/data/items_shuffle.json

echo "=== 1) rebuild pilot 200 subset (complete_official_split must be False) ==="
$PY scripts/build_official_task_manifest.py --benchmark webshop --split test --purpose evaluate \
  --revision $WS_REV --output $MON/webshop-pilot-200.json \
  --webshop-file-path $WS_FILE --sample-count 200 --sample-seed 20260805

echo "=== verify pilot ==="
$PY -c "import json; d=json.load(open('$MON/webshop-pilot-200.json')); print('pilot200 tasks=', len(d['tasks']), 'complete=', d['complete_official_split'])"

echo "=== 2) build official test 500 (complete_official_split must be True, count==500) ==="
$PY scripts/build_official_task_manifest.py --benchmark webshop --split test --purpose evaluate \
  --revision $WS_REV --output $MON/webshop-test.json \
  --webshop-file-path $WS_FILE

echo "=== verify official test ==="
$PY -c "import json; d=json.load(open('$MON/webshop-test.json')); print('official-test tasks=', len(d['tasks']), 'complete=', d['complete_official_split'])"