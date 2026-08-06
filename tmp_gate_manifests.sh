#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
cd /root/autodl-tmp/AgentRelay
MON=/root/autodl-tmp/AgentRelay/results/manifests
WS_REV=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
WS_FILE=/root/autodl-tmp/AgentRelay/repositories/webshop/data/items_shuffle.json

echo "=== build WebShop gate50 (purpose=tune, sample 50) ==="
$PY scripts/build_official_task_manifest.py --benchmark webshop --split test --purpose tune \
  --revision $WS_REV --output $MON/webshop-gate50.json \
  --webshop-file-path $WS_FILE --sample-count 50 --sample-seed 20260805

echo "=== slice ALFWorld valid-seen to gate20 (purpose=tune preserved) ==="
$PY scripts/slice_task_manifest.py \
  $MON/alfworld-valid-seen.json $MON/alfworld-gate20.json --count 20

echo "=== verify ==="
$PY - <<'PY'
import json
for f in ["webshop-gate50", "alfworld-gate20"]:
    d=json.load(open("/root/autodl-tmp/AgentRelay/results/manifests/%s.json"%f))
    print(f, "complete=", d.get("complete_official_split"), "tasks=", len(d["tasks"]), "purposes=", sorted({t.get("purpose") for t in d["tasks"]}))
PY