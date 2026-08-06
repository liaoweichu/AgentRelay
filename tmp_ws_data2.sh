#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
R=/root/autodl-tmp/AgentRelay
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
for f in "$R"/repositories/webshop/data/*_goals.json "$R"/repositories/webshop/data/items_shuffle_1000.json; do
  [ -e "$f" ] || continue
  n=$($PY -c "import json; d=json.load(open('$f')); print(len(d))" 2>/dev/null)
  echo "$(basename $f): $n"
done
echo "=== look for goals jsons ==="
$PY - <<'EOF'
import glob, os, json
base="/root/autodl-tmp/AgentRelay/repositories/webshop/data"
for f in sorted(glob.glob(base+"/*")):
    if f.endswith(".json"):
        try:
            d=json.load(open(f))
            print(os.path.basename(f), "type=", type(d).__name__, "len=", len(d) if hasattr(d,'__len__') else '?')
        except Exception as e:
            print(os.path.basename(f), "ERR", repr(e))
EOF