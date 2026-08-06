#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
set -euo pipefail
export NLTK_DISABLE_IMPORT_SECURITY=1
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
DATA=$R/repositories/webshop/data

echo "=== get_human_goals source ==="
sed -n '/def get_human_goals/,/^def get_synthetic_goals/p' \
  /root/autodl-tmp/AgentRelay/repositories/webshop/web_agent_site/engine/goal.py | head -60

echo ""
echo "=== items_human_ins.json structure ==="
$PY - <<'PYEOF'
import json
p = "/root/autodl-tmp/AgentRelay/repositories/webshop/data/items_human_ins.json"
d = json.load(open(p))
print("type=", type(d).__name__, "len=", len(d))
if isinstance(d, dict):
    k = next(iter(d))
    print("sample key=", k, "value_type=", type(d[k]).__name__)
    print("value keys=", list(d[k].keys())[:8] if isinstance(d[k], dict) else d[k])
PYEOF