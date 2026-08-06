#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay

echo "=== locked config: datasets + benchmarks + limits ==="
$PY - <<'EOF'
import json
d = json.load(open('/root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json'))
print("datasets:", json.dumps(d.get("datasets"), indent=2))
print("benchmarks:", json.dumps(d.get("benchmarks"), indent=2))
print("limits:", json.dumps(d.get("limits"), indent=2))
print("controller:", json.dumps(d.get("controller"), indent=2))
EOF

echo ""
echo "=== network trace assets ==="
find $R/datasets -maxdepth 3 -iname "*network*" -o -iname "*trace*" 2>/dev/null | head
ls -la $R/datasets/ 2>&1 | head -20

echo ""
echo "=== implemented methods ==="
ls $R/artifacts/*/results/implemented-methods*.json 2>/dev/null | head
$PY -c "import json; d=json.load(open(sorted(__import__('glob').glob('/root/autodl-tmp/AgentRelay/artifacts/local-data/results/implemented-methods-v2.json'))[0])); print(json.dumps(d, indent=2)[:1500])" 2>&1 | head -40