#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay

echo "=== results dir ==="
ls -la $R/results/ 2>&1

echo ""
echo "=== locked config: model + dataset + official repo revisions ==="
$PY - <<'EOF'
import json
d = json.load(open('/root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json'))
print("models:")
for role, m in d.get("models", {}).items():
    print("  ", role, m.get("model_id"), "rev=", m.get("revision"))
print("repositories:")
repos = d.get("official_repositories") or d.get("repositories") or {}
for name, info in repos.items():
    if isinstance(info, dict):
        print("  ", name, "revision=", info.get("revision") or info.get("commit"))
    else:
        print("  ", name, "=", info)
print("benchmarks keys:", list(d.get("benchmarks", {}).keys()))
EOF

echo ""
echo "=== models on disk ==="
du -sh $R/models/* 2>/dev/null || ls -la $R/models

echo ""
echo "=== repositories ==="
ls $R/repositories/

echo ""
echo "=== WebShop index ==="
du -sh $R/repositories/webshop/indexes 2>&1

echo ""
echo "=== AppWorld task count ==="
ls $R/repositories/appworld/data/tasks/ | wc -l

echo ""
echo "=== disk usage ==="
du -sh $R/ 2>/dev/null
df -h /root/autodl-tmp 2>/dev/null | tail -2