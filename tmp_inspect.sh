#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay

echo "=== locked config top-level keys ==="
$PY - <<'EOF'
import json
d = json.load(open('/root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json'))
print("top keys:", list(d.keys()))
print("repositories type:", type(d.get("repositories")), d.get("repositories"))
print("models:", json.dumps(d.get("models"), indent=2))
print("data_root:", d.get("data_root"))
print("run_mode:", d.get("run_mode"))
EOF

echo ""
echo "=== find Qwen model checkpoints ==="
find $R/cache $R/models -maxdepth 6 -type d -name "*Qwen*" 2>/dev/null | head
echo "--- snapshots ---"
find $R/cache/huggingface/hub -maxdepth 3 -type d 2>/dev/null | head -20

echo ""
echo "=== find webshop index dirs ==="
find $R/repositories/webshop -maxdepth 3 -type d -name "*index*" 2>/dev/null
echo "--- webshop data file ---"
find $R/repositories/webshop -maxdepth 3 -name "items_shuffle.json" 2>/dev/null

echo ""
echo "=== results dirs anywhere ==="
find $R -maxdepth 2 -type d -name "results" 2>/dev/null
ls -la $R/experiments 2>/dev/null | head