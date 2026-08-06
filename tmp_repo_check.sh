#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay

echo "=== repo HEAD vs locked revisions ==="
for repo in alfworld webshop appworld; do
  dir=$R/repositories/$repo
  if [ -d "$dir/.git" ]; then
    head=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
    echo "$repo: HEAD=$head"
  else
    echo "$repo: no .git (source dir)"
  fi
done

echo ""
echo "=== locked revisions ==="
$PY - <<'EOF'
import json
d = json.load(open('/root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json'))
for item in d["repositories"]:
    print(item["name"], item["revision"])
EOF

echo ""
echo "=== run download_public_data (idempotent) ==="
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null
$PY $R/scripts/download_public_data.py $R/formal-autodl-4090d.locked.json 2>&1 | tail -15

echo ""
echo "=== run prepare_official_benchmarks (idempotent) ==="
bash $R/scripts/prepare_official_benchmarks.sh 2>&1 | tail -15