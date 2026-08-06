#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
for repo in "YWZBrandon/webshop-data" "axon-rl/webshop" "Yuxuan13/webshop_dataset" "webgoose/webshop_data"; do
  echo "===== ${repo} ====="
  timeout 20 curl -s "https://hf-mirror.com/api/datasets/${repo}" -o /tmp/r.json 2>&1
  $PY - <<'EOF'
import json
try:
    d = json.load(open('/tmp/r.json'))
    print("id:", d.get('id'))
    for s in (d.get('siblings') or []):
        print("  ", s.get('rfilename'))
except Exception as e:
    print("ERR", e, open('/tmp/r.json').read()[:200])
EOF
done