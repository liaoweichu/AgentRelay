#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
echo "---HFMIRROR---"
timeout 12 curl -s -o /dev/null -w "%{http_code}\n" https://hf-mirror.com 2>&1
echo "---HFMIRROR-WEBSHOP-API---"
timeout 20 curl -s "https://hf-mirror.com/api/datasets?search=webshop&limit=30" -o /tmp/ws_search.json 2>&1
$PY - <<'EOF'
import json
try:
    d = json.load(open('/tmp/ws_search.json'))
    for x in d:
        print(x.get('id'))
except Exception as e:
    print("PARSE_ERR", e)
    print(open('/tmp/ws_search.json').read()[:500])
EOF
echo "---HFMIRROR-PRINCETON-WEBSHOP---"
timeout 15 curl -s "https://hf-mirror.com/api/datasets/princeton-nlp/WebShop" -o /tmp/ws_repo.json 2>&1
$PY - <<'EOF'
import json
try:
    d = json.load(open('/tmp/ws_repo.json'))
    print("repo:", d.get('id'), "| files:", d.get('siblings'))
except Exception as e:
    print("PARSE_ERR", e)
EOF