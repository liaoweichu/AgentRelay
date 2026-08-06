#!/usr/bin/env bash
set -uo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
for f in webshop-test-200 webshop-pilot-200 alfworld-valid-seen alfworld-train; do
  p=/root/autodl-tmp/AgentRelay/results/manifests/$f.json
  if [ -f "$p" ]; then
    echo -n "$f: "
    $PY - <<PY
import json
d=json.load(open("$p"))
print("complete=", d.get("complete_official_split"), "tasks=", len(d.get("tasks",[])), "purposes=", sorted({t.get("purpose") for t in d["tasks"]}))
PY
  else
    echo "$f: MISSING"
  fi
done