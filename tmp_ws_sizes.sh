#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
repo="YWZBrandon/webshop-data"
for f in items_human_ins.json items_ins_v2.json items_shuffle.json; do
  echo "--- ${f} ---"
  timeout 20 curl -sI "https://hf-mirror.com/datasets/${repo}/resolve/main/${f}" 2>&1 | grep -iE "content-length|location|HTTP"
done