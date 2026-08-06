#!/usr/bin/env bash
set -euo pipefail
repo="YWZBrandon/webshop-data"
dest="/root/autodl-tmp/AgentRelay/repositories/webshop/data"
mkdir -p "$dest"
for f in items_shuffle.json items_ins_v2.json items_human_ins.json; do
  echo "=== downloading ${f} ==="
  curl -L --retry 3 --retry-delay 5 -o "${dest}/${f}" \
    "https://hf-mirror.com/datasets/${repo}/resolve/main/${f}"
  echo "=== ${f} size: $(du -h "${dest}/${f}" | cut -f1) ==="
done
echo "=== DONE ==="
ls -la "$dest"