#!/usr/bin/env bash
repo="YWZBrandon/webshop-data"
dest="/root/autodl-tmp/AgentRelay/repositories/webshop/data"
for f in items_shuffle_1000.json items_ins_v2_1000.json; do
  echo "=== downloading ${f} ==="
  curl -L --retry 3 --retry-delay 5 -o "${dest}/${f}" \
    "https://hf-mirror.com/datasets/${repo}/resolve/main/${f}"
  echo "=== ${f} size: $(du -h "${dest}/${f}" | cut -f1) ==="
done
echo "=== data dir ==="
ls -la "$dest"