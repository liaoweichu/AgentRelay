#!/usr/bin/env bash
set -e
R=/root/autodl-tmp/AgentRelay
SRC=$R/cache/huggingface/hub
DST=$R/models
mkdir -p "$DST"
for m in Qwen--Qwen2.5-14B-Instruct Qwen--Qwen2.5-1.5B-Instruct; do
  if [ -d "$SRC/models--$m" ]; then
    echo "moving models--$m"
    mv "$SRC/models--$m" "$DST/"
  fi
done
echo "=== models dir ==="
ls -la "$DST"
echo "=== verify snapshot exists ==="
ls "$DST/models--Qwen--Qwen2.5-14B-Instruct/snapshots/" 2>&1
ls "$DST/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/" 2>&1