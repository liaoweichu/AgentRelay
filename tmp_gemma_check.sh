#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
set -euo pipefail
echo "=== HF token present? ==="
ls -la /root/.cache/huggingface/token 2>/dev/null && echo "TOKEN_FILE_EXISTS" || echo "NO_TOKEN_FILE"
echo "HF_TOKEN env: ${HF_TOKEN:+SET}"
echo "HF_ENDPOINT: ${HF_ENDPOINT:-unset}"

echo ""
echo "=== existing models in storage ==="
ls -la /root/autodl-tmp/AgentRelay/models/ 2>/dev/null || echo "no models dir"

echo ""
echo "=== cache dirs ==="
ls -d /root/autodl-tmp/AgentRelay/models/* 2>/dev/null || true