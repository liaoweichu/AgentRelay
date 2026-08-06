#!/usr/bin/env bash
cd /root/autodl-tmp/AgentRelay || exit 1
echo "---REVISIONS---"
for r in alfworld webshop appworld; do
  printf '%s: %s\n' "$r" "$(git -C repositories/$r rev-parse HEAD 2>&1)"
done
echo "---VENV KEY PACKAGES---"
venv/bin/pip list 2>/dev/null | grep -iE 'torch|transformers|faiss|pyserini|gym|flask|beautifulsoup|gdown|rank_bm25|cleantext|datasets|peft|bitsandbytes|accelerate|webshop|alfworld|appworld'
echo "---MODELS---"
ls -la models
echo "---TMP HEAD---"
ls /root/autodl-tmp/AgentRelay/tmp | head -20
echo "---ALFWORLD DATA---"
ls /root/autodl-tmp/AgentRelay/datasets/alfworld 2>/dev/null | head
echo "---WEBSHOP DATA---"
ls repositories/webshop/data 2>/dev/null | head
echo "---APPWORLD DATA---"
ls repositories/appworld/data 2>/dev/null | head