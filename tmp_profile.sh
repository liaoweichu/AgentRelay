#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
cd /root/autodl-tmp/AgentRelay
$PY scripts/profile_models.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json \
  /root/autodl-tmp/AgentRelay/results/service-profile.json \
  --repeats 3 \
  --network-trace /root/autodl-tmp/AgentRelay/datasets/network/trace.csv \
  --rate-column mbps --sample-period-ms 1000 \
  --trace-source 'Real network throughput recorded on the AutoDL RTX 4090D instance (connect.cqa1.seetacloud.com:30697) streaming Qwen/Qwen2.5-1.5B-Instruct/model.safetensors from hf-mirror.com, 2026-08-05, 1s sampling (70 samples)'