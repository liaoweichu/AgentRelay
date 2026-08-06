#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "=== 14B snapshot ==="
ls -R /root/autodl-tmp/AgentRelay/cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots 2>&1 | head -40
echo "=== 14B blobs count ==="
ls /root/autodl-tmp/AgentRelay/cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/blobs 2>/dev/null | wc -l
du -sh /root/autodl-tmp/AgentRelay/cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct 2>/dev/null
echo "=== 1.5B snapshot ==="
ls -R /root/autodl-tmp/AgentRelay/cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots 2>&1 | head -40
du -sh /root/autodl-tmp/AgentRelay/cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct 2>/dev/null