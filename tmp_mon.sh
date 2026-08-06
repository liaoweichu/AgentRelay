#!/usr/bin/env bash
echo "=== python procs ==="
ps -eo pid,pcpu,pmem,etime,cmd | grep python | grep -v grep
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== log tail ==="
tail -5 /root/autodl-tmp/AgentRelay/results/train-rollout.log