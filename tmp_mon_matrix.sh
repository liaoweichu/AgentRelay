#!/usr/bin/env bash
RUNID=$(ls -dt /root/autodl-tmp/AgentRelay/runs/formal-matrix-* 2>/dev/null | head -1)
echo "RUNID=$RUNID"
echo "episode_json_count=$(find "$RUNID" -name '*.json' 2>/dev/null | wc -l)"
echo "method_dirs=$(ls "$RUNID/webshop/test" 2>/dev/null | tr '\n' ' ')"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
pgrep -f run_autodl_matrix >/dev/null && echo "STATUS=RUNNING" || echo "STATUS=NOT_RUNNING"
echo "traceback_count=$(grep -c Traceback /root/autodl-tmp/AgentRelay/results/webshop-matrix.log 2>/dev/null)"
tail -c 300 /root/autodl-tmp/AgentRelay/results/webshop-matrix.log | tr '\r' '\n' | grep -v 'it/s]' | tail -n 5