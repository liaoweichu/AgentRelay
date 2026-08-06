#!/usr/bin/env bash
# Poll the gate job every 5 minutes and print a status line.
set -u
cd /root/autodl-tmp/AgentRelay || exit 1
for ((i=1; i<=36; i++)); do
  sleep 300
  echo "=== CHECK $((i*5))min $(date +%H:%M:%S) ==="
  who=$(pgrep -c -f 'venv/bin/python tmp_run_gate' 2>/dev/null || echo 0)
  if [ -f results/gate-webshop50-full.json ]; then
    res="DONE"
  else
    res="RUNNING"
  fi
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,nounits 2>/dev/null | tail -n1)
  echo "proc=$who result=$res vram=${vram}MiB"
  echo "last=$(tail -n 1 gate_full.log 2>/dev/null)"
  if [ "$res" = "DONE" ]; then
    echo '*** GATE COMPLETE ***'
    break
  fi
  if [ "$who" = "0" ]; then
    echo '*** PROC DIED ***'
    break
  fi
done