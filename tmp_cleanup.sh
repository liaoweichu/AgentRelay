#!/usr/bin/env bash
set -uo pipefail
echo "=== before ==="
df -h /root/autodl-tmp | tail -1
echo "=== purge pip cache ==="
/root/autodl-tmp/AgentRelay/venv/bin/pip cache purge 2>&1 | tail -2
echo "=== rm tmp/ (pip unpack+install leftovers) ==="
rm -rf /root/autodl-tmp/AgentRelay/tmp/tmpieisckw_ /root/autodl-tmp/AgentRelay/tmp/pip-* 2>/dev/null
rm -rf /root/autodl-tmp/AgentRelay/tmp/* 2>/dev/null
echo "=== after ==="
df -h /root/autodl-tmp | tail -1
du -sh /root/autodl-tmp/AgentRelay/cache /root/autodl-tmp/AgentRelay/tmp 2>/dev/null