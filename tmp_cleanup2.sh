#!/usr/bin/env bash
set -uo pipefail
echo "=== cache breakdown ==="
du -sh /root/autodl-tmp/AgentRelay/cache/* 2>/dev/null | sort -rh | head
echo "=== remove pip cache dir ==="
rm -rf /root/autodl-tmp/AgentRelay/cache/pip 2>/dev/null
echo "=== after ==="
df -h /root/autodl-tmp | tail -1
du -sh /root/autodl-tmp/AgentRelay/cache 2>/dev/null