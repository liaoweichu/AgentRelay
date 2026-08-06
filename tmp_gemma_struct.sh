#!/usr/bin/env bash
set -uo pipefail
echo "=== models/models structure ==="
ls -la /root/autodl-tmp/AgentRelay/models/models/ 2>/dev/null
find /root/autodl-tmp/AgentRelay/models/models -maxdepth 3 -type d 2>/dev/null | head
echo "=== disk ==="
df -h /root/autodl-tmp | tail -1