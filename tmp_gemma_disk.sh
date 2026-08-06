#!/usr/bin/env bash
set -uo pipefail
echo "=== biggest dirs under /root/autodl-tmp/AgentRelay ==="
du -sh /root/autodl-tmp/AgentRelay/* 2>/dev/null | sort -rh | head -25
echo ""
echo "=== disk free now ==="
df -h /root/autodl-tmp | tail -1
echo ""
echo "=== /root/.cache if any (HF curl) ==="
du -sh /root/.cache 2>/dev/null | sort -rh | head