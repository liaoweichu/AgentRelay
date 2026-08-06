#!/usr/bin/env bash
set -uo pipefail
echo "=== cache/ contents ==="
du -sh /root/autodl-tmp/AgentRelay/cache/* 2>/dev/null | sort -rh | head -20
echo ""
echo "=== tmp/ contents ==="
du -sh /root/autodl-tmp/AgentRelay/tmp/* 2>/dev/null | sort -rh | head -20
echo ""
echo "=== models/ contents ==="
du -sh /root/autodl-tmp/AgentRelay/models/* 2>/dev/null | sort -rh | head