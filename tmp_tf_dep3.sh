#!/usr/bin/env bash
set -uo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
echo "=== alfworld transformers imports ==="
grep -rl "transformers" /root/autodl-tmp/AgentRelay/repositories/alfworld/ 2>/dev/null | head || echo "NONE"
echo "=== appworld transformers imports ==="
grep -rl "transformers" /root/autodl-tmp/AgentRelay/repositories/appworld/ 2>/dev/null | head || echo "NONE"
echo "=== webshop web_agent_site transformers imports ==="
grep -rl "transformers" /root/autodl-tmp/AgentRelay/repositories/webshop/web_agent_site/ 2>/dev/null | head || echo "NONE"
echo "=== done ==="