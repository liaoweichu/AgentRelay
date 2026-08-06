#!/usr/bin/env bash
set -uo pipefail
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null
echo "=== web_agent_site import (with env PATH) ==="
/root/autodl-tmp/AgentRelay/venv/bin/python -c "import web_agent_site; print('web_agent_site OK')" 2>&1 | tail -2
echo "=== ALFWorld transformers usage ==="
grep -rl "import transformers\|from transformers" /root/autodl-tmp/AgentRelay/repositories/alfworld/ 2>/dev/null | head
echo "=== AppWorld transformers usage ==="
grep -rl "import transformers\|from transformers" /root/autodl-tmp/AgentRelay/repositories/appworld/ 2>/dev/null | head
echo "=== WebShop transformers usage (non-pyserini) ==="
grep -rl "import transformers\|from transformers" /root/autodl-tmp/AgentRelay/repositories/webshop/web_agent_site/ 2>/dev/null | head