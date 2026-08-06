#!/usr/bin/env bash
set -uo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
echo "=== does pyserini/web_agent_site import transformers at runtime? ==="
$PY - <<'PY'
import sys
# check whether pyserini is importable and whether it imports transformers
try:
    import pyserini
    print("pyserini import OK, version=", getattr(pyserini, "__version__", "?"))
except Exception as e:
    print("pyserini import ERR:", str(e)[:200])
# check web_agent_site
try:
    import web_agent_site
    print("web_agent_site import OK")
except Exception as e:
    print("web_agent_site import ERR:", str(e)[:200])
PY
echo "=== search_for transformers import in pyserini packaging ==="
grep -rl "import transformers\|from transformers" /root/autodl-tmp/AgentRelay/venv/lib/python*/site-packages/pyserini/ 2>/dev/null | head