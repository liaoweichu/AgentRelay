#!/usr/bin/env bash
# Check ModelScope availability + disk + candidate Gemma 4 repo IDs
set -euo pipefail
echo "=== disk ==="
df -h /root/autodl-tmp | tail -1
echo "=== modelscope installed? ==="
/root/autodl-tmp/AgentRelay/venv/bin/pip show modelscope 2>/dev/null | head -2 || echo "modelscope NOT installed"
echo "=== pinned transformers version ==="
/root/autodl-tmp/AgentRelay/venv/bin/pip show transformers 2>/dev/null | head -2
echo "=== try list candidate repos via modelscope Hub (if available) ==="
/root/autodl-tmp/AgentRelay/venv/bin/python - <<'PY'
try:
    from modelscope.hub.api import HubApi
    api = HubApi()
    for mid in ["google/gemma-4-E4B-it", "google/gemma-4-E4B", "google/gemma-4-12b-it"]:
        try:
            info = api.get_model(model_id=mid)
            print("EXISTS", mid, "| files=", len(info.files or []))
        except Exception as e:
            print("CHECK", mid, "ERR", str(e)[:120])
except Exception as e:
    print("modelscope import/hub not usable:", str(e)[:200])
PY