#!/usr/bin/env bash
set -uo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
PIP=/root/autodl-tmp/AgentRelay/venv/bin/pip
echo "=== install modelscope_hub --no-deps ==="
$PIP install --no-deps modelscope_hub 2>&1 | tail -4
echo "=== verify repo IDs ==="
$PY - <<'PY'
try:
    from modelscope.hub.api import HubApi
    api = HubApi()
    for mid in ["google/gemma-4-E4B-it", "google/gemma-4-E4B", "google/gemma-4-12b-it"]:
        try:
            info = api.get_model(model_id=mid)
            files = getattr(info, "files", None) or []
            print("EXISTS", mid, "| files=", len(files))
        except Exception as e:
            print("CHECK", mid, "ERR", str(e)[:150])
except Exception as e:
    print("hub unusable:", str(e)[:250])
PY