#!/usr/bin/env bash
set -uo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
$PY - <<'PY'
from modelscope.hub.api import HubApi
api = HubApi()
for mid in ["google/gemma-4-E4B-it", "google/gemma-4-12b-it"]:
    print("==== %s ====" % mid)
    try:
        files = api.get_model_files(model_id=mid)
        total = 0
        for f in files:
            size = getattr(f, "size", 0) or 0
            total += size
            print("  %-60s %10.1f MB" % (f.path, size/1e6))
        print("  TOTAL: %.1f GB" % (total/1e9))
    except Exception as e:
        print("  ERR", str(e)[:200])
PY