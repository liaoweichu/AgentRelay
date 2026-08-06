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
        print("  type=", type(files), "count=", len(files))
        total = 0
        for f in files:
            if isinstance(f, dict):
                path = f.get("Path") or f.get("path") or f.get("Name") or str(f)
                size = f.get("Size") or f.get("size") or 0
            else:
                path = getattr(f, "path", None) or getattr(f, "Path", None) or str(f)
                size = getattr(f, "size", None) or getattr(f, "Size", None) or 0
            total += int(size or 0)
            print("  %-60s %10.1f MB" % (str(path)[:60], int(size or 0)/1e6))
        print("  TOTAL: %.1f GB" % (total/1e9))
    except Exception as e:
        print("  ERR", str(e)[:200])
PY