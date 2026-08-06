#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
$PY - <<'EOF'
import json, urllib.request
for model in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]:
    url = f"https://hf-mirror.com/api/models/{model}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"x"})
        data = json.load(urllib.request.urlopen(req, timeout=20))
        files = []
        for s in data.get("siblings", []):
            rfn = s.get("rfilename","")
            if rfn.endswith((".safetensors",".bin",".pt")):
                files.append(rfn)
        print(model, "->", files[:6])
    except Exception as e:
        print(model, "FAIL", repr(e))
EOF