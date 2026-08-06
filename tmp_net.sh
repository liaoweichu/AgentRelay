#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
$PY - <<'EOF'
import socket
hosts = ["hf-mirror.com", "cdn-lfs.hf-mirror.com", "huggingface.co",
         "www.baidu.com", "mirrors.aliyun.com", "pypi.org", "files.pythonhosted.org"]
for h in hosts:
    try:
        ip = socket.gethostbyname(h)
        print(h, "->", ip)
    except Exception as e:
        print(h, "FAIL", repr(e))
EOF