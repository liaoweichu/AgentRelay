#!/usr/bin/env bash
echo "=== appworld/__init__.py 30-60 ==="
sed -n '30,60p' /root/autodl-tmp/AgentRelay/repositories/appworld/src/appworld/__init__.py
echo "=== path_store.py ==="
cat /root/autodl-tmp/AgentRelay/repositories/appworld/src/appworld/common/path_store.py 2>&1 | head -80