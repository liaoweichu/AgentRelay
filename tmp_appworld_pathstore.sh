#!/usr/bin/env bash
echo "=== path_store definition ==="
grep -rn "path_store" /root/autodl-tmp/AgentRelay/repositories/appworld/src/appworld/*.py 2>/dev/null | grep -iE "=|class|PathStore|data|root" | head -20
echo "=== find path.py ==="
ls /root/autodl-tmp/AgentRelay/repositories/appworld/src/appworld/path.py 2>&1
echo "=== path.py content ==="
cat /root/autodl-tmp/AgentRelay/repositories/appworld/src/appworld/path.py 2>&1 | head -80