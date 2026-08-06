#!/usr/bin/env bash
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
echo "---WS ROOT TOP LEVEL---"
ls -la "$WS" | awk '{print $NF}'
echo "---FIND locale dirs---"
find "$WS" -maxdepth 3 -type d -name "locale" 2>/dev/null
echo "---FIND locale files---"
find "$WS" -maxdepth 3 -type f -name "locale*" 2>/dev/null | head
echo "---PYTHONPATH---"
/root/autodl-tmp/AgentRelay/venv/bin/python -c "import sys; print('\n'.join(sys.path))"