#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh
echo "ALFWORLD_DATA=$ALFWORLD_DATA"
echo "APPWORLD_ROOT=$APPWORLD_ROOT"
echo "PYTHONPATH has webshop:"
echo "$PYTHONPATH" | tr ':' '\n' | grep -c webshop
echo "python=$(which python)"
python -c "import sys; print('venv py:', sys.prefix)"