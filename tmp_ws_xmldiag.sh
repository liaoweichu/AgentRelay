#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
export PYTHONPATH="$WS"
cd /root
echo "---import nltk standalone---"
$PY -c "import nltk; print('nltk OK')" 2>&1 | tail -4
echo "---sys.path + xml resolution---"
$PY -c "import sys; print('CWD', __import__('os').getcwd()); [print(repr(p)) for p in sys.path]; import importlib.util as u; s=u.find_spec('xml'); print('xml spec', s.origin if s else None)" 2>&1 | tail -20
echo "---find xml dirs on path---"
for p in /root/miniconda3/lib/python3.12 /root/autodl-tmp/AgentRelay/venv/lib/python3.12 /root; do
  echo "== $p =="; ls -d "$p"/xml "$p"/locale 2>/dev/null
done