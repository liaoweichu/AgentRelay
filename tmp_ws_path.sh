#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
PIP=/root/autodl-tmp/AgentRelay/venv/bin/pip
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
echo "---SITE-PACKAGES---"
SP=$($PY -c "import site; print(site.getsitepackages()[0])")
echo "sitepkg: $SP"
echo "---SPACY SM?---"
$PY -c "import en_core_web_sm; print('sm present')" 2>&1 | tail -1
echo "---SETUP .pth---"
echo "$WS" > "$SP/webshop.pth"
cat "$SP/webshop.pth"
echo "---VERIFY IMPORT after pth---"
$PY -c "import web_agent_site; print('web_agent_site import OK', web_agent_site.__file__)" 2>&1 | tail -3