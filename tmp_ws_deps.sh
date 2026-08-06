#!/usr/bin/env bash
PIP=/root/autodl-tmp/AgentRelay/venv/bin/pip
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
echo "---INSTALL thefuzz rich pyyaml scikit-learn-----"
$PIP install -q thefuzz rich pyyaml scikit-learn 2>&1 | tail -5
echo "---VERIFY COVERAGE---"
$PY -c "import thefuzz, rich, yaml, sklearn; print('deps OK')" 2>&1 | tail -3