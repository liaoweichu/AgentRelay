#!/usr/bin/env bash
set -e
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"

PY=/root/autodl-tmp/AgentRelay/venv/bin/python
echo "=== appworld package location ==="
$PY -c "import appworld, os; print(os.path.dirname(appworld.__file__))"

echo "=== APPWORLD_ROOT ==="
echo "$APPWORLD_ROOT"

echo "=== AppWorldAdapter smoke test ==="
$PY - <<'EOF'
import os
print("APPWORLD_ROOT env =", os.environ.get("APPWORLD_ROOT"))
from agentrelay.official_adapters import AppWorldAdapter
adapter = AppWorldAdapter(
    task_id="50e1ac9_1",
    dataset_revision="a072b7a86e7c1d5b1d7175659d750ebb9b79f10a",
    split="dev",
    experiment_name="smoke-test",
    allow_official_evaluation=False,
)
obs = adapter.reset()
print("RESET OK")
print("goal:", obs.text[:150])
adapter.close()
print("ALL OK")
EOF