#!/usr/bin/env bash
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
AW=/root/autodl-tmp/AgentRelay/repositories/appworld
export PYTHONPATH="$AW/src:/root/autodl-tmp/AgentRelay/src:$PYTHONPATH"
echo "---dev.txt tasks---"
head -3 "$AW/data/datasets/dev.txt"
FIRST=$(head -1 "$AW/data/datasets/dev.txt" | awk '{print $1}')
echo "FIRST TASK: $FIRST"
cd /root
$PY - <<EOF
import traceback
try:
    from agentrelay.official_adapters import AppWorldAdapter
    task_id = "$FIRST"
    adapter = AppWorldAdapter(task_id=task_id, dataset_revision="a072b7a86e7c1d5b1d7175659d750ebb9b79f10a",
                              split="dev", experiment_name="smoke-test", allow_official_evaluation=False)
    obs = adapter.reset()
    print("RESET OK. goal:", obs.text[:150])
    print("valid_actions:", obs.valid_actions)
    print("resources:", dict(obs.resources))
    adapter.close()
except Exception as e:
    print("ERR:", type(e).__name__)
    traceback.print_exc()
EOF