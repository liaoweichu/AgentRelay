#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python

$PY - <<'EOF'
from agentrelay.official_adapters import ALFWorldAdapter
cfg = "/root/autodl-tmp/AgentRelay/repositories/alfworld/configs/base_config.yaml"
adapter = ALFWorldAdapter(
    config_path=cfg,
    train_eval="train",
    dataset_revision="aaba6870f86c5be6a08a491f32a50b906227bc3e",
    split="train",
    task_index=0,
)
obs = adapter.reset()
print("ALFWORLD RESET OK")
print("goal:", obs.text[:120])
print("valid_actions:", obs.valid_actions[:3])
adapter.close()
print("ALL OK")
EOF