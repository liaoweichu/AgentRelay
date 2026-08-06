#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
MON=$R/results/manifests
cd $R
AW_REV=aaba6870f86c5be6a08a491f32a50b906227bc3e
AW_CFG=$R/repositories/alfworld/configs/base_config.yaml

echo "=== rebuild ALFWorld eval manifests (correct train_eval) ==="
$PY scripts/build_official_task_manifest.py --benchmark alfworld --split valid_seen --purpose tune --revision $AW_REV --output $MON/alfworld-valid-seen.json --alfworld-config $AW_CFG --train-eval eval_in_distribution
$PY scripts/build_official_task_manifest.py --benchmark alfworld --split valid_unseen --purpose evaluate --revision $AW_REV --output $MON/alfworld-valid-unseen.json --alfworld-config $AW_CFG --train-eval eval_out_of_distribution

echo "=== WebShop goal count with FULL items file ==="
$PY - <<'EOF'
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
for fp in ["/root/autodl-tmp/AgentRelay/repositories/webshop/data/items_shuffle.json",
           "/root/autodl-tmp/AgentRelay/repositories/webshop/data/items_shuffle_1000.json"]:
    env = WebAgentTextEnv(observation_mode="text", human_goals=1, file_path=fp)
    try:
        print(fp.split("/")[-1], "-> goals:", len(env.server.goals))
    finally:
        env.close()
EOF