#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH=/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
cd $R

echo "=== JAVA check ==="
echo "JAVA_HOME=$JAVA_HOME"
$PY -c "import jnius; print('jnius ok', jnius.get_jdk_home('posix'))"

echo ""
echo "=== webshop full shuffle goal count ==="
$PY -c "from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv; e=WebAgentTextEnv(observation_mode='text', human_goals=1, file_path='$R/repositories/webshop/data/items_shuffle.json'); print('FULL_SHUFFLE_GOALS', len(e.server.goals)); e.close()"

echo ""
echo "=== webshop human_ins goal count ==="
$PY -c "from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv; e=WebAgentTextEnv(observation_mode='text', human_goals=1, file_path='$R/repositories/webshop/data/items_human_ins.json'); print('HUMAN_INS_GOALS', len(e.server.goals)); e.close()"