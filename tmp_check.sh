#!/usr/bin/env bash
R=/root/autodl-tmp/AgentRelay
PY=$R/venv/bin/python
cd $R
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH=$R/repositories/webshop:$PYTHONPATH

echo "=== manifest task counts ==="
for f in results/manifests/*.json; do
  n=$($PY -c "import json,sys;d=json.load(open('$f'));print(len(d['tasks']),d['complete_official_split'])")
  echo "$(basename $f): $n"
done

echo ""
echo "=== webshop goal counts per file ==="
for f in repositories/webshop/data/items_shuffle_1000.json repositories/webshop/data/items_shuffle.json; do
  echo -n "$(basename $f): "
  $PY -c "import json;d=json.load(open('$f'));print(len(d))"
done

echo ""
echo "=== webshop env goal load (1000 file) ==="
$PY -c "from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv; e=WebAgentTextEnv(observation_mode='text', human_goals=1, file_path='$R/repositories/webshop/data/items_shuffle_1000.json'); print('goals', len(e.server.goals)); e.close()"

echo ""
echo "=== webshop env goal load (full shuffle) ==="
$PY -c "from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv; e=WebAgentTextEnv(observation_mode='text', human_goals=1, file_path='$R/repositories/webshop/data/items_shuffle.json'); print('goals', len(e.server.goals)); e.close()"