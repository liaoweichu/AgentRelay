#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH=/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
R=/root/autodl-tmp/AgentRelay
cd $R

$PY - <<'PYEOF'
import time, gc
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

# Time full-shuffle env creation (products + search engine)
t0 = time.time()
e = WebAgentTextEnv(observation_mode='text', human_goals=1,
                    file_path='/root/autodl-tmp/AgentRelay/repositories/webshop/data/items_shuffle_1000.json')
t1 = time.time()
print(f"ENV_CREATE_1000FILE_SEC={t1-t0:.1f} goals={len(e.server.goals)}")
e.close(); gc.collect()

# Time a single episode: reset -> search goal keywords -> click -> buy
t0 = time.time()
e = WebAgentTextEnv(observation_mode='text', human_goals=1,
                    file_path='/root/autodl-tmp/AgentRelay/repositories/webshop/data/items_shuffle_1000.json')
t1 = time.time()
print(f"ENV_CREATE_AGAIN_SEC={t1-t0:.1f}")
e.reset(session=0)
goal = getattr(e, 'instruction_text', '')
print(f"GOAL={goal[:80]}")
# search using first few words
import re
words = re.findall(r"[a-zA-Z]+", goal)[:6]
kw = words[0] if words else 'shirt'
t0 = time.time()
e.step(f"search[{kw}]")
t1 = time.time()
print(f"STEP_SEARCH_SEC={t1-t0:.4f}")
t0 = time.time()
e.step("click[buy now]")
t1 = time.time()
print(f"STEP_BUY_SEC={t1-t0:.4f}")
e.close(); gc.collect()
PYEOF