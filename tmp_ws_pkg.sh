#!/usr/bin/env bash
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
echo "---SETUP FILES---"
ls "$WS"/*.py "$WS"/setup.* "$WS"/pyproject.toml 2>/dev/null
echo "---ENV __init__---"
sed -n '1,60p' "$WS/web_agent_site/envs/web_agent_text_env.py" 2>/dev/null | head -60
echo "---num_products default---"
grep -n "num_products\|search_engine\|init_search_engine" "$WS/web_agent_site/envs/web_agent_text_env.py" | head