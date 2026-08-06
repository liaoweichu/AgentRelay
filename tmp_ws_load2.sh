#!/usr/bin/env bash
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
echo "---load_products---"
sed -n '215,300p' "$WS/web_agent_site/engine/engine.py"
echo "---goal.py get_goals---"
sed -n '1,80p' "$WS/web_agent_site/engine/goal.py"