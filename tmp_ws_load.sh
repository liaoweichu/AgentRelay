#!/usr/bin/env bash
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
echo "---utils.py DEFAULT_FILE_PATH---"
grep -n "DEFAULT_FILE_PATH\|BASE_DIR\|INDEX" "$WS/web_agent_site/utils.py" 2>/dev/null | head -20
sed -n '1,60p' "$WS/web_agent_site/utils.py"
echo "---engine.py 180-215---"
sed -n '180,215p' "$WS/web_agent_site/engine/engine.py"