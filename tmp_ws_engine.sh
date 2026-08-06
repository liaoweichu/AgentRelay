#!/usr/bin/env bash
WS=/root/autodl-tmp/AgentRelay/repositories/webshop
echo "---SEARCH ENGINE CODE---"
cat "$WS/search_engine/lucene_searcher.py"
echo "---convert_product_file_format.py---"
cat "$WS/search_engine/convert_product_file_format.py"
echo "---web_agent_site search python---"
ls -R "$WS/web_agent_site/engine" 2>/dev/null
echo "---search engine init usage---"
grep -rn "load_index\|resources\|indexes\|SearchEngine\|searcher" "$WS/web_agent_site/engine" 2>/dev/null | head -40