#!/usr/bin/env bash
EOF_LINE='EOF'
cat >> /root/autodl-tmp/AgentRelay/env.sh <<'APPEND'
# WebShop official environment (Java 21 for pyserini Lucene, NLTK import security off)
export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
APPEND
echo "---updated env.sh---"
cat /root/autodl-tmp/AgentRelay/env.sh