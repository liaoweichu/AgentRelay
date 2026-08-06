#!/usr/bin/env bash
export AGENTRELAY_DATA_ROOT="/root/autodl-tmp/AgentRelay"
export HF_HOME="/root/autodl-tmp/AgentRelay/cache/huggingface"
export HF_HUB_CACHE="/root/autodl-tmp/AgentRelay/cache/huggingface/hub"
export HF_DATASETS_CACHE="/root/autodl-tmp/AgentRelay/datasets"
export PIP_CACHE_DIR="/root/autodl-tmp/AgentRelay/cache/pip"
export TMPDIR="/root/autodl-tmp/AgentRelay/tmp"
# WebShop official environment (Java 21 for pyserini Lucene, NLTK import security off)
export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$JAVA_HOME/lib/server"
export NLTK_DISABLE_IMPORT_SECURITY=1
export PYTHONPATH="/root/autodl-tmp/AgentRelay/repositories/webshop:$PYTHONPATH"
source "/root/autodl-tmp/AgentRelay/venv/bin/activate"
export APPWORLD_ROOT="/root/autodl-tmp/AgentRelay/repositories/appworld"
export ALFWORLD_DATA="/root/autodl-tmp/AgentRelay/datasets/alfworld"
# Force offline inference from the local model cache to avoid huggingface.co timeouts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1