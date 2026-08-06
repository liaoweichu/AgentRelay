#!/usr/bin/env bash
# Upgrade transformers to 5.10.1 for Gemma 4 (E4B needs >=5.5, 12B needs >=5.10).
set -uo pipefail
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
PIP=/root/autodl-tmp/AgentRelay/venv/bin/pip
echo "=== pre-upgrade versions ==="
$PY -c "import torch, transformers, bitsandbytes; print('torch', torch.__version__); print('transformers', transformers.__version__); print('bnb', bitsandbytes.__version__)"
echo "=== installing transformers 5.10.1 ==="
$PIP install --no-cache-dir "transformers==5.10.1" 2>&1 | tail -20
echo "=== post-upgrade versions ==="
$PY -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__); import transformers; print('has multimodal:', hasattr(transformers, 'AutoModelForMultimodalLM'))"
echo "=== setup done at $(date) ==="