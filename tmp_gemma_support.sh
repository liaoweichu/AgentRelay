#!/usr/bin/env bash
set -uo pipefail
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
echo "=== transformers version ==="
$PY -c "import transformers; print(transformers.__version__)"
echo "=== does transformers know Gemma4 arch? ==="
$PY - <<'PY'
from transformers.models import auto
cls = auto.auto_factory._LazyAutoMapping if hasattr(auto.auto_factory,'_LazyAutoMapping') else None
import transformers.models.auto.modeling_auto as ma
names = [k for k in dir(ma) if 'GEMMA' in k.upper()]
print("GEMMA refs:", names[:20])
# check AutoModelForMultimodalLM availability
import transformers
print("AutoModelForMultimodalLM:", hasattr(transformers, "AutoModelForMultimodalLM"))
print("AutoProcessor:", hasattr(transformers, "AutoProcessor"))
PY