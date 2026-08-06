#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
PY=/root/autodl-tmp/AgentRelay/venv/bin/python
echo "=== nvidia-smi ==="
nvidia-smi 2>&1 | head -15
echo ""
echo "=== torch cuda ==="
$PY - <<'EOF'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    p = torch.cuda.get_device_properties(0)
    print("total mem GiB:", round(p.total_memory/1024**3, 2))
    print("cuda version:", torch.version.cuda)
print("torch:", torch.__version__)
EOF
echo ""
echo "=== locked config exists ==="
ls -la /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json 2>&1