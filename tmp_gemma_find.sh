#!/usr/bin/env bash
set -uo pipefail
echo "=== find large temp files being written ==="
find /root/autodl-tmp/AgentRelay/models /root/.cache /tmp -name "*.safetensors*" -o -name "*.tmp" 2>/dev/null | head
echo "=== du on modelscope cache locations ==="
du -sh /root/autodl-tmp/AgentRelay/models/* 2>/dev/null | sort -rh | head
du -sh /root/.cache/modelscope 2>/dev/null
echo "=== running download processes ==="
ps aux | grep -i "snapshot_download\|modelscope\|huggingface_hub" | grep -v grep | head