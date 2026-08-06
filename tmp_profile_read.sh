#!/usr/bin/env bash
set -uo pipefail
echo "=== existing service-profile.json ==="
cat /root/autodl-tmp/AgentRelay/results/service-profile.json
echo ""
echo "=== tmp_profile.sh ==="
cat /root/autodl-tmp/AgentRelay/tmp_profile.sh 2>/dev/null
echo "=== profile script in scripts/ ==="
ls /root/autodl-tmp/AgentRelay/scripts/ | grep -i profile