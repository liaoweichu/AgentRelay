#!/usr/bin/env bash
source /root/autodl-tmp/AgentRelay/env.sh 2>/dev/null || true
export NLTK_DISABLE_IMPORT_SECURITY=1
R=/root/autodl-tmp/AgentRelay
echo "=== webshop data files ==="
ls -la $R/repositories/webshop/data/ 2>&1 | head
echo ""
echo "=== goal counts per json ==="
$PY=/root/autodl-tmp/AgentRelay/venv/bin/python
for f in $R/repositories/webshop/data/*_goals.json; do
  [ -e "$f" ] || continue
  n=$($PY -c "import json,sys; d=json.load(open('$f')); print(len(d))" 2>/dev/null)
  echo "$(basename $f): $n"
done
echo ""
echo "=== ALFWorld data dirs ==="
ls $R/datasets/alfworld/json_2.1.1/ 2>&1