#!/usr/bin/env bash
cd /root/autodl-tmp/AgentRelay
for s in build_router_rows.py fit_router.py build_calibration_rows.py calibrate_router.py run_autodl_matrix.py build_official_task_manifest.py; do
  echo "===== $s ====="
  grep -n "add_argument" scripts/$s
done