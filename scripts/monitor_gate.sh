#!/usr/bin/env bash
# Monitor the WebShop train/dev learnability gate and report every 20 minutes.
# Usage: bash scripts/monitor_gate.sh [--once]
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG="${LOG:-$PROJECT_ROOT/gate_ws_traindev.log}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/webshop-train-dev-gate-gemma4}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1200}"

report_status() {
  now="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "============================================================"
  echo "[$now] G8 WebShop train/dev gate monitor"
  echo "------------------------------------------------------------"

  # 1. Is the gate orchestrator still alive?
  gate_pids="$(pgrep -f 'run_webshop_train_dev_gate.py' || true)"
  if [ -n "$gate_pids" ]; then
    echo "gate=RUNNING pids=$(echo $gate_pids | tr '\n' ' ')"
  else
    echo "gate=STOPPED (orchestrator no longer running)"
  fi

  # 2. Which matrix sub-run is active right now?
  active="$(pgrep -f 'run_autodl_matrix.py' | head -n1 || true)"
  if [ -n "$active" ]; then
    active_cmd="$(tr '\0' ' ' < /proc/$active/cmdline 2>/dev/null || true)"
    echo "active_matrix_pid=$active"
    echo "active_matrix_cmd=${active_cmd:-unknown}"
  else
    echo "active_matrix=none"
  fi

  # 3. Completed episodes per matrix run dir.
  echo "runs:"
  if ls -d runs/formal-matrix-webshop-gate-* >/dev/null 2>&1; then
    for d in runs/formal-matrix-webshop-gate-*; do
      n="$(find "$d" -name '*.json' 2>/dev/null | wc -l)"
      echo "  $d episodes=$n"
    done
  else
    echo "  (none yet)"
  fi

  # 4. Current stage of the gate output directory.
  echo "output_dir:"
  if [ -d "$RESULTS_DIR" ]; then
    for f in "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.jsonl "$RESULTS_DIR"/*.joblib; do
      [ -e "$f" ] && echo "  $(basename "$f")  ($(stat -c '%y' "$f" | cut -d. -f1))"
    done
    if [ -f "$RESULTS_DIR/receipt.json" ]; then
      echo -n "  runs_recorded="
      python3 -c "
import json
try:
    r=json.load(open('$RESULTS_DIR/receipt.json'))
    print(sorted(r.get('runs',{}).keys()) or 'none')
except Exception as e:
    print('unreadable:', e)
"
    fi
  else
    echo "  (not created yet)"
  fi

  # 5. Latest significant log lines (drop per-step progress bars).
  echo "log_tail:"
  if [ -f "$LOG" ]; then
    tail -200 "$LOG" 2>/dev/null | grep -v -E '^\s*[0-9]+%\|' | tail -15
  else
    echo "  (no log at $LOG)"
  fi
  echo "============================================================"
  echo ""
}

if [ "${1:-}" = "--once" ]; then
  report_status
  exit 0
fi

echo "Monitoring $LOG every ${INTERVAL_SECONDS}s (Ctrl-C to stop)."
while true; do
  report_status
  sleep "$INTERVAL_SECONDS"
done