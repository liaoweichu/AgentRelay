#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
AGENTRELAY_ROOT="/root/autodl-tmp/AgentRelay"
VENV_DIR="${AGENTRELAY_ROOT}/venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export AGENTRELAY_DATA_ROOT="${AGENTRELAY_ROOT}"
export HF_HOME="${AGENTRELAY_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${AGENTRELAY_ROOT}/datasets"
export PIP_CACHE_DIR="${AGENTRELAY_ROOT}/cache/pip"
export TMPDIR="${AGENTRELAY_ROOT}/tmp"

mkdir -p "${AGENTRELAY_ROOT}"/{cache,datasets,models,runs,results,repositories,tmp}
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install \
  torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${VENV_DIR}/bin/python" -m pip install -e "${PROJECT_DIR}[ml,dev]"

cat > "${AGENTRELAY_ROOT}/env.sh" <<EOF
export AGENTRELAY_DATA_ROOT="${AGENTRELAY_ROOT}"
export HF_HOME="${AGENTRELAY_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${AGENTRELAY_ROOT}/cache/huggingface/hub"
export HF_DATASETS_CACHE="${AGENTRELAY_ROOT}/datasets"
export PIP_CACHE_DIR="${AGENTRELAY_ROOT}/cache/pip"
export TMPDIR="${AGENTRELAY_ROOT}/tmp"
source "${VENV_DIR}/bin/activate"
EOF

cat <<EOF
AutoDL environment prepared.
Activate with: source ${AGENTRELAY_ROOT}/env.sh
Next: lock a config, download pinned public data, then run preflight.
EOF
