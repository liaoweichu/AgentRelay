#!/usr/bin/env bash
set -euo pipefail

# Run only after checkout_repositories.py has verified every locked commit.
AGENTRELAY_ROOT="/root/autodl-tmp/AgentRelay"
REPOSITORY_ROOT="${AGENTRELAY_ROOT}/repositories"
PYTHON_BIN="${AGENTRELAY_ROOT}/venv/bin/python"

for required in alfworld webshop appworld; do
  if [[ ! -d "${REPOSITORY_ROOT}/${required}/.git" ]]; then
    echo "missing locked official repository: ${REPOSITORY_ROOT}/${required}" >&2
    exit 1
  fi
done

export ALFWORLD_DATA="${AGENTRELAY_ROOT}/datasets/alfworld"
mkdir -p "${ALFWORLD_DATA}"
"${PYTHON_BIN}" -m pip install -e "${REPOSITORY_ROOT}/alfworld[full]"
"${PYTHON_BIN}" "${REPOSITORY_ROOT}/alfworld/scripts/alfworld-download"

# WebShop's official setup script installs its pinned requirements, downloads
# the complete product/instruction data, and builds the search index.  Keep its
# generated files in the locked repository below autodl-tmp.
(
  cd "${REPOSITORY_ROOT}/webshop"
  bash setup.sh -d all
)

# AppWorld source installation requires its encrypted bundle to be unpacked in
# the repository, followed by the official benchmark-data download.
(
  cd "${REPOSITORY_ROOT}/appworld"
  git lfs pull
  "${PYTHON_BIN}" -m pip install -e .
  export APPWORLD_ROOT="${REPOSITORY_ROOT}/appworld"
  "${PYTHON_BIN}" -m appworld.cli install --repo
  "${PYTHON_BIN}" -m appworld.cli download data
  "${PYTHON_BIN}" -m appworld.cli verify tests
)

echo "Official benchmark environments prepared under ${REPOSITORY_ROOT}."
