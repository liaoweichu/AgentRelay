#!/usr/bin/env bash
set -euo pipefail

TAU2_REPOSITORY=${1:?usage: prepare_tau2_bench.sh /absolute/path/to/tau2-bench}
TAU2_REVISION=0ed2fd8d830a20657d89ae9c2efcc94838aa7129
TAU2_ORIGIN=https://github.com/sierra-research/tau2-bench.git

test -d "$TAU2_REPOSITORY/.git"
test "$(git -C "$TAU2_REPOSITORY" rev-parse HEAD)" = "$TAU2_REVISION"
test "$(git -C "$TAU2_REPOSITORY" remote get-url origin)" = "$TAU2_ORIGIN"
python -m pip install --editable "$TAU2_REPOSITORY"
python -c "import tau2; print('tau2 import ok')"
