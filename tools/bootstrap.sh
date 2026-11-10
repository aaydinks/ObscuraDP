#!/usr/bin/env bash
# Create .venv and install the package with its development tools.
#
#   bash tools/bootstrap.sh            # core + dev tools
#   bash tools/bootstrap.sh legacy     # also what the old research code needs (comparison tests)
#
# A CPU-only PyTorch build is used when its download server is reachable. Otherwise pip falls
# back to the default build from PyPI, which on Linux also downloads several gigabytes of GPU
# libraries that are never used here.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-python3}"
EXTRAS="dev"
if [ "${1:-}" = "legacy" ]; then EXTRAS="dev,legacy"; fi

if [ ! -d .venv ]; then "$PYTHON" -m venv .venv; fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip

CPU_INDEX="https://download.pytorch.org/whl/cpu"
if python -m pip download --quiet --no-deps --dest "$(mktemp -d)" --index-url "$CPU_INDEX" "torch" >/dev/null 2>&1; then
  echo "bootstrap: using the CPU-only PyTorch index"
  export PIP_EXTRA_INDEX_URL="$CPU_INDEX"
else
  echo "bootstrap: CPU-only PyTorch index not reachable, using the default build from PyPI"
fi

python -m pip install -c constraints/dev.txt -e ".[${EXTRAS}]"
python -c "import obscuradp; print('obscuradp', obscuradp.__version__)"
echo "bootstrap: done. Activate with:  . .venv/bin/activate"
