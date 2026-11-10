#!/usr/bin/env bash
# Regenerate constraints/dev.txt: pinned versions of the core and development dependencies.
# Needs uv (https://docs.astral.sh/uv/). Resolves for the oldest supported Python so the pins
# work on every supported version. GPU helper packages are left out: constraints only pin, and
# CI installs the CPU-only PyTorch build.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
uv pip compile pyproject.toml --extra dev --python-version 3.11 --universal --quiet \
  --no-annotate --no-header \
  | grep -v -i -E '^(nvidia|cuda|triton)' > constraints/dev.tmp
{
  echo "# Pinned versions of the core and development dependencies."
  echo "# Regenerate with: bash tools/update_constraints.sh"
  echo "# Use with: pip install -c constraints/dev.txt -e \".[dev]\""
  cat constraints/dev.tmp
} > constraints/dev.txt
rm constraints/dev.tmp
echo "wrote constraints/dev.txt"
