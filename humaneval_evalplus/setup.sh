#!/bin/bash
# setup.sh — build the EvalPlus environment used to benchmark FinchMoE tiers
# against the 3090's llama.cpp numbers.
#
# Pins Python 3.12 to match the 3090 run (humaneval_3090/README.md). The system
# default here is 3.14, which is newer than anything evalplus 0.3.1 pins against.
#
# The two Windows patches the 3090 README describes (time_limit/SIGALRM and
# reliability_guard/resource) are NOT needed on macOS — both are Unix-only APIs
# that exist here.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"

if ! command -v uv >/dev/null 2>&1; then
    echo "ABORT: uv not found (expected /opt/homebrew/bin/uv)" >&2
    exit 1
fi

echo "[setup] creating $VENV (python 3.12)"
uv venv --python 3.12 "$VENV"

echo "[setup] installing evalplus==0.3.1 + openai"
VIRTUAL_ENV="$PWD/$VENV" uv pip install "evalplus==0.3.1" openai

mkdir -p results

# macOS needs its own compatibility patch (different from the 3090's two
# Windows ones): reliability_guard's setrlimit(RLIMIT_AS) raises here and
# kills every test subprocess.
echo "[setup] applying the macOS evalplus patch"
"$VENV/bin/python" patch_evalplus.py

echo "[setup] pre-warming the HumanEval+ dataset cache (needs network)"
"$VENV/bin/python" -c "
from evalplus.data import get_human_eval_plus
d = get_human_eval_plus()
print(f'[setup] dataset OK: {len(d)} problems')
"

echo "[setup] done. Environment: $PWD/$VENV"
"$VENV/bin/python" -c "import evalplus, openai, sys; print('  python  ', sys.version.split()[0]); print('  evalplus', evalplus.__version__); print('  openai  ', openai.__version__)"
