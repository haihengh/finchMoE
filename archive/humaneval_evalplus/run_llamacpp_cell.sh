#!/bin/bash
# run_llamacpp_cell.sh — E1 of the gap investigation:
# llama-server + the 3090's exact GGUF + the SAME EvalPlus harness, on this Mac.
#
#   ./run_llamacpp_cell.sh            # full 164
#   ./run_llamacpp_cell.sh 0 20       # smoke (id range)
#
# If this reproduces ~90%: the harness and file are vindicated on this hardware
# and the bug is definitively in FinchMoE. If it lands ~13%: the shim or
# harness protocol is the problem.
set -uo pipefail
cd "$(dirname "$0")"

START="${1:-}"; END="${2:-}"
GGUF="../finchmoe/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
MODEL_ID="llamacpp-mac-3090gguf"
PORT=8080
VENV=".venv"; PY="$PWD/$VENV/bin/python"
SERVER_LOG="/tmp/llamacpp_e1.log"

[ -x "$PY" ] || { echo "ABORT: no venv — run ./setup.sh first" >&2; exit 2; }
[ -f "$GGUF" ] || { echo "ABORT: 3090 GGUF not found at $GGUF" >&2; exit 2; }
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ABORT: port $PORT in use" >&2; exit 1
fi
pgrep -fl finchmoe-infer && { echo "ABORT: finchmoe engine running — one model at a time" >&2; exit 1; }

SRV=""
cleanup() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null; [ -n "$SRV" ] && wait "$SRV" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "[e1] starting llama-server (reasoning off, jinja, Metal, -ngl 99)"
../llama.cpp/build/bin/llama-server -m "$GGUF" -c 32768 -ngl 99 \
    --reasoning off --host 127.0.0.1 --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SRV=$!

echo "[e1] waiting for server readiness"
UP=0
for i in $(seq 1 900); do
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/health"; then UP=1; break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then
        echo "ABORT: llama-server died — tail:" >&2
        tail -30 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done
[ "$UP" = 1 ] || { echo "ABORT: server not ready after 900s" >&2; tail -30 "$SERVER_LOG" >&2; exit 1; }
echo "[e1] server up. load line: $(grep -m1 'model loaded' "$SERVER_LOG" || echo '(check log)')"

export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"
SAMPLES="results/humaneval/${MODEL_ID}_openai_temp_0.0.jsonl"

echo "[e1] generating${START:+ (ids $START-$END)} ..."
"$PY" humaneval_gen.py "$MODEL_ID" $START $END || { echo "ABORT: generation failed" >&2; exit 1; }

if [ -n "$START" ]; then
    "$PY" smoke_check.py "$MODEL_ID"
else
    echo "[e1] evaluating"
    "$PY" -m evalplus.evaluate --dataset humaneval --samples "$SAMPLES" \
        --i-just-wanna-run 2>&1 | tee "results/${MODEL_ID}_eval.txt"
fi

echo "=== E1 done — samples: $SAMPLES ==="
