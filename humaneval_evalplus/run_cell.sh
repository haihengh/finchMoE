#!/bin/bash
# run_cell.sh — run ONE cell of the HumanEval matrix end-to-end.
#
#   run_cell.sh <3bit|4bit|gguf> [start end]
#
# Brings up finchmoe-infer for the tier, asserts the engine really loaded that
# tier, starts shim.py, runs the EvalPlus codegen + evaluate, then tears
# everything down. Passing `start end` limits the run to that id range (smoke).
#
# Everything is torn down by an EXIT trap, including ~/.flash-moe/system.md,
# which is shared with the interactive chat client.
set -uo pipefail
cd "$(dirname "$0")"

CELL="${1:?usage: run_cell.sh <3bit|4bit|gguf> [start end]}"
START="${2:-}"
END="${3:-}"

PORT_ENGINE="${PORT_ENGINE:-9000}"
PORT_SHIM="${PORT_SHIM:-8080}"
VENV=".venv"
PY="$PWD/$VENV/bin/python"
ENGINE_LOG="/tmp/finchmoe_eval_${CELL}.log"
SHIM_LOG="/tmp/finchmoe_shim_${CELL}.log"
SYSMD="$HOME/.flash-moe/system.md"
SYSMD_BAK="$HOME/.flash-moe/system.md.evalbak"

case "$CELL" in
  3bit) MODEL_ID="finchmoe-3bit"
        ENGINE_ARGS="-m . -e 0 --top-k 1 --no-think --rep-penalty 1.0"
        EXPECT="3-bit experts (1376256 bytes each)" ;;
  4bit) MODEL_ID="finchmoe-4bit"
        ENGINE_ARGS="-m . --4bit -e 0 --top-k 1 --no-think --rep-penalty 1.0"
        EXPECT="4-bit experts (1769472 bytes each)" ;;
  gguf) MODEL_ID="finchmoe-gguf-q4km"
        ENGINE_ARGS="--gguf ../models/Qwen3.6-35B-A3B-Q4_K_M.gguf -e 0 --top-k 1 --no-think --rep-penalty 1.0 --low-memory"
        EXPECT="[gguf] magic OK" ;;
  gguf3090) MODEL_ID="finchmoe-gguf-q4km-3090"
        # the EXACT lmstudio-community Q4_K_M the 3090 ran (733 tensors,
        # imatrix-quantized; 21,166,757,728 bytes) — cell D prime, the true
        # control against the published 91.5%.
        ENGINE_ARGS="--gguf Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf -e 0 --top-k 1 --no-think --rep-penalty 1.0 --low-memory"
        EXPECT="[gguf] magic OK" ;;
  *) echo "ABORT: unknown cell '$CELL' (want 3bit|4bit|gguf|gguf3090)" >&2; exit 2 ;;
esac

[ -x "$PY" ] || { echo "ABORT: no venv — run ./setup.sh first" >&2; exit 2; }

# ---- guards -------------------------------------------------------------
# A second engine on a busy port dies on bind while the harness quietly talks
# to the FIRST one — that silently mislabelled a whole tier on 2026-08-20.
for p in "$PORT_ENGINE" "$PORT_SHIM"; do
    if lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "ABORT: port $p already in use — another run is live" >&2
        exit 1
    fi
done

ENGINE_PID=""; SHIM_PID=""
cleanup() {
    [ -n "$SHIM_PID" ]   && kill "$SHIM_PID"   2>/dev/null
    [ -n "$ENGINE_PID" ] && kill "$ENGINE_PID" 2>/dev/null
    [ -n "$ENGINE_PID" ] && wait "$ENGINE_PID" 2>/dev/null
    # restore the shared system prompt
    if [ -f "$SYSMD_BAK" ]; then mv -f "$SYSMD_BAK" "$SYSMD"
    else rm -f "$SYSMD"; fi
}
trap cleanup EXIT INT TERM

# ---- system-prompt parity ----------------------------------------------
# The engine ignores the request's system message (extract_last_content takes
# the LAST "content"), so the 3090's system prompt has to arrive this way.
mkdir -p "$HOME/.flash-moe"
[ -f "$SYSMD" ] && cp -f "$SYSMD" "$SYSMD_BAK"
printf 'You are a helpful assistant good at coding.' > "$SYSMD"

# ---- engine -------------------------------------------------------------
# Started from finchmoe/: the engine resolves shaders.metal, vocab.bin and
# packed_experts_* against the PROCESS cwd, not -m.
echo "[run] cell=$CELL model=$MODEL_ID engine args: $ENGINE_ARGS"
rm -f "$ENGINE_LOG"
( cd ../finchmoe && exec ./finchmoe-infer -R "$PORT_ENGINE" $ENGINE_ARGS ) \
    > "$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!

echo "[run] waiting for engine on :$PORT_ENGINE (log: $ENGINE_LOG)"
UP=0
for i in $(seq 1 600); do
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT_ENGINE/health"; then
        UP=1; break
    fi
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo "ABORT: engine died during startup — last 30 lines:" >&2
        tail -30 "$ENGINE_LOG" >&2
        exit 1
    fi
    sleep 1
done
[ "$UP" = 1 ] || { echo "ABORT: engine not ready after 600s" >&2; exit 1; }

# ---- tier assertion (mandatory) ----------------------------------------
# --4bit was silently overridden by expert auto-detect before commit 6c2e264;
# never trust the flag, trust the engine's own startup log.
if ! grep -qF "$EXPECT" "$ENGINE_LOG"; then
    echo "ABORT: engine did not load the '$CELL' tier — expected: $EXPECT" >&2
    grep -E "Quant:|\[auto\]|\[gguf\]" "$ENGINE_LOG" >&2 || true
    exit 1
fi
if [ "$CELL" != "gguf" ] && grep -q '\[auto\] Using' "$ENGINE_LOG"; then
    echo "ABORT: expert auto-detect fired — it can override the tier flag" >&2
    grep -E "Quant:|\[auto\]" "$ENGINE_LOG" >&2
    exit 1
fi
echo "[run] tier verified: $(grep -m1 -E 'Quant:|\[gguf\] magic' "$ENGINE_LOG")"

# ---- shim ---------------------------------------------------------------
"$PY" shim.py --port "$PORT_SHIM" --upstream-port "$PORT_ENGINE" \
    > "$SHIM_LOG" 2>&1 &
SHIM_PID=$!
for i in $(seq 1 30); do
    curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT_SHIM/health" && break
    sleep 1
done
echo "[run] shim up on :$PORT_SHIM (log: $SHIM_LOG)"

# ---- generate + evaluate ------------------------------------------------
export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"
SAMPLES="results/humaneval/${MODEL_ID}_openai_temp_0.0.jsonl"

echo "[run] generating${START:+ (ids $START-$END)} ..."
"$PY" humaneval_gen.py "$MODEL_ID" $START $END || {
    echo "ABORT: generation failed" >&2; exit 1; }

if [ -n "$START" ]; then
    # Partial run: evalplus.evaluate asserts every problem is present, so a
    # smoke run cannot be scored. Report protocol diagnostics instead — that
    # is what the smoke run is for.
    echo "[run] partial run (ids $START-$END) — diagnostics instead of pass@1"
    "$PY" smoke_check.py "$MODEL_ID"
else
    echo "[run] evaluating $SAMPLES"
    "$PY" -m evalplus.evaluate --dataset humaneval --samples "$SAMPLES" \
        --i-just-wanna-run 2>&1 | tee "results/${MODEL_ID}_eval.txt"
fi

echo "=== cell $CELL done — samples: $SAMPLES ==="
