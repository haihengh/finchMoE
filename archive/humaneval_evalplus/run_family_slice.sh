#!/bin/bash
# run_family_slice.sh <model_id> <extra_engine_args...>  — eval-config 20-task
# slice with an extra engine-arg family swap (e.g. --cpu-experts), scored by
# score_slice.py. Strictly sequential engine use.
set -uo pipefail
cd "$(dirname "$0")"
MODEL_ID="${1:?model_id}"
shift
EXTRA_ARGS="$*"
PORT_ENGINE="${PORT_ENGINE:-9000}"
PORT_SHIM="${PORT_SHIM:-8080}"
PY="$PWD/.venv/bin/python"
ENGINE_LOG="/tmp/finchmoe_fam_${MODEL_ID}.log"
SHIM_LOG="/tmp/finchmoe_shim_${MODEL_ID}.log"
SYSMD="$HOME/.flash-moe/system.md"
SYSMD_BAK="$HOME/.flash-moe/system.md.evalbak"
START=0; END=20
RAW="results/humaneval/${MODEL_ID}_openai_temp_0.0.raw.jsonl"
MAIN="results/humaneval/${MODEL_ID}_openai_temp_0.0.jsonl"

for p in "$PORT_ENGINE" "$PORT_SHIM"; do
    if lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "ABORT: port $p busy" >&2; exit 1
    fi
done
ENGINE_PID=""; SHIM_PID=""
cleanup() {
    [ -n "$SHIM_PID" ]   && kill "$SHIM_PID" 2>/dev/null
    [ -n "$ENGINE_PID" ] && kill "$ENGINE_PID" 2>/dev/null
    [ -n "$ENGINE_PID" ] && wait "$ENGINE_PID" 2>/dev/null
    if [ -f "$SYSMD_BAK" ]; then mv -f "$SYSMD_BAK" "$SYSMD"; else rm -f "$SYSMD"; fi
}
trap cleanup EXIT INT TERM

mkdir -p "$HOME/.flash-moe"
[ -f "$SYSMD" ] && cp -f "$SYSMD" "$SYSMD_BAK"
printf 'You are a helpful assistant good at coding.' > "$SYSMD"

echo "[run] $MODEL_ID extra: $EXTRA_ARGS"
rm -f "$ENGINE_LOG" "$RAW" "$MAIN"
( cd ../finchmoe && HOME=/tmp exec ./finchmoe-infer -R "$PORT_ENGINE" \
    --gguf Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -e 0 --top-k 1 --no-think --rep-penalty 1.05 --low-memory $EXTRA_ARGS ) \
    > "$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!

UP=0
for i in $(seq 1 600); do
    curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT_ENGINE/health" && { UP=1; break; }
    kill -0 "$ENGINE_PID" 2>/dev/null || { echo "ABORT: engine died" >&2; tail -20 "$ENGINE_LOG" >&2; exit 1; }
    sleep 1
done
[ "$UP" = 1 ] || { echo "ABORT: engine not ready" >&2; tail -20 "$ENGINE_LOG" >&2; exit 1; }
grep -qF "[gguf] magic OK" "$ENGINE_LOG" || { echo "ABORT: not GGUF" >&2; exit 1; }

"$PY" shim.py --port "$PORT_SHIM" --upstream-port "$PORT_ENGINE" > "$SHIM_LOG" 2>&1 &
SHIM_PID=$!
for i in $(seq 1 30); do
    curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT_SHIM/health" && break
    sleep 1
done

export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"
echo "[run] generating $MODEL_ID ids $START-$END ..."
"$PY" humaneval_gen.py "$MODEL_ID" $START $END || { echo "ABORT: gen failed" >&2; exit 1; }
echo "[run] scoring with score_slice.py"
"$PY" score_slice.py "$RAW" $START $END 2>&1 | tail -3
echo "[run] done — $RAW"
