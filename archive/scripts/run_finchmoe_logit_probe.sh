#!/bin/bash
# run_finchmoe_logit_probe.sh — start finchmoe-infer for ONE tier, send the
# HumanEval/0 chat request, capture the --logit-diag dump, then tear down.
#
#   run_finchmoe_logit_probe.sh <3bit|4bit|gguf3090> <out_prefix>
#
# Mirrors run_cell.sh's engine invocation (cwd=finchmoe, system prompt via
# ~/.flash-moe/system.md) so the prompt tokens match the real eval run exactly.
# Strictly sequential on purpose: concurrent model loads on this Mac have
# caused kernel panics (documented in humaneval_evalplus/README.md).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TIER="${1:?usage: run_finchmoe_logit_probe.sh <3bit|4bit|gguf3090> <out_prefix>}"
OUT="${2:?usage: run_finchmoe_logit_probe.sh <3bit|4bit|gguf3090> <out_prefix>}"
PORT_ENGINE=9000
PROMPT_FILE="/tmp/he0_prompt.txt"
SYSMD="$HOME/.flash-moe/system.md"
SYSMD_BAK="$HOME/.flash-moe/system.md.evalbak"
MAX_TOKENS=400

[ -f "$PROMPT_FILE" ] || { echo "ABORT: $PROMPT_FILE missing (run §15.1 first)" >&2; exit 2; }

case "$TIER" in
  3bit)     ENGINE_ARGS="-m . -e 0 --top-k 1 --no-think --rep-penalty 1.05" ;;
  4bit)     ENGINE_ARGS="-m . --4bit -e 0 --top-k 1 --no-think --rep-penalty 1.05" ;;
  gguf3090) ENGINE_ARGS="--gguf Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf -e 0 --top-k 1 --no-think --rep-penalty 1.05 --low-memory" ;;
  *) echo "ABORT: unknown tier '$TIER'" >&2; exit 2 ;;
esac

for p in "$PORT_ENGINE"; do
    if lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "ABORT: port $p already in use" >&2; exit 1
    fi
done

ENGINE_LOG="$OUT.engine.log"
RESPONSE="$OUT.response.json"

ENGINE_PID=""
cleanup() {
    [ -n "$ENGINE_PID" ] && kill "$ENGINE_PID" 2>/dev/null
    [ -n "$ENGINE_PID" ] && wait "$ENGINE_PID" 2>/dev/null
    if [ -f "$SYSMD_BAK" ]; then mv -f "$SYSMD_BAK" "$SYSMD"; else rm -f "$SYSMD"; fi
}
trap cleanup EXIT INT TERM

mkdir -p "$HOME/.flash-moe"
[ -f "$SYSMD" ] && cp -f "$SYSMD" "$SYSMD_BAK"
printf 'You are a helpful assistant good at coding.' > "$SYSMD"

echo "[probe] tier=$TIER out=$OUT engine args: $ENGINE_ARGS"
rm -f "$ENGINE_LOG" "$RESPONSE"
( cd "$REPO_ROOT/finchmoe" && exec ./finchmoe-infer -R "$PORT_ENGINE" $ENGINE_ARGS --logit-diag 1 ) \
    > "$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!

UP=0
for i in $(seq 1 600); do
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT_ENGINE/health"; then UP=1; break; fi
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo "ABORT: engine died — tail:" >&2
        tail -30 "$ENGINE_LOG" >&2
        exit 1
    fi
    sleep 1
done
[ "$UP" = 1 ] || { echo "ABORT: engine not ready after 600s" >&2; tail -30 "$ENGINE_LOG" >&2; exit 1; }
echo "[probe] engine up: $(grep -m1 -E 'Quant:|\[gguf\] magic' "$ENGINE_LOG")"

python3 -c "
import json
prompt = open('$PROMPT_FILE').read()
body = json.dumps({'messages':[{'role':'user','content':prompt}],'max_tokens':$MAX_TOKENS,'temperature':0})
open('$OUT.request.json','w').write(body)
"

curl -s -m 900 "http://127.0.0.1:$PORT_ENGINE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @"$OUT.request.json" > "$RESPONSE"

echo "[probe] done — logit-diag blocks: $(grep -c '\[logit-diag\] step=' "$ENGINE_LOG")"
echo "[probe] log: $ENGINE_LOG   response: $RESPONSE"
