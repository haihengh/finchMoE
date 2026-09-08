#!/bin/bash
# run_llamacpp_logit_probe.sh — llama-server reference probe for the same
# HumanEval/0 request the finchMoE probe used. Captures per-token top-20
# (n_probs=20, pre-sampling) logprobs on the 3090's exact GGUF file.
#
#   run_llamacpp_logit_probe.sh <out_prefix>
#
# System prompt is matched to whatever finchMoE actually used (the built-in
# default "You are a helpful assistant." — the ~/.flash-moe/system.md write is
# sandbox-denied). Consistency across engines is what matters for the diff.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUT="${1:?usage: run_llamacpp_logit_probe.sh <out_prefix>}"
PORT=8090
GGUF="$REPO_ROOT/finchmoe/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
PROMPT_FILE="/tmp/he0_prompt.txt"
SYSTEM="You are a helpful assistant."
MAX_TOKENS=400
SERVER_LOG="$OUT.server.log"

[ -f "$GGUF" ] || { echo "ABORT: GGUF not found: $GGUF" >&2; exit 2; }
[ -f "$PROMPT_FILE" ] || { echo "ABORT: $PROMPT_FILE missing" >&2; exit 2; }
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ABORT: port $PORT in use" >&2; exit 1
fi

SRV=""
cleanup() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null; [ -n "$SRV" ] && wait "$SRV" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "[llama] starting llama-server on :$PORT"
rm -f "$SERVER_LOG"
# Memory note: this is a 16 GB machine holding a 21 GB GGUF, with the user's
# apps holding several GB of active RAM. GPU offload (-ngl 99) puts the ~20 GB
# of expert weights in file-backed mapped Metal buffers; prefill touches the
# whole expert set and the GPU command buffer dies with
# kIOGPUCommandBufferCallbackErrorOutOfMemory (reproduced 2026-08-22 with
# -c 8192, and with -c 4096 -b 256 -ub 256). -cmoe is WORSE: it allocates an
# 18.6 GB real-RAM CPU_REPACK copy of the experts — impossible on 16 GB.
# The proven layout is the E2 CPU reference: -ngl 0 keeps the whole model in
# the zero-copy CPU_Mapped mmap (E2 log: "CPU_Mapped model buffer size =
# 20175.71 MiB", no REPACK), streams weights through the page cache, and
# needs no GPU working set at all. Slower (CPU decode), but it works under
# any memory pressure and is the config the E2 parity test was built on.
# -b 256 + -c 4096: the prompt here is ~145 tokens so 4096 KV is ample.
"$REPO_ROOT/llama.cpp/build/bin/llama-server" -m "$GGUF" -c 4096 -ngl 0 -b 256 -ub 256 \
    --reasoning off --temp 0 --top-k 1 --repeat-penalty 1.0 \
    --host 127.0.0.1 --port "$PORT" --verbose \
    > "$SERVER_LOG" 2>&1 &
SRV=$!

# /health answers 503 ("Loading model") until the model is actually loaded;
# check the HTTP status, not just curl's exit code.
UP=0
for i in $(seq 1 900); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    if [ "$CODE" = "200" ]; then UP=1; break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then
        echo "ABORT: server died — tail:" >&2
        tail -30 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done
[ "$UP" = 1 ] || { echo "ABORT: server not ready after 900s" >&2; tail -30 "$SERVER_LOG" >&2; exit 1; }
echo "[llama] server up (health 200)"

python3 -c "
import json
prompt = open('$PROMPT_FILE').read()
body = json.dumps({
    'model':'x',
    'messages':[
        {'role':'system','content':'$SYSTEM'},
        {'role':'user','content':prompt},
    ],
    'max_tokens':$MAX_TOKENS,
    'temperature':0,
    'n_probs':20,
})
open('$OUT.request.json','w').write(body)
"

echo "[llama] sending request (max_tokens=$MAX_TOKENS, n_probs=20) ..."
# /health can read 200 a moment before the slot is fully ready; retry the
# actual request a few times on "Loading model" 503s.
for attempt in 1 2 3 4 5; do
    curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary @"$OUT.request.json" > "$OUT.response.json"
    if ! grep -q '"Loading model"' "$OUT.response.json" 2>/dev/null; then break; fi
    echo "[llama] still loading (attempt $attempt), waiting 15s ..."
    sleep 15
done

python3 -c "
import json
r = json.load(open('$OUT.response.json'))
ch = r.get('choices',[{}])[0]
lp = ch.get('logprobs')
if lp is None:
    print('[llama] NO logprobs in response; keys:', list(ch.keys()), 'top keys:', list(r.keys()))
else:
    content = lp.get('content') or []
    print(f'[llama] logprobs content tokens: {len(content)}')
    if content:
        print('[llama] first entry keys:', list(content[0].keys()))
        print('[llama] first entry:', json.dumps(content[0])[:400])
print('[llama] usage:', r.get('usage'))
"
echo "[llama] response: $OUT.response.json   server log: $SERVER_LOG"
