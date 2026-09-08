#!/bin/bash
# e2_run.sh — E2 differential logit dump: FinchMoE vs llama.cpp on the same
# chat prompt, same 3090 GGUF. Finds the first divergent token.
#
# Step 1: dump the exact token IDs our engine uses for the benchmark chat
#         prompt (system + user, via the engine's own tokenizer)
# Step 2: llama.cpp logit_dump on those tokens  -> first-token reference logits
# Step 3: FinchMoE --dump-logits on the same tokens -> first-token logits
# Step 4: compare: cosine, top-20, argmax, and per-position max-abs diff
set -euo pipefail
cd "/Volumes/samsung 2t/code/finchMoE/e2_diff"

ENGINE="../finchmoe/finchmoe-infer"
GGUF="../finchmoe/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
ENGINE_DIR="../finchmoe"
LOGIT_DUMP="../llama.cpp/build/bin/logit_dump"
SYSMD="$HOME/.flash-moe/system.md"
SYSMD_BAK="$HOME/.flash-moe/system.md.e2bak"
TOKS_BIN="/tmp/e2_tokens.bin"
LCPP_LOGITS="/tmp/e2_llamacpp_logits.bin"
FM_LOGITS="/tmp/e2_finchmoe_logits.bin"
FM_STDOUT="/tmp/e2_finchmoe_out.txt"

echo "[e2] building the benchmark prompt (HumanEval/0)"
python3 e2_build_prompt.py

# --- Step 0: make the engine's system prompt match the benchmark's ---------
mkdir -p "$HOME/.flash-moe"
[ -f "$SYSMD" ] && cp -f "$SYSMD" "$SYSMD_BAK"
trap 'if [ -f "$SYSMD_BAK" ]; then mv -f "$SYSMD_BAK" "$SYSMD"; else rm -f "$SYSMD"; fi' EXIT
cp /tmp/e2_system.txt "$SYSMD"

# --- Step 1: token dump via the engine's own chat tokenizer -----------------
# -P runs tokenize_chat_message (system prompt + user turn + assistant/think
# suffix) and FINCHMOE_DUMP_PROMPT_TOKENS writes the raw IDs. Kill it right
# after the dump lands — we don't need a full generation for the IDs.
echo "[e2] step 1: dumping prompt tokens from the engine's chat tokenizer"
rm -f "$TOKS_BIN"
( cd "$ENGINE_DIR" && exec env FINCHMOE_DUMP_PROMPT_TOKENS="$TOKS_BIN" \
    "$ENGINE" --gguf "$GGUF" -P "$(cat /tmp/e2_prompt.txt)" \
    --no-think --low-memory -e 0 --top-k 1 --rep-penalty 1.0 -t 1 \
    > /dev/null 2>/tmp/e2_step1.log ) &
P1=$!
for i in $(seq 1 300); do
    [ -s "$TOKS_BIN" ] && break
    if ! kill -0 $P1 2>/dev/null; then break; fi
    sleep 2
done
[ -s "$TOKS_BIN" ] || { echo "[e2] FAIL: token dump not produced" >&2; tail -20 /tmp/e2_step1.log >&2; exit 1; }
kill $P1 2>/dev/null || true
wait $P1 2>/dev/null || true

python3 - "$TOKS_BIN" <<'EOF'
import struct, sys
d = open(sys.argv[1], 'rb').read()
n = struct.unpack('<I', d[:4])[0]
ids = struct.unpack(f'<{n}I', d[4:])
print(f"[e2] engine tokenized the chat prompt into {n} tokens")
print(f"[e2] first 15 ids: {ids[:15]}")
print(f"[e2] last  15 ids: {ids[-15:]}")
open('/tmp/e2_tokens_csv.txt','w').write(','.join(map(str, ids)))
EOF

# --- Step 2: llama.cpp first-token logits (reference) ----------------------
echo "[e2] step 2: llama.cpp logit_dump (CPU reference)"
rm -f "$LCPP_LOGITS" /tmp/llama_layers.bin
"$LOGIT_DUMP" "$GGUF" "$(cat /tmp/e2_tokens_csv.txt)" "$LCPP_LOGITS" \
    > /tmp/e2_step2.log 2>&1

# --- Step 3: FinchMoE first-token logits ------------------------------------
# -p feeds the SAME token IDs, bypassing its tokenizer entirely, and
# --dump-logits writes the pre-sampling logits after prefill.
echo "[e2] step 3: finchmoe --dump-logits"
rm -f "$FM_LOGITS"
( cd "$ENGINE_DIR" && exec "$ENGINE" --gguf "$GGUF" -p "$TOKS_BIN" \
    --dump-logits "$FM_LOGITS" --no-think --low-memory \
    -e 0 --top-k 1 --rep-penalty 1.0 -t 1 \
    > "$FM_STDOUT" 2>/tmp/e2_step3.log )
grep -m1 "dump-logits" /tmp/e2_step3.log || { echo "[e2] FAIL: no finchmoe logits" >&2; tail -20 /tmp/e2_step3.log >&2; exit 1; }

# --- Step 4: compare --------------------------------------------------------
python3 - "$LCPP_LOGITS" "$FM_LOGITS" <<'EOF'
import struct, sys
import numpy as np

a = np.frombuffer(open(sys.argv[1], 'rb').read(), np.float32)
b = np.frombuffer(open(sys.argv[2], 'rb').read(), np.float32)
assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
n = a.shape[0]

cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
maxd = float(np.max(np.abs(a - b)))
arg_a, arg_b = int(np.argmax(a)), int(np.argmax(b))
top_a = set(np.argsort(a)[-20:].tolist())
top_b = set(np.argsort(b)[-20:].tolist())

print(f"\n[e2] vocab size: {n}")
print(f"[e2] cosine similarity : {cos:.6f}")
print(f"[e2] max abs diff       : {maxd:.4f}")
print(f"[e2] argmax             : llama.cpp={arg_a}  finchmoe={arg_b}  "
      f"{'MATCH' if arg_a == arg_b else '*** MISMATCH ***'}")
print(f"[e2] top-20 overlap     : {len(top_a & top_b)}/20")

# worst positions
idx = np.argsort(np.abs(a - b))[-10:][::-1]
print("[e2] worst 10 positions:")
for i in idx:
    print(f"     id={int(i):6d}  llama={float(a[i]):.4f}  finchmoe={float(b[i]):.4f}  d={float(a[i]-b[i]):.4f}")

open('/tmp/e2_result.txt', 'w').write(
    f"cos={cos:.6f} maxd={maxd:.4f} argmax_llamacpp={arg_a} argmax_finchmoe={arg_b} "
    f"top20_overlap={len(top_a & top_b)}/20\n")
EOF

cat /tmp/e2_result.txt
echo "[e2] done. finchmoe generation stdout: $FM_STDOUT"
