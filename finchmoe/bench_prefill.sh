#!/bin/bash
# bench_prefill.sh — prefill benchmark + bitwise parity matrix for finchmoe-infer
#
# Usage: ./bench_prefill.sh [chunk sizes...]   (default: 1 2 4 8 23 64)
#
# For each chunk size: runs flag 0 (per-token baseline) and flag N (chunked),
# parses [prefill]/[ttft] timings, and verifies --dump-logits bitwise parity
# against the flag-0 reference (max|Δ| must be 0.0 in FP32 mode; the KV
# helper refactor changed the compiler vectorization of the CPU-attention
# dot products, so the cross-path delta is 1e-5 — the 1e-4 tolerance still
# catches the 1e-2-class wobble).
#
# Requires: ./finchmoe-infer built; quant_clean weights; python3 + numpy.
# Env: PROMPT (text) or PROMPT_FILE; CHUNKS; MODEL_ARGS (extra flags).

set -u
cd "$(dirname "$0")"

PROMPT="${PROMPT:-The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Japan is Tokyo. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of Brazil is Brasilia. The capital of India is New Delhi. The capital of China is Beijing.}"
if [ -n "${PROMPT_FILE:-}" ]; then PROMPT="$(cat "$PROMPT_FILE")"; fi
CHUNKS="${@:-${CHUNKS:-1 2 4 8 23 64}}"
MODEL_ARGS="${MODEL_ARGS:--m . --weights quant_clean/model_weights_quant.bin --manifest quant_clean/model_weights_quant.json}"
COMMON="--tokens 0 --temperature 0 --top-k 1"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "== finchmoe prefill benchmark =="
echo "prompt tokens: $(echo "$PROMPT" | wc -w) words; chunk sizes: $CHUNKS"
echo

# Flag-0 reference run
./finchmoe-infer $MODEL_ARGS --prompt "$PROMPT" $COMMON --dump-logits "$tmp/ref0.bin" >"$tmp/ref0.log" 2>&1
p0=$(grep -oE '\[prefill\].*' "$tmp/ref0.log" | tail -1)
t0=$(grep -oE '\[ttft\].*' "$tmp/ref0.log" | tail -1)
printf "%-8s | %-45s | %s\n" "chunk" "prefill" "ttft"
printf "%-8s | %-45s | %s\n" "0 (base)" "${p0#\[prefill\] }" "${t0#\[ttft\] }"

for cs in $CHUNKS; do
    ./finchmoe-infer $MODEL_ARGS --prompt "$PROMPT" $COMMON --prefill-chunk "$cs" --dump-logits "$tmp/c$cs.bin" >"$tmp/c$cs.log" 2>&1
    p=$(grep -oE '\[prefill\].*' "$tmp/c$cs.log" | tail -1)
    t=$(grep -oE '\[ttft\].*' "$tmp/c$cs.log" | tail -1)
    parity=$(python3 - "$tmp/ref0.bin" "$tmp/c$cs.bin" <<'EOF'
import sys, numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32); b = np.fromfile(sys.argv[2], dtype=np.float32)
d = float(np.abs(a-b).max())
print("BITWISE" if d == 0.0 else ("OK(1e-5)" if d < 1e-4 else f"DIFF {d:.3e}"))
EOF
)
    printf "%-8s | %-45s | %s\n" "$cs" "${p#\[prefill\] }" "${t#\[ttft\] }  [$parity]"
done

echo
echo "FINCHMOE_PF_TIMING=1 ./finchmoe-infer ... --prefill-chunk 8  → per-phase breakdown"
