#!/bin/bash
# wobble_hunt.sh — 10x chunk-4-vs-flag-0 prefill parity repeat with dumps.
# Captures hidden_dump/hidden_new + logits for any DIFF iteration into
# /tmp/wobble_hunt/ for the (token, layer) bisection.
set -u
cd "$(dirname "$0")"

P="The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Japan is Tokyo. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of Brazil is Brasilia. The capital of India is New Delhi. The capital of China is Beijing."
MODEL="-m . --weights quant_clean/model_weights_quant.bin --manifest quant_clean/model_weights_quant.json"
COMMON="--tokens 0 --temperature 0 --top-k 1"
OUT=/tmp/wobble_hunt
mkdir -p "$OUT"
rm -f "$OUT"/*.bin

for i in $(seq 1 10); do
    rm -f /tmp/hidden_dump.bin /tmp/hidden_new.bin /tmp/wref.bin /tmp/wc4.bin

    FINCHMOE_DUMP_HIDDEN=1 ./finchmoe-infer $MODEL --prompt "$P" $COMMON \
        --prefill-chunk 0 --dump-logits /tmp/wref.bin >"$OUT/ref_$i.log" 2>&1
    mv /tmp/hidden_dump.bin "$OUT/ref_hidden_$i.bin" 2>/dev/null

    FINCHMOE_DUMP_HIDDEN=1 ./finchmoe-infer $MODEL --prompt "$P" $COMMON \
        --prefill-chunk 4 --dump-logits /tmp/wc4.bin >"$OUT/c4_$i.log" 2>&1
    mv /tmp/hidden_new.bin "$OUT/c4_hidden_$i.bin" 2>/dev/null
    cp /tmp/wref.bin "$OUT/ref_logits_$i.bin" 2>/dev/null
    cp /tmp/wc4.bin  "$OUT/c4_logits_$i.bin"  2>/dev/null

    d=$(python3 -c "
import numpy as np
a=np.fromfile('/tmp/wref.bin',dtype=np.float32); b=np.fromfile('/tmp/wc4.bin',dtype=np.float32)
if a.size!=b.size: print(f'SIZE-MISMATCH {a.size} vs {b.size}')
else:
    m=np.abs(a-b).max()
    print(f'{m:.3e}' + ('  <-- DIFF' if m>0 else '  bitwise'))
")
    echo "iter $i: max|dlogits| = $d"
done
echo "hunt done — artifacts in $OUT"
