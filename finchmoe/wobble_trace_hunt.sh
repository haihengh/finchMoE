#!/bin/bash
# wobble_trace_hunt.sh — repeat chunk-4-vs-flag-0 with Phase-B traces until DIFF.
# Saves pb_ref.bin / pb_new.bin + logits for every iteration into /tmp/wobble_trace/.
set -u
cd "$(dirname "$0")"

P="The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Japan is Tokyo. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of Brazil is Brasilia. The capital of India is New Delhi. The capital of China is Beijing."
MODEL="-m . --weights quant_clean/model_weights_quant.bin --manifest quant_clean/model_weights_quant.json"
COMMON="--tokens 0 --temperature 0 --top-k 1"
OUT=/tmp/wobble_trace
mkdir -p "$OUT"
rm -f "$OUT"/*.bin

for i in $(seq 1 30); do
    rm -f /tmp/pb_ref.bin /tmp/pb_new.bin /tmp/wtref.bin /tmp/wtc4.bin

    FINCHMOE_DUMP_PHASEB=1 ./finchmoe-infer $MODEL --prompt "$P" $COMMON \
        --prefill-chunk 0 --dump-logits /tmp/wtref.bin >"$OUT/ref_$i.log" 2>&1
    cp /tmp/pb_ref.bin "$OUT/pb_ref_$i.bin" 2>/dev/null

    FINCHMOE_DUMP_PHASEB=1 ./finchmoe-infer $MODEL --prompt "$P" $COMMON \
        --prefill-chunk 4 --dump-logits /tmp/wtc4.bin >"$OUT/c4_$i.log" 2>&1
    cp /tmp/pb_new.bin "$OUT/pb_new_$i.bin" 2>/dev/null
    cp /tmp/wtref.bin "$OUT/ref_logits_$i.bin" 2>/dev/null
    cp /tmp/wtc4.bin  "$OUT/c4_logits_$i.bin"  2>/dev/null

    d=$(python3 -c "
import numpy as np
a=np.fromfile('/tmp/wtref.bin',dtype=np.float32); b=np.fromfile('/tmp/wtc4.bin',dtype=np.float32)
if a.size!=b.size: print(f'SIZE-MISMATCH {a.size} vs {b.size}')
else:
    m=np.abs(a-b).max()
    print(f'{m:.3e}' + ('  <-- DIFF' if m>0 else '  bitwise'))
")
    echo "iter $i: max|dlogits| = $d"
    if echo "$d" | grep -q DIFF; then
        echo "DIFF captured at iter $i — traces in $OUT"
        exit 0
    fi
done
echo "no diff in 20 iterations"
exit 1
