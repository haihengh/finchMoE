#!/bin/bash
# bench.sh — M1 mini benchmark suite for finchmoe-infer (run from this dir)
# Covers fp32 / --kv-fp16 / --kv-turbo with the same tests as the M4 runs.
set -u
cd "$(dirname "$0")"

MODEL="-m . --weights quant_clean/model_weights_quant.bin --manifest quant_clean/model_weights_quant.json"
P_CAPS="The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Japan is Tokyo. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of Brazil is Brasilia. The capital of India is New Delhi. The capital of China is Beijing."

for MODE in "" "--kv-fp16" "--kv-turbo"; do
    echo ""
    echo "======================================================"
    echo "== MODE: ${MODE:-fp32}"
    echo "======================================================"

    echo "== 1. Decode baseline (3x, 50 tokens) =="
    for i in 1 2 3; do
        ./finchmoe-infer $MODEL $MODE -P "The capital of France is" --tokens 50 2>&1 \
            | grep -E "Generation" | head -1
    done

    echo "== 2. Prefill: 90-token prompt, chunk 8 =="
    ./finchmoe-infer $MODEL $MODE --prompt "$P_CAPS" --tokens 0 --temperature 0 --top-k 1 \
        --prefill-chunk 8 2>&1 | grep -E "ttft" | head -1

    echo "== 3. Prefill: per-token baseline (chunk 0) =="
    ./finchmoe-infer $MODEL $MODE --prompt "$P_CAPS" --tokens 0 --temperature 0 --top-k 1 \
        --prefill-chunk 0 2>&1 | grep -E "ttft" | head -1

    echo "== 4. Long decode (200 tokens, attn-heavy) =="
    ./finchmoe-infer $MODEL $MODE -P "Write a short story about a lighthouse keeper." \
        --tokens 200 2>&1 | grep -E "Generation" | head -1

    echo "== 5. pread_wait (chunked, timing) =="
    FINCHMOE_PF_TIMING=1 ./finchmoe-infer $MODEL $MODE --prompt "$P_CAPS" --tokens 0 \
        --temperature 0 --top-k 1 --prefill-chunk 8 2>&1 | grep -E "pread_wait" | head -1
done

echo ""
echo "M4 references (warm cache): decode fp32 16-22 / fp16 21.9-22.6 / turbo 17.0-21.9 tok/s;"
echo "  prefill 90-tok chunk-8 ~1.9-2.4 s all modes; flag-0 ~5.0-5.5 s; 200-tok decode ~17-19 tok/s."
echo "On the M4 the quant modes are within noise — the expert I/O dominates."
