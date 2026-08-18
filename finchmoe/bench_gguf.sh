#!/bin/bash
# bench_gguf.sh — Phase C gate runner for the GGUF path.
# Runs: (a) llama.cpp reference (logit_dump), (b) finchmoe CPU reference
# (--low-memory), (c) finchmoe GPU run, compares cos, prints tok/s.
# Usage: ./bench_gguf.sh [tokens] [prompt-file]
set -u
cd "$(dirname "$0")"

TOKENS=${1:-13}
PROMPT=${2:-prompt_tokens_gguf.bin}
GGUF=../models/Qwen3.6-35B-A3B-Q4_K_M.gguf
REF=logits_ref.bin
CPUOUT=logits_fm_cpu.bin
GPUOUT=logits_fm_gpu.bin

# token ids for llama.cpp reference
IDS=$(python3 -c "
import struct
with open('$PROMPT','rb') as f:
    n = struct.unpack('I', f.read(4))[0]
    ids = struct.unpack('<%dI' % n, f.read(4*n))
print(','.join(str(i) for i in ids))
")

# memory watchdog: kill the child if free collapses AND compressor balloons
watch() {
    local pid=$1
    while kill -0 $pid 2>/dev/null; do
        f=$(vm_stat | awk '/Pages free/{gsub(/\./,"");print $3}')
        c=$(vm_stat | awk '/Pages occupied by compressor/{gsub(/\./,"");print $5}')
        if [ "${f:-0}" -lt 12000 ] && [ "${c:-0}" -gt 250000 ]; then
            echo "WATCHDOG: free=${f}*16KB compressed=${c}*16KB — killing run" >&2
            kill -9 $pid; return 3
        fi
        sleep 2
    done
    wait $pid
}

echo "=== (a) llama.cpp reference ==="
( cd ../llama.cpp && ./build/bin/logit_dump "$GGUF" "$IDS" ../finchmoe/$REF ) 2>&1 | grep -E "wrote|error" || { echo "REF FAILED"; exit 1; }

echo "=== (b) finchmoe CPU reference (--low-memory) ==="
./finchmoe-infer --gguf "$GGUF" --low-memory --prefill-chunk 0 -L -p "$PROMPT" -t 1 -I $CPUOUT > /tmp/bench_cpu.log 2>&1 &
watch $! || exit 1
grep -E "Tokens:|Generation" /tmp/bench_cpu.log | tail -2

echo "=== (c) finchmoe GPU run ==="
./finchmoe-infer --gguf "$GGUF" --prefill-chunk 0 -L -p "$PROMPT" -t 1 -I $GPUOUT > /tmp/bench_gpu.log 2>&1 &
watch $! || exit 1
grep -E "Tokens:|Generation" /tmp/bench_gpu.log | tail -2

echo "=== comparisons ==="
python3 compare_gguf_logits.py $REF $CPUOUT | grep -E "cos_sim|argmax"
python3 compare_gguf_logits.py $CPUOUT $GPUOUT | grep -E "cos_sim|argmax"
