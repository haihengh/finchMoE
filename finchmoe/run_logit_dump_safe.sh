#!/bin/bash
# Reference logit dump for GGUF cross-validation.
# SAFETY: must be run ALONE on the 16GB mini (see crash forensics 2026-08-15).
# Watches vm_stat and aborts the run if free memory collapses.
#
# Usage: run_logit_dump_safe.sh [prompt-tokens-file] [out-file]
#   prompt-tokens-file: [n u32][n × u32] token IDs (e.g. from
#     FINCHMOE_DUMP_PROMPT_TOKENS=file — the 90-token soak flow). Defaults to
#     the hardcoded 13-token prompt.
set -u
cd "$(dirname "$0")/../llama.cpp" || exit 1

MODEL=../models/Qwen3.6-35B-A3B-Q4_K_M.gguf
OUT=${2:-../finchmoe/logits_ref.bin}
if [ -n "${1:-}" ]; then
    TOKENS=$(python3 -c "
import struct, sys
d = open(sys.argv[1], 'rb').read()
n = struct.unpack('I', d[:4])[0]
ids = struct.unpack('<%dI' % n, d[4:4+4*n])
print(','.join(str(i) for i in ids))
" "$1")
    echo "[safe-run] prompt: $(wc -c < "$1") bytes, $(echo "$TOKENS" | tr ',' '\n' | wc -l | tr -d ' ') tokens"
else
    TOKENS="248045,846,198,9419,248046,198,248045,74455,198,248068,271,248069,271"
fi

# pre-flight: require >= 6GB reclaimable (free+speculative+inactive; 16KB pages)
# note: +0 coerces vm_stat's trailing-period numbers to integers (gsub doesn't re-split fields)
reclaim_kb=$(vm_stat | awk '/Pages free/{free=$3+0} /Pages speculative/{spec=$3+0} /Pages inactive/{inact=$3+0} END{print int((free+spec+inact)*16/1024)}')
if [ "${reclaim_kb:-0}" -lt 6000000 ]; then
    echo "ABORT: only ${reclaim_kb} MB reclaimable — close more apps first" >&2
    exit 2
fi

./build/bin/logit_dump "$MODEL" "$TOKENS" "$OUT" &
PID=$!
LOW=0
# Kill condition = BOTH free collapsed AND compressor ballooned (the panic
# signature: compressor segments 100% + free ~0). Bare free-at-floor with a
# big inactive file cache is normal macOS behavior on a fresh boot — the
# kernel serves allocations from inactive. The compressor is the real tell.
while kill -0 $PID 2>/dev/null; do
    f=$(vm_stat | awk '/Pages free/{print $3+0}')
    c=$(vm_stat | awk '/Pages occupied by compressor/{print $5+0}')
    if [ "${f:-0}" -lt 12000 ] && [ "${c:-0}" -gt 250000 ]; then  # <190MB free AND >4GB compressed
        LOW=$((LOW+1))
        [ "$LOW" -ge 3 ] && { echo "ABORT: free=${f}*16KB compressed=${c}*16KB — killing logit_dump" >&2; kill -9 $PID; exit 3; }
    else
        LOW=0
    fi
    sleep 2
done
wait $PID
echo "[safe-run] logit_dump exit: $?  output: $OUT"
