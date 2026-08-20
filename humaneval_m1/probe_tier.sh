#!/bin/bash
# Probe a weight tier on HumanEval (raw protocol, T=0).
# Usage: probe_tier.sh <name> [extra infer args...]
#   name: short label (used in the results filename)
#   extra args: passed to finchmoe-infer (e.g. --int8-experts)
#   env: EXTRA_WEIGHTS="--weights PATH --manifest PATH" for a non-default
#        bin/manifest — paths resolve against finchmoe/ (the server's cwd)
#   env: THINK_ALLOWED=1 drops --no-think (E3' protocol probe)
#   env: TEMP=0.3 overrides the default -e 0 (E3' sampling probe)
set -e
cd "$(dirname "$0")"
NAME="$1"; shift || true
PORT="${PORT:-9000}"
LIMIT="${LIMIT:-20}"
INFER_ARGS="$@"
THINK_ARGS="--no-think"
[ "${THINK_ALLOWED:-0}" = "1" ] && THINK_ARGS=""
TEMP_E="${TEMP:-0}"

RESULTS="he_results_${NAME}.jsonl"
rm -f "$RESULTS" /tmp/probe_server.log

# Engine-contract check: layer norms must carry the Qwen3.5 (1 + w) fold.
# A raw-HF build (rms << 1) passes quantize_non_experts.py --verify but
# soups the engine — the 2026-08-20 E1 build lost a full probe this way.
if [ -n "$EXTRA_WEIGHTS" ]; then
( cd ../finchmoe && python3 - "$EXTRA_WEIGHTS" <<'EOF'
import json, sys
paths = dict(zip(sys.argv[1].split()[::2], sys.argv[1].split()[1::2]))
manifest = paths.get('--manifest')
if manifest:
    m = json.load(open(manifest))['tensors']
    t = m.get('model.layers.0.input_layernorm.weight')
    if t and t.get('dtype') == 'BF16':
        import numpy as np
        wf = paths.get('--weights') or manifest.replace('.json', '.bin')
        raw = open(wf, 'rb').read()
        d = raw[t['offset']:t['offset'] + t['size']]
        v = (np.frombuffer(d, np.uint16).astype(np.uint32) << 16).view(np.float32)
        rms = float(np.sqrt(np.mean(v**2)))
        print(f"[norm-check] input_layernorm rms={rms:.4f} (expect ~1.0)")
        if rms < 0.5:
            sys.exit('ABORT: layer norms are raw (no +1 fold) — '
                     'rebuild with FINCHMOE_NORM_PLUS1=1')
EOF
)
fi

# start server from finchmoe/ — the engine resolves shaders.metal, vocab.bin,
# and {model_path}/packed_experts_* against the PROCESS CWD (not -m), and all
# of those live in finchmoe/ (packs are symlinks there). Run from
# humaneval_m1 and the server boots GPU-less and dies on vocab.bin.
( cd ../finchmoe && exec ./finchmoe-infer -R "$PORT" -m . -e "$TEMP_E" --top-k 1 $THINK_ARGS --rep-penalty 1.0 \
    $EXTRA_WEIGHTS $INFER_ARGS > /tmp/probe_server.log 2>&1 ) &
SRV=$!
for i in $(seq 1 60); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/" && break
    sleep 1
done

python3 generate.py --limit "$LIMIT" --port "$PORT" --results "$RESULTS"
python3 evaluate.py --results "$RESULTS"
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
echo "=== probe $NAME done: $RESULTS ==="
