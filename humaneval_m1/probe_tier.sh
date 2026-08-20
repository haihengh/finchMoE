#!/bin/bash
# Probe a weight tier on HumanEval (raw protocol, T=0).
# Usage: probe_tier.sh <name> [extra infer args...]
#   name: short label (used in the results filename)
#   extra args: passed to finchmoe-infer (e.g. --int8-experts)
#   env: EXTRA_WEIGHTS="-w PATH -j PATH" for a non-default bin/manifest
set -e
cd "$(dirname "$0")"
NAME="$1"; shift || true
PORT="${PORT:-9000}"
LIMIT="${LIMIT:-20}"
INFER_ARGS="$@"

RESULTS="he_results_${NAME}.jsonl"
rm -f "$RESULTS" /tmp/probe_server.log

# start server (default tier unless INFER_ARGS given)
../finchmoe/finchmoe-infer -R "$PORT" -m . -e 0 --top-k 1 --no-think --rep-penalty 1.0 \
    $EXTRA_WEIGHTS $INFER_ARGS > /tmp/probe_server.log 2>&1 &
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
