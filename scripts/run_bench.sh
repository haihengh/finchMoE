#!/usr/bin/env bash
# Run one .finch install across the three frozen real-generation-v1 cases using
# the published host-row protocol (the one the M6 mini rows use): app sampling
# defaults, a 128-token cap, a discarded warmup per case, then the measured pass
# in a fresh process. Appends each case's timing footer to <out>/summary.txt.
#
# usage: scripts/run_bench.sh <model-dir> <result-prefix> <out-dir>
#   e.g. scripts/run_bench.sh models/Qwen3.6-35B-A3B-4bit.finch qwen36-35b-base \
#          benchmark-results/m4pro-20260930-XXXX
set -euo pipefail

MODEL="${1:?model dir}"
PREFIX="${2:?result prefix, e.g. qwen36-35b-base}"
OUT="${3:?output dir}"

CLI="$(pwd)/.build/release/FinchMoECLI"
PROMPTS="$(pwd)/docs/benchmark-prompts/real-generation-v1"
mkdir -p "$OUT"

run_one() {
  local case_id="$1" dest="$2"
  "$CLI" \
    --model "$MODEL" \
    --messages-file "$PROMPTS/${case_id}.json" \
    --max-new 128 \
    --max-context 4096 \
    --temperature 0.2 \
    --top-k 64 \
    --top-p 0.95 \
    --counters \
    > "$dest.stdout" 2> "$dest.stderr"
  local footer
  footer="$(grep -h '^\[stop=' "$dest.stderr" | head -1)"
  echo "[$PREFIX/$case_id] $footer"
}

for case_id in short-explanation medium-review long-synthesis; do
  echo "=== $PREFIX / $case_id : warmup (discarded) ==="
  run_one "$case_id" "$OUT/warmup/${PREFIX}-${case_id}"
  echo "=== $PREFIX / $case_id : measured ==="
  run_one "$case_id" "$OUT/${PREFIX}-${case_id}"
  grep -h '^\[stop=' "$OUT/${PREFIX}-${case_id}.stderr" | head -1 \
    | tee -a "$OUT/summary.txt"
done
