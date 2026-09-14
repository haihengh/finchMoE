#!/bin/bash
#
# Capture `FQ_EXPERT_TRACE` for a *corpus* of prompts, both installs, one trace
# per prompt per install. The traces are the input to `hot_set_transfer.py`,
# which builds a static per-layer hot set from some prompts and prices it on the
# held-out one -- the out-of-sample number that decides whether pinning a hot
# set resident can work.
#
# This is not `capture-traces.sh`. That one interleaves two runs of a single
# prompt to pair the engine against its own replay; here there is nothing to
# pair, because the quantity being measured (which experts a prompt routes to)
# does not depend on the drive at all. One run per cell is enough, and each
# prompt is run at temperature 0 so the trace is reproducible.
#
# Usage: capture-corpus-traces.sh [prompt-dir] [output-dir]
#   defaults: docs/benchmark-prompts/hot-set-transfer, /tmp/fq-corpus
set -u
cd "/Volumes/samsung 2t/code/finchMoE"
BIN=./.build/release/FinchMoECLI
PDIR=${1:-docs/benchmark-prompts/hot-set-transfer}
OUT=${2:-/tmp/fq-corpus}
COMMON="--max-context 2048 --max-new 32 --temperature 0 --expert-cache-slots 16 --counters"
mkdir -p "$OUT"
: > "$OUT/summary.txt"

# The baseline prompt lives in the sibling real-generation-v1 set and is used
# unchanged, so the corpus is the four domains and nothing else differs.
PROMPTS=(
  "a-coastal:docs/benchmark-prompts/real-generation-v1/short-explanation.json"
  "b-debt:$PDIR/b-debt-covenants.json"
  "c-protein:$PDIR/c-protein-folding.json"
  "d-counterpoint:$PDIR/d-counterpoint.json"
)
MODELS=("36:models/Qwen3.6-35B-A3B-4bit.finch" "38:models/Qwen3.8-Flash-Next-125B.finch")

for p in "${PROMPTS[@]}"; do
  key=${p%%:*}; prompt=${p#*:}
  for m in "${MODELS[@]}"; do
    mk=${m%%:*}; model=${m#*:}
    label="$key-$mk"
    echo "### $label start $(date +%T)" >> "$OUT/summary.txt"
    FQ_EXPERT_TRACE="$OUT/trace-$label.txt" \
      ./tools/memguard.sh --min-free 20 $BIN --model "$model" \
        --messages-file "$prompt" $COMMON \
        > "$OUT/$label.out" 2> "$OUT/$label.err"
    echo "exit=$? $(date +%T) reads=$(wc -l < "$OUT/trace-$label.txt" | tr -d ' ')" >> "$OUT/summary.txt"
    grep -oE "hits=[0-9]+|misses=[0-9]+|io_read_wall_ms/step=[0-9.]+" \
      "$OUT/$label.err" | tr '\n' ' ' >> "$OUT/summary.txt"
    echo >> "$OUT/summary.txt"
  done
done
echo "ALL DONE $(date +%T)" >> "$OUT/summary.txt"
