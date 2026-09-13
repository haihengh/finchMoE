#!/bin/bash
#
# Capture `FQ_EXPERT_TRACE` and the io counters for both installs, interleaved.
#
# The traces are the input to `replay_dest.py`; the counters are the engine half
# of any engine-vs-replay comparison. Both are needed from the *same session*:
# this drive drifts within a session by more than most effects measured here
# (METH-15), so an engine number from one hour and a replay from another cannot
# be subtracted. Re-run this immediately before replaying.
#
# Usage: capture-traces.sh [output-dir] [label-prefix]
#   defaults: /tmp/fq-trace, no prefix
set -u
cd "/Volumes/samsung 2t/code/finchMoE"
BIN=./.build/release/FinchMoECLI
PROMPT=docs/benchmark-prompts/real-generation-v1/short-explanation.json
COMMON="--messages-file $PROMPT --max-context 2048 --max-new 32 --temperature 0 --expert-cache-slots 16 --counters"
OUT=${1:-/tmp/fq-trace}
PFX=${2:-}
mkdir -p "$OUT"

run() {
  local label=$1 model=$2
  echo "### ${label} start $(date +%T)" >> $OUT/summary.txt
  FQ_EXPERT_TRACE=$OUT/trace-$label.txt \
    ./tools/memguard.sh --min-free 20 $BIN --model "$model" $COMMON \
      > $OUT/$label.out 2> $OUT/$label.err
  echo "exit=$? $(date +%T)" >> $OUT/summary.txt
  grep -oE "io_wall_ms/step=[0-9.]+|io_read_wall_ms/step=[0-9.]+|io_conc=[0-9.]+|io_read_identity=[a-z]+|hits=[0-9]+|misses=[0-9]+|io_mb/step=[0-9.]+" \
    $OUT/$label.err | tr '\n' ' ' >> $OUT/summary.txt
  echo >> $OUT/summary.txt
}

# Paired and interleaved: 3.6, 3.8, 3.6, 3.8. Same prompt, same slots, same hour.
# Pass the same prefix twice with the order reversed to build a palindrome.
run ${PFX}36-t1 models/Qwen3.6-35B-A3B-4bit.finch
run ${PFX}38-t1 models/Qwen3.8-Flash-Next-125B.finch
run ${PFX}36-t2 models/Qwen3.6-35B-A3B-4bit.finch
run ${PFX}38-t2 models/Qwen3.8-Flash-Next-125B.finch
echo "ALL DONE $(date +%T)" >> $OUT/summary.txt
