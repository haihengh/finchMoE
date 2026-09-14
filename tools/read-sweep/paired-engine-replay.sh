#!/bin/bash
#
# Alternate the engine and the replay of its own trace, one round at a time.
#
# METH-15: this drive drifts within a session by more than most effects measured
# here -- re-running the same capture twenty minutes apart moved one synthetic
# cell from 1.825 to 3.163 GB/s, and the engine's own two back-to-back captures
# differ by 1.2x (`io_read_wall_ms/step` 46.20 against 57.15 on 3.6, 207.00
# against 250.39 on 3.8, 2026-09-13). An engine number from one part of a
# session and a replay number from another part cannot be subtracted. So they
# are not subtracted: they are interleaved, and every engine window is read
# against a replay that ran within a minute of it.
#
# The engine run is cheap -- ~10 s on 3.6, ~30 s on 3.8, model load included --
# so the alternation can be tight. Each round emits one engine window and one
# replay per selected condition; the per-round ratio is the result, not the
# ratio of the two minima.
#
# usage: paired-engine-replay.sh <install> <trace> <depth> [rounds] [conds]
#   install  3.6 | 3.8
#   rounds   default 5
#   conds    default "1,3" -- allowed+warm (the IO-17 condition) and
#            allowed+slot (the engine's own shape). Neither is bypassed: the
#            engine opens its layer files without F_NOCACHE, and F_NOCACHE is
#            not honoured on APFS here anyway -- 3.6's bypassed arm reads at
#            27.8 GB/s, which is RAM, not NAND.
set -u
cd "/Volumes/samsung 2t/code/finchMoE"

INSTALL=${1:?usage: paired-engine-replay.sh <install> <trace> <depth> [rounds] [conds]}
TRACE=${2:?}
DEPTH=${3:?}
ROUNDS=${4:-5}
CONDS=${5:-1,3}

case "$INSTALL" in
    3.6) MODEL=models/Qwen3.6-35B-A3B-4bit.finch; SLOTARG=--depth36 ;;
    3.8) MODEL=models/Qwen3.8-Flash-Next-125B.finch; SLOTARG=--depth38 ;;
    *)   echo "unknown install $INSTALL" >&2; exit 2 ;;
esac

PROMPT=docs/benchmark-prompts/real-generation-v1/short-explanation.json
COMMON="--messages-file $PROMPT --max-context 2048 --max-new 32 --temperature 0 --expert-cache-slots 16 --counters"
ENGINE_OUT=/tmp/paired-engine-$INSTALL.err

# The engine's own split of the read window, so per-request service time can be
# compared and not just the window: `io_thread_wall / misses` is the average
# pread, and `io_conc` is what was actually in flight. Two arms with the same
# in-flight bytes and different service times differ in the read path, not in
# the queueing.
engine_round() {
  ./tools/memguard.sh --min-free 20 ./.build/release/FinchMoECLI \
      --model "$MODEL" $COMMON >/dev/null 2>"$ENGINE_OUT"
  grep -oE "io_read_wall_ms/step=[0-9.]+|io_thread_wall_ms/step=[0-9.]+|io_conc=[0-9.]+" \
    "$ENGINE_OUT" | cut -d= -f2 | tr '\n' ' '
}

printf '# paired-engine-replay  install=%s  trace=%s  depth=%s  conds=%s\n' \
       "$INSTALL" "$TRACE" "$DEPTH" "$CONDS"
printf '%-6s %-30s %s\n' round "engine: read thread conc" "replay (in run order)"

for r in $(seq 1 "$ROUNDS"); do
  eng=$(engine_round)
  # Alternate condition order with round parity. Without this the first
  # condition in the list is always the cold pass and the second always the
  # warm one, which on 3.6 alone is 25-56 against 11.7 ms/step.
  if [ $((r % 2)) -eq 0 ]; then
    c="$CONDS"
  else
    c=$(echo "$CONDS" | awk -F, '{for(i=NF;i>=1;i--) printf "%s%s", $i, (i>1?",":"")}')
  fi
  rep=$(python3 tools/read-sweep/replay_dest.py --only "$INSTALL" \
          --trace36 "$TRACE" --trace38 "$TRACE" "$SLOTARG" "$DEPTH" \
          --rounds 1 --conds "$c" --tag paired 2>/dev/null \
        | awk '/ms\/step/ {for(i=1;i<=NF;i++) if($i=="ms/step") printf "%s ", $(i-1)}')
  printf '%-6s %-30s %s\n' "$r" "$eng" "$rep"
done
