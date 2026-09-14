#!/bin/bash
#
# Add the GPU to the replay, to price the one cell IO-20 left open.
#
# The engine cannot answer this about itself: `FQ_EXPERT_TRACE` is capture-only,
# so nop'ing the routed readback changes the token stream and with it the reads,
# and the arm stops being comparable to the 2.10 ms baseline. So the experiment
# is inverted. The offline replay already holds the engine's ring
# (`makeBuffer(bytesNoCopy:)`, same posix_memalign) and issues the engine's own
# preads, and has no GPU at all -- which is exactly the missing ingredient. Give
# it one, at the engine's measured dose, and the trace is fixed by construction.
#
# The dose is per-step bytes and burst length, not a rate, because 3.8's 16.7
# GB/s is compute-bound (648.8 MB to read, 38.8 ms spent reading it), not a
# bandwidth ceiling. Matching bytes while reading them 5x faster would hit the
# memory system 5x harder over a 5x shorter window and test a dose no engine run
# produces.
#
#   install  burst MB   burst GB/s   period ms   why
#   3.6      227.5      11.9         57          227.5 MB / 19.1 ms routed phase
#   3.8      648.8      16.7         147         648.8 MB / 38.8 ms routed phase
#
# Period is the replay's own pass, so one burst lands per pass -- one routed
# phase per engine step, which is the engine's structure.
#
# Prediction under the contention hypothesis: the 3.8 loaded arm's p50 moves
# from ~1.05 back toward the engine's 2.10, and the 3.6 arm moves less. A null
# on both closes the cell. Note the prediction is 1.05, not 0.52: 0.52 is 3.6's
# p50, and 3.8's replay under this condition measured 1.05 in four rounds.
#
# usage: gpu-contention.sh <install> [rounds] [extra replay args...]
set -u
cd "/Volumes/samsung 2t/code/finchMoE"

INSTALL=${1:?usage: gpu-contention.sh <install> [rounds] [replay args]}
ROUNDS=${2:-3}
shift $(( $# > 1 ? 2 : 1 ))

case "$INSTALL" in
    3.6) D_GIB=1.05; D_BURST=227.5; D_GBPS=11.9; D_PERIOD=57 ;;
    3.8) D_GIB=1.98; D_BURST=${BURST38:-648.8}; D_GBPS=${GBP38:-16.7}; D_PERIOD=147 ;;
    *)   echo "unknown install $INSTALL" >&2; exit 2 ;;
esac

# LOAD_PROG lets the same interleaved design run the CPU control (mem_load),
# which reads the same bytes in the same burst shape without a GPU. Without it
# a positive result cannot be attributed to the GPU rather than to DRAM traffic.
LOAD_PROG=${LOAD_PROG:-gpu_load}
LOAD=${LOAD_PROG:+tools/read-sweep/$LOAD_PROG}
[ -x "$LOAD" ] || { echo "build $LOAD_PROG first" >&2; exit 2; }

# The flag set is per-program, not shared. mem_load has no --gbps (its rate is
# whatever the CPUs deliver, not a throttle) and no --quiet, and passing it
# gpu_load's DOSE made it exit 2 on the *first* argument -- so the "loaded" arm
# ran with nothing in it and reported a clean null. That is the failure this
# case statement exists to prevent: an instrument that refuses to start and an
# instrument that changes nothing are the same reading otherwise.
case "$LOAD_PROG" in
    gpu_load) DOSE="--gib $D_GIB --burst-mb $D_BURST --gbps $D_GBPS \
--period-ms $D_PERIOD ${LOAD_EXTRA:-} --quiet" ;;
    mem_load) DOSE="--gib $D_GIB --burst-mb $D_BURST --period-ms $D_PERIOD \
--threads 4 ${LOAD_EXTRA:-}" ;;
    *)        DOSE="--gib $D_GIB --burst-mb $D_BURST --period-ms $D_PERIOD \
${LOAD_EXTRA:-}" ;;
esac

# Condition 3 only: "allowed, slot dest" is the engine's own shape -- the ring
# allocated and the pread landing in it -- which is what IO-20's 1.05 ms p50 was
# measured under. The other conditions answer different questions.
REPLAY=(python3 tools/read-sweep/replay_dest.py --only "$INSTALL" --conds 3
        --rounds 1 --hist --evict-gib 17 --tag contention "$@")

# The span line comes first and the per-round `reads` line second; `tail` would
# take the summary's restatement of the same percentiles and drop the span,
# which is the one field that says whether the *drive* moved or only the
# process's accounting of it.
line() { grep -E "span +[0-9.]+ ms/step|^      reads p50" | head -2 | tr '\n' ' '; }

printf '# gpu-contention  install=%s  rounds=%s  dose: %s\n' "$INSTALL" "$ROUNDS" "$DOSE"
printf '%-6s %-8s %s\n' round arm "p50..n / span"

for r in $(seq 1 "$ROUNDS"); do
  # Alternate which arm goes first. The drive drifts within a session by more
  # than most effects measured here (METH-15), so an arm that is always second
  # is an arm that is always measured against a different drive state.
  if [ $((r % 2)) -eq 0 ]; then arms="alone loaded"; else arms="loaded alone"; fi

  for a in $arms; do
    if [ "$a" = loaded ]; then
      "$LOAD" $DOSE --seconds 600 >/dev/null 2>&1 &
      LOADPID=$!
      sleep 2                      # let the ring fault in before the replay starts
      # Fail loudly instead of measuring an empty arm. A loader that died on
      # its own arguments leaves an arm holding a plain unloaded run, and
      # nothing downstream can tell that reading from a real null.
      if ! kill -0 "$LOADPID" 2>/dev/null; then
        echo "!! $LOAD_PROG exited during startup -- this arm would be a false" \
             "null, so the run is stopping. Flags were: $DOSE" >&2
        exit 2
      fi
    fi
    out=$("${REPLAY[@]}" 2>&1 | line)
    if [ "$a" = loaded ]; then
      kill "$LOADPID" 2>/dev/null
      wait "$LOADPID" 2>/dev/null
      # The pilot for IO-21 read 2.10 ms in an `alone` arm that ran immediately
      # after a kill -- the load still winding down, since killing the process
      # does not drain work already queued on the GPU. That is the failure this
      # sleep exists to prevent: it is short enough to cost nothing and long
      # enough that an arm which follows a kill is not measuring the last one.
      sleep 3
    fi
    printf '%-6s %-8s %s\n' "$r" "$a" "$out"
  done
done
