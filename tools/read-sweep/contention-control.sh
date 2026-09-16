#!/bin/bash
#
# contention-control.sh — is the drive penalty specific to the GPU, and how much
# of the CPU arm's cost is thread count rather than memory traffic?
#
# IO-21 established that adding the GPU to the replay costs the drive ~+15 ms/step,
# and that a CPU loader costs roughly half that — but it recorded the CPU half as
# *not a matched control* for two reasons, and this script removes both:
#
#   1. `mem_load` burned four CPU cores where `gpu_load` runs at ~1.1%, so the
#      arms differed in CPU occupancy as well as in traffic. Here the CPU arm
#      runs at two thread counts, and a third arm burns four cores with no
#      memory traffic at all (`yes`), which is the discriminator: if that also
#      costs the drive, the price is thread/scheduler pressure, not bytes moved.
#   2. The two halves were measured in different sessions, and this box's drive
#      drifts between sessions by more than the effect (alone arms 184-194
#      ms/step in the CPU session against 167-173 in the GPU one). Here every
#      arm alternates inside one session, so all of them see the same states.
#
# The statistic is `thread_mean`, not p50: IO-21 found the replay's per-read
# distribution is bimodal with the median sitting in a 4.1%-full valley, so a
# p50 here is a knife-edge. `span` is reported beside it because it says whether
# the *drive* moved or only the process's accounting of it.
#
# usage: contention-control.sh <install> [rounds]
set -u
cd "/Volumes/samsung 2t/code/finchMoE"

INSTALL=${1:?usage: contention-control.sh <install> [rounds]}
ROUNDS=${2:-3}

case "$INSTALL" in
    3.6) D_GIB=1.05; D_BURST=227.5; D_GBPS=11.9; D_PERIOD=57 ;;
    3.8) D_GIB=1.98; D_BURST=648.8; D_GBPS=16.7; D_PERIOD=147 ;;
    *)   echo "unknown install $INSTALL" >&2; exit 2 ;;
esac

GPU_LOAD=tools/read-sweep/gpu_load
CPU_LOAD=tools/read-sweep/mem_load
for p in "$GPU_LOAD" "$CPU_LOAD"; do
    [ -x "$p" ] || { echo "build $p first" >&2; exit 2; }
done

REPLAY=(python3 tools/read-sweep/replay_dest.py --only "$INSTALL" --conds 3
        --rounds 1 --hist --evict-gib 17 --tag contention-control)

# The flag set is per-program. mem_load has no --gbps (its rate is what the CPUs
# deliver, not a throttle) and no --quiet; handing it gpu_load's flags is the
# mistake that made IO-21's first CPU arms run empty, so it is spelled out.
loader_cmd() {
    case "$1" in
        gpu)  echo "$GPU_LOAD --gib $D_GIB --burst-mb $D_BURST --gbps $D_GBPS --period-ms $D_PERIOD --quiet" ;;
        cpu4) echo "$CPU_LOAD --gib $D_GIB --burst-mb $D_BURST --period-ms $D_PERIOD --threads 4" ;;
        cpu1) echo "$CPU_LOAD --gib $D_GIB --burst-mb $D_BURST --period-ms $D_PERIOD --threads 1" ;;
    esac
}

ARMS="alone gpu cpu4 cpu1 burn"
RESULTS=$(mktemp)
trap 'rm -f "$RESULTS"' EXIT

start_arm() {   # $1 = arm; sets LOADPIDS
    LOADPIDS=""
    case "$1" in
        alone) ;;
        burn)  for _ in 1 2 3 4; do yes > /dev/null & LOADPIDS="$LOADPIDS $!"; done ;;
        *)     $(loader_cmd "$1") --seconds 900 >/dev/null 2>&1 &
               LOADPIDS="$!" ;;
    esac
    [ -z "$LOADPIDS" ] && return 0
    sleep 2                       # reach steady state before the replay starts
    for p in $LOADPIDS; do
        if ! kill -0 "$p" 2>/dev/null; then
            echo "!! arm '$1' exited during startup — that arm would be a false" \
                 "null, so the run is stopping." >&2
            exit 2
        fi
    done
}

stop_arm() {
    for p in ${LOADPIDS:-}; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
    LOADPIDS=""
    sleep 2                       # a killed loader still winds down; an arm that
                                  # starts into the tail measures the previous one
}

printf '# contention-control  install=%s  rounds=%s  dose: %s MB per %s ms (%s GB/s)\n' \
       "$INSTALL" "$ROUNDS" "$D_BURST" "$D_PERIOD" "$D_GBPS"
printf '%-6s %-6s %14s %16s\n' round arm span_ms_per_step thread_mean_ms

for r in $(seq 1 "$ROUNDS"); do
    # Rotate the starting arm each round: the drive drifts within a session, so
    # an arm always in the same slot is measured against a different drive state.
    # (`off` is captured first because `set --` overwrites the positional params,
    # including the argument this was called with.)
    off=$((r - 1))
    set -- $ARMS
    n=$#
    order=""
    i=0
    while [ $i -lt $n ]; do
        j=$(( (i + off) % n ))
        eval "order=\"$order \${$((j + 1))}\""
        i=$((i + 1))
    done
    for a in $order; do
        start_arm "$a"
        out=$("${REPLAY[@]}" 2>&1)
        span=$(printf '%s\n' "$out" | grep -oE "span +[0-9.]+ ms/step" | grep -oE "[0-9.]+" | head -1)
        tmean=$(printf '%s\n' "$out" | grep -oE "thread_mean=[0-9.]+" | head -1 | cut -d= -f2)
        stop_arm
        printf '%-6s %-6s %14s %16s\n' "$r" "$a" "${span:-?}" "${tmean:-?}"
        echo "$a ${span:-0} ${tmean:-0}" >> "$RESULTS"
    done
done

echo
echo "# per-arm means over $ROUNDS rounds (and the delta against this session's alone)"
python3 - "$RESULTS" <<'PY'
import sys, collections
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    arm, span, tmean = line.split()
    rows[arm].append((float(span), float(tmean)))
base_span = sum(s for s, _ in rows.get('alone', [(0, 0)])) / max(len(rows.get('alone', [1])), 1)
base_tm = sum(t for _, t in rows.get('alone', [(0, 0)])) / max(len(rows.get('alone', [1])), 1)
print(f"{'arm':<7}{'span mean':>11}{'vs alone':>10}{'thread_mean':>13}{'vs alone':>10}  n")
for arm in ('alone', 'gpu', 'cpu4', 'cpu1', 'burn'):
    if arm not in rows:
        continue
    v = rows[arm]
    s = sum(x for x, _ in v) / len(v)
    t = sum(y for _, y in v) / len(v)
    ds = f"{s - base_span:+.2f}" if arm != 'alone' else '-'
    dt = f"{t - base_tm:+.2f}" if arm != 'alone' else '-'
    spread = f"  span {min(x for x,_ in v):.1f}-{max(x for x,_ in v):.1f}"
    print(f"{arm:<7}{s:>11.2f}{ds:>10}{t:>13.2f}{dt:>10}  {len(v)}{spread}")
PY
