#!/bin/bash
#
# memguard — run a heavy build/test under this machine's 16 GB protocol.
#
# The panics on this box (2026-09-05 ×2, 2026-09-10) were all the same
# signature: `watchdog timeout: no checkins from watchdogd in ~90 seconds`.
# That is a system-wide hang from memory/IO thrash, not a process crash — the
# box goes so deep into compression and paging that watchdogd stops checking
# in and the kernel force-restarts. Nothing gets a crash report, so a build
# that allocates past physical memory does not fail; it takes the machine
# down with it and leaves a stale `.build/.lock` behind.
#
# This wrapper turns the prose protocol in docs/QWEN36_PORT.md ("gate heavy
# runs on memory_pressure -Q >= ~60% free, watch RSS, abort if swapping") into
# something mechanical, in two parts:
#
#   1. It refuses to start a command while the box is already under pressure
#      (default: below 60% free), before the command has allocated anything.
#   2. It samples while the command runs and kills it — process group and
#      all — if free memory collapses or the compressor balloons, which is
#      the state the kernel would otherwise panic in. Better to lose a build
#      than the machine: the build is resumable, and a panic costs a reboot,
#      a Spotlight reindex of the external volume, and the session.
#
# Usage:
#   tools/memguard.sh swift build -j 3
#   tools/memguard.sh swift test -c debug --no-parallel
#   tools/memguard.sh --floor 20 --max-compressed 3 swift build -j 3
#
# Exit codes: 0 child's own status; 2 refused at the gate; 3 killed mid-run.
# Environment overrides: MEMGUARD_MIN_FREE_BEFORE, MEMGUARD_FLOOR,
# MEMGUARD_MAX_COMPRESSED_GB, MEMGUARD_MAX_SWAP_MB, MEMGUARD_INTERVAL.

set -uo pipefail

MIN_FREE_BEFORE=${MEMGUARD_MIN_FREE_BEFORE:-60}
FLOOR=${MEMGUARD_FLOOR:-12}
MAX_COMPRESSED_GB=${MEMGUARD_MAX_COMPRESSED_GB:-4}
MAX_SWAP_MB=${MEMGUARD_MAX_SWAP_MB:-2048}
INTERVAL=${MEMGUARD_INTERVAL:-5}

while [ $# -gt 0 ]; do
    case "$1" in
        --floor)              FLOOR=$2; shift 2 ;;
        --min-free)           MIN_FREE_BEFORE=$2; shift 2 ;;
        --max-compressed)     MAX_COMPRESSED_GB=$2; shift 2 ;;
        --interval)           INTERVAL=$2; shift 2 ;;
        --)                   shift; break ;;
        -*)                   echo "memguard: unknown option $1" >&2; exit 2 ;;
        *)                    break ;;
    esac
done
[ $# -gt 0 ] || { echo "usage: memguard.sh [options] -- command [args...]" >&2; exit 2; }

# `memory_pressure -Q` is the documented gate: one line, one percentage.
free_pct() {
    memory_pressure -Q 2>/dev/null \
        | awk -F': *' '/free percentage/ { gsub(/%/, "", $2); print int($2) }'
}

# The compressor is the leading indicator. Free percentage can still read
# healthy while the kernel is compressing gigabytes it cannot evict, and that
# is the state the panic notes call out ("compressor segments 100% (BAD)").
compressed_gb() {
    # `exit` on the first match matters: vm_stat also reports "Pages stored in
    # compressor" (the uncompressed size), which is a different number.
    vm_stat 2>/dev/null | awk '
        /Pages occupied by compressor/ { gsub(/\./, "", $5); printf "%.1f", $5 * 16384 / 1e9; exit }
        /Pages used by compressor/     { gsub(/\./, "", $5); printf "%.1f", $5 * 16384 / 1e9; exit }'
}

swap_used_mb() {
    sysctl -n vm.swapusage 2>/dev/null \
        | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | cut -d. -f1
}

gate_free=$(free_pct)
gate_comp=$(compressed_gb)
if [ -z "$gate_free" ]; then
    echo "memguard: cannot read memory_pressure — running without the gate" >&2
elif [ "$gate_free" -lt "$MIN_FREE_BEFORE" ]; then
    echo "memguard: REFUSING — ${gate_free}% free, need ${MIN_FREE_BEFORE}%." >&2
    echo "memguard: compressor holds ${gate_comp} GB. Close VS Code / Chrome / simulators," >&2
    echo "memguard: wait a few minutes, or lower the gate with --min-free N." >&2
    exit 2
fi

echo "memguard: starting at ${gate_free}% free, ${gate_comp} GB compressed — $*"

# `set -m` gives the job its own process group, so the kill below reaches the
# whole tree (swift build's frontend and Metal-compiler children included)
# rather than just the driver.
set -m
"$@" &
child=$!
set +m

killed=0
peak_comp=$gate_comp
min_free=$gate_free

while kill -0 "$child" 2>/dev/null; do
    sleep "$INTERVAL"
    kill -0 "$child" 2>/dev/null || break
    f=$(free_pct); c=$(compressed_gb); s=$(swap_used_mb)
    [ -n "$f" ] && [ "$f" -lt "$min_free" ] && min_free=$f
    [ -n "$c" ] && awk -v a="$c" -v b="$peak_comp" 'BEGIN { exit !(a > b) }' && peak_comp=$c

    reason=""
    if [ -n "$f" ] && [ "$f" -lt "$FLOOR" ]; then
        reason="free memory collapsed to ${f}%"
    elif [ -n "$c" ] && awk -v c="$c" -v m="$MAX_COMPRESSED_GB" 'BEGIN { exit !(c > m) }'; then
        reason="compressor reached ${c} GB"
    elif [ -n "$s" ] && [ "$s" -gt "$MAX_SWAP_MB" ]; then
        reason="swap in use: ${s} MB"
    fi

    if [ -n "$reason" ]; then
        echo "memguard: KILLING — ${reason} (was ${gate_free}% free at start)" >&2
        kill -TERM -- "-$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null
        for _ in $(seq 12); do
            kill -0 "$child" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$child" 2>/dev/null || true
        killed=1
        break
    fi
done

wait "$child"; status=$?

if [ "$killed" = 1 ]; then
    echo "memguard: killed the command before the kernel had to." >&2
    # The kernel is not the only thing that leaves debris behind.
    if [ -f .build/.lock ]; then
        echo "memguard: note — .build/.lock is now stale; the next build may need it removed." >&2
    fi
    exit 3
fi

echo "memguard: done (status ${status}); low-water ${min_free}% free, peak ${peak_comp} GB compressed"
exit "$status"
