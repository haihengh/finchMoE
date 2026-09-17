#!/bin/bash
#
# heavy-tests — run the whole test suite in a shape this 16 GB box can finish.
#
# Why this exists, and why it looks like this:
#
# `swift test --no-parallel` runs every test in ONE process. That works fine
# for 854 of the 866 tests (~72 s, 1.5 GB compressor peak under the default
# memguard ceiling). It does not work for the twelve in QwenLayer0DebugTests,
# which each load a real install, build fp32 references and sweep layers.
#
# Measured, one test per process, under memguard's DEFAULT 4 GB ceiling:
#
#     kernelsMatchReferenceOnRealWeights         1.5 GB
#     layer0PostAttnMatchesFp32Reference         1.4 GB
#     allLayerMixerSweep                         1.4 GB
#     prefillSweepFindsFirstDivergence           1.4 GB
#     prefillLayerTailsMatchReference            0.9 GB
#     repackWeightsMatchBf16Checkpoint           2.6 GB (see the note in phase 2:
#                                                      measured up to 5.3 GB since)
#     ... all twelve pass
#
# but all twelve in one process reached 8.8 GB and the guard killed it — and
# raising the ceiling only moved the kill later, because the peak tracks the
# ceiling: each test's working set is retained as compressed pages (free
# memory stays ~78%, so macOS never has a reason to drain them) and the next
# test allocates on top. It is cumulative, not a leak: the suite is a struct
# with no stored properties, MetalContext holds only a per-instance pipeline
# cache, and every test passes alone under the default guard.
#
# So the fix is not a bigger ceiling — it is recycling the process between the
# heavy tests. Two phases, both under memguard's defaults:
#
#   1. everything except QwenLayer0DebugTests, in one process
#   2. each heavy test in its own process
#
# Usage:
#   tools/heavy-tests.sh              # both phases
#   tools/heavy-tests.sh --part-a     # the 854 tests only
#   tools/heavy-tests.sh --part-b     # the twelve heavy tests only
#
# Exit codes: 0 all green; 1 something failed; 2 refused at memguard's gate.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

SUITE=QwenLayer0DebugTests
SRC=Tests/FinchMoE/Core/Infrastructure/ModelIO/${SUITE}.swift
GUARD=tools/memguard.sh

# Read the test names out of the source so the list cannot drift from it.
# (No `mapfile`: this runs on macOS's bash 3.2.)
TESTS=()
while IFS= read -r _t; do
    [ -n "$_t" ] && TESTS+=("$_t")
done < <(grep -A2 '^    @Test' "$SRC" | grep -oE 'func [a-zA-Z0-9_]+' | sed 's/func //')
if [ "${#TESTS[@]}" -eq 0 ]; then
    echo "heavy-tests: no tests found in $SRC — did the declaration style change?" >&2
    exit 1
fi

PART_A=1 PART_B=1
case "${1:-}" in
    --part-a) PART_B=0 ;;
    --part-b) PART_A=0 ;;
    "") ;;
    *) echo "usage: $0 [--part-a|--part-b]" >&2; exit 2 ;;
esac

rc_all=0

if [ "$PART_A" -eq 1 ]; then
    echo "=== phase 1: everything except $SUITE ==="
    "$GUARD" -- swift test --no-parallel --skip "$SUITE" > /tmp/heavytests_a.log 2>&1
    rc=$?
    grep -E "Test run with .* (passed|failed)" /tmp/heavytests_a.log | tail -1
    grep -E "memguard:" /tmp/heavytests_a.log | tail -1
    if [ $rc -ne 0 ]; then
        echo "phase 1 FAILED (exit $rc) — see /tmp/heavytests_a.log"
        tail -20 /tmp/heavytests_a.log
        rc_all=1
    fi
fi

if [ "$PART_B" -eq 1 ]; then
    echo
    echo "=== phase 2: $SUITE, one test per guarded process (${#TESTS[@]} tests) ==="
    pass=0; fail=0
    for t in "${TESTS[@]}"; do
        # Most of these fit the default guard. One does not any more:
        # repackWeightsMatchBf16Checkpoint has been measured at 2.6 GB (the
        # header's number), 3.7 GB and 5.3 GB on the same box, and it is the
        # test's own working set plus whatever the compressor is already holding
        # from other activity — on a quiet box it lands under 4 GB, after hours
        # of prefill runs it does not. It gets a ceiling of its own so the
        # verdict tracks the test rather than the box's mood.
        #
        # This is NOT the cumulative case the header warns about, where a bigger
        # ceiling only moves the kill later because each test's pages stay
        # compressed on top of the last one's. Phase 2 recycles the process per
        # test precisely to keep that from happening; this is one test, alone,
        # whose peak has grown past a ceiling written for it long ago.
        guard_args=()
        case "$t" in
            repackWeightsMatchBf16Checkpoint) guard_args=(--max-compressed 6) ;;
        esac
        "$GUARD" ${guard_args[@]+"${guard_args[@]}"} -- \
            swift test --no-parallel --filter "$SUITE/$t" > "/tmp/heavytests_$t.log" 2>&1
        rc=$?
        peak=$(grep -oE "peak [0-9.]+ GB compressed" "/tmp/heavytests_$t.log" | tail -1 | sed 's/peak //;s/ GB compressed//')
        secs=$(grep -oE "passed after [0-9.]+ seconds" "/tmp/heavytests_$t.log" | tail -1 | sed 's/passed after //;s/ seconds//')
        if [ $rc -eq 0 ]; then
            verdict=PASS; pass=$((pass + 1))
        else
            verdict="FAIL(exit=$rc)"; fail=$((fail + 1)); rc_all=1
        fi
        printf "  %-42s %-12s peak=%-5s %ss\n" "$t" "$verdict" "$peak" "$secs"
    done
    echo "  ---"
    echo "  $pass passed, $fail failed"
    [ $fail -gt 0 ] && echo "  failing logs: /tmp/heavytests_<name>.log"
fi

exit $rc_all
