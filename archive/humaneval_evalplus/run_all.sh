#!/bin/bash
# run_all.sh — sequential sweep over the three FinchMoE cells.
#
#   ./run_all.sh              # 3bit, 4bit, gguf  (all 164 problems)
#   ./run_all.sh 3bit 4bit    # a subset of cells
#
# STRICTLY sequential and lock-guarded. Two concurrent model runs on this
# 16 GB machine have caused kernel panics three times (2026-08-15, -08-20);
# one engine at a time is a hard rule, not a preference.
#
# EvalPlus resumes by default, so an interrupted sweep picks up where it left
# off rather than restarting.
set -uo pipefail
cd "$(dirname "$0")"

CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=(3bit 4bit gguf)

LOCK="/tmp/finchmoe_evalplus.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "ABORT: another sweep holds $LOCK — remove it if that is stale" >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

START_TS=$(date +%s)
echo "=== sweep start: ${CELLS[*]} ==="

FAILED=()
for cell in "${CELLS[@]}"; do
    echo
    echo "======================================================================"
    echo "  CELL $cell   ($(date '+%H:%M:%S'))"
    echo "======================================================================"
    if ./run_cell.sh "$cell"; then
        echo "[sweep] $cell OK"
    else
        echo "[sweep] $cell FAILED (rc=$?) — continuing with the next cell" >&2
        FAILED+=("$cell")
    fi
    # let the machine settle between tiers (page cache, thermals)
    sleep 10
done

echo
echo "=== sweep done in $(( ($(date +%s) - START_TS) / 60 )) min ==="
for f in results/*_eval.txt; do
    [ -e "$f" ] || continue
    echo "--- $f"
    grep -E "pass@1|humaneval" "$f" || true
done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED cells: ${FAILED[*]}" >&2
    exit 1
fi
