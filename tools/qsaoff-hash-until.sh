#!/bin/bash
#
# qsaoff-hash-until — catch the intermittent selector-off divergence, with the
# row fingerprints on, and stop the moment a pair actually diverges.
#
# WHY THIS EXISTS
#
# The engine has had two long-prefill non-reproducibilities. The first was a
# store bug: `encodeQKPost` was given both a `kRawOffset` binding and a `pos`
# while the kernel adds `pos*idxDim` itself, so every prefill key landed at
# 2*pos — fixed, and the configuration that always diverged now agrees
# bit-for-bit (docs/experiments/summaries/05-attention-and-kv-cache.md, KV-15).
#
# The second is still open, and this script is how it is being chased. It
# lives in the *dense* path — it reproduces with `FQ_QSA_OFF=1`, a runner built
# with no indexer at all, so no store, no pool and no ranking run — and it is
# **intermittent**: five pairs at chunk 512 on this prompt after the fix gave
# one divergence and four exact agreements, and the diverging pair was the
# *fastest* of the five. A single pair is therefore as likely as not to agree
# and say nothing, which is why this loops.
#
# WHAT IT ESTABLISHED SO FAR
#
# On the one diverging pair, the map's first difference is layer 3, stage
# `attn`, row 1024 — one row — with `qkv`, `krot` and `vrot` showing no
# differing row at that layer at all: the attention's output differs on
# bit-identical q/k/v *stage* rows. That is the same landing the first
# (indexer-on) map produced, which the store bug never explained.
#
# HOW TO READ A HIT
#
# Stages 9-11 are the indexer's and stay zero here (stages 12-13 are the K/V
# cache — see `PrefillRowHash.stageNames`, fourteen in all). On a diverging
# pair, look at what the first-differing row shows beside the attention output:
#
#   kcache/vcache rows EQUAL, attention output differs
#       -> the attention kernel, on inputs the cache actually held. The audit
#          belongs in `Attention.encodeFull`'s split path (`encodeSplit`,
#          attention_decode_partial/combine) around the row where the first
#          difference lands.
#   kcache/vcache rows DIFFER
#       -> the copy that fills them, `copyPrefillKVToCache` / `copyPrefillKV`:
#          its ring arithmetic, its offsets, or its ordering against the
#          attention that reads it.
#
# The row to reason about is the *first* one the diff prints, not the layer's
# total: from that row on, a causal pipeline carries the difference.
#
# WHAT IT DOES
#
# Two runs of the same prompt and flags as the pairs recorded in KV-15 —
# chunk 512, 4606 tokens, temperature 0, `--max-context 8192` — with
# `FQ_QSA_OFF=1` (no indexer), `FQ_ROW_HASH` (the fingerprints) and a prefill
# logits dump as the control: if the logits agree, the runs did not diverge and
# the dumps must agree too, so an agreeing pair is evidence about the rate
# rather than about the mechanism. Both runs go through `tools/memguard.sh`,
# which is the rule on this box.
#
# Usage:
#   tools/qsaoff-hash-until.sh [maxPairs]     # default 6; each pair ~11 min
#
# Output: the per-pair verdicts, then on the first divergence the full stage
# map and the interpretation above, via tools/row-hash-diff.py. Exit 0 when a
# pair diverged (and the map is on stdout), 1 when every pair agreed — which
# means the rate is lower than one in five, not that the bug is gone.
set -u

MAX=${1:-6}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2

CLI=".build/arm64-apple-macosx/release/FinchMoECLI"
PROMPT="docs/benchmark-prompts/real-generation-v1/long-matched.json"
MODEL="models/Qwen3.8-Flash-Next-125B-ple4bit.finch"
OUT=/tmp/qsaoff_hash_until_last.txt

[ -x "$CLI" ] || { echo "ABORT: build the release CLI first (swift build -c release)"; exit 2; }
[ -f "$PROMPT" ] || { echo "ABORT: $PROMPT is missing"; exit 2; }
[ -f "$MODEL/manifest.json" ] || { echo "ABORT: no manifest under $MODEL"; exit 2; }

for pair in $(seq 1 "$MAX"); do
    echo "=== pair $pair of $MAX  $(date '+%H:%M:%S') ==="
    for i in a b; do
        rm -f "/tmp/qor_$i.bin" "/tmp/qor_${i}_logits.bin"
        FQ_QSA_OFF=1 \
        FQ_ROW_HASH="/tmp/qor_$i.bin" \
        FQ_DUMP_PREFILL_LOGITS="/tmp/qor_${i}_logits.bin" \
          tools/memguard.sh --max-compressed 7 -- "$CLI" \
          --model "$MODEL" --messages-file "$PROMPT" \
          --temperature 0 --max-new 4 --max-context 8192 \
          --prefill-chunk-tokens 512 > "/tmp/qor_$i.log" 2>&1
        printf '  run %s: exit=%s  %s\n' "$i" "$?" \
          "$(grep -oE 'prefill=[0-9.]+s \([0-9.]+tok/s\)' "/tmp/qor_$i.log" | tail -1)"
    done

    verdict=$(python3 - <<'PY'
import numpy as np
a = np.fromfile('/tmp/qor_a_logits.bin', dtype='<f4')
b = np.fromfile('/tmp/qor_b_logits.bin', dtype='<f4')
if a.size and a.size == b.size:
    n = int((a != b).sum())
    print(f"{n}/{a.size} " + ("DIVERGED" if n else "identical"))
else:
    print(f"sizes {a.size} vs {b.size} MISSING")
PY
)
    echo "  control (logits): $verdict"
    case "$verdict" in
        *DIVERGED*) : ;;
        *) echo "  pair agreed — the mechanism did not fire; going again"; continue ;;
    esac

    echo
    echo "=== diverged on pair $pair — what moved, and which side that implicates ==="
    python3 tools/row-hash-diff.py /tmp/qor_a.bin /tmp/qor_b.bin | tee "$OUT"
    echo
    echo "Full diff kept in $OUT (the dumps are in /tmp/qor_{a,b}.bin)."
    exit 0
done

echo "=== $MAX pairs, none diverged — the rate is below one in five; raise maxPairs ==="
exit 1
