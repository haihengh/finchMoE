#!/usr/bin/env python3
"""Price a static per-layer hot set out of sample, across prompt domains.

The archived engine's hot-set prefetch (8c9b496) built its table from a
four-prompt corpus and reported 81% coverage corpus-internal, but 26.2% of
unique experts / 39.5% of per-token requests in flight. That gap is domain
shift, and it is the whole question: a hot set is only worth holding resident
if its coverage survives being built on prompts it was not built from.

So this is leave-one-domain-out. For each held-out prompt, the table is built
from the *other* prompts' traces and scored against the held-out one. The
in-sample column (build and score on the same prompt) is the ceiling that a
past measurement would have quoted.

Scoring is against the trace's own reads, which are the engine's *misses*
(`RealForwardRunner.swift:830` -- `[layer, missCount, expert...]`). That is the
number that matters, because the 16-slot LRU ring already serves 49-58% of the
routed experts; a pinned set is additional capacity and is only credited with
the reads the ring did not already avoid.

Usage: hot_set_transfer.py [--dir /tmp/fq-corpus] [--sizes 16,32,64]
"""
import argparse
import collections
import glob
import os
import re
import sys

SIZES = (16, 32, 64)


def parse(path):
    """`[layer, missCount, expert...]` repeated -> [(layer, [expert, ...]), ...]."""
    raw = [int(x) for x in open(path)]
    out, i = [], 0
    while i < len(raw):
        c = raw[i + 1]
        out.append((raw[i], raw[i + 2:i + 2 + c]))
        i += 2 + c
    return out


def coverage(train, test, n):
    """Fraction of `test`'s reads served by the top-`n` per layer of `train`."""
    freq = collections.defaultdict(collections.Counter)
    for layer, experts in train:
        freq[layer].update(experts)
    top = {L: {e for e, _ in c.most_common(n)} for L, c in freq.items()}
    hit = sum(1 for L, ex in test if L in top for e in ex if e in top[L])
    total = sum(len(ex) for _, ex in test)
    return hit, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/fq-corpus")
    ap.add_argument("--sizes", default="16,32,64")
    args = ap.parse_args()
    sizes = tuple(int(s) for s in args.sizes.split(","))

    traces = {}
    for path in sorted(glob.glob(os.path.join(args.dir, "trace-*.txt"))):
        m = re.match(r"trace-(.+)-(36|38)\.txt$", os.path.basename(path))
        if m:
            traces.setdefault(m.group(2), {})[m.group(1)] = parse(path)
    if not traces:
        sys.exit(f"no traces in {args.dir} -- run capture-corpus-traces.sh first")

    width = max(20, max(len(k) for t in traces.values() for k in t))
    verdicts = []
    for model in sorted(traces):
        by_prompt = traces[model]
        keys = sorted(by_prompt)
        if len(keys) < 2:
            print(f"--- {model}: only {len(keys)} prompt(s), no fold possible")
            continue
        n_reads = sum(len(ex) for _, ex in by_prompt[keys[0]])
        print(f"--- Qwen 3.{model}  ({len(keys)} prompts, {n_reads} reads in the first)")
        head = "held out".ljust(width) + "".join(f"  top-{n:<4}" for n in sizes)
        print(head)
        fold = {n: [] for n in sizes}
        for held in keys:
            train = [t for k, t in by_prompt.items() if k != held]
            flat = [b for t in train for b in t]
            row = held.ljust(width)
            for n in sizes:
                hit, total = coverage(flat, by_prompt[held], n)
                pct = 100.0 * hit / total
                fold[n].append(pct)
                row += f"  {pct:6.1f}%"
            print(row)
        means = {n: sum(v) / len(v) for n, v in fold.items()}
        print("MEAN out-of-sample".ljust(width) + "".join(f"  {means[n]:6.1f}%" for n in sizes))

        # Ceiling: build and score on the same prompt. This is the number a
        # same-prompt measurement would report, and the gap to MEAN is the
        # domain shift the archived 81% -> 26.2%/39.5% collapse was made of.
        ins = {}
        for n in sizes:
            tot_hit = tot = 0
            for k in keys:
                h, t = coverage(by_prompt[k], by_prompt[k], n)
                tot_hit += h
                tot += t
            ins[n] = 100.0 * tot_hit / tot
        print("CEILING in-sample".ljust(width) + "".join(f"  {ins[n]:6.1f}%" for n in sizes))
        spread = {n: max(fold[n]) - min(fold[n]) for n in sizes}
        print("fold spread (max-min)".ljust(width) + "".join(f"  {spread[n]:6.1f}pp" for n in sizes))
        print()
        verdicts.append((model, means, ins))

    print("Decision rule (top-32 out-of-sample mean):")
    for model, means, ins in verdicts:
        m32 = means[32]
        if m32 >= 40:
            call = ">=40% -> build static hot-set pinning"
        elif m32 < 25:
            call = "<25%  -> fall through to the Step 2 CB1 overlap probe"
        else:
            call = "25-40% -> between the bars; size a hybrid, keep the LRU ring for the rest"
        print(f"  3.{model}: {m32:5.1f}% out-of-sample (ceiling {ins[32]:.1f}%)  {call}")


if __name__ == "__main__":
    main()
