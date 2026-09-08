#!/usr/bin/env python3
"""build_hot_sets.py — build per-layer static expert hot sets from routing logs.

Input: one or more routing log files produced by finchmoe-infer
       --collect-routing F (record: int32 layer, int32 K, float[2048] hidden,
       int32[K] top-K indices, int32[24] top-24 indices).
Output: hot_sets.bin — [40][64] int32, the 64 most-frequently-selected
        experts per layer across the corpus, in descending frequency order.
        Used by the prefill expert prefetcher (layer L+1's hot set is
        prefetched into the prefetch pool while layer L computes).

Usage: python3 build_hot_sets.py [--top N] [--out hot_sets.bin] log1.bin [log2.bin ...]
"""
import argparse
import struct
import sys
from collections import Counter

HIDDEN_DIM = 2048
NUM_LAYERS = 40


def read_log(path):
    """Yield (layer, topk_indices) per routing record."""
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            layer, k = struct.unpack('<ii', hdr)
            f.seek(HIDDEN_DIM * 4, 1)  # skip hidden
            idx = struct.unpack(f'<{k}i', f.read(k * 4))
            f.read(24 * 4)  # skip extended top-24
            yield layer, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=64, help='hot-set size per layer')
    parser.add_argument('--out', default='hot_sets.bin')
    parser.add_argument('logs', nargs='+')
    args = parser.parse_args()

    counts = [Counter() for _ in range(NUM_LAYERS)]
    total = 0
    for path in args.logs:
        for layer, idx in read_log(path):
            if 0 <= layer < NUM_LAYERS:
                counts[layer].update(idx)
                total += 1
    print(f"records: {total} across {len(args.logs)} logs")

    with open(args.out, 'wb') as f:
        for layer in range(NUM_LAYERS):
            hot = [e for e, _ in counts[layer].most_common(args.top)]
            if len(hot) < args.top:
                print(f"warning: layer {layer} has only {len(hot)} distinct experts")
                hot += [0] * (args.top - len(hot))
            f.write(struct.pack(f'<{args.top}i', *hot))

    # Coverage report (corpus-internal)
    cov = []
    for layer in range(NUM_LAYERS):
        hot = {e for e, _ in counts[layer].most_common(args.top)}
        hits = sum(c for e, c in counts[layer].items() if e in hot)
        tot = sum(counts[layer].values())
        cov.append(hits / tot if tot else 0.0)
    print(f"hot-{args.top} coverage: mean {sum(cov)/NUM_LAYERS:.3f}, "
          f"min layer {min(cov):.3f} (corpus-internal)")
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
