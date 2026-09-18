#!/usr/bin/env python3
"""Diff two FQ_ROW_HASH dumps and report where the two runs first diverge.

Usage: row-hash-diff.py a.bin b.bin [--rows N]

The dump is `[layer][stage][row]` little-endian UInt64 behind a 16-byte header
of four UInt32s (magic, layerCount, stageCount, rowCount), written by
`RealForwardRunner.writeRowHashDump`; see docs/RUNTIME_CONTROLS.md.

What it reports, and why in that order:

  * The first (layer, stage, row) that differs, scanning layer then stage then
    row. "in" at layer L is the plane on entry to L, which is layer L-1's
    output, so agreement at L and disagreement at L+1 localizes the divergence
    to layer L -- and `attn`/`post` then split that layer into its attention
    block and its routed-expert tail.
  * Per (layer, stage), how many rows differ and whether they form a contiguous
    range. A contiguous run starting at a chunk boundary points at the tiling; a
    scattered or all-rows pattern points at something the whole chunk shares.
  * Which rows of the first affected stage moved: a single row is a per-token
    story, a prefix is an ordering story.

Exit code is 0 when the two runs agree everywhere, 1 when they differ, 2 on a
malformed input -- so it composes with `if` in a sweep script.
"""
import struct
import sys

MAGIC = 0x5248_4831
STAGES = ["in", "attn", "post", "qkv", "idxcells", "core", "oproj", "krot", "vrot"]


def load(path):
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) < 16:
        raise ValueError(f"{path}: shorter than its header")
    magic, layers, stages, rows = struct.unpack_from("<IIII", blob, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic 0x{magic:08x}")
    want = 16 + layers * stages * rows * 8
    if len(blob) != want:
        raise ValueError(f"{path}: {len(blob)} bytes, expected {want} "
                         f"({layers} layers x {stages} stages x {rows} rows)")
    values = struct.unpack_from(f"<{layers * stages * rows}Q", blob, 16)
    return {"layers": layers, "stages": stages, "rows": rows,
            "v": values, "path": path}


def index(d, layer, stage, row):
    return ((layer * d["stages"] + stage) * d["rows"]) + row


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    try:
        a, b = load(args[0]), load(args[1])
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return 2
    if (a["layers"], a["stages"], a["rows"]) != (b["layers"], b["stages"], b["rows"]):
        print(f"error: shapes differ: "
              f"{a['layers']}x{a['stages']}x{a['rows']} vs "
              f"{b['layers']}x{b['stages']}x{b['rows']}")
        return 2

    layers, stages, rows = a["layers"], a["stages"], a["rows"]
    names = STAGES if stages == len(STAGES) else [f"s{i}" for i in range(stages)]

    diffs = []
    for L in range(layers):
        for s in range(stages):
            base = index(a, L, s, 0)
            for r in range(rows):
                if a["v"][base + r] != b["v"][base + r]:
                    diffs.append((L, s, r))
    if not diffs:
        print(f"identical: {layers} layers x {stages} stages x {rows} rows "
              f"({layers * stages * rows} hashes)")
        return 0

    L0, s0, r0 = diffs[0]
    print(f"differ: {len(diffs)} of {layers * stages * rows} hashes "
          f"({100.0 * len(diffs) / (layers * stages * rows):.2f}%)")
    print(f"first: layer={L0} stage={names[s0]} row={r0} (token position {r0})")

    # The stage carries the localization. `in` is the plane on entry, i.e. the
    # previous layer's output, so a first difference there belongs to layer
    # L-1; `attn` and `post` split layer L into its attention block and its
    # routed-expert tail; and 3-6 go inside the attention block, where `qkv` equal
    # with `idxcells` different means the QSA ranking moved and not the
    # projections.
    if s0 == 0:
        where = (f"inside layer {L0 - 1} (its own `post` would have shown it)"
                 if L0 > 0 else "at the first layer's entry")
    elif s0 == 1:
        where = f"inside layer {L0}'s attention block (its input agreed)"
    elif s0 == 2:
        where = f"inside layer {L0}'s routed-expert tail (its attention output agreed)"
    elif s0 in (3, 7, 8):
        which = {3: "queries", 7: "keys", 8: "values"}[s0]
        where = (f"in layer {L0}'s {which} after the RoPE/norm epilogue "
                 f"(the layer's input agreed)")
    elif s0 == 4:
        where = (f"in layer {L0}'s QSA selection — the projections matched and the "
                 f"chosen cells did not, so the ranking moved")
    elif s0 == 5:
        where = (f"in layer {L0}'s attention itself (projections and selection agreed, "
                 f"its output did not)")
    else:
        where = f"in layer {L0}'s output gate or o_proj (the attention agreed)"
    print(f"  -> the divergence is {where}")

    per_layer = {}
    for L, s, r in diffs:
        per_layer.setdefault(L, {}).setdefault(s, []).append(r)
    print(f"\n{'layer':>5}  " + "  ".join(f"{n:>14}" for n in names) + "   first row")
    for L in sorted(per_layer):
        cells = []
        for s in range(stages):
            rs = per_layer[L].get(s)
            cells.append(f"{len(rs):>6}/{rows:<7}" if rs else f"{'-':>14}")
        first = min(r for rs in per_layer[L].values() for r in rs)
        print(f"{L:>5}  " + "  ".join(cells) + f"   {first}")

    for L in sorted(per_layer):
        for s in sorted(per_layer[L]):
            rs = per_layer[L][s]
            contiguous = (rs == list(range(rs[0], rs[0] + len(rs))))
            span = f"{rs[0]}..{rs[-1]}" if len(rs) > 1 else f"{rs[0]}"
            shape = "contiguous" if contiguous else "scattered"
            print(f"  layer {L} {names[s]}: rows {span} ({len(rs)}) {shape}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
