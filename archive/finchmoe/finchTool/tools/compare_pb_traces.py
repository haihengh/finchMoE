#!/usr/bin/env python3
"""compare_pb_traces.py — compare per-(token, layer) Phase-B traces.

Usage: python3 compare_pb_traces.py <pb_ref.bin> <pb_new.bin>

Record format (ref): [pos i32][layer i32][1 i32][h_post 2048][gates 256][seg 1]
                     [gdef_hidden 2048][h_mid 2048]
Record format (new): [cb i32][layer i32][M i32][h_post M*2048][gates M*256][seg M]
                     [moe_hidden M*2048][oproj M*2048][h_mid M*2048][input M*2048]

Locates the first divergent (token, layer) on h_post/gates, then reports the
boundary-field diffs at that record: new.moe_hidden vs ref.gdef_hidden
(= CMD3(L-1) output, layers > 0) and new.h_mid vs ref.h_mid.
"""
import sys
import numpy as np

H = 2048
NE = 256

def parse_ref(path):
    data = np.fromfile(path, dtype=np.float32)
    rec_size = 3 + H + NE + 1 + H + H
    n = data.size // rec_size
    assert data.size % rec_size == 0, f"ref size {data.size} not divisible by {rec_size}"
    recs = {}
    for i in range(n):
        base = i * rec_size
        pos, layer, one = [int(x) for x in data[base:base+3].view(np.int32)]
        assert one == 1
        o = base + 3
        h_post = data[o:o+H]; o += H
        gates  = data[o:o+NE]; o += NE
        seg    = data[o:o+1]; o += 1
        gdef   = data[o:o+H]; o += H
        h_mid  = data[o:o+H]
        recs[(pos, layer)] = dict(h_post=h_post, gates=gates, seg=seg,
                                  moe=gdef, h_mid=h_mid)
    return recs

def parse_new(path):
    data = np.fromfile(path, dtype=np.float32)
    off = 0
    recs = {}
    CONV = 8192
    ZDIM = 4096
    per_m = H + NE + 1 + 4 * H
    while off + 3 <= data.size:
        cb, layer, M = [int(x) for x in data[off:off+3].view(np.int32)]
        off += 3
        body = M * per_m + M * (CONV + ZDIM + 64) + M * 4096 + 2
        if off + body > data.size:
            print(f"WARNING: truncated at off={off}, expected body {body}")
            break
        for m in range(M):
            pos = cb + m
            o = off + m * H
            h_post = data[o:o+H]
            o = off + M * H + m * NE
            gates = data[o:o+NE]
            o = off + M * (H + NE) + m
            seg = data[o:o+1]
            o = off + M * (H + NE + 1)
            moe  = data[o + m*H : o + (m+1)*H]
            opr  = data[o + M*H + m*H : o + M*H + (m+1)*H]
            mid  = data[o + 2*M*H + m*H : o + 2*M*H + (m+1)*H]
            inp  = data[o + 3*M*H + m*H : o + 3*M*H + (m+1)*H]
            q = off + M * per_m
            qkv = data[q + m*CONV : q + (m+1)*CONV]
            z   = data[q + M*CONV + m*ZDIM : q + M*CONV + (m+1)*ZDIM]
            ba  = data[q + M*(CONV+ZDIM) + m*64 : q + M*(CONV+ZDIM) + (m+1)*64]
            oin = data[q + M*(CONV+ZDIM+64) + m*4096 : q + M*(CONV+ZDIM+64) + (m+1)*4096]
            st  = data[q + M*(CONV+ZDIM+64+4096) : q + M*(CONV+ZDIM+64+4096) + 2]
            recs[(pos, layer)] = dict(h_post=h_post, gates=gates, seg=seg,
                                      moe=moe, oproj=opr, h_mid=mid, inp=inp,
                                      qkv=qkv, z=z, ba=ba, state=st, oproj_in=oin)
        off += body
    return recs

def main():
    ref = parse_ref(sys.argv[1])
    new = parse_new(sys.argv[2])
    keys = sorted(set(ref) | set(new))
    print(f"ref records: {len(ref)}, new records: {len(new)}")
    missing = [k for k in keys if k not in ref or k not in new]
    if missing:
        print(f"MISSING KEYS (first 10): {missing[:10]}")

    ndiff = 0
    first = None
    for k in keys:
        if k not in ref or k not in new:
            continue
        dh = np.abs(ref[k]['h_post'] - new[k]['h_post']).max()
        dg = np.abs(ref[k]['gates'] - new[k]['gates']).max()
        if dh > 0 or dg > 0:
            ndiff += 1
            if first is None:
                first = k
    if first is None:
        print("ALL RECORDS BITWISE IDENTICAL")
        return
    t, L = first
    r, n = ref[first], new[first]
    print(f"records diverging: {ndiff}")
    print(f"FIRST DIVERGENT: token={t} layer={L}")
    dh = np.abs(r['h_post'] - n['h_post'])
    dg = np.abs(r['gates'] - n['gates'])
    print(f"  h_post: {np.count_nonzero(dh)}/{H} differ, max {dh.max():.3e}")
    print(f"  gates : {np.count_nonzero(dg)}/{NE} differ, max {dg.max():.3e}")
    if L > 0:
        dm = np.abs(r['moe'] - n['moe'])
        dhmid = np.abs(r['h_mid'] - n['h_mid'])
        print(f"  moe_hidden(L-1): {np.count_nonzero(dm)}/{H} differ, max {dm.max():.3e}"
              + ("   <-- CMD3(L-1) CORRUPT" if dm.max() > 0 else "   clean"))
        print(f"  h_mid(L): {np.count_nonzero(dhmid)}/{H} differ, max {dhmid.max():.3e}"
              + ("   <-- cmdA/cmdB(L) CORRUPT" if dhmid.max() > 0 else "   clean"))

if __name__ == "__main__":
    main()
