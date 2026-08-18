#!/usr/bin/env python3
"""Analyze the crashed session's /tmp debug dumps from the Phase C S4 C3
chunked-GGUF parity hunt (run: 13 tokens, chunk=1, FINCHMOE_GGUF_CHUNK=1,
FINCHMOE_GGUF_DBG=1, FINCHMOE_PF_DUMP=1). Each dump holds CPU-reference vs
GPU/chunked arrays; print cos + maxdiff per entry and per field."""
import numpy as np, os, sys

D = dict(HIDDEN=2048, MOE_INT=512, SHARED_INT=512, CONV=8192, TVAL=4096,
         TKEY=2048, STATE=32*128*128, K=8)
P = 13  # positions (tokens), chunk size 1

def chunks(fname, nfloats, stride=None, skip_bytes=0):
    data = np.fromfile(fname, dtype=np.float32, offset=skip_bytes)
    s = stride or nfloats
    out = []
    for i in range(0, len(data) - nfloats + 1, s):
        out.append(data[i:i+nfloats])
    return out

def report(name, a, b, label):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    with np.errstate(all='ignore'):
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    d = np.abs(a - b)
    print(f"  {label:34s} cos={cos:9.6f}  maxdiff={d.max():.3e}  "
          f"(a{rms(a):.4f} b{rms(b):.4f})")

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float32)**2)))

def fin(name):
    return f"/tmp/{name}"

print("=== chain2_dbg.bin: CPU-BLAS ref vs GPU delta_net_step (chunked chain, layer 2) ===")
d = np.fromfile(fin("chain2_dbg.bin"), np.float32)
print(f"  entries: {len(d)//(2*D['TVAL'])}")
for i in range(len(d)//(2*D['TVAL'])):
    cpu, gpu = d[i*2*D['TVAL']:(i+1)*2*D['TVAL']-D['TVAL']], d[(i+1)*2*D['TVAL']-D['TVAL']:(i+1)*2*D['TVAL']]
    report("chain2", cpu, gpu, f"pos {i} out")

print("\n=== qkv2_dbg.bin: CPU vs GPU batched QK matvecs (qkv 8192, z 4096) ===")
d = np.fromfile(fin("qkv2_dbg.bin"), np.float32)
E = D['CONV']*2 + D['TVAL']*2
print(f"  entries: {len(d)//E}")
for i in range(len(d)//E):
    e = d[i*E:(i+1)*E]
    report("qkv", e[:D['CONV']], e[D['CONV']:2*D['CONV']], f"pos {i} qkv")
    report("z", e[2*D['CONV']:2*D['CONV']+D['TVAL']], e[-D['TVAL']:], f"pos {i} z")

print("\n=== conv2_entry vs conv2_new: chunked conv state (3x8192), sequential check ===")
entry = np.fromfile(fin("conv2_entry.bin"), np.float32)
new = np.fromfile(fin("conv2_new.bin"), np.float32)
CE = 3*D['CONV']
print(f"  entry entries: {len(entry)//CE}, new entries: {len(new)//CE}")
for i in range(min(len(entry)//CE, len(new)//CE)):
    report("conv", entry[i*CE:(i+1)*CE], new[i*CE:(i+1)*CE], f"pos {i} entry vs new")
# cross: entry[i+1] should share rows with new[i] (shift by one qkv row)
if len(new)//CE >= 2:
    # new[i] rows = [r1, r2, qkv_i]; new[i+1] rows = [r2, qkv_i, qkv_{i+1}]
    # → new[i+1][0:2*CONV] must equal new[i][CONV:3*CONV] bitwise
    for i in range(len(new)//CE - 1):
        a = new[(i+1)*CE:(i+1)*CE+2*D['CONV']]
        b = new[i*CE+D['CONV']:(i+1)*CE]
        report("conv-shift", a, b, f"new[{i+1}][:2] vs new[{i}][1:3]")

print("\n=== state2_new.bin: GPU delta state snapshots (32x128x128) ===")
st = np.fromfile(fin("state2_new.bin"), np.float32)
S = D['STATE']
print(f"  entries: {len(st)//S}")
for i in range(1, min(len(st)//S, 5)):
    report("state", st[(i-1)*S:i*S], st[i*S:(i+1)*S], f"state {i-1} vs {i}")

print("\n=== inputnorm_cpu.bin: CPU vs GPU C3 input norm / residual norm (layer 2) ===")
d = np.fromfile(fin("inputnorm_cpu.bin"), np.float32)
E = D['HIDDEN']*5
print(f"  entries: {len(d)//E}")
for i in range(len(d)//E):
    e = d[i*E:(i+1)*E]
    report("norm", e[:2048], e[2048:4096], f"pos {i} C3 input norm (CPU vs GPU)")
    report("resnorm", e[4096:6144], e[6144:8192], f"pos {i} residual norm (CPU vs GPU)")

print("\n=== shared_cpu.bin: CPU vs GPU shared-down (layers 0/1) ===")
d = np.fromfile(fin("shared_cpu.bin"), np.float32)
E = D['HIDDEN']*2 + D['SHARED_INT']
print(f"  entries: {len(d)//E}")
for i in range(len(d)//E):
    e = d[i*E:(i+1)*E]
    report("shared", e[:2048], e[2048:4096], f"entry {i} (layer {i%2})")

print("\n=== poolslot.bin: pread pool slot bytes vs mmap (layer 1 expert slabs) ===")
raw = open(fin("poolslot.bin"), "rb").read()
per = len(raw)//6
for j, name in enumerate(["gate slot vs mmap", "up slot vs mmap", "down slot vs mmap"]):
    a = np.frombuffer(raw[j*2*per:(j*2+1)*per], np.uint8)
    b = np.frombuffer(raw[(j*2+1)*per:(j*2+2)*per], np.uint8)
    if len(a) != len(b):
        print(f"  {name}: SIZE MISMATCH {len(a)} vs {len(b)}"); continue
    nb = np.count_nonzero(a != b)
    print(f"  {name:24s} {len(a)} bytes, mismatches={nb} ({100.0*nb/len(a):.4f}%)")

print("\n=== expert_cpu0.bin: CPU expert outputs (layers 0/1/39) ===")
raw = open(fin("expert_cpu0.bin"), "rb").read()
per = 8 + D['K']*(4 + D['HIDDEN']*4)
print(f"  calls: {len(raw)//per}")
for i in range(min(len(raw)//per, 9)):
    base = i*per
    layer, gM = np.frombuffer(raw[base:base+8], np.int32)
    if layer == 39:
        print(f"  call {i}: layer={layer} gM={gM} (K={D['K']})")

print("\n=== cmd3_components.bin: GPU CMD3 components (layer 0/1, m=0) ===")
raw = open(fin("cmd3_components.bin"), "rb").read()
print(f"  size: {len(raw)} bytes")

print("\n=== dbg26/27/28.bin: hidden-state dumps (993280 bytes each) ===")
for n in ("26", "27", "28"):
    pth = fin(f"dbg{n}.bin")
    if not os.path.exists(pth):
        print(f"  dbg{n}: missing"); continue
    d = np.fromfile(pth, np.float32)
    print(f"  dbg{n}: {len(d)} floats = {len(d)/2048:.1f} hidden states, rms={rms(d[:2048]):.4f}")
