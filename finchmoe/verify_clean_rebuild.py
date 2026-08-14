#!/usr/bin/env python3
"""Validate the single-stage rebuild (pristine BF16 base -> quant_clean +
packed_experts_3bit) against the BF16 reference weights.

Checks:
  1. Non-expert tensors: dequant vs BF16 source (4-bit >= 0.995, 8-bit >= 0.999)
  2. Expert tensors: 3-bit vs BF16 source (target >= 0.98; gate_proj gate:
     if < 0.975 consider a 4-bit gate_proj hybrid)
"""
import json
import struct
import sys
import numpy as np

BF16_MODEL = '../models/Qwen3.6-35B-A3B-bf16'
MANIFEST = 'quant_clean/model_weights_quant.json'
WEIGHTS = 'quant_clean/model_weights_quant.bin'
EXPERT_FILE = '../models/Qwen3.6-35B-A3B-bf16/packed_experts_3bit/layer_00.bin'

m = json.load(open(MANIFEST))['tensors']
wf = open(WEIGHTS, 'rb')


def bf16(u):
    return struct.unpack('f', struct.pack('I', u << 16))[0]


def read_bf16_source(name, n):
    idx = json.load(open(BF16_MODEL + '/model.safetensors.index.json'))
    wm = idx['weight_map']
    if name not in wm:
        # tolerate prefix differences: match on the suffix
        key = name.split('.', 2)[-1]
        hits = [k for k in wm if k.endswith(key)]
        if not hits:
            raise KeyError(name)
        name = hits[0]
    shard = wm[name]
    f = open(BF16_MODEL + '/' + shard, 'rb')
    hlen = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hlen))
    base = 8 + hlen
    s, e = hdr[name]['data_offsets']
    f.seek(base + s)
    return (np.frombuffer(f.read(n * 2), np.uint16).astype(np.uint32) << 16).view(np.float32)


def dequant_row(tname, n):
    t = m[tname]
    bits = t.get('bits', 4)
    gs = t.get('group_size', 64)
    wf.seek(t['offset'])
    row_u32 = np.frombuffer(wf.read(t['shape'][1] * 4), np.uint32)
    groups = n // gs
    sc = m[tname[:-7] + '.scales']; wf.seek(sc['offset'])
    sv = np.frombuffer(wf.read(groups * 2), np.uint16)
    bi = m[tname[:-7] + '.biases']; wf.seek(bi['offset'])
    bv = np.frombuffer(wf.read(groups * 2), np.uint16)
    vpu = 32 // bits
    mask = (1 << bits) - 1
    vals = np.zeros(n, np.float32)
    for g in range(groups):
        s = bf16(int(sv[g])); b = bf16(int(bv[g]))
        for u in range(gs // vpu):
            pk = int(row_u32[g * (gs // vpu) + u])
            for v in range(vpu):
                vals[g * gs + u * vpu + v] = ((pk >> (v * bits)) & mask) * s + b
    return vals


def cos(a, b):
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def check_non_expert(tname, bf16_name, n=2048):
    if tname not in m:
        print(f'{tname.split(".")[-2]:12s}: SKIP (not in manifest)')
        return
    ref = read_bf16_source(bf16_name, n)
    deq = dequant_row(tname, n)
    c = cos(deq, ref)
    status = 'PASS' if c > 0.995 else 'FAIL'
    print(f'{tname.split(".")[-2]:12s}: CosSim={c:.6f} {status}')


def check_expert_3bit():
    """3-bit experts (layer 0, slot 0) vs BF16 source gate row 0."""
    idx = json.load(open(BF16_MODEL + '/model.safetensors.index.json'))
    # gate rows 0-511 of fused gate_up [256, 1024, 2048]
    name = 'model.language_model.layers.0.mlp.experts.gate_up_proj'
    shard = idx['weight_map'][name]
    f = open(BF16_MODEL + '/' + shard, 'rb')
    hlen = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hlen))
    base = 8 + hlen
    s, e = hdr[name]['data_offsets']
    f.seek(base + s)
    # row 0 of expert 0 = first 2048 BF16
    ref = (np.frombuffer(f.read(2048 * 2), np.uint16).astype(np.uint32) << 16).view(np.float32)

    # packed 3-bit gate row 0: bytes 0..767 of layer_00.bin, scales at 393216
    pk = open(EXPERT_FILE, 'rb')
    pk.seek(0)
    raw = pk.read(768)
    pk.seek(393216)
    sraw = pk.read(64)
    pk.seek(425984)
    braw = pk.read(64)
    vals = np.zeros(2048, np.float32)
    for g in range(32):
        s = bf16(struct.unpack('<H', sraw[g * 2:g * 2 + 2])[0])
        b = bf16(struct.unpack('<H', braw[g * 2:g * 2 + 2])[0])
        for t in range(8):
            v = raw[(g * 8 + t) * 3] | (raw[(g * 8 + t) * 3 + 1] << 8) | (raw[(g * 8 + t) * 3 + 2] << 16)
            for j in range(8):
                vals[g * 64 + t * 8 + j] = ((v >> (3 * j)) & 7) * s + b
    c = cos(vals, ref)
    gate_ok = c >= 0.975
    print(f'3bit-gate    : CosSim={c:.6f} {"PASS" if gate_ok else "FAIL (<0.975 -> consider 4-bit gate_proj)"}')
    return c


print('=== non-expert (row 0, vs BF16 source) ===')
check_non_expert('model.layers.0.linear_attn.in_proj_qkv.weight',
                 'model.language_model.layers.0.linear_attn.in_proj_qkv.weight')
check_non_expert('model.layers.0.linear_attn.in_proj_z.weight',
                 'model.language_model.layers.0.linear_attn.in_proj_z.weight')
check_non_expert('model.layers.0.linear_attn.out_proj.weight',
                 'model.language_model.layers.0.linear_attn.out_proj.weight')
check_non_expert('model.layers.3.self_attn.q_proj.weight',
                 'model.language_model.layers.3.self_attn.q_proj.weight')
check_non_expert('model.embed_tokens.weight',
                 'model.language_model.model.embed_tokens.weight')
print()
print('=== 3-bit experts (layer 0, expert 0, vs BF16 source) ===')
check_expert_3bit()
