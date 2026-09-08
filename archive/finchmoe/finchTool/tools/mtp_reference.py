#!/usr/bin/env python3
"""mtp_reference.py — numpy reference for the Qwen3.6-35B-A3B MTP head.

Reads the MTP tensors from the pristine BF16 safetensors and replays the
dumped records (/tmp/mtp_ref_input.bin — FINCHMOE_MTP_DUMP=1) through the
reference forward, printing the draft-vs-main logit cos + argmax per gen.

Usage: python3 mtp_reference.py [dump_file] [model_dir]
"""
import json, os, struct, sys
import numpy as np

H = 2048
V = 248320
NE = 256
MOE_INT = 512
NQ, NKV, HD, ROT = 16, 2, 256, 64
ROPE_THETA = 10000000.0
EPS = 1e-6

dump_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/mtp_ref_input.bin'
model_dir = sys.argv[2] if len(sys.argv) > 2 else '../models/Qwen3.6-35B-A3B-bf16'

idx = json.load(open(os.path.join(model_dir, 'model.safetensors.index.json')))
wm = idx['weight_map']

_shard_cache = {}
def read_tensor(name):
    """Read a BF16 tensor from the shard files (cached headers)."""
    shard_name = wm[name]
    if shard_name not in _shard_cache:
        path = os.path.join(model_dir, shard_name)
        with open(path, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(n))
        _shard_cache[shard_name] = (path, hdr, 8 + n)
    path, hdr, data_base = _shard_cache[shard_name]
    info = hdr[name]
    shape = info['shape']
    with open(path, 'rb') as f:
        f.seek(data_base + info['data_offsets'][0])
        raw = f.read(int(np.prod(shape)) * 2)
    u = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    return (u.astype(np.uint32) << 16).view(np.float32)

def bf16_vec(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def rms_norm(x, w):
    s = np.sqrt((x * x).mean() + EPS)
    return (x / s) * w

print("loading tensors...")
embed   = read_tensor('model.language_model.embed_tokens.weight')          # [V, H]
lm_head = read_tensor('lm_head.weight')                      # [V, H]
enorm   = read_tensor('mtp.pre_fc_norm_embedding.weight')
hnorm   = read_tensor('mtp.pre_fc_norm_hidden.weight')
in_norm = read_tensor('mtp.layers.0.input_layernorm.weight')
post_norm = read_tensor('mtp.layers.0.post_attention_layernorm.weight')
fnorm   = read_tensor('mtp.norm.weight')
q_w = read_tensor('mtp.layers.0.self_attn.q_proj.weight')    # [8192, H]
k_w = read_tensor('mtp.layers.0.self_attn.k_proj.weight')    # [512, H]
v_w = read_tensor('mtp.layers.0.self_attn.v_proj.weight')    # [512, H]
o_w = read_tensor('mtp.layers.0.self_attn.o_proj.weight')    # [H, 4096]
qn_w = read_tensor('mtp.layers.0.self_attn.q_norm.weight')   # [256]
kn_w = read_tensor('mtp.layers.0.self_attn.k_norm.weight')   # [256]
gate_w = read_tensor('mtp.layers.0.mlp.gate.weight')         # [256, H]
sg_gate = read_tensor('mtp.layers.0.mlp.shared_expert_gate.weight')  # [1, H]
sg_w = read_tensor('mtp.layers.0.mlp.shared_expert.gate_proj.weight')
su_w = read_tensor('mtp.layers.0.mlp.shared_expert.up_proj.weight')
sd_w = read_tensor('mtp.layers.0.mlp.shared_expert.down_proj.weight')
fc_w = read_tensor('mtp.fc.weight')                          # [H, 4096]
gu = read_tensor('mtp.layers.0.mlp.experts.gate_up_proj')    # [256, 1024, H]
dn = read_tensor('mtp.layers.0.mlp.experts.down_proj')       # [256, H, 512]
print("tensors loaded")

def rope(q, pos):
    half = ROT // 2
    i = np.arange(half, dtype=np.float64)
    freq = 1.0 / np.power(ROPE_THETA, (2 * i) / ROT)
    ang = pos * freq
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    q0 = q[:half].copy(); q1 = q[half:ROT].copy()
    q[:half] = q0 * c - q1 * s
    q[half:ROT] = q0 * s + q1 * c
    return q

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def mtp_forward(tok, hidden, kcache, vcache, n_ctx):
    e = embed[tok]
    e_norm = rms_norm(e, enorm)
    h_norm = rms_norm(hidden, hnorm)
    h = e_norm + h_norm
    normed = rms_norm(h, in_norm)

    q_raw = q_w @ normed                 # [8192]
    q = q_raw[0::2].reshape(NQ, HD).copy()      # element-interleaved
    gate = q_raw[1::2].reshape(NQ, HD)
    for hh in range(NQ):
        q[hh] = rms_norm(q[hh], qn_w)
    k = k_w @ normed                      # [512]
    k2 = k.reshape(NKV, HD).copy()
    for hh in range(NKV):
        k2[hh] = rms_norm(k2[hh], kn_w)
    v = v_w @ normed                      # [512]
    pos = n_ctx
    for hh in range(NQ): rope(q[hh], pos)
    for hh in range(NKV): rope(k2[hh], pos)
    kcache[pos] = k2.reshape(-1); vcache[pos] = v

    attn = np.zeros((NQ, HD), dtype=np.float32)
    scale = 1.0 / np.sqrt(HD)
    for qh in range(NQ):
        kvh = qh // 8
        scores = (q[qh] @ kcache[:n_ctx+1, kvh*HD:(kvh+1)*HD].T) * scale
        p = softmax(scores.astype(np.float64)).astype(np.float32)
        out = p @ vcache[:n_ctx+1, kvh*HD:(kvh+1)*HD]
        out = out / (1.0 + np.exp(-gate[qh]))
        attn[qh] = out
    o = o_w @ attn.reshape(-1)
    h = h + o

    h_post = rms_norm(h, post_norm)

    gs = softmax((gate_w @ h_post).astype(np.float64)).astype(np.float32)
    top = np.argsort(gs)[-8:]
    ws = gs[top]; ws = ws / ws.sum()

    moe = np.zeros(H, dtype=np.float32)
    for wi, ei in zip(ws, top):
        wm2 = gu[ei]                     # [1024, H]
        act = wm2 @ h_post               # [1024]
        g_, u_ = act[:512], act[512:]
        act = g_ / (1.0 + np.exp(-g_)) * u_
        moe += wi * (dn[ei] @ act)

    sg_score = 1.0 / (1.0 + np.exp(-(sg_gate @ h_post)[0]))
    s_act = (sg_w @ h_post) / (1.0 + np.exp(-(sg_w @ h_post))) * (su_w @ h_post)
    s_out = sd_w @ s_act
    h = h + moe + sg_score * s_out

    final = rms_norm(h, fnorm)
    fc_in = np.concatenate([final, e_norm])
    fc_out = fc_w @ fc_in
    return lm_head @ fc_out

rec_size = 2 + H + V
data = np.fromfile(dump_file, dtype=np.float32)
n_rec = data.size // rec_size
print(f"{n_rec} records")

kcache = np.zeros((4096, NKV * HD), dtype=np.float32)
vcache = np.zeros((4096, NKV * HD), dtype=np.float32)
for r in range(n_rec):
    rec = data[r*rec_size:(r+1)*rec_size]
    pos, tok = rec[:2].view(np.int32)
    hidden = rec[2:2+H]
    main_logits = rec[2+H:2+H+V]
    if pos < n_rec - 1 and np.abs(main_logits).sum() == 0:
        # prefill record: only fills the cache — recompute K/V directly
        e = embed[tok]
        e_norm = rms_norm(e, enorm)
        h = e_norm + rms_norm(hidden, hnorm)
        normed = rms_norm(h, in_norm)
        k = (k_w @ normed).reshape(NKV, HD).copy()
        for hh in range(NKV):
            k[hh] = rms_norm(k[hh], kn_w); rope(k[hh], pos)
        kcache[pos] = k.reshape(-1)
        vcache[pos] = v_w @ normed
        continue
    draft = mtp_forward(tok, hidden, kcache, vcache, pos)
    cos = float((draft * main_logits).sum() / np.sqrt((draft**2).sum() * (main_logits**2).sum()))
    print(f"pos {pos}: ref cos = {cos:.4f}  draft argmax = {int(np.argmax(draft))}  main argmax = {int(np.argmax(main_logits))}")
