#!/usr/bin/env python3
"""Numpy reference for layer-0 token-0 GDN chain, compared against the
engine's FINCHMOE_DUMP_STAGES output (/tmp/stage_dump.bin).

Mirrors the Metal shader semantics exactly:
  conv1d_step, rms_norm_qk, compute_decay_beta, gated_delta_net_step,
  gated_rms_norm, dequant_matvec_4bit_v3, rms_norm_apply.

Usage:
  FINCHMOE_DUMP_STAGES=1 ./finchmoe-infer -t 1 -k 8 -e 0 -P "..."   # engine
  cp /tmp/stage_dump.bin /tmp/stage_quant.bin
  python3 debug_gdn_reference.py /tmp/stage_quant.bin
"""
import json
import os
import struct
import sys
import math
import numpy as np

HIDDEN = 2048
LINEAR_CONV_DIM = 8192
LINEAR_TOTAL_VALUE = 4096
LINEAR_NUM_V_HEADS = 32
LINEAR_NUM_K_HEADS = 16
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
EPS = 1e-6

MANIFEST = os.environ.get('FINCHMOE_REF_MANIFEST', 'quant_test/model_weights.json')
WEIGHTS = os.environ.get('FINCHMOE_REF_WEIGHTS', 'quant_test/model_weights.bin')

m = json.load(open(MANIFEST))['tensors']
wf = open(WEIGHTS, 'rb')


def bf16(u):
    return struct.unpack('f', struct.pack('I', u << 16))[0]


def load(name):
    t = m[name]
    wf.seek(t['offset'])
    raw = wf.read(t['size'])
    if t['dtype'] == 'U32':
        return np.frombuffer(raw, np.uint32).reshape(t['shape'])
    if t['dtype'] in ('BF16', 'U16'):
        u = np.frombuffer(raw, np.uint16)
        return (u.astype(np.uint32) << 16).view(np.float32).reshape(t['shape'])
    if t['dtype'] == 'F32':
        return np.frombuffer(raw, np.float32).reshape(t['shape'])
    raise ValueError(t['dtype'])


def dequant_matvec(W, S, B, x, bits=4):
    """W [out, packed_cols] u32, S/B [out, groups] bf16-f32, x [in]."""
    out_dim, packed_cols = W.shape
    vals = 32 // bits
    groups = S.shape[1]
    group_size = (packed_cols * vals) // groups
    xg = x.reshape(groups, group_size)
    acc = np.zeros(out_dim, dtype=np.float64)
    for g in range(groups):
        sg = S[:, g].astype(np.float64)
        bg = B[:, g].astype(np.float64)
        xgv = xg[g].astype(np.float64)
        for p in range(group_size // vals):
            packed = W[:, g * (group_size // vals) + p]
            for n in range(vals):
                q = ((packed >> (n * bits)) & ((1 << bits) - 1)).astype(np.float64)
                acc += q * (sg * xgv[p * vals + n]) + bg * xgv[p * vals + n]
    return acc.astype(np.float32)


def proj_bits(base):
    """Detect packing bits from the manifest (explicit field or shape rule)."""
    wi = m[base + '.weight']
    si = m[base + '.scales']
    if 'bits' in wi:
        return wi['bits']
    return 8 if wi['shape'][1] == si['shape'][1] * 16 else 4


def embed(tok):
    W = load('model.embed_tokens.weight')      # [248320, 256]
    S = load('model.embed_tokens.scales')      # [248320, 32]
    B = load('model.embed_tokens.biases')      # [248320, 32]
    row = W[tok].reshape(1, 256)
    s = S[tok].reshape(1, 32)
    b = B[tok].reshape(1, 32)
    return dequant_matvec(row, s, b, np.zeros(1, dtype=np.float32))  # dummy: handle below


def embed_vec(tok):
    W = load('model.embed_tokens.weight')[tok]  # [packed_cols]
    S = load('model.embed_tokens.scales')[tok]  # [groups]
    B = load('model.embed_tokens.biases')[tok]  # [groups]
    bits = proj_bits('model.embed_tokens')
    vpu = 32 // bits
    groups = len(S)
    gs = (len(W) * vpu) // groups
    vals = []
    for g in range(groups):
        s = float(S[g]); b = float(B[g])
        for p in range(gs // vpu):
            pk = int(W[g * (gs // vpu) + p])
            for n in range(vpu):
                vals.append(((pk >> (bits * n)) & ((1 << bits) - 1)) * s + b)
    return np.array(vals, dtype=np.float32)


def rms_norm(x, w):
    r = math.sqrt(float(np.mean(x.astype(np.float64) ** 2)) + EPS)
    return (x / r) * w


def main():
    stage_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/stage_quant.bin'
    eng = np.fromfile(stage_file, dtype=np.float32)
    spec = [('qkv', 8192), ('z', 4096), ('beta', 32), ('alpha', 32),
            ('conv', 8192), ('delta_out', 4096), ('gated', 4096),
            ('o_proj', 2048), ('h_mid', 2048), ('h_post', 2048)]
    eng_stages = {}
    off = 0
    for name, n in spec:
        eng_stages[name] = eng[off:off + n]
        off += n
    print(f'engine stage file: {stage_file} ({off} floats)')

    # ---- token 0 of the chat template = <|im_start|> = 248045 ----
    tok = 248045
    emb = embed_vec(tok)
    print(f'emb rms={np.sqrt(np.mean(emb**2)):.4f}')

    inw = load('model.layers.0.input_layernorm.weight')
    normed = rms_norm(emb, inw)

    def proj(name, out_dim):
        W = load(f'{name}.weight')
        if name + '.scales' not in m:
            # BF16 raw matvec
            x64 = normed.astype(np.float64)
            return (W.astype(np.float64) @ x64).astype(np.float32)
        S = load(f'{name}.scales')
        B = load(f'{name}.biases')
        return dequant_matvec(W, S, B, normed.astype(np.float32), proj_bits(name))

    qkv = proj('model.layers.0.linear_attn.in_proj_qkv', LINEAR_CONV_DIM)
    z = proj('model.layers.0.linear_attn.in_proj_z', LINEAR_TOTAL_VALUE)
    beta = proj('model.layers.0.linear_attn.in_proj_b', LINEAR_NUM_V_HEADS)
    alpha = proj('model.layers.0.linear_attn.in_proj_a', LINEAR_NUM_V_HEADS)

    # ---- conv1d (token 0: zero history) ----
    convw = load('model.layers.0.linear_attn.conv1d.weight').reshape(LINEAR_CONV_DIM, 4)
    conv = qkv * convw[:, 3]
    conv = conv / (1.0 + np.exp(-conv))  # SiLU

    # ---- rms_norm_qk ----
    inv_scale = 1.0 / math.sqrt(LINEAR_KEY_DIM)
    q_ = conv[:LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM].reshape(LINEAR_NUM_K_HEADS, LINEAR_KEY_DIM).copy()
    k_ = conv[LINEAR_TOTAL_VALUE // 2:LINEAR_TOTAL_VALUE // 2 + LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM].reshape(
        LINEAR_NUM_K_HEADS, LINEAR_KEY_DIM).copy()
    for h in range(LINEAR_NUM_K_HEADS):
        rq = math.sqrt(float(np.mean(q_[h].astype(np.float64) ** 2)) + EPS)
        rk = math.sqrt(float(np.mean(k_[h].astype(np.float64) ** 2)) + EPS)
        q_[h] = (q_[h] / rq) * inv_scale * inv_scale
        k_[h] = (k_[h] / rk) * inv_scale
    q_ = q_.reshape(-1)
    k_ = k_.reshape(-1)
    v_ = conv[2 * (LINEAR_TOTAL_VALUE // 2):2 * (LINEAR_TOTAL_VALUE // 2) + LINEAR_TOTAL_VALUE]

    # ---- compute_decay_beta ----
    A_log = load('model.layers.0.linear_attn.A_log').reshape(-1)
    dt_bias = load('model.layers.0.linear_attn.dt_bias').reshape(-1)
    softplus = np.log(1.0 + np.exp(alpha.astype(np.float64) + dt_bias.astype(np.float64)))
    g_decay = np.exp(-np.exp(A_log.astype(np.float64)) * softplus)
    beta_gate = 1.0 / (1.0 + np.exp(-beta.astype(np.float64)))

    # ---- gated_delta_net_step (fresh state = 0, token 0) ----
    khpv = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS  # 2
    delta_out = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float64)
    for head in range(LINEAR_NUM_V_HEADS):
        kh = head // khpv
        kb = kh * LINEAR_KEY_DIM
        vb = head * LINEAR_VALUE_DIM
        g = g_decay[head]
        b = beta_gate[head]
        for vi in range(LINEAR_VALUE_DIM):
            kv_mem = 0.0
            for ki in range(LINEAR_KEY_DIM):
                kv_mem += 0.0  # state zero
            delta = (v_[vb + vi] - kv_mem) * b
            out = 0.0
            for ki in range(LINEAR_KEY_DIM):
                s = k_[kb + ki] * delta  # S = k*delta with zero initial state
                out += s * q_[kb + ki]
            delta_out[vb + vi] = out

    # ---- gated_rms_norm ----
    gnorm_w = load('model.layers.0.linear_attn.norm.weight').reshape(-1)
    zs = z.reshape(LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM)
    gated = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float64)
    for h in range(LINEAR_NUM_V_HEADS):
        base = h * LINEAR_VALUE_DIM
        r = math.sqrt(float(np.mean(delta_out[base:base + LINEAR_VALUE_DIM] ** 2)) + EPS)
        for vi in range(LINEAR_VALUE_DIM):
            normed_v = delta_out[base + vi] / r
            zval = zs[h, vi]
            gate = zval / (1.0 + math.exp(-zval))
            gated[base + vi] = normed_v * gate * float(gnorm_w[vi])
    gated = gated.astype(np.float32)

    # ---- o_proj ----
    oW = load('model.layers.0.linear_attn.out_proj.weight')
    oS = load('model.layers.0.linear_attn.out_proj.scales')
    oB = load('model.layers.0.linear_attn.out_proj.biases')
    oproj = dequant_matvec(oW, oS, oB, gated, proj_bits('model.layers.0.linear_attn.out_proj'))
    h_mid = emb + oproj
    paw = load('model.layers.0.post_attention_layernorm.weight')
    h_post = rms_norm(h_mid.astype(np.float32), paw).astype(np.float32)

    def cos(a, b):
        return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))

    print(f'{"stage":10s} {"CosSim":>9s} {"rms ref":>9s} {"rms eng":>9s}')
    for name, ref in [('qkv', qkv), ('z', z), ('beta', beta), ('alpha', alpha),
                      ('conv', conv), ('delta_out', delta_out.astype(np.float32)),
                      ('gated', gated), ('o_proj', oproj), ('h_mid', h_mid),
                      ('h_post', h_post)]:
        e = eng_stages[name]
        print(f'{name:10s} {cos(ref, e):9.6f} {np.sqrt(np.mean(ref**2)):9.4f} {np.sqrt(np.mean(e**2)):9.4f}')


if __name__ == '__main__':
    main()
