#!/usr/bin/env python3
"""
Cell A vs Cell B: Compare BF16 vs 4-bit forward pass in Python.
If layer 0 output diverges significantly, quantization is the root cause.
"""
import json, struct, numpy as np, math, sys, time

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_4BIT = os.path.join(ROOT, 'finchmoe/model_weights.json')
WEIGHTS_4BIT = os.path.join(ROOT, 'finchmoe/model_weights.bin')
BF16_DIR = os.path.join(ROOT, 'models/Qwen3.6-35B-A3B-bf16')

# === Load BF16 index ===
with open(f'{BF16_DIR}/model.safetensors.index.json') as f:
    bf_idx = json.load(f)

# Cache for opened safetensors files
bf_files = {}

def load_bf16_tensor(safe_name):
    """Load a single tensor directly from BF16 safetensors."""
    fname = bf_idx['weight_map'][safe_name]
    if fname not in bf_files:
        path = f'{BF16_DIR}/{fname}'
        with open(path, 'rb') as f:
            hl = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(hl))
        bf_files[fname] = (path, 8 + hl, hdr)

    path, data_start, hdr = bf_files[fname]
    offsets = hdr[safe_name]['data_offsets']
    shape = hdr[safe_name]['shape']
    with open(path, 'rb') as f:
        f.seek(data_start + offsets[0])
        data = f.read(offsets[1] - offsets[0])
    # BF16 -> float32
    return (np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()

# === Load 4-bit manifest ===
with open(MANIFEST_4BIT) as f:
    m4 = json.load(f)
tensors_4bit = m4['tensors']

def load_4bit_tensor(name):
    """Load and dequantize a 4-bit tensor."""
    info = tensors_4bit[name]
    with open(WEIGHTS_4BIT, 'rb') as f:
        f.seek(info['offset'])
        data = f.read(info['size'])
    dtype, shape = info['dtype'], info['shape']
    if dtype == 'U16':
        arr = (np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        return arr.reshape(shape).copy()
    elif dtype == 'U32':
        W = np.frombuffer(data, dtype=np.uint32).reshape(shape).copy()
        s_name = name.replace('.weight', '.scales')
        b_name = name.replace('.weight', '.biases')
        si = tensors_4bit[s_name]; bi = tensors_4bit[b_name]
        with open(WEIGHTS_4BIT, 'rb') as f:
            f.seek(si['offset']); sd = f.read(si['size'])
            f.seek(bi['offset']); bd = f.read(bi['size'])
        S = (np.frombuffer(sd, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(si['shape'])
        B = (np.frombuffer(bd, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(bi['shape'])

        out_dim = shape[0]
        vals_per_u32 = 32 // 4  # 8 for 4-bit, 4 for 8-bit
        # Detect 8-bit: if S.shape[1] != shape[1] / (in_dim // 64) pattern...
        # For now assume 4-bit unless name matches 8-bit patterns
        bits = 8 if ('.mlp.gate.weight' in name or '.mlp.shared_expert_gate.weight' in name) else 4
        vals_per_u32 = 32 // bits
        num_groups = S.shape[1]
        packed_per_group = shape[1] // num_groups  # uint32 per group
        group_size = packed_per_group * vals_per_u32

        deq = np.zeros((out_dim, num_groups * group_size), dtype=np.float32)
        for row in range(out_dim):
            for g in range(num_groups):
                scale = S[row][g]; bias = B[row][g]
                for p in range(packed_per_group):
                    packed = int(W[row][g*packed_per_group + p])
                    for n in range(vals_per_u32):
                        val = (packed >> (n*bits)) & ((1<<bits)-1)
                        idx = g*group_size + p*vals_per_u32 + n
                        deq[row][idx] = float(val)*scale + bias
        return deq
    elif dtype == 'F32':
        return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()

def rms_norm(x, weight):
    rms = math.sqrt(np.mean(x**2) + 1e-6)
    return (x / rms) * weight

# === Map BF16 names to 4-bit names ===
def bf16_to_4bit_name(bf16_name):
    """model.language_model.layers.X... -> model.layers.X..."""
    return bf16_name.replace('model.language_model.', 'model.')

# === Compute layer 0 forward pass with both BF16 and 4-bit weights ===
HIDDEN = 2048
token_id = 760  # "The"

print("=== Cell A: BF16 forward pass ===")
t0 = time.time()

# BF16 embedding
emb_bf16 = load_bf16_tensor('model.language_model.embed_tokens.weight')
hidden_bf16 = emb_bf16[token_id].copy().astype(np.float32)
print(f"Embed rms: {np.sqrt(np.mean(hidden_bf16**2)):.6f}")

# Layer 0 BF16
l0_in_norm = load_bf16_tensor('model.language_model.layers.0.input_layernorm.weight')
normed = rms_norm(hidden_bf16, l0_in_norm)
residual = hidden_bf16.copy()

# QKV projection
qkv_w = load_bf16_tensor('model.language_model.layers.0.linear_attn.in_proj_qkv.weight')
qkv_out = qkv_w @ normed  # [8192]

# Z projection
z_w = load_bf16_tensor('model.language_model.layers.0.linear_attn.in_proj_z.weight')
z_out = z_w @ normed  # [4096]

# Beta/Alpha projections
b_w = load_bf16_tensor('model.language_model.layers.0.linear_attn.in_proj_b.weight')
b_out = b_w @ normed  # [32]
a_w = load_bf16_tensor('model.language_model.layers.0.linear_attn.in_proj_a.weight')
a_out = a_w @ normed  # [32]

# Conv1d
conv_w = load_bf16_tensor('model.language_model.layers.0.linear_attn.conv1d.weight')
# conv_w shape: [8192, 1, 4] -> reshape to [8192, 4]
conv_w_2d = conv_w.reshape(8192, 4)
# For token 0, conv state is zero, so we only use the last weight position
conv_out = np.zeros(8192, dtype=np.float32)
for c in range(8192):
    conv_out[c] = qkv_out[c] * conv_w_2d[c, 3]  # new input * w[3]
# SiLU
conv_out = conv_out / (1.0 + np.exp(-conv_out))

# Split Q, K, V
q = conv_out[:2048].copy()
k = conv_out[2048:4096].copy()
v = conv_out[4096:8192].copy()

# Q/K norm + scale
inv_scale = 1.0 / math.sqrt(128)
q = q.reshape(16, 128)
k = k.reshape(16, 128)
for h in range(16):
    q[h] = rms_norm(q[h], np.ones(128)) * (inv_scale * inv_scale)
    k[h] = rms_norm(k[h], np.ones(128)) * inv_scale
v = v.reshape(32, 128)

# Delta-net state init (zero for first token)
# Skip full delta-net recurrence for speed — just do the first step
A_log = load_bf16_tensor('model.language_model.layers.0.linear_attn.A_log')
dt_bias = load_bf16_tensor('model.language_model.layers.0.linear_attn.dt_bias')

k_heads_per_v = 2
out_vals_bf16 = np.zeros(4096, dtype=np.float32)
for vh in range(32):
    kh = vh // k_heads_per_v
    A_val = math.exp(float(A_log[vh]))
    softplus_val = math.log(1.0 + math.exp(float(a_out[vh]) + float(dt_bias[vh])))
    g_decay = math.exp(-A_val * softplus_val)
    beta_gate = 1.0 / (1.0 + math.exp(-float(b_out[vh])))

    # S starts at 0, so S@k = 0, delta = v * beta
    delta = v[vh] * beta_gate
    # S_new = delta @ k^T (outer product, since S was 0)
    # output = S_new @ q = (delta @ k^T) @ q = delta * (k^T @ q)
    k_dot_q = np.dot(k[kh], q[kh])
    out_vals_bf16[vh*128:(vh+1)*128] = delta * k_dot_q

# Gated norm
gated_norm_w = load_bf16_tensor('model.language_model.layers.0.linear_attn.norm.weight')
gated_bf16 = np.zeros(4096, dtype=np.float32)
for vh in range(32):
    oh = out_vals_bf16[vh*128:(vh+1)*128]
    zh = z_out[vh*128:(vh+1)*128]
    rms = math.sqrt(np.mean(oh**2) + 1e-6)
    silu_z = zh / (1.0 + np.exp(-zh))
    gated_bf16[vh*128:(vh+1)*128] = (oh / rms) * gated_norm_w * silu_z

# Out proj
out_w = load_bf16_tensor('model.language_model.layers.0.linear_attn.out_proj.weight')
attn_proj_bf16 = out_w @ gated_bf16
hidden_bf16 = residual + attn_proj_bf16

rms_bf16 = np.sqrt(np.mean(hidden_bf16**2))
print(f"Layer 0 output rms (BF16): {rms_bf16:.6f}")
print(f"BF16 time: {time.time()-t0:.1f}s")

# === Cell B: 4-bit forward pass (same computation) ===
print("\n=== Cell B: 4-bit forward pass ===")
t0 = time.time()

# 4-bit embedding
emb_4bit = load_4bit_tensor('model.embed_tokens.weight')
hidden_4bit = emb_4bit[token_id].copy().astype(np.float32)
print(f"Embed rms: {np.sqrt(np.mean(hidden_4bit**2)):.6f}")

# Layer 0 4-bit
l0_in = load_4bit_tensor('model.layers.0.input_layernorm.weight')
normed4 = rms_norm(hidden_4bit, l0_in)
residual4 = hidden_4bit.copy()

qkv_w4 = load_4bit_tensor('model.layers.0.linear_attn.in_proj_qkv.weight')
qkv_out4 = qkv_w4 @ normed4

z_w4 = load_4bit_tensor('model.layers.0.linear_attn.in_proj_z.weight')
z_out4 = z_w4 @ normed4

b_w4 = load_4bit_tensor('model.layers.0.linear_attn.in_proj_b.weight')
b_out4 = b_w4 @ normed4
a_w4 = load_4bit_tensor('model.layers.0.linear_attn.in_proj_a.weight')
a_out4 = a_w4 @ normed4

# Conv1d (BF16, same as above)
conv_out4 = np.zeros(8192, dtype=np.float32)
for c in range(8192):
    conv_out4[c] = qkv_out4[c] * conv_w_2d[c, 3]
conv_out4 = conv_out4 / (1.0 + np.exp(-conv_out4))

q4 = conv_out4[:2048].copy()
k4 = conv_out4[2048:4096].copy()
v4 = conv_out4[4096:8192].copy()

q4 = q4.reshape(16, 128); k4 = k4.reshape(16, 128)
for h in range(16):
    q4[h] = rms_norm(q4[h], np.ones(128)) * (inv_scale * inv_scale)
    k4[h] = rms_norm(k4[h], np.ones(128)) * inv_scale
v4 = v4.reshape(32, 128)

A_log4 = load_4bit_tensor('model.layers.0.linear_attn.A_log')
dt_bias4 = load_4bit_tensor('model.layers.0.linear_attn.dt_bias')

out_vals_4bit = np.zeros(4096, dtype=np.float32)
for vh in range(32):
    kh = vh // k_heads_per_v
    A_val = math.exp(float(A_log4[vh]))
    softplus_val = math.log(1.0 + math.exp(float(a_out4[vh]) + float(dt_bias4[vh])))
    g_decay = math.exp(-A_val * softplus_val)
    beta_gate = 1.0 / (1.0 + math.exp(-float(b_out4[vh])))
    delta = v4[vh] * beta_gate
    k_dot_q = np.dot(k4[kh], q4[kh])
    out_vals_4bit[vh*128:(vh+1)*128] = delta * k_dot_q

gated_norm_w4 = load_4bit_tensor('model.layers.0.linear_attn.norm.weight')
gated_4bit = np.zeros(4096, dtype=np.float32)
for vh in range(32):
    oh = out_vals_4bit[vh*128:(vh+1)*128]
    zh = z_out4[vh*128:(vh+1)*128]
    rms = math.sqrt(np.mean(oh**2) + 1e-6)
    silu_z = zh / (1.0 + np.exp(-zh))
    gated_4bit[vh*128:(vh+1)*128] = (oh / rms) * gated_norm_w4 * silu_z

out_w4 = load_4bit_tensor('model.layers.0.linear_attn.out_proj.weight')
attn_proj_4bit = out_w4 @ gated_4bit
hidden_4bit = residual4 + attn_proj_4bit

rms_4bit = np.sqrt(np.mean(hidden_4bit**2))
print(f"Layer 0 output rms (4-bit): {rms_4bit:.6f}")
print(f"4-bit time: {time.time()-t0:.1f}s")

# === COMPARISON ===
diff = hidden_bf16 - hidden_4bit
print(f"\n=== BF16 vs 4-bit Layer 0 Output ===")
print(f"BF16 rms: {rms_bf16:.6f}")
print(f"4-bit rms: {rms_4bit:.6f}")
print(f"Diff rms: {np.sqrt(np.mean(diff**2)):.6f}")
print(f"Relative RMS error: {np.sqrt(np.mean(diff**2)) / rms_bf16 * 100:.2f}%")
print(f"Max abs diff: {np.abs(diff).max():.6f}")
print(f"Cosine similarity: {np.dot(hidden_bf16, hidden_4bit) / (np.linalg.norm(hidden_bf16) * np.linalg.norm(hidden_4bit)):.6f}")
print(f"\nBF16 first 8: {hidden_bf16[:8]}")
print(f"4-bit first 8: {hidden_4bit[:8]}")
print(f"Diff first 8: {diff[:8]}")
