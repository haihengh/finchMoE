#!/usr/bin/env python3
"""
Compare FinchMoE engine output against Python reference computation
for a single transformer layer. This identifies whether the engine's
computation diverges from the model weights.

Usage:
    python debug_layer_compare.py [--layer N] [--prompt "text"]
"""

import json
import struct
import sys
import argparse
import numpy as np
import math
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/Qwen3.6-35B-A3B-4bit-custom")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "finchmoe/model_weights.json")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "finchmoe/model_weights.bin")

# Model constants
HIDDEN_DIM = 2048
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
FULL_ATTN_INTERVAL = 4
RMS_NORM_EPS = 1e-6
LINEAR_NUM_V_HEADS = 32
LINEAR_NUM_K_HEADS = 16
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
LINEAR_TOTAL_KEY = LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM  # 2048
LINEAR_TOTAL_VALUE = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM  # 4096
LINEAR_CONV_DIM = LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE  # 8192
CONV_KERNEL_SIZE = 4
ROTARY_DIM = 64
ROPE_THETA = 10_000_000.0
GROUP_SIZE = 64

# Load manifest
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
tensors = manifest['tensors']

# Load weight file
weights_fd = open(WEIGHTS_PATH, 'rb')
weights_data = weights_fd.read()  # mmap equivalent — load into memory for simplicity

def load_tensor(name):
    """Load a tensor from model_weights.bin by name."""
    info = tensors[name]
    offset = info['offset']
    size = info['size']
    data = weights_data[offset:offset + size]
    dtype = info['dtype']
    shape = info['shape']

    if dtype == 'U32':
        arr = np.frombuffer(data, dtype=np.uint32).reshape(shape).copy()
    elif dtype == 'U16':
        arr = np.frombuffer(data, dtype=np.uint16).copy()
        # BF16 -> float32
        f32 = (arr.astype(np.uint32) << 16).view(np.float32)
        arr = f32.reshape(shape).copy()
    elif dtype == 'F32':
        arr = np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
    return arr

def bf16_to_f32(u16_arr):
    """Convert uint16 BF16 array to float32."""
    return (u16_arr.astype(np.uint32) << 16).view(np.float32)

def rms_norm(x, weight_bf16):
    """Qwen3_5RMSNorm: out = x / rms(x) * weight"""
    rms = math.sqrt(np.mean(x**2) + RMS_NORM_EPS)
    normed = x / rms
    return normed * weight_bf16

def dequant_matvec(W_packed, scales_bf16, biases_bf16, x, bits=4):
    """MLX affine 4-bit dequant matvec — vectorized with numpy for speed."""
    out_dim, packed_cols = W_packed.shape
    num_groups = scales_bf16.shape[1]
    vals_per_u32 = 32 // bits
    in_dim = packed_cols * vals_per_u32  # dequantized input dimension
    group_size = in_dim // num_groups

    # Step 1: Unpack all weights at once
    # W_packed: [out_dim, packed_cols] uint32
    # Each uint32 has 8 int4 values
    # Result: [out_dim, in_dim] float32

    # Broadcast x into group structure
    # x: [in_dim] → reshape to [num_groups, group_size]
    x_groups = x.reshape(num_groups, group_size)  # [32, 64]

    # Pre-scale x: scale_x[g] = scale[out_dim, g] * x_groups[g]
    # Pre-bias x: bias_x[g] = bias[out_dim, g] * x_groups[g]
    # scales: [out_dim, num_groups], biases: [out_dim, num_groups]

    scale_x_sum = np.zeros(out_dim, dtype=np.float32)
    bias_x_sum = np.zeros(out_dim, dtype=np.float32)

    for g in range(num_groups):
        sg = scales_bf16[:, g]  # [out_dim]
        bg = biases_bf16[:, g]  # [out_dim]
        # Sum of x over this group
        xg = x_groups[g]  # [group_size]
        bias_x_sum += bg * np.sum(xg)

    # Now unpack quantized weights and compute
    # To vectorize: unpack each uint32 into 8 int4 values
    # W_unpacked: [out_dim, in_dim] — but this is 8192*2048 = 16M floats = 64MB, manageable

    shifts = np.arange(vals_per_u32, dtype=np.uint32) * bits  # [0, 4, 8, 12, 16, 20, 24, 28]
    mask = (1 << bits) - 1

    # Reshape W to [out_dim * packed_cols] for vectorized unpack
    W_flat = W_packed.reshape(-1).astype(np.uint32)  # [out_dim * packed_cols]

    # For each group, unpack and accumulate
    out = np.zeros(out_dim, dtype=np.float32)
    packed_per_group = group_size // vals_per_u32

    for g in range(num_groups):
        scale = scales_bf16[:, g].astype(np.float32)  # [out_dim]
        bias = biases_bf16[:, g].astype(np.float32)   # [out_dim]
        xg = x_groups[g].astype(np.float32)           # [group_size]

        # For each uint32 in this group, across all rows
        for p in range(packed_per_group):
            # Column index in packed weight matrix
            col = g * packed_per_group + p
            packed_vals = W_packed[:, col].astype(np.uint32)  # [out_dim]

            # Unpack 8 int4 values per uint32: [out_dim, 8]
            unpacked = np.zeros((out_dim, vals_per_u32), dtype=np.float32)
            for n in range(vals_per_u32):
                unpacked[:, n] = ((packed_vals >> (n * bits)) & mask).astype(np.float32)

            # For each of the 8 values, dequant and multiply by x
            x_slice = xg[p * vals_per_u32:(p + 1) * vals_per_u32]  # [8]
            # [out_dim, 8] @ [8] → [out_dim]
            out += unpacked @ (scale[:, np.newaxis] * x_slice[np.newaxis, :]).T  # wrong, need rethink

    # Actually simpler approach: pre-compute scale*x and bias*x per group
    # Then dot product of unpacked weights with processed x

    # Cleaner approach:
    out2 = np.zeros(out_dim, dtype=np.float32)
    for g in range(num_groups):
        scale = scales_bf16[:, g].astype(np.float32)    # [out_dim]
        bias = biases_bf16[:, g].astype(np.float32)     # [out_dim]
        xg = x_groups[g].astype(np.float32)             # [group_size]

        # Sum of x over this group (for bias term)
        out2 += bias * np.sum(xg)

        # For each packed uint32 position in the group
        for p in range(packed_per_group):
            col = g * packed_per_group + p
            packed_vals = W_packed[:, col].astype(np.uint32)  # [out_dim]

            for n in range(vals_per_u32):
                vals = ((packed_vals >> (n * bits)) & mask).astype(np.float32)  # [out_dim]
                x_val = xg[p * vals_per_u32 + n]
                out2 += vals * scale * x_val

    return out2

def apply_rotary_emb(q_heads, k_heads, pos):
    """Apply RoPE to q and k. q: [num_q_heads, head_dim], k: [num_kv_heads, head_dim]"""
    half = ROTARY_DIM // 2
    for h in range(q_heads.shape[0]):
        for i in range(half):
            freq = 1.0 / (ROPE_THETA ** (2.0 * i / ROTARY_DIM))
            angle = pos * freq
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            q0 = q_heads[h, i]
            q1 = q_heads[h, i + half]
            q_heads[h, i] = q0 * cos_a - q1 * sin_a
            q_heads[h, i + half] = q0 * sin_a + q1 * cos_a
    for h in range(k_heads.shape[0]):
        for i in range(half):
            freq = 1.0 / (ROPE_THETA ** (2.0 * i / ROTARY_DIM))
            angle = pos * freq
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            k0 = k_heads[h, i]
            k1 = k_heads[h, i + half]
            k_heads[h, i] = k0 * cos_a - k1 * sin_a
            k_heads[h, i + half] = k0 * sin_a + k1 * cos_a

def cpu_silu(x):
    return x / (1.0 + np.exp(-x))

def cpu_swiglu(gate, up):
    return cpu_silu(gate) * up

def cpu_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def embed_lookup(token_id):
    """Look up the embedding for a token."""
    W = load_tensor('model.embed_tokens.weight')
    S = load_tensor('model.embed_tokens.scales')
    B = load_tensor('model.embed_tokens.biases')

    num_groups = S.shape[1]
    group_size = HIDDEN_DIM // num_groups
    vals_per_u32 = 8

    hidden = np.zeros(HIDDEN_DIM, dtype=np.float32)
    w_row = W[token_id]
    s_row = S[token_id]
    b_row = B[token_id]

    for g in range(num_groups):
        scale = s_row[g]
        bias = b_row[g]
        for p in range(group_size // vals_per_u32):
            packed = int(w_row[g * (group_size // vals_per_u32) + p])
            for n in range(vals_per_u32):
                val = (packed >> (n * 4)) & 0xF
                idx = g * group_size + p * vals_per_u32 + n
                hidden[idx] = float(val) * scale + bias
    return hidden

def compute_layer_forward(hidden, layer_idx, pos, kv_cache_k=None, kv_cache_v=None, conv_state=None, ssm_state=None):
    """
    Compute a single transformer layer's forward pass in Python.
    Returns (output_hidden, kv_cache_k, kv_cache_v, conv_state, ssm_state, debug_dict)
    """
    is_full = ((layer_idx + 1) % FULL_ATTN_INTERVAL == 0)
    debug = {}
    residual = hidden.copy()

    # Load norm weights
    input_norm_w = load_tensor(f"model.layers.{layer_idx}.input_layernorm.weight")
    post_attn_norm_w = load_tensor(f"model.layers.{layer_idx}.post_attention_layernorm.weight")

    # Step 1: Input LayerNorm
    normed = rms_norm(hidden, input_norm_w)
    debug['normed'] = normed.copy()

    attn_projected = None

    if is_full:
        # ---- Full attention ----
        # QKV projections
        q_w = load_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
        q_s = load_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.scales")
        q_b = load_tensor(f"model.layers.{layer_idx}.self_attn.q_proj.biases")
        k_w = load_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
        k_s = load_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.scales")
        k_b = load_tensor(f"model.layers.{layer_idx}.self_attn.k_proj.biases")
        v_w = load_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.weight")
        v_s = load_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.scales")
        v_b = load_tensor(f"model.layers.{layer_idx}.self_attn.v_proj.biases")
        o_w = load_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
        o_s = load_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.scales")
        o_b = load_tensor(f"model.layers.{layer_idx}.self_attn.o_proj.biases")
        q_norm_w = load_tensor(f"model.layers.{layer_idx}.self_attn.q_norm.weight")
        k_norm_w = load_tensor(f"model.layers.{layer_idx}.self_attn.k_norm.weight")

        q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2  # 8192
        kv_dim = NUM_KV_HEADS * HEAD_DIM  # 512

        q_all = dequant_matvec(q_w, q_s, q_b, normed)  # [8192]
        k = dequant_matvec(k_w, k_s, k_b, normed)  # [512]
        v = dequant_matvec(v_w, v_s, v_b, normed)  # [512]

        # Split Q into query and gate
        q_proj = q_all.reshape(NUM_ATTN_HEADS, 2 * HEAD_DIM)
        q = q_proj[:, :HEAD_DIM].copy()  # [16, 256]
        q_gate = q_proj[:, HEAD_DIM:].copy()  # [16, 256]
        k = k.reshape(NUM_KV_HEADS, HEAD_DIM)
        v = v.reshape(NUM_KV_HEADS, HEAD_DIM)

        debug['q_rms'] = float(np.sqrt(np.mean(q**2)))
        debug['k_rms'] = float(np.sqrt(np.mean(k**2)))
        debug['v_rms'] = float(np.sqrt(np.mean(v**2)))
        debug['gate_mean'] = float(np.mean(q_gate))
        debug['gate_rms'] = float(np.sqrt(np.mean(q_gate**2)))

        # Q/K per-head norm
        for h in range(NUM_ATTN_HEADS):
            q[h] = rms_norm(q[h], q_norm_w)
        for h in range(NUM_KV_HEADS):
            k[h] = rms_norm(k[h], k_norm_w)

        # RoPE
        apply_rotary_emb(q, k, pos)

        # Update KV cache
        if kv_cache_k is None:
            kv_cache_k = []
            kv_cache_v = []
        kv_cache_k.append(k.copy())
        kv_cache_v.append(v.copy())

        # Scaled dot-product attention (GQA)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        heads_per_kv = NUM_ATTN_HEADS // NUM_KV_HEADS
        attn_out = np.zeros((NUM_ATTN_HEADS, HEAD_DIM), dtype=np.float32)

        for h in range(NUM_ATTN_HEADS):
            kv_h = h // heads_per_kv
            seq_len = len(kv_cache_k)
            scores = np.zeros(seq_len, dtype=np.float32)
            for p in range(seq_len):
                scores[p] = np.dot(q[h], kv_cache_k[p][kv_h]) * scale
            # Softmax
            scores = scores - np.max(scores)
            probs = np.exp(scores)
            probs = probs / np.sum(probs)
            for p in range(seq_len):
                attn_out[h] += probs[p] * kv_cache_v[p][kv_h]

        # Apply sigmoid gate
        flat_gate = q_gate.reshape(-1)
        sigmoid_gate = 1.0 / (1.0 + np.exp(-flat_gate))
        attn_flat = attn_out.reshape(-1)
        attn_flat *= sigmoid_gate
        debug['sigmoid_gate_mean'] = float(np.mean(sigmoid_gate))
        debug['attn_gated_rms'] = float(np.sqrt(np.mean(attn_flat**2)))
    else:
        # ---- Linear attention (GatedDeltaNet) ----
        # QKV/Z/Beta/Alpha projections
        qkv_w = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight")
        qkv_s = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_qkv.scales")
        qkv_b = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_qkv.biases")
        z_w = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_z.weight")
        z_s = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_z.scales")
        z_b = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_z.biases")
        a_w = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_a.weight")
        a_s = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_a.scales")
        a_b = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_a.biases")
        b_w = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_b.weight")
        b_s = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_b.scales")
        b_b = load_tensor(f"model.layers.{layer_idx}.linear_attn.in_proj_b.biases")
        conv_w_raw = load_tensor(f"model.layers.{layer_idx}.linear_attn.conv1d.weight")
        # conv1d weight shape: [8192, 1, 4] — flatten to 1D
        conv_w = conv_w_raw.reshape(-1)
        A_log = load_tensor(f"model.layers.{layer_idx}.linear_attn.A_log")
        dt_bias = load_tensor(f"model.layers.{layer_idx}.linear_attn.dt_bias")
        gated_norm_w = load_tensor(f"model.layers.{layer_idx}.linear_attn.norm.weight")
        out_w = load_tensor(f"model.layers.{layer_idx}.linear_attn.out_proj.weight")
        out_s = load_tensor(f"model.layers.{layer_idx}.linear_attn.out_proj.scales")
        out_b = load_tensor(f"model.layers.{layer_idx}.linear_attn.out_proj.biases")

        qkv_out = dequant_matvec(qkv_w, qkv_s, qkv_b, normed)  # [8192]
        z_out = dequant_matvec(z_w, z_s, z_b, normed)  # [4096]
        a_out = dequant_matvec(a_w, a_s, a_b, normed)  # [32]
        b_out = dequant_matvec(b_w, b_s, b_b, normed)  # [32]

        debug['qkv_rms'] = float(np.sqrt(np.mean(qkv_out**2)))
        debug['z_rms'] = float(np.sqrt(np.mean(z_out**2)))

        # Conv1d step
        if conv_state is None:
            conv_state = np.zeros((CONV_KERNEL_SIZE - 1, LINEAR_CONV_DIM), dtype=np.float32)

        # Build full window: [s1, s2, s3, new] for depthwise conv
        window = np.concatenate([conv_state, qkv_out.reshape(1, -1)], axis=0)  # [4, 8192]
        conv_out = np.zeros(LINEAR_CONV_DIM, dtype=np.float32)
        for c in range(LINEAR_CONV_DIM):
            acc = 0.0
            for k in range(CONV_KERNEL_SIZE):
                acc += window[k, c] * conv_w[c * CONV_KERNEL_SIZE + k]
            conv_out[c] = acc
        # SiLU activation
        conv_out = cpu_silu(conv_out)

        # Update conv state
        conv_state[:-1] = conv_state[1:]
        conv_state[-1] = qkv_out

        # Split into Q, K, V
        lin_q = conv_out[:LINEAR_TOTAL_KEY].copy()  # [2048]
        lin_k = conv_out[LINEAR_TOTAL_KEY:2*LINEAR_TOTAL_KEY].copy()  # [2048]
        lin_v = conv_out[2*LINEAR_TOTAL_KEY:].copy()  # [4096]

        # RMS norm + scale Q and K
        inv_scale = 1.0 / math.sqrt(LINEAR_KEY_DIM)
        q_heads = lin_q.reshape(LINEAR_NUM_K_HEADS, LINEAR_KEY_DIM)
        k_heads = lin_k.reshape(LINEAR_NUM_K_HEADS, LINEAR_KEY_DIM)
        v_heads = lin_v.reshape(LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM)

        for h in range(LINEAR_NUM_K_HEADS):
            rms = math.sqrt(np.mean(q_heads[h]**2) + RMS_NORM_EPS)
            q_heads[h] = q_heads[h] / rms * (inv_scale * inv_scale)
            rms = math.sqrt(np.mean(k_heads[h]**2) + RMS_NORM_EPS)
            k_heads[h] = k_heads[h] / rms * inv_scale

        # Delta-net recurrence
        k_heads_per_v = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS

        # Compute decay and beta
        g_decay = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
        beta_gate = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
        for vh in range(LINEAR_NUM_V_HEADS):
            A_val = math.exp(A_log[vh])
            softplus_val = math.log(1.0 + math.exp(a_out[vh] + dt_bias[vh]))
            g_decay[vh] = math.exp(-A_val * softplus_val)
            beta_gate[vh] = cpu_sigmoid(b_out[vh])

        # Recurrent update
        if ssm_state is None:
            ssm_state = np.zeros((LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM), dtype=np.float32)

        out_values = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)
        for vh in range(LINEAR_NUM_V_HEADS):
            kh = vh // k_heads_per_v
            S = ssm_state[vh]  # [128, 128]
            k_vec = k_heads[kh]  # [128]
            v_vec = v_heads[vh]  # [128]
            q_vec = q_heads[kh]  # [128]

            # Step 1: Decay
            S *= g_decay[vh]
            # Step 2: kv_mem = S @ k
            kv_mem = S @ k_vec  # [128]
            # Step 3: delta = (v - kv_mem) * beta
            delta = (v_vec - kv_mem) * beta_gate[vh]
            # Step 4: S += delta @ k^T
            S += np.outer(delta, k_vec)
            # Step 5: output = S @ q
            out_values[vh * LINEAR_VALUE_DIM:(vh + 1) * LINEAR_VALUE_DIM] = S @ q_vec

        debug['delta_out_rms'] = float(np.sqrt(np.mean(out_values**2)))

        # Gated RMS norm
        gated_out = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)
        for vh in range(LINEAR_NUM_V_HEADS):
            oh = out_values[vh * LINEAR_VALUE_DIM:(vh + 1) * LINEAR_VALUE_DIM]
            zh = z_out[vh * LINEAR_VALUE_DIM:(vh + 1) * LINEAR_VALUE_DIM]
            rms = math.sqrt(np.mean(oh**2) + RMS_NORM_EPS)
            silu_z = cpu_silu(zh)
            gated_out[vh * LINEAR_VALUE_DIM:(vh + 1) * LINEAR_VALUE_DIM] = oh / rms * gated_norm_w * silu_z

        debug['gated_out_rms'] = float(np.sqrt(np.mean(gated_out**2)))

        # Out projection
        attn_projected = dequant_matvec(out_w, out_s, out_b, gated_out)

    # Step 2: Residual add after attention
    if attn_projected is None:
        raise RuntimeError(f"attn_projected not set for layer {layer_idx}")
    hidden = residual + attn_projected
    debug['h_mid_rms'] = float(np.sqrt(np.mean(hidden**2)))

    # Step 3: Post-attention LayerNorm
    residual2 = hidden.copy()
    h_post = rms_norm(hidden, post_attn_norm_w)
    debug['h_post_rms'] = float(np.sqrt(np.mean(h_post**2)))

    # Step 4: MoE block
    # Shared expert
    sg_w = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight")
    sg_s = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.scales")
    sg_b = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.biases")
    su_w = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight")
    su_s = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.scales")
    su_b = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.biases")
    sd_w = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight")
    sd_s = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.scales")
    sd_b = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.biases")
    seg_w = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight")
    seg_s = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert_gate.scales")
    seg_b = load_tensor(f"model.layers.{layer_idx}.mlp.shared_expert_gate.biases")

    shared_gate = dequant_matvec(sg_w, sg_s, sg_b, h_post)  # [512]
    shared_up = dequant_matvec(su_w, su_s, su_b, h_post)  # [512]
    shared_act = cpu_swiglu(shared_gate, shared_up)  # [512]
    shared_out = dequant_matvec(sd_w, sd_s, sd_b, shared_act)  # [2048]

    # Shared expert gate (8-bit)
    seg_score = dequant_matvec(seg_w, seg_s, seg_b, h_post, bits=8)[0]  # scalar
    shared_out *= cpu_sigmoid(seg_score)

    debug['shared_out_rms'] = float(np.sqrt(np.mean(shared_out**2)))
    debug['shared_gate_score'] = float(seg_score)

    # For simplicity, skip routed experts (K=0 mode for comparison)
    moe_out = np.zeros(HIDDEN_DIM, dtype=np.float32)

    # Step 5: Residual add after MoE
    hidden = residual2 + shared_out + moe_out

    debug['output_rms'] = float(np.sqrt(np.mean(hidden**2)))

    return hidden, kv_cache_k, kv_cache_v, conv_state, ssm_state, debug


def main():
    parser = argparse.ArgumentParser(description='Compare layer output with engine')
    parser.add_argument('--layer', type=int, default=0, help='Layer to compute')
    parser.add_argument('--prompt', default='The capital of France is', help='Prompt text')
    args = parser.parse_args()

    # Get prompt tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    token_ids = tok.encode(args.prompt)
    print(f"Prompt: {args.prompt!r}")
    print(f"Tokens: {token_ids}")

    # Process each token through the specified layer
    hidden = None
    kv_cache_k, kv_cache_v = None, None
    conv_state, ssm_state = None, None

    for pos, tok_id in enumerate(token_ids):
        hidden = embed_lookup(tok_id)
        print(f"\n--- Token {pos} (id={tok_id}) ---")
        print(f"  Embedding rms: {np.sqrt(np.mean(hidden**2)):.6f}")

        for layer in range(args.layer + 1):
            hidden, kv_cache_k, kv_cache_v, conv_state, ssm_state, debug = \
                compute_layer_forward(hidden, layer, pos, kv_cache_k, kv_cache_v, conv_state, ssm_state)

            if layer == args.layer:
                print(f"\n  Layer {layer} debug:")
                for k, v in sorted(debug.items()):
                    print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    print(f"\n=== Final hidden state after layer {args.layer} ===")
    print(f"  rms: {np.sqrt(np.mean(hidden**2)):.6f}")
    print(f"  first 8: [{hidden[0]:.6f}, {hidden[1]:.6f}, {hidden[2]:.6f}, {hidden[3]:.6f}, "
          f"{hidden[4]:.6f}, {hidden[5]:.6f}, {hidden[6]:.6f}, {hidden[7]:.6f}]")


if __name__ == '__main__':
    main()
