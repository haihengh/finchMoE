#!/usr/bin/env python3
"""
Layer-by-layer differential debug: FinchMoE vs HF reference.
Compares at 4 probe points per layer to identify exact divergence location.

Probes:
  1. GDN/Attention output (after residual: x + attn_out)
  2. Shared expert output (before gating)
  3. Routed MoE output (weighted sum of top-K experts)
  4. Final layer output (x + attn_out + shared_out + routed_out)

Usage:
    python3 finchmoe/debug_layer_diff.py [--layer N] [--all-layers]
"""

import argparse, json, os, struct, sys, time
import numpy as np

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "Qwen3.6-35B-A3B-bf16")
Q8_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "models", "Qwen3.6-35B-A3B-8bit-custom")

# Architecture constants
HIDDEN_DIM = 2048
NUM_LAYERS = 40
FULL_ATTN_INTERVAL = 4
NUM_EXPERTS = 256
NUM_ACTIVE_EXPERTS = 8
MOE_INTERMEDIATE = 512
SHARED_INTERMEDIATE = 512
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
LINEAR_NUM_V_HEADS = 32
LINEAR_NUM_K_HEADS = 16
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
LINEAR_TOTAL_KEY = 2048
LINEAR_TOTAL_VALUE = 4096
LINEAR_CONV_DIM = 8192
CONV_KERNEL_SIZE = 4
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000000.0

# ============================================================================
# Weight loading
# ============================================================================

def load_tensors(model_dir):
    """Load all tensors from BF16 safetensors."""
    import torch
    from safetensors import safe_open
    tensors = {}
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith('.safetensors'):
            continue
        full = os.path.join(model_dir, fname)
        with safe_open(full, framework='pt') as st:
            for key in st.keys():
                tensors[key] = st.get_tensor(key)
    return tensors

def to_f32(t):
    import torch
    if t.dtype == torch.bfloat16:
        return t.to(torch.float32).numpy()
    return t.numpy()

def bf16_to_f32(bf16_val):
    return struct.unpack('f', struct.pack('I', int(bf16_val) << 16))[0]

# ============================================================================
# HF Reference implementation
# ============================================================================

def hf_rms_norm(x, weight, eps=RMS_NORM_EPS):
    """HF RMS norm: x * weight / rms(x). weight already includes the +1 adjustment."""
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)
    return (x / rms).astype(np.float32) * weight

def hf_linear_attn_forward(hidden, weights, conv_state, ssm_state, layer_idx):
    """HF reference implementation of linear attention."""
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."

    # QKV projection
    qkv_w = to_f32(weights[prefix + "in_proj_qkv.weight"])
    z_w = to_f32(weights[prefix + "in_proj_z.weight"])
    beta_w = to_f32(weights[prefix + "in_proj_b.weight"])
    alpha_w = to_f32(weights[prefix + "in_proj_a.weight"])

    qkv_out = qkv_w @ hidden  # [8192]
    z_out = z_w @ hidden      # [4096]
    beta_out = beta_w @ hidden # [32]
    alpha_out = alpha_w @ hidden # [32]

    # Conv1d
    conv_w = to_f32(weights[prefix + "conv1d.weight"]).squeeze()
    if conv_w.ndim == 1:
        conv_w = conv_w.reshape(-1, CONV_KERNEL_SIZE)  # handle [8192, 1, 4] squeeze -> [8192, 4]
    out = np.zeros(LINEAR_CONV_DIM, dtype=np.float32)
    for c in range(LINEAR_CONV_DIM):
        acc = 0.0
        for k in range(CONV_KERNEL_SIZE - 1):
            acc += conv_state[k * LINEAR_CONV_DIM + c] * conv_w[c, k]
        acc += qkv_out[c] * conv_w[c, CONV_KERNEL_SIZE - 1]
        out[c] = acc
    # SiLU
    conv_out = out / (1.0 + np.exp(-out))

    # Update conv state
    new_conv = np.zeros_like(conv_state)
    for k in range(CONV_KERNEL_SIZE - 2):
        new_conv[k * LINEAR_CONV_DIM:(k+1) * LINEAR_CONV_DIM] = \
            conv_state[(k+1) * LINEAR_CONV_DIM:(k+2) * LINEAR_CONV_DIM]
    new_conv[(CONV_KERNEL_SIZE - 2) * LINEAR_CONV_DIM:] = qkv_out

    # Split q, k, v
    lin_q = conv_out[:LINEAR_TOTAL_KEY].copy()
    lin_k = conv_out[LINEAR_TOTAL_KEY:2*LINEAR_TOTAL_KEY].copy()
    lin_v = conv_out[2*LINEAR_TOTAL_KEY:].copy()

    # HF: L2 normalize q and k per head
    for h in range(LINEAR_NUM_K_HEADS):
        start = h * LINEAR_KEY_DIM
        end = start + LINEAR_KEY_DIM
        q_norm = np.sqrt(np.sum(lin_q[start:end]**2) + 1e-6)
        k_norm = np.sqrt(np.sum(lin_k[start:end]**2) + 1e-6)
        lin_q[start:end] = lin_q[start:end] / q_norm / np.sqrt(LINEAR_KEY_DIM)
        lin_k[start:end] = lin_k[start:end] / k_norm

    # Compute g_decay and beta_gate
    A_log = to_f32(weights[prefix + "A_log"])
    dt_bias = to_f32(weights[prefix + "dt_bias"])

    g_decay = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
    beta_gate = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
    for vh in range(LINEAR_NUM_V_HEADS):
        softplus_val = np.log(1.0 + np.exp(float(alpha_out[vh]) + float(dt_bias[vh])))
        g_decay[vh] = np.exp(-np.exp(float(A_log[vh])) * softplus_val)
        beta_gate[vh] = 1.0 / (1.0 + np.exp(-float(beta_out[vh])))

    # GDN recurrence
    k_heads_per_v = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS
    output = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)

    for vh in range(LINEAR_NUM_V_HEADS):
        kh = vh // k_heads_per_v
        g = g_decay[vh]
        beta = beta_gate[vh]
        S = ssm_state[vh]
        k_h = lin_k[kh * LINEAR_KEY_DIM:(kh+1) * LINEAR_KEY_DIM]
        v_h = lin_v[vh * LINEAR_VALUE_DIM:(vh+1) * LINEAR_VALUE_DIM]
        q_h = lin_q[kh * LINEAR_KEY_DIM:(kh+1) * LINEAR_KEY_DIM]

        for vi in range(LINEAR_VALUE_DIM):
            # Decay + compute kv_mem
            kv_mem = 0.0
            for ki in range(LINEAR_KEY_DIM):
                s_val = S[vi, ki] * g
                S[vi, ki] = s_val
                kv_mem += s_val * k_h[ki]
            # Delta update
            delta = (v_h[vi] - kv_mem) * beta
            for ki in range(LINEAR_KEY_DIM):
                S[vi, ki] += k_h[ki] * delta

        for vi in range(LINEAR_VALUE_DIM):
            out_val = 0.0
            for ki in range(LINEAR_KEY_DIM):
                out_val += S[vi, ki] * q_h[ki]
            output[vh * LINEAR_VALUE_DIM + vi] = out_val

    # Gated RMS norm
    norm_w = to_f32(weights[prefix + "norm.weight"])
    gated_out = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)
    for vh in range(LINEAR_NUM_V_HEADS):
        start = vh * LINEAR_VALUE_DIM
        end = start + LINEAR_VALUE_DIM
        rms = np.sqrt(np.mean(output[start:end]**2) + 1e-6)
        normed = output[start:end] / rms
        silu_z = z_out[start:end] / (1.0 + np.exp(-z_out[start:end]))
        gated_out[start:end] = normed * silu_z * norm_w

    # Output projection
    out_w = to_f32(weights[prefix + "out_proj.weight"])
    attn_out = out_w @ gated_out  # [2048]

    return attn_out, new_conv, ssm_state

def hf_full_attn_forward(hidden, weights, kv_cache, layer_idx, pos):
    """HF reference for full attention (simplified: single token, cached KV)."""
    prefix = f"model.language_model.layers.{layer_idx}.self_attn."

    q_w = to_f32(weights[prefix + "q_proj.weight"])
    k_w = to_f32(weights[prefix + "k_proj.weight"])
    v_w = to_f32(weights[prefix + "v_proj.weight"])
    o_w = to_f32(weights[prefix + "o_proj.weight"])

    q_norm_w = to_f32(weights[prefix + "q_norm.weight"])
    k_norm_w = to_f32(weights[prefix + "k_norm.weight"])

    # QKV projections
    q_proj = q_w @ hidden  # [num_heads * head_dim * 2]
    k_out = k_w @ hidden   # [num_kv_heads * head_dim]
    v_out = v_w @ hidden   # [num_kv_heads * head_dim]

    # Split Q and gate
    q = q_proj[:NUM_ATTN_HEADS * HEAD_DIM].copy()
    gate = q_proj[NUM_ATTN_HEADS * HEAD_DIM:].copy()

    # Q norm (per head)
    for h in range(NUM_ATTN_HEADS):
        start = h * HEAD_DIM
        end = start + HEAD_DIM
        rms = np.sqrt(np.mean(q[start:end]**2) + RMS_NORM_EPS)
        q[start:end] = q[start:end] / rms * q_norm_w

    # K norm (per head)
    k = k_out.reshape(NUM_KV_HEADS, HEAD_DIM)
    for h in range(NUM_KV_HEADS):
        rms = np.sqrt(np.mean(k[h]**2) + RMS_NORM_EPS)
        k[h] = k[h] / rms * k_norm_w
    k = k.flatten()

    # RoPE (simplified)
    PARTIAL = 0.25
    rotary_dim = int(HEAD_DIM * PARTIAL)
    half = rotary_dim // 2

    def apply_rope(x, num_heads, pos):
        x = x.reshape(num_heads, HEAD_DIM)
        for h in range(num_heads):
            for i in range(half):
                freq = 1.0 / (ROPE_THETA ** (2.0 * i / rotary_dim))
                angle = pos * freq
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                x0, x1 = x[h, i], x[h, i + half]
                x[h, i] = x0 * cos_a - x1 * sin_a
                x[h, i + half] = x0 * sin_a + x1 * cos_a
        return x.flatten()

    q = apply_rope(q, NUM_ATTN_HEADS, pos)
    k = apply_rope(k, NUM_KV_HEADS, pos)

    # Update KV cache
    cache_pos = kv_cache['len']
    kv_cache['k'][cache_pos] = k.copy()
    kv_cache['v'][cache_pos] = v_out.copy()
    kv_cache['len'] += 1

    # Attention
    scale = 1.0 / np.sqrt(HEAD_DIM)
    heads_per_kv = NUM_ATTN_HEADS // NUM_KV_HEADS
    attn_out = np.zeros(NUM_ATTN_HEADS * HEAD_DIM, dtype=np.float32)

    for h in range(NUM_ATTN_HEADS):
        kv_h = h // heads_per_kv
        qh = q[h * HEAD_DIM:(h+1) * HEAD_DIM]

        scores = np.zeros(kv_cache['len'], dtype=np.float32)
        for p in range(kv_cache['len']):
            kp = kv_cache['k'][p][kv_h * HEAD_DIM:(kv_h+1) * HEAD_DIM]
            scores[p] = np.dot(qh, kp) * scale

        # Softmax
        scores -= scores.max()
        scores = np.exp(scores)
        scores /= scores.sum()

        oh = attn_out[h * HEAD_DIM:(h+1) * HEAD_DIM]
        for p in range(kv_cache['len']):
            vp = kv_cache['v'][p][kv_h * HEAD_DIM:(kv_h+1) * HEAD_DIM]
            oh += scores[p] * vp

    # Sigmoid gate
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
    attn_out *= sigmoid_gate

    # Output projection
    attn_out = o_w @ attn_out  # [2048]

    return attn_out

def hf_moe_forward(hidden, weights, layer_idx):
    """HF reference MoE forward pass for one layer."""
    moe_prefix = f"model.language_model.layers.{layer_idx}.mlp."

    # Router gate
    gate_w = to_f32(weights[moe_prefix + "gate.weight"])
    gate_scores = gate_w @ hidden  # [256]

    # Softmax
    gate_scores -= gate_scores.max()
    gate_scores = np.exp(gate_scores)
    gate_scores /= gate_scores.sum()

    # Top-K
    K = NUM_ACTIVE_EXPERTS
    top_k_idx = np.argpartition(gate_scores, -K)[-K:]
    top_k_vals = gate_scores[top_k_idx]
    top_k_vals = top_k_vals / top_k_vals.sum()  # normalize

    # Routed experts
    routed_out = np.zeros(HIDDEN_DIM, dtype=np.float32)

    gate_up_w = to_f32(weights[moe_prefix + "experts.gate_up_proj"])  # [256, 1024, 2048]
    down_w = to_f32(weights[moe_prefix + "experts.down_proj"])  # [256, 2048, 512]

    for i, expert_id in enumerate(top_k_idx):
        weight = top_k_vals[i]
        e_gate_w = gate_up_w[expert_id, :512, :]  # gate half
        e_up_w = gate_up_w[expert_id, 512:, :]    # up half
        e_down_w = down_w[expert_id]               # down

        gate_out = e_gate_w @ hidden
        up_out = e_up_w @ hidden
        act = gate_out / (1.0 + np.exp(-gate_out)) * up_out  # SwiGLU
        expert_out = e_down_w @ act
        routed_out += weight * expert_out

    # Shared expert
    seg_w = to_f32(weights[moe_prefix + "shared_expert_gate.weight"])
    sg_w = to_f32(weights[moe_prefix + "shared_expert.gate_proj.weight"])
    su_w = to_f32(weights[moe_prefix + "shared_expert.up_proj.weight"])
    sd_w = to_f32(weights[moe_prefix + "shared_expert.down_proj.weight"])

    shared_gate_score = float((seg_w.reshape(-1) @ hidden))
    shared_weight = 1.0 / (1.0 + np.exp(-shared_gate_score))  # sigmoid

    shared_gate = sg_w @ hidden
    shared_up = su_w @ hidden
    shared_act = shared_gate / (1.0 + np.exp(-shared_gate)) * shared_up
    shared_out = sd_w @ shared_act * shared_weight

    return routed_out, shared_out, top_k_idx, top_k_vals

def hf_full_layer_forward(hidden, weights, conv_states, ssm_states, kv_caches, layer_idx, pos):
    """HF reference: full layer forward pass with probe points."""
    prefix = f"model.language_model.layers.{layer_idx}."

    # ---- Pre-attention norm ----
    input_norm_w = to_f32(weights[prefix + "input_layernorm.weight"])
    normed = hf_rms_norm(hidden, input_norm_w)

    # ---- Attention ----
    is_full = ((layer_idx + 1) % FULL_ATTN_INTERVAL == 0)

    if is_full:
        attn_out = hf_full_attn_forward(normed, weights, kv_caches[layer_idx], layer_idx, pos)
    else:
        attn_out, new_conv, new_ssm = hf_linear_attn_forward(
            normed, weights, conv_states[layer_idx], ssm_states[layer_idx], layer_idx)
        conv_states[layer_idx] = new_conv
        ssm_states[layer_idx] = new_ssm

    # Probe 1: after attention residual
    probe1 = hidden + attn_out

    # ---- Post-attention norm ----
    post_norm_w = to_f32(weights[prefix + "post_attention_layernorm.weight"])
    h_post = hf_rms_norm(probe1, post_norm_w)

    # ---- MoE ----
    routed_out, shared_out, top_k, top_w = hf_moe_forward(h_post, weights, layer_idx)

    # Probe 2: shared expert output
    probe2 = shared_out.copy()
    # Probe 3: routed MoE output
    probe3 = routed_out.copy()

    # Probe 4: final layer output
    probe4 = probe1 + routed_out + shared_out

    return {
        'probe1': probe1,
        'probe2': probe2,
        'probe3': probe3,
        'probe4': probe4,
        'attn_out': attn_out,
        'routed_out': routed_out,
        'shared_out': shared_out,
        'top_k': top_k,
        'top_w': top_w,
    }


# ============================================================================
# FinchMoE C-engine simulator (matches infer.m exactly)
# ============================================================================

def finchmoe_rms_norm(x, weight, eps=RMS_NORM_EPS):
    """FinchMoE RMS norm: x * weight / rms(x)"""
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)
    return (x / rms).astype(np.float32) * weight

def finchmoe_conv1d_step(conv_state, new_input, conv_weight, channels):
    """Exact match of C cpu_conv1d_step."""
    out = np.zeros(channels, dtype=np.float32)
    for c in range(channels):
        acc = 0.0
        for k in range(CONV_KERNEL_SIZE - 1):
            acc += conv_state[k * channels + c] * conv_weight[c, k]
        acc += new_input[c] * conv_weight[c, CONV_KERNEL_SIZE - 1]
        out[c] = acc
    # SiLU
    out = out / (1.0 + np.exp(-out))
    # Shift state
    for c in range(channels):
        conv_state[0 * channels + c] = conv_state[1 * channels + c]
        conv_state[1 * channels + c] = conv_state[2 * channels + c]
        conv_state[2 * channels + c] = new_input[c]
    return out

def finchmoe_linear_forward(hidden, weights, conv_state, ssm_state, layer_idx):
    """Exact match of C linear attention (cpu_linear_attn_layer)."""
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."

    qkv_w = to_f32(weights[prefix + "in_proj_qkv.weight"])
    z_w = to_f32(weights[prefix + "in_proj_z.weight"])
    beta_w = to_f32(weights[prefix + "in_proj_b.weight"])
    alpha_w = to_f32(weights[prefix + "in_proj_a.weight"])

    qkv_out = qkv_w @ hidden
    z_out = z_w @ hidden
    beta_out = beta_w @ hidden
    alpha_out = alpha_w @ hidden

    conv_w = to_f32(weights[prefix + "conv1d.weight"]).squeeze()
    conv_out = finchmoe_conv1d_step(conv_state, qkv_out, conv_w, LINEAR_CONV_DIM)

    lin_q = conv_out[:LINEAR_TOTAL_KEY].copy()
    lin_k = conv_out[LINEAR_TOTAL_KEY:2*LINEAR_TOTAL_KEY].copy()
    lin_v = conv_out[2*LINEAR_TOTAL_KEY:].copy()

    # RMS norm + scaling (matching C code: q *= 1/128, k *= 1/sqrt(128))
    inv_scale = 1.0 / np.sqrt(LINEAR_KEY_DIM)
    q_scale = 1.0 / LINEAR_KEY_DIM
    for h in range(LINEAR_NUM_K_HEADS):
        start = h * LINEAR_KEY_DIM
        end = start + LINEAR_KEY_DIM
        rms_q = np.sqrt(np.mean(lin_q[start:end]**2) + 1e-6)
        rms_k = np.sqrt(np.mean(lin_k[start:end]**2) + 1e-6)
        lin_q[start:end] = lin_q[start:end] / rms_q * q_scale
        lin_k[start:end] = lin_k[start:end] / rms_k * inv_scale

    # g_decay and beta_gate
    A_log = to_f32(weights[prefix + "A_log"])
    dt_bias = to_f32(weights[prefix + "dt_bias"])

    g_decay = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
    beta_gate = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
    for vh in range(LINEAR_NUM_V_HEADS):
        softplus_val = np.log(1.0 + np.exp(float(alpha_out[vh]) + float(dt_bias[vh])))
        g_decay[vh] = np.exp(-np.exp(float(A_log[vh])) * softplus_val)
        beta_gate[vh] = 1.0 / (1.0 + np.exp(-float(beta_out[vh])))

    # GDN recurrence
    k_heads_per_v = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS
    output = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)

    for vh in range(LINEAR_NUM_V_HEADS):
        kh = vh // k_heads_per_v
        g = g_decay[vh]
        beta = beta_gate[vh]
        S = ssm_state[vh]
        k_h = lin_k[kh * LINEAR_KEY_DIM:(kh+1) * LINEAR_KEY_DIM]
        v_h = lin_v[vh * LINEAR_VALUE_DIM:(vh+1) * LINEAR_VALUE_DIM]
        q_h = lin_q[kh * LINEAR_KEY_DIM:(kh+1) * LINEAR_KEY_DIM]

        for vi in range(LINEAR_VALUE_DIM):
            kv_mem = 0.0
            for ki in range(LINEAR_KEY_DIM):
                s_val = S[vi, ki] * g
                S[vi, ki] = s_val
                kv_mem += s_val * k_h[ki]
            delta = (v_h[vi] - kv_mem) * beta
            for ki in range(LINEAR_KEY_DIM):
                S[vi, ki] += k_h[ki] * delta

        for vi in range(LINEAR_VALUE_DIM):
            out_val = 0.0
            for ki in range(LINEAR_KEY_DIM):
                out_val += S[vi, ki] * q_h[ki]
            output[vh * LINEAR_VALUE_DIM + vi] = out_val

    # Gated RMS norm
    norm_w = to_f32(weights[prefix + "norm.weight"])
    gated_out = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)
    for vh in range(LINEAR_NUM_V_HEADS):
        start = vh * LINEAR_VALUE_DIM
        end = start + LINEAR_VALUE_DIM
        rms = np.sqrt(np.mean(output[start:end]**2) + 1e-6)
        normed = output[start:end] / rms
        silu_z = z_out[start:end] / (1.0 + np.exp(-z_out[start:end]))
        gated_out[start:end] = normed * silu_z * norm_w

    # Output projection
    out_w = to_f32(weights[prefix + "out_proj.weight"])
    attn_out = out_w @ gated_out  # [2048]

    return attn_out


# ============================================================================
# Main comparison
# ============================================================================

def compare_layer(layer_idx, hidden, weights, conv_states_f, ssm_states_f,
                   conv_states_hf, ssm_states_hf, kv_caches_f, kv_caches_hf, pos, verbose=True):
    """Compare FinchMoE vs HF for one layer, returning probe point comparisons."""
    prefix = f"model.language_model.layers.{layer_idx}."
    is_full = ((layer_idx + 1) % FULL_ATTN_INTERVAL == 0)

    # ---- FinchMoE path ----
    # Pre-attention norm
    input_norm_w = to_f32(weights[prefix + "input_layernorm.weight"])
    normed_f = finchmoe_rms_norm(hidden, input_norm_w)

    if is_full:
        # Full attention (use same implementation for both — algorithm matches)
        attn_out_f = hf_full_attn_forward(normed_f, weights, kv_caches_f[layer_idx], layer_idx, pos)
    else:
        attn_out_f = finchmoe_linear_forward(
            normed_f, weights, conv_states_f[layer_idx], ssm_states_f[layer_idx], layer_idx)

    probe1_f = hidden + attn_out_f

    # Post-attention norm
    post_norm_w = to_f32(weights[prefix + "post_attention_layernorm.weight"])
    h_post_f = finchmoe_rms_norm(probe1_f, post_norm_w)

    # MoE forward (same weights, same algorithm)
    routed_out_f, shared_out_f, top_k_f, top_w_f = hf_moe_forward(h_post_f, weights, layer_idx)

    probe2_f = shared_out_f.copy()
    probe3_f = routed_out_f.copy()
    probe4_f = probe1_f + routed_out_f + shared_out_f

    # ---- HF Reference path ----
    normed_hf = hf_rms_norm(hidden, input_norm_w)

    if is_full:
        attn_out_hf = hf_full_attn_forward(normed_hf, weights, kv_caches_hf[layer_idx], layer_idx, pos)
    else:
        attn_out_hf, _, _ = hf_linear_attn_forward(
            normed_hf, weights, conv_states_hf[layer_idx], ssm_states_hf[layer_idx], layer_idx)

    probe1_hf = hidden + attn_out_hf

    h_post_hf = hf_rms_norm(probe1_hf, post_norm_w)
    routed_out_hf, shared_out_hf, _, _ = hf_moe_forward(h_post_hf, weights, layer_idx)

    probe2_hf = shared_out_hf.copy()
    probe3_hf = routed_out_hf.copy()
    probe4_hf = probe1_hf + routed_out_hf + shared_out_hf

    # ---- Compare ----
    def compare(name, a, b):
        diff = np.abs(a - b)
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        rms_a = np.sqrt(np.mean(a**2))
        rms_b = np.sqrt(np.mean(b**2))
        return {
            'name': name,
            'max_diff': float(diff.max()),
            'mean_diff': float(diff.mean()),
            'cos_sim': float(cos_sim),
            'rms_a': float(rms_a),
            'rms_b': float(rms_b),
        }

    results = [
        compare('Probe1 (attn residual)', probe1_f, probe1_hf),
        compare('Probe2 (shared expert)', probe2_f, probe2_hf),
        compare('Probe3 (routed MoE)', probe3_f, probe3_hf),
        compare('Probe4 (final output)', probe4_f, probe4_hf),
    ]

    if verbose:
        for r in results:
            status = "✅" if r['cos_sim'] > 0.9999 else ("⚠️" if r['cos_sim'] > 0.999 else "❌")
            print(f"  {status} {r['name']}: cos={r['cos_sim']:.6f}, max_diff={r['max_diff']:.6f}, "
                  f"rms=({r['rms_a']:.4f},{r['rms_b']:.4f})")

    return results, probe4_f, top_k_f, top_w_f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=0, help='Single layer to test')
    parser.add_argument('--all-layers', action='store_true')
    parser.add_argument('--num-layers', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("Loading BF16 model weights...")
    weights = load_tensors(MODEL_DIR)
    print(f"Loaded {len(weights)} tensors")

    # Initialize states
    conv_states_f = [np.zeros((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, dtype=np.float32)
                      for _ in range(NUM_LAYERS)]
    ssm_states_f = [np.zeros((LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM), dtype=np.float32)
                     for _ in range(NUM_LAYERS)]
    conv_states_hf = [np.zeros((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, dtype=np.float32)
                       for _ in range(NUM_LAYERS)]
    ssm_states_hf = [np.zeros((LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM), dtype=np.float32)
                      for _ in range(NUM_LAYERS)]

    # KV caches (separate for each path)
    kv_caches_f = [None] * NUM_LAYERS
    kv_caches_hf = [None] * NUM_LAYERS
    for i in range(NUM_LAYERS):
        if (i + 1) % FULL_ATTN_INTERVAL == 0:
            kv_caches_f[i] = {
                'k': np.zeros((64, NUM_KV_HEADS * HEAD_DIM), dtype=np.float32),
                'v': np.zeros((64, NUM_KV_HEADS * HEAD_DIM), dtype=np.float32),
                'len': 0,
            }
            kv_caches_hf[i] = {
                'k': np.zeros((64, NUM_KV_HEADS * HEAD_DIM), dtype=np.float32),
                'v': np.zeros((64, NUM_KV_HEADS * HEAD_DIM), dtype=np.float32),
                'len': 0,
            }

    # Random input
    np.random.seed(args.seed)
    hidden = np.random.randn(HIDDEN_DIM).astype(np.float32) * 0.1

    layers_to_test = range(args.num_layers) if args.all_layers else [args.layer]

    print(f"\n{'='*70}")
    print(f"Layer-by-layer comparison: FinchMoE (C-sim) vs HF Reference")
    print(f"Input: random (seed={args.seed}), K={NUM_ACTIVE_EXPERTS} experts")
    print(f"{'='*70}")

    all_results = []
    first_divergence = None

    for layer_idx in layers_to_test:
        is_full = ((layer_idx + 1) % FULL_ATTN_INTERVAL == 0)
        layer_type = "FULL-ATTN" if is_full else "LINEAR"

        print(f"\n--- Layer {layer_idx} ({layer_type}) ---")

        results, hidden, top_k, top_w = compare_layer(
            layer_idx, hidden, weights,
            conv_states_f, ssm_states_f,
            conv_states_hf, ssm_states_hf,
            kv_caches_f, kv_caches_hf, pos=0, verbose=True
        )
        all_results.append(results)

        if first_divergence is None:
            for r in results:
                if r['cos_sim'] < 0.999:
                    first_divergence = (layer_idx, r['name'], r['cos_sim'])
                    print(f"  🔴 FIRST DIVERGENCE at Layer {layer_idx}, {r['name']}: cos={r['cos_sim']:.6f}")
                    break

    if first_divergence:
        print(f"\n🔴 First divergence: Layer {first_divergence[0]}, {first_divergence[1]}, "
              f"cos_sim={first_divergence[2]:.6f}")
    else:
        print(f"\n✅ All layers pass (cos_sim > 0.999 for all probes)")


if __name__ == '__main__':
    main()
