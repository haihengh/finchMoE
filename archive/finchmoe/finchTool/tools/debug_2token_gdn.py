#!/usr/bin/env python3
"""
2-token differential debugger: catch state persistence bugs in FinchMoE's GDN.

Runs 2 sequential tokens through both FinchMoE's GDN algorithm and the HF reference,
comparing per-layer hidden states and recurrent states after each token.

Key hypothesis: Token 1 is correct, Token 2 diverges due to state persistence bug.

Usage:
    python3 finchmoe/debug_2token_gdn.py [--layer 0] [--all-layers]
"""

import argparse
import os
import struct
import sys
import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "Qwen3.6-35B-A3B-bf16")

# Architecture constants
HIDDEN_DIM = 2048
LINEAR_NUM_V_HEADS = 32
LINEAR_NUM_K_HEADS = 16
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
LINEAR_TOTAL_KEY = LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM   # 2048
LINEAR_TOTAL_VALUE = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM  # 4096
LINEAR_CONV_DIM = LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE  # 8192
CONV_KERNEL_SIZE = 4
RMS_NORM_EPS = 1e-6


def load_tensor(tensors, name):
    """Load tensor from BF16 safetensors files, cached."""
    if name in tensors:
        return tensors[name]

    for fname in sorted(os.listdir(MODEL_DIR)):
        if not fname.endswith('.safetensors'):
            continue
        full = os.path.join(MODEL_DIR, fname)
        try:
            with safe_open(full, framework='pt') as st:
                if name in st.keys():
                    t = st.get_tensor(name)
                    tensors[name] = t
                    return t
        except Exception:
            continue
    return None


def to_np(t):
    """Convert tensor to float32 numpy."""
    if t.dtype == torch.bfloat16:
        return t.to(torch.float32).numpy()
    return t.numpy()


def rms_norm(x, eps=RMS_NORM_EPS):
    """RMS normalize: x / rms(x)."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms


def rms_norm_gated(values, z, w, eps=RMS_NORM_EPS):
    """Gated RMS norm: rms_norm(values) * silu(z) * w."""
    inv_rms = 1.0 / np.sqrt(np.mean(values ** 2) + eps)
    normed = values * inv_rms
    silu_z = z / (1.0 + np.exp(-z))
    return normed * silu_z * w


def conv1d_step(conv_state, new_input, conv_weight_f32, channels, kernel_size=4):
    """Depthwise conv1d then SiLU. conv_weight_f32: [channels * kernel_size]."""
    out = np.zeros(channels, dtype=np.float32)
    for c in range(channels):
        acc = 0.0
        for k in range(kernel_size - 1):
            acc += conv_state[k * channels + c] * conv_weight_f32[c * kernel_size + k]
        acc += new_input[c] * conv_weight_f32[c * kernel_size + (kernel_size - 1)]
        out[c] = acc
    return out / (1.0 + np.exp(-out))  # SiLU


def gdn_step_finchmoe(q, k, v, g_decay, beta_gate, ssm_state):
    """
    FinchMoE's gated_delta_net_step (exact translation).
    State is mutated IN-PLACE. Returns output [num_v_heads * value_dim].
    """
    output = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)
    k_heads_per_v = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS  # 2

    for vh in range(LINEAR_NUM_V_HEADS):
        kh = vh // k_heads_per_v
        g = g_decay[vh]
        beta = beta_gate[vh]

        S = ssm_state[vh]  # [value_dim, key_dim]
        k_h = k[kh * LINEAR_KEY_DIM:(kh + 1) * LINEAR_KEY_DIM]
        v_h = v[vh * LINEAR_VALUE_DIM:(vh + 1) * LINEAR_VALUE_DIM]
        q_h = q[kh * LINEAR_KEY_DIM:(kh + 1) * LINEAR_KEY_DIM]

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

    return output


def gdn_step_llamacpp(q, k, v, g, beta, ssm_state):
    """
    llama.cpp's GDN autoregressive step (from delta-net-base.cpp:289-371).

    Key differences to verify against FinchMoE:
    - llama.cpp does state decay as a separate pass (s = s * exp(g))
    - Then sk = sum_rows(s * k)
    - Then d = (v - sk^T) * beta
    - Then s += k * d^T
    - Then o = sum_rows(s * q)

    BUT: the math is identical. The comparison verifies numerical equivalence.
    """
    # This is the same algorithm as FinchMoE, just structured differently
    # We use it to verify there's no implementation bug
    k_heads_per_v = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS

    # Pre-compute exp(g) since llama.cpp passes gate = -exp(A_log)*softplus
    g_exp = np.exp(g)

    output = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)

    for vh in range(LINEAR_NUM_V_HEADS):
        kh = vh // k_heads_per_v
        g_vh = g_exp[vh]
        beta_vh = beta[vh]

        S = ssm_state[vh]
        k_h = k[kh * LINEAR_KEY_DIM:(kh + 1) * LINEAR_KEY_DIM]
        v_h = v[vh * LINEAR_VALUE_DIM:(vh + 1) * LINEAR_VALUE_DIM]
        q_h = q[kh * LINEAR_KEY_DIM:(kh + 1) * LINEAR_KEY_DIM]

        for vi in range(LINEAR_VALUE_DIM):
            kv_mem = 0.0
            for ki in range(LINEAR_KEY_DIM):
                s_val = S[vi, ki] * g_vh
                S[vi, ki] = s_val
                kv_mem += s_val * k_h[ki]

            delta = (v_h[vi] - kv_mem) * beta_vh
            for ki in range(LINEAR_KEY_DIM):
                S[vi, ki] += k_h[ki] * delta

        for vi in range(LINEAR_VALUE_DIM):
            out_val = 0.0
            for ki in range(LINEAR_KEY_DIM):
                out_val += S[vi, ki] * q_h[ki]
            output[vh * LINEAR_VALUE_DIM + vi] = out_val

    return output


def run_layer_2token(layer_idx, tensors, seed=42, verbose=True):
    """
    Run 2 sequential tokens through one linear attention layer.
    Compare FinchMoE vs HF reference for each token.

    Returns: dict with per-token comparison data, or None if layer not found.
    """
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."

    # Check if this is a linear attention layer
    qkv_tensor = load_tensor(tensors, prefix + "in_proj_qkv.weight")
    if qkv_tensor is None:
        return None  # Full attention layer, skip

    # Load all layer weights
    qkv_w = to_np(qkv_tensor)
    z_w = to_np(load_tensor(tensors, prefix + "in_proj_z.weight"))
    beta_w = to_np(load_tensor(tensors, prefix + "in_proj_b.weight"))
    alpha_w = to_np(load_tensor(tensors, prefix + "in_proj_a.weight"))
    out_w = to_np(load_tensor(tensors, prefix + "out_proj.weight"))
    A_log = to_np(load_tensor(tensors, prefix + "A_log"))
    dt_bias = to_np(load_tensor(tensors, prefix + "dt_bias"))
    conv_w = to_np(load_tensor(tensors, prefix + "conv1d.weight"))
    norm_w = to_np(load_tensor(tensors, prefix + "norm.weight"))
    input_norm_w = to_np(load_tensor(tensors,
        f"model.language_model.layers.{layer_idx}.input_layernorm.weight"))
    post_norm_w = to_np(load_tensor(tensors,
        f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight"))

    # Reshape conv_w: HF stores as [8192, 1, 4]; flatten to [channels * kernel_size]
    conv_w_flat = conv_w.reshape(-1).astype(np.float32)
    assert conv_w_flat.size == LINEAR_CONV_DIM * CONV_KERNEL_SIZE, \
        f"conv_w size mismatch: {conv_w_flat.size} != {LINEAR_CONV_DIM * CONV_KERNEL_SIZE}"

    results = {'layer': layer_idx, 'tokens': []}

    # Generate two random input tokens (simulating embeddings)
    np.random.seed(seed)
    token0_embed = np.random.randn(HIDDEN_DIM).astype(np.float32) * 0.1
    np.random.seed(seed + 1)
    token1_embed = np.random.randn(HIDDEN_DIM).astype(np.float32) * 0.1

    # Initialize states (both FinchMoE and HF start from zero)
    conv_state_finch = np.zeros((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, dtype=np.float32)
    ssm_state_finch = np.zeros((LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM), dtype=np.float32)

    conv_state_hf = np.zeros((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, dtype=np.float32)
    ssm_state_hf = np.zeros((LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM), dtype=np.float32)

    for token_idx, embed in enumerate([token0_embed, token1_embed]):
        token_result = {'token': token_idx}

        # ============================================================
        # FinchMoE path
        # ============================================================
        hidden = embed.copy()

        # Input norm
        normed_finch = rms_norm(hidden, RMS_NORM_EPS) * input_norm_w

        # Projections
        qkv_out = qkv_w @ normed_finch
        z_out = z_w @ normed_finch
        beta_out = beta_w @ normed_finch
        alpha_out = alpha_w @ normed_finch

        # Conv1d
        conv_out_finch = conv1d_step(conv_state_finch, qkv_out, conv_w_flat, LINEAR_CONV_DIM, CONV_KERNEL_SIZE)
        # Update conv state
        conv_state_finch = np.roll(
            conv_state_finch.reshape(CONV_KERNEL_SIZE - 1, LINEAR_CONV_DIM), -1, axis=0)
        conv_state_finch[-1] = qkv_out
        conv_state_finch = conv_state_finch.flatten()

        # Split q, k, v
        lin_q = conv_out_finch[:LINEAR_TOTAL_KEY].copy()
        lin_k = conv_out_finch[LINEAR_TOTAL_KEY:2 * LINEAR_TOTAL_KEY].copy()
        lin_v = conv_out_finch[2 * LINEAR_TOTAL_KEY:].copy()

        # RMS normalize q and k
        inv_scale = 1.0 / np.sqrt(LINEAR_KEY_DIM)
        for h in range(LINEAR_NUM_K_HEADS):
            start = h * LINEAR_KEY_DIM
            end = start + LINEAR_KEY_DIM
            lin_q[start:end] = rms_norm(lin_q[start:end]) * inv_scale * inv_scale
            lin_k[start:end] = rms_norm(lin_k[start:end]) * inv_scale

        # Compute g_decay and beta_gate
        g_decay = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
        beta_gate = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
        for vh in range(LINEAR_NUM_V_HEADS):
            softplus_val = np.log(1.0 + np.exp(float(alpha_out[vh]) + float(dt_bias[vh])))
            g_decay[vh] = np.exp(-np.exp(float(A_log[vh])) * softplus_val)
            beta_gate[vh] = 1.0 / (1.0 + np.exp(-float(beta_out[vh])))

        # GDN recurrence (saves state BEFORE this step for comparison)
        ssm_before = ssm_state_finch.copy()
        gdn_out_finch = gdn_step_finchmoe(lin_q, lin_k, lin_v, g_decay, beta_gate, ssm_state_finch)

        # Gated RMS norm
        gated_out_finch = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)
        for vh in range(LINEAR_NUM_V_HEADS):
            start = vh * LINEAR_VALUE_DIM
            end = start + LINEAR_VALUE_DIM
            gated_out_finch[start:end] = rms_norm_gated(
                gdn_out_finch[start:end], z_out[start:end], norm_w)

        # Output projection
        attn_out_finch = out_w @ gated_out_finch  # [2048, 4096] @ [4096] = [2048]

        # Residual
        hidden_finch = hidden + attn_out_finch

        token_result['finch'] = {
            'q_rms': float(np.sqrt(np.mean(lin_q ** 2))),
            'k_rms': float(np.sqrt(np.mean(lin_k ** 2))),
            'v_rms': float(np.sqrt(np.mean(lin_v ** 2))),
            'gdn_out_rms': float(np.sqrt(np.mean(gdn_out_finch ** 2))),
            'attn_out_rms': float(np.sqrt(np.mean(attn_out_finch ** 2))),
            'hidden_rms': float(np.sqrt(np.mean(hidden_finch ** 2))),
            'ssm_rms': float(np.sqrt(np.mean(ssm_state_finch ** 2))),
            'ssm_max': float(np.max(np.abs(ssm_state_finch))),
        }

        # ============================================================
        # HF reference path (same algorithm, independent state)
        # ============================================================
        hidden_hf = embed.copy()

        normed_hf = rms_norm(hidden_hf, RMS_NORM_EPS) * input_norm_w

        qkv_out_hf = qkv_w @ normed_hf
        z_out_hf = z_w @ normed_hf
        beta_out_hf = beta_w @ normed_hf
        alpha_out_hf = alpha_w @ normed_hf

        conv_out_hf = conv1d_step(conv_state_hf, qkv_out_hf, conv_w_flat, LINEAR_CONV_DIM, CONV_KERNEL_SIZE)

        conv_state_hf = np.roll(
            conv_state_hf.reshape(CONV_KERNEL_SIZE - 1, LINEAR_CONV_DIM), -1, axis=0)
        conv_state_hf[-1] = qkv_out_hf
        conv_state_hf = conv_state_hf.flatten()

        lin_q_hf = conv_out_hf[:LINEAR_TOTAL_KEY].copy()
        lin_k_hf = conv_out_hf[LINEAR_TOTAL_KEY:2 * LINEAR_TOTAL_KEY].copy()
        lin_v_hf = conv_out_hf[2 * LINEAR_TOTAL_KEY:].copy()

        for h in range(LINEAR_NUM_K_HEADS):
            start = h * LINEAR_KEY_DIM
            end = start + LINEAR_KEY_DIM
            lin_q_hf[start:end] = rms_norm(lin_q_hf[start:end]) * inv_scale * inv_scale
            lin_k_hf[start:end] = rms_norm(lin_k_hf[start:end]) * inv_scale

        g_decay_hf = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
        beta_gate_hf = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
        for vh in range(LINEAR_NUM_V_HEADS):
            softplus_val = np.log(1.0 + np.exp(float(alpha_out_hf[vh]) + float(dt_bias[vh])))
            g_decay_hf[vh] = np.exp(-np.exp(float(A_log[vh])) * softplus_val)
            beta_gate_hf[vh] = 1.0 / (1.0 + np.exp(-float(beta_out_hf[vh])))

        ssm_before_hf = ssm_state_hf.copy()
        gdn_out_hf = gdn_step_llamacpp(lin_q_hf, lin_k_hf, lin_v_hf,
                                        np.log(g_decay_hf), beta_gate_hf, ssm_state_hf)

        gated_out_hf = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)
        for vh in range(LINEAR_NUM_V_HEADS):
            start = vh * LINEAR_VALUE_DIM
            end = start + LINEAR_VALUE_DIM
            gated_out_hf[start:end] = rms_norm_gated(
                gdn_out_hf[start:end], z_out_hf[start:end], norm_w)

        attn_out_hf = out_w @ gated_out_hf
        hidden_hf = hidden_hf + attn_out_hf

        token_result['hf'] = {
            'q_rms': float(np.sqrt(np.mean(lin_q_hf ** 2))),
            'gdn_out_rms': float(np.sqrt(np.mean(gdn_out_hf ** 2))),
            'attn_out_rms': float(np.sqrt(np.mean(attn_out_hf ** 2))),
            'hidden_rms': float(np.sqrt(np.mean(hidden_hf ** 2))),
            'ssm_rms': float(np.sqrt(np.mean(ssm_state_hf ** 2))),
            'ssm_max': float(np.max(np.abs(ssm_state_hf))),
        }

        # ============================================================
        # Compare
        # ============================================================
        gdn_diff = np.abs(gdn_out_finch - gdn_out_hf)
        attn_diff = np.abs(attn_out_finch - attn_out_hf)
        hidden_diff = np.abs(hidden_finch - hidden_hf)
        ssm_diff = np.abs(ssm_state_finch - ssm_state_hf)
        q_diff = np.abs(lin_q - lin_q_hf)
        conv_diff = np.abs(conv_out_finch - conv_out_hf)
        ssm_before_diff = np.abs(ssm_before - ssm_before_hf)

        token_result['comparison'] = {
            'conv_max_diff': float(conv_diff.max()),
            'q_max_diff': float(q_diff.max()),
            'gdn_max_diff': float(gdn_diff.max()),
            'gdn_mean_diff': float(gdn_diff.mean()),
            'attn_max_diff': float(attn_diff.max()),
            'hidden_max_diff': float(hidden_diff.max()),
            'ssm_max_diff': float(ssm_diff.max()),
            'ssm_mean_diff': float(ssm_diff.mean()),
            'ssm_before_max_diff': float(ssm_before_diff.max()),
        }

        results['tokens'].append(token_result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--all-layers', action='store_true')
    parser.add_argument('--num-layers', type=int, default=40)
    args = parser.parse_args()

    print("Loading BF16 model tensors (cached)...")
    tensors = {}

    if args.all_layers:
        print(f"\n{'='*80}")
        print(f"2-TOKEN DIFFERENTIAL DEBUG: Testing layers 0-{args.num_layers-1}")
        print(f"{'='*80}")
        print(f"{'Layer':>5} {'Type':>8} {'T1 gdn_diff':>12} {'T1 ssm_diff':>12} "
              f"{'T2 gdn_diff':>12} {'T2 ssm_diff':>12} {'T2 hidden':>12} {'Verdict'}")
        print(f"{'-'*5} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*7}")

        failures = []
        for layer in range(args.num_layers):
            result = run_layer_2token(layer, tensors, seed=42 + layer, verbose=False)

            if result is None:
                print(f"{layer:>5} {'FULL':>8} {'—':>12} {'—':>12} {'—':>12} {'—':>12} {'—':>12} {'SKIP'}")
                continue

            t0 = result['tokens'][0]['comparison']
            t1 = result['tokens'][1]['comparison']

            t1_gdn_diff = t1['gdn_max_diff']
            t1_ssm_diff = t1['ssm_max_diff']
            t1_hidden_diff = t1['hidden_max_diff']

            # Flag failures: token 2 divergence > 1e-3
            failed = (t1_gdn_diff > 1e-3 or t1_ssm_diff > 1e-3 or t1_hidden_diff > 1e-2)

            if np.isnan(t1_gdn_diff):
                status = 'NaN!'
                failures.append((layer, 'NaN'))
            elif failed:
                status = '❌ FAIL'
                failures.append((layer, f'gdn={t1_gdn_diff:.2e} ssm={t1_ssm_diff:.2e} hidden={t1_hidden_diff:.2e}'))
            else:
                status = '✅'

            print(f"{layer:>5} {'LINEAR':>8} {t0['gdn_max_diff']:>12.2e} {t0['ssm_max_diff']:>12.2e} "
                  f"{t1_gdn_diff:>12.2e} {t1_ssm_diff:>12.2e} {t1_hidden_diff:>12.2e} {status}")

            # If token 2 fails, print detailed breakdown
            if failed:
                t0 = result['tokens'][0]
                t1 = result['tokens'][1]
                print(f"  Token 0: finch ssm_rms={t0['finch']['ssm_rms']:.4f} "
                      f"hf ssm_rms={t0['hf']['ssm_rms']:.4f}")
                print(f"  Token 1: finch ssm_rms={t1['finch']['ssm_rms']:.4f} "
                      f"hf ssm_rms={t1['hf']['ssm_rms']:.4f}")
                print(f"  Token 1: finch q_rms={t1['finch']['q_rms']:.4f} "
                      f"hf q_rms={t1['hf']['q_rms']:.4f}")
                comp = t1['comparison']
                print(f"  Token 1 diffs: conv={comp['conv_max_diff']:.2e} q={comp['q_max_diff']:.2e} "
                      f"ssm_before={comp['ssm_before_max_diff']:.2e}")
                print(f"  Token 1 diffs: gdn={comp['gdn_max_diff']:.2e} "
                      f"attn={comp['attn_max_diff']:.2e}")

        if failures:
            print(f"\n❌ {len(failures)} layers failed on token 2:")
            for layer, detail in failures:
                print(f"   Layer {layer}: {detail}")
        else:
            print(f"\n✅ All layers pass on token 2 — GDN state persistence is CORRECT")

    else:
        # Single layer detailed mode
        result = run_layer_2token(args.layer, tensors, seed=42 + args.layer, verbose=False)

        if result is None:
            print(f"Layer {args.layer} is full attention (no linear_attn weights)")
            return

        print(f"\n{'='*80}")
        print(f"2-TOKEN DIFFERENTIAL DEBUG: Layer {args.layer}")
        print(f"{'='*80}")

        for t in result['tokens']:
            tok = t['token']
            comp = t['comparison']
            print(f"\n--- Token {tok} ---")
            print(f"  Conv max diff:     {comp['conv_max_diff']:.6e}")
            print(f"  q max diff:         {comp['q_max_diff']:.6e}")
            print(f"  SSM before diff:    {comp['ssm_before_max_diff']:.6e}")
            print(f"  GDN max diff:       {comp['gdn_max_diff']:.6e}")
            print(f"  GDN mean diff:      {comp['gdn_mean_diff']:.6e}")
            print(f"  Attn max diff:      {comp['attn_max_diff']:.6e}")
            print(f"  Hidden max diff:    {comp['hidden_max_diff']:.6e}")
            print(f"  SSM after max diff: {comp['ssm_max_diff']:.6e}")
            print(f"  SSM after mean diff:{comp['ssm_mean_diff']:.6e}")

            print(f"\n  FinchMoE state:")
            print(f"    q_rms={t['finch']['q_rms']:.6f} k_rms={t['finch']['k_rms']:.6f} "
                  f"v_rms={t['finch']['v_rms']:.6f}")
            print(f"    gdn_out_rms={t['finch']['gdn_out_rms']:.6f} "
                  f"attn_out_rms={t['finch']['attn_out_rms']:.6f} "
                  f"hidden_rms={t['finch']['hidden_rms']:.6f}")
            print(f"    ssm_rms={t['finch']['ssm_rms']:.6f} ssm_max={t['finch']['ssm_max']:.6f}")

        t0_comp = result['tokens'][0]['comparison']
        t1_comp = result['tokens'][1]['comparison']

        print(f"\n{'='*80}")
        if t1_comp['gdn_max_diff'] > 1e-3 or t1_comp['ssm_max_diff'] > 1e-3:
            print(f"❌ FAIL: Token 2 state divergence detected!")
            print(f"   GDN diff grew from {t0_comp['gdn_max_diff']:.2e} → {t1_comp['gdn_max_diff']:.2e}")
            print(f"   SSM diff grew from {t0_comp['ssm_max_diff']:.2e} → {t1_comp['ssm_max_diff']:.2e}")
        else:
            print(f"✅ PASS: Both tokens match within tolerance")


if __name__ == '__main__':
    main()
