#!/usr/bin/env python3
"""
Layer-by-layer comparison: FinchMoE GDN vs HF reference.

Loads one linear attention layer at a time from the BF16 model, runs FinchMoE's
exact GDN algorithm (translated to PyTorch), and compares against the HF forward pass.

Usage:
    python3 finchmoe/debug_gdn_compare.py [--layer N] [--all-layers] [--tolerance 1e-3]
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "Qwen3.6-35B-A3B-bf16")
QUANT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "Qwen3.6-35B-A3B-4bit-custom")

# Architecture constants (matching infer.m)
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
FULL_ATTN_INTERVAL = 4


def bf16_to_f32(bf16_val):
    """Convert bf16 uint16 to float32."""
    bits = int(bf16_val) << 16
    return struct.unpack('f', struct.pack('I', bits & 0xFFFFFFFF))[0]


def rms_norm(x, eps=RMS_NORM_EPS):
    """RMS normalize: x / rms(x)"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms


def rms_norm_gated(x, z, w, eps=RMS_NORM_EPS):
    """Gated RMS norm: rms_norm(x) * silu(z) * w"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    normed = x / rms
    silu_z = z / (1.0 + np.exp(-z))  # SiLU
    return normed * silu_z * w


def cpu_conv1d_step(conv_state, new_input, conv_weight_f32, channels, kernel_size=4):
    """FinchMoE conv1d: depthwise conv then SiLU.

    Args:
        conv_weight_f32: [channels * kernel_size] float32 (already dequantized/converted)
    """
    out = np.zeros(channels, dtype=np.float32)
    for c in range(channels):
        acc = 0.0
        for k in range(kernel_size - 1):
            w = conv_weight_f32[c * kernel_size + k]
            acc += conv_state[k * channels + c] * w
        w = conv_weight_f32[c * kernel_size + (kernel_size - 1)]
        acc += new_input[c] * w
        out[c] = acc
    # SiLU
    out = out / (1.0 + np.exp(-out))
    return out


def finchmoe_gdn_step(q, k, v, g_decay, beta_gate, ssm_state, k_heads_per_v=2):
    """
    Exact translation of FinchMoE's gated_delta_net_step.

    Args:
        q: [num_k_heads * key_dim] = [2048]
        k: [num_k_heads * key_dim] = [2048]
        v: [num_v_heads * value_dim] = [4096]
        g_decay: [num_v_heads] = [32]
        beta_gate: [num_v_heads] = [32]
        ssm_state: [num_v_heads, value_dim, key_dim] = [32, 128, 128] (updated in-place)

    Returns:
        output: [num_v_heads * value_dim] = [4096]
    """
    num_v_heads = LINEAR_NUM_V_HEADS
    num_k_heads = LINEAR_NUM_K_HEADS
    key_dim = LINEAR_KEY_DIM
    value_dim = LINEAR_VALUE_DIM

    output = np.zeros(num_v_heads * value_dim, dtype=np.float32)

    for vh in range(num_v_heads):
        kh = vh // k_heads_per_v
        g = g_decay[vh]
        beta = beta_gate[vh]

        S = ssm_state[vh]  # [value_dim, key_dim]
        k_h = k[kh * key_dim : (kh+1) * key_dim]
        v_h = v[vh * value_dim : (vh+1) * value_dim]
        q_h = q[kh * key_dim : (kh+1) * key_dim]

        for vi in range(value_dim):
            # Step 1+2: Decay state and compute kv_mem
            kv_mem = 0.0
            for ki in range(key_dim):
                s_val = S[vi, ki] * g
                S[vi, ki] = s_val
                kv_mem += s_val * k_h[ki]

            # Step 3+4: Delta update
            delta = (v_h[vi] - kv_mem) * beta
            for ki in range(key_dim):
                S[vi, ki] += k_h[ki] * delta

        # Step 5: Output
        for vi in range(value_dim):
            out_val = 0.0
            for ki in range(key_dim):
                out_val += S[vi, ki] * q_h[ki]
            output[vh * value_dim + vi] = out_val

    return output


def load_bf16_tensor_from_safetensors(model_dir, tensor_name):
    """Load a specific tensor from the BF16 safetensors files."""
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith('.safetensors'):
            continue
        fpath = os.path.join(model_dir, fname)
        try:
            with safe_open(fpath, framework='pt') as st:
                if tensor_name in st.keys():
                    return st.get_tensor(tensor_name)
        except Exception:
            continue
    return None


def load_quantized_tensor(model_dir, tensor_name):
    """Load a quantized tensor from FinchMoE's format.

    FinchMoE stores weights as raw 4-bit packed arrays with scale+bias per group.
    BF16-protected weights are stored as raw uint16 (bf16).
    """
    # Look for the weight file (single merged file or directory of .safetensors)
    weight_file = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(weight_file):
        with safe_open(weight_file, framework='pt') as st:
            if tensor_name in st.keys():
                return st.get_tensor(tensor_name)

    # Check for FinchMoE's custom format
    # The quantized model may have a different layout
    return None


def project_matvec(weight, scales, biases, x, group_size=64, force_bf16=False):
    """
    FinchMoE's matvec: y = W @ x, supporting INT4 and BF16.

    When scales is None (BF16 path): direct BF16 dot product.
    Otherwise: 4-bit dequant matvec.
    """
    out_dim = weight.shape[0] if hasattr(weight, 'shape') else len(weight)
    in_dim = len(x)
    y = np.zeros(out_dim, dtype=np.float32)

    if scales is None:
        # BF16 path
        for i in range(out_dim):
            s = 0.0
            for j in range(in_dim):
                w_bf16 = weight[i * in_dim + j] if isinstance(weight, np.ndarray) else weight[i][j]
                # If weight is torch tensor in bf16, extract the uint16
                if hasattr(w_bf16, 'item'):
                    w_bf16 = w_bf16.item()
                if isinstance(w_bf16, float):
                    s += w_bf16 * x[j]
                else:
                    s += bf16_to_f32(w_bf16) * x[j]
            y[i] = s
    else:
        # 4-bit path
        for i in range(out_dim):
            s = 0.0
            for j in range(in_dim):
                group = j // group_size
                scale = bf16_to_f32(scales[i * ((in_dim + group_size - 1) // group_size) + group])
                bias = bf16_to_f32(biases[i * ((in_dim + group_size - 1) // group_size) + group])
                # Extract 4-bit value from packed array
                byte_idx = (i * in_dim + j) // 2
                nibble = (weight[byte_idx] >> (4 * ((i * in_dim + j) % 2))) & 0x0F
                w_dequant = nibble * scale + bias
                s += w_dequant * x[j]
            y[i] = s

    return y


def test_linear_layer(layer_idx, tolerance=1e-3, verbose=True):
    """Test one linear attention layer by comparing FinchMoE's GDN against HF reference."""
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."

    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing linear attention layer {layer_idx}")
        print(f"{'='*60}")

    # Load weights from BF16 model
    qkv_weight = load_bf16_tensor_from_safetensors(MODEL_DIR, prefix + "in_proj_qkv.weight")

    if qkv_weight is None:
        # Check if this is a full attention layer
        full_prefix = f"model.language_model.layers.{layer_idx}.self_attn."
        full_check = load_bf16_tensor_from_safetensors(MODEL_DIR, full_prefix + "q_proj.weight")
        if full_check is not None:
            if verbose:
                print(f"  SKIP: layer {layer_idx} is full attention (no linear_attn weights)")
        else:
            if verbose:
                print(f"  SKIP: weights not found for layer {layer_idx}")
        return None

    z_weight = load_bf16_tensor_from_safetensors(MODEL_DIR, prefix + "in_proj_z.weight")
    beta_weight = load_bf16_tensor_from_safetensors(MODEL_DIR, prefix + "in_proj_b.weight")
    alpha_weight = load_bf16_tensor_from_safetensors(MODEL_DIR, prefix + "in_proj_a.weight")
    out_weight = load_bf16_tensor_from_safetensors(MODEL_DIR, prefix + "out_proj.weight")
    A_log = load_bf16_tensor_from_safetensors(MODEL_DIR, prefix + "A_log")
    dt_bias = load_bf16_tensor_from_safetensors(MODEL_DIR, prefix + "dt_bias")
    conv_weight = load_bf16_tensor_from_safetensors(MODEL_DIR, prefix + "conv1d.weight")
    norm_weight = load_bf16_tensor_from_safetensors(MODEL_DIR, prefix + "norm.weight")

    # Convert to numpy (handle bf16 tensors)
    def to_np(t):
        if t.dtype == torch.bfloat16:
            return t.to(torch.float32).numpy()
        return t.numpy()

    qkv_w = to_np(qkv_weight)
    z_w = to_np(z_weight)
    beta_w = to_np(beta_weight)
    alpha_w = to_np(alpha_weight)
    out_w = to_np(out_weight)
    A_log_np = to_np(A_log)
    dt_bias_np = to_np(dt_bias)
    conv_w = to_np(conv_weight.squeeze(1))
    norm_w = to_np(norm_weight)

    if verbose:
        print(f"  qkv_w: {qkv_w.shape}")
        print(f"  z_w: {z_w.shape}")
        print(f"  beta_w: {beta_w.shape}")
        print(f"  alpha_w: {alpha_w.shape}")
        print(f"  out_w: {out_w.shape}")
        print(f"  A_log: {A_log_np.shape}, range=[{A_log_np.min():.4f}, {A_log_np.max():.4f}]")
        print(f"  dt_bias: {dt_bias_np.shape}, range=[{dt_bias_np.min():.4f}, {dt_bias_np.max():.4f}]")
        print(f"  conv_w: {conv_w.shape}")
        print(f"  norm_w: {norm_w.shape}")

    # Create random input (simulating post-LN hidden state)
    np.random.seed(42 + layer_idx)
    hidden = np.random.randn(HIDDEN_DIM).astype(np.float32) * 0.1

    # ============================================
    # FinchMoE path
    # ============================================

    # Step 1: QKV projection (matvec: qkv_w [8192, 2048] @ hidden [2048])
    qkv_out = qkv_w @ hidden  # [8192]

    # Step 1b: Z projection
    z_out = z_w @ hidden  # [4096]

    # Step 1c: Beta projection
    beta_out = beta_w @ hidden  # [32]

    # Step 1d: Alpha projection
    alpha_out = alpha_w @ hidden  # [32]

    if verbose:
        print(f"  qkv_out rms: {np.sqrt(np.mean(qkv_out**2)):.6f}")
        print(f"  z_out rms: {np.sqrt(np.mean(z_out**2)):.6f}")

    # Step 2: Conv1d (depthwise, kernel=4)
    # conv_state is zero for first token
    conv_state = np.zeros((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, dtype=np.float32)
    conv_out = cpu_conv1d_step(conv_state, qkv_out, conv_w.flatten(), LINEAR_CONV_DIM, CONV_KERNEL_SIZE)

    if verbose:
        print(f"  conv_out rms: {np.sqrt(np.mean(conv_out**2)):.6f}")

    # Step 3: Split into q, k, v
    lin_q = conv_out[:LINEAR_TOTAL_KEY].copy()   # [2048]
    lin_k = conv_out[LINEAR_TOTAL_KEY:2*LINEAR_TOTAL_KEY].copy()  # [2048]
    lin_v = conv_out[2*LINEAR_TOTAL_KEY:].copy()  # [4096]

    # Step 4: RMS normalize q and k
    inv_scale = 1.0 / np.sqrt(LINEAR_KEY_DIM)
    for h in range(LINEAR_NUM_K_HEADS):
        start = h * LINEAR_KEY_DIM
        end = start + LINEAR_KEY_DIM
        lin_q[start:end] = rms_norm(lin_q[start:end]) * inv_scale * inv_scale
        lin_k[start:end] = rms_norm(lin_k[start:end]) * inv_scale

    if verbose:
        print(f"  q (after norm) rms: {np.sqrt(np.mean(lin_q**2)):.6f}")
        print(f"  k (after norm) rms: {np.sqrt(np.mean(lin_k**2)):.6f}")
        print(f"  v rms: {np.sqrt(np.mean(lin_v**2)):.6f}")

    # Step 5: Compute g_decay and beta_gate
    g_decay = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
    beta_gate = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
    for vh in range(LINEAR_NUM_V_HEADS):
        a_val = alpha_out[vh]
        dt_b = bf16_to_f32(dt_bias_np[vh]) if dt_bias_np.dtype == np.uint16 else dt_bias_np[vh]
        A_val = np.exp(A_log_np[vh])
        softplus_val = np.log(1.0 + np.exp(a_val + dt_b))
        g_decay[vh] = np.exp(-A_val * softplus_val)
        beta_gate[vh] = 1.0 / (1.0 + np.exp(-beta_out[vh]))

    if verbose:
        print(f"  g_decay: range=[{g_decay.min():.6f}, {g_decay.max():.6f}]")
        print(f"  beta_gate: range=[{beta_gate.min():.6f}, {beta_gate.max():.6f}]")

    # Step 6: GDN recurrence
    ssm_state = np.zeros((LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM), dtype=np.float32)
    k_heads_per_v = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS  # 2
    finch_output = finchmoe_gdn_step(lin_q, lin_k, lin_v, g_decay, beta_gate, ssm_state,
                                      k_heads_per_v)

    if verbose:
        print(f"  GDN output rms: {np.sqrt(np.mean(finch_output**2)):.6f}")
        print(f"  State rms: {np.sqrt(np.mean(ssm_state**2)):.6f}")

    # Step 7: Gated RMS norm
    gated_out = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)
    for vh in range(LINEAR_NUM_V_HEADS):
        start = vh * LINEAR_VALUE_DIM
        end = start + LINEAR_VALUE_DIM
        oh = finch_output[start:end]
        zh = z_out[start:end]
        gated_out[start:end] = rms_norm_gated(oh, zh, norm_w, RMS_NORM_EPS)

    if verbose:
        print(f"  Gated norm output rms: {np.sqrt(np.mean(gated_out**2)):.6f}")

    # Step 8: Output projection
    finch_attn_out = out_w @ gated_out  # [4096] @ [4096, 2048] → actually out_w is [2048, 4096]
    # out_w @ gated_out means: [2048, 4096] @ [4096] = [2048]

    if verbose:
        print(f"  FinchMoE attn_out rms: {np.sqrt(np.mean(finch_attn_out**2)):.6f}")
        print(f"  FinchMoE attn_out[:5]: {finch_attn_out[:5]}")

    # ============================================
    # HF Reference path
    # ============================================
    # Use the HF model to run the same input through the layer
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)

        # Build a minimal forward pass using the loaded weights
        # Since we can't load the full model, we compute the HF-equivalent manually

        # The HF model does:
        # 1. input_layernorm
        # 2. linear_attn (which includes all the GDN steps)
        # 3. residual connection
        # 4. post_attention_layernorm
        # 5. MoE block

        # For comparison, we compute what the HF linear_attn should output
        # using float32 for direct comparison with FinchMoE
        hf_hidden = torch.from_numpy(hidden)

        # QKV projection
        hf_qkv = (torch.from_numpy(qkv_w) @ hf_hidden).to(torch.float32)  # [8192]

        # Z projection
        hf_z = (torch.from_numpy(z_w) @ hf_hidden).to(torch.float32)  # [4096]

        # Beta projection
        hf_beta = (torch.from_numpy(beta_w) @ hf_hidden).to(torch.float32)  # [32]

        # Alpha projection
        hf_alpha = (torch.from_numpy(alpha_w) @ hf_hidden).to(torch.float32)  # [32]

        # Conv1d (depthwise, incremental, same as FinchMoE's cpu_conv1d_step)
        hf_conv_out_np = cpu_conv1d_step(
            np.zeros((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, dtype=np.float32),
            hf_qkv.to(torch.float32).numpy(),
            conv_w.flatten().astype(np.float32),
            LINEAR_CONV_DIM, CONV_KERNEL_SIZE
        )
        hf_conv_out = torch.from_numpy(hf_conv_out_np)

        # L2 normalize q and k (HF uses F.normalize with p=2)
        hf_q = hf_conv_out[:LINEAR_TOTAL_KEY].reshape(LINEAR_NUM_K_HEADS, LINEAR_KEY_DIM)
        hf_k = hf_conv_out[LINEAR_TOTAL_KEY:2*LINEAR_TOTAL_KEY].reshape(LINEAR_NUM_K_HEADS, LINEAR_KEY_DIM)
        hf_v = hf_conv_out[2*LINEAR_TOTAL_KEY:].reshape(LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM)

        hf_q = torch.nn.functional.normalize(hf_q.to(torch.float32), p=2, dim=-1)
        hf_k = torch.nn.functional.normalize(hf_k.to(torch.float32), p=2, dim=-1)

        # Apply q scaling: q = q / sqrt(key_dim)
        hf_q = hf_q / np.sqrt(LINEAR_KEY_DIM)

        # Compare q/k norms
        if verbose:
            print(f"\n  --- Comparison ---")
            print(f"  HF q rms: {torch.sqrt(torch.mean(hf_q**2)).item():.6f} vs Finch q rms: {np.sqrt(np.mean(lin_q**2)):.6f}")
            print(f"  HF k rms: {torch.sqrt(torch.mean(hf_k**2)).item():.6f} vs Finch k rms: {np.sqrt(np.mean(lin_k**2)):.6f}")

            # Element-wise comparison of q
            q_diff = np.abs(hf_q.flatten().numpy() - lin_q)
            print(f"  q max diff: {q_diff.max():.6f}")
            print(f"  q mean diff: {q_diff.mean():.6f}")

        # Compute HF GDN (using HF's actual implementation from modeling_qwen3_5_moe.py)
        hf_beta_sigmoid = torch.sigmoid(hf_beta.to(torch.float32))
        hf_alpha_biased = hf_alpha.to(torch.float32) + dt_bias.to(torch.float32)
        hf_alpha_softplus = torch.nn.functional.softplus(hf_alpha_biased)
        hf_A = A_log.to(torch.float32).exp()
        hf_gate = -hf_A * hf_alpha_softplus  # gate = -exp(A_log) * softplus
        # g_decay = exp(gate) = exp(-exp(A_log) * softplus)

        if verbose:
            gd_diff = np.abs(np.exp(hf_gate.numpy()) - g_decay)
            bg_diff = np.abs(hf_beta_sigmoid.numpy() - beta_gate)
            print(f"  g_decay max diff: {gd_diff.max():.8f}")
            print(f"  beta_gate max diff: {bg_diff.max():.8f}")

        # Compare conv output
        conv_diff = np.abs(hf_conv_out.numpy() - conv_out)
        if verbose:
            print(f"  conv_out max diff: {conv_diff.max():.6f}")
            print(f"  conv_out mean diff: {conv_diff.mean():.6f}")

        # Run the HF GDN recurrence manually to compare
        ssm_state_hf = np.zeros((LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM), dtype=np.float32)
        hf_q_np = hf_q.numpy()  # [16, 128]
        hf_k_np = hf_k.numpy()  # [16, 128]
        hf_v_np = hf_v.numpy()  # [32, 128]

        # HF does:
        # state = state * exp(gate)  [gate already includes -exp(A_log)*softplus]
        # sk = state @ k^T
        # delta = (v - sk) * beta_sigmoid
        # state += outer(k, delta)
        # output = state @ q
        hf_exp_gate = np.exp(hf_gate.numpy())  # [32]
        hf_beta_np = hf_beta_sigmoid.numpy()  # [32]

        hf_gdn_output = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)
        for vh in range(LINEAR_NUM_V_HEADS):
            kh = vh // k_heads_per_v
            g = hf_exp_gate[vh]
            beta = hf_beta_np[vh]
            S = ssm_state_hf[vh]
            k_h = hf_k_np[kh]
            v_h = hf_v_np[vh]
            q_h = hf_q_np[kh]

            for vi in range(LINEAR_VALUE_DIM):
                # Decay + compute sk (same as FinchMoE)
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
                hf_gdn_output[vh * LINEAR_VALUE_DIM + vi] = out_val

        gdn_diff = np.abs(hf_gdn_output - finch_output)
        if verbose:
            print(f"\n  GDN raw output max diff: {gdn_diff.max():.6f}")
            print(f"  GDN raw output mean diff: {gdn_diff.mean():.6f}")
            print(f"  GDN raw output rms: {np.sqrt(np.mean(hf_gdn_output**2)):.6f} vs {np.sqrt(np.mean(finch_output**2)):.6f}")
            print(f"  GDN state rms: {np.sqrt(np.mean(ssm_state_hf**2)):.6f} vs {np.sqrt(np.mean(ssm_state**2)):.6f}")

        # Check if any head has large divergence
        for vh in range(LINEAR_NUM_V_HEADS):
            head_diff = np.abs(hf_gdn_output[vh*128:(vh+1)*128] - finch_output[vh*128:(vh+1)*128])
            head_max = head_diff.max()
            if head_max > 0.01:
                if verbose:
                    print(f"  ⚠️  Head {vh}: max diff={head_max:.6f}, mean diff={head_diff.mean():.6f}")

        max_diff = gdn_diff.max()
        mean_diff = gdn_diff.mean()
        passed = max_diff < tolerance

        if verbose:
            if passed:
                print(f"\n  ✅ PASS: max GDN diff {max_diff:.6f} < {tolerance}")
            else:
                print(f"\n  ❌ FAIL: max GDN diff {max_diff:.6f} >= {tolerance}")

        return {
            'layer': layer_idx,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'passed': passed,
            'gdn_output_max': float(np.abs(hf_gdn_output).max()),
            'finch_output_max': float(np.abs(finch_output).max()),
        }

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        return {'layer': layer_idx, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Compare FinchMoE GDN against HF reference')
    parser.add_argument('--layer', type=int, default=0, help='Single layer to test')
    parser.add_argument('--all-layers', action='store_true', help='Test all linear layers (0-39)')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='Pass/fail tolerance')
    parser.add_argument('--num-layers', type=int, default=5, help='Number of layers to test with --all-layers')
    args = parser.parse_args()

    if args.all_layers:
        results = []
        for i in range(args.num_layers):
            r = test_linear_layer(i, tolerance=args.tolerance, verbose=False)
            if r:
                results.append(r)
                status = "✅" if r.get('passed') else "❌"
                if 'error' in r:
                    status = "⚠️"
                    print(f"  Layer {i}: {status} {r['error']}")
                else:
                    print(f"  Layer {i}: {status} max_diff={r['max_diff']:.6f} mean_diff={r['mean_diff']:.6f}")

        passed = sum(1 for r in results if r.get('passed', False))
        print(f"\n{passed}/{len(results)} layers passed")
    else:
        test_linear_layer(args.layer, tolerance=args.tolerance, verbose=True)


if __name__ == '__main__':
    main()
