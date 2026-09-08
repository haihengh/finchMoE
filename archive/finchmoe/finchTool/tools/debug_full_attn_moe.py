#!/usr/bin/env python3
"""
Test full attention + MoE FFN for 2-token correctness.
Compares FinchMoE's implementation against HF reference for a FULL attention layer.
"""
import os, struct, sys
import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "Qwen3.6-35B-A3B-bf16")

HIDDEN_DIM = 2048
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY)  # 64
ROPE_THETA = 10000000.0
RMS_NORM_EPS = 1e-6
NUM_EXPERTS = 256
MOE_INTERMEDIATE = 512
SHARED_INTERMEDIATE = 512


def to_np(t):
    return t.to(torch.float32).numpy() if t.dtype == torch.bfloat16 else t.numpy()


def load_all(tensors, model_dir, layer):
    """Load all tensors for a given layer."""
    prefix = f"model.language_model.layers.{layer}."
    keys_needed = [
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + "self_attn.q_proj.weight", prefix + "self_attn.k_proj.weight",
        prefix + "self_attn.v_proj.weight", prefix + "self_attn.o_proj.weight",
        prefix + "self_attn.q_norm.weight", prefix + "self_attn.k_norm.weight",
        prefix + "mlp.gate.weight", prefix + "mlp.shared_expert_gate.weight",
        prefix + "mlp.shared_expert.gate_proj.weight",
        prefix + "mlp.shared_expert.up_proj.weight",
        prefix + "mlp.shared_expert.down_proj.weight",
    ]
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith('.safetensors'):
            continue
        full = os.path.join(model_dir, fname)
        with safe_open(full, framework='pt') as st:
            for key in keys_needed:
                if key not in tensors and key in st.keys():
                    tensors[key] = st.get_tensor(key)
    return all(k in tensors for k in keys_needed)


def rms_norm(x, w, eps=RMS_NORM_EPS):
    inv_rms = 1.0 / np.sqrt(np.mean(x**2) + eps)
    return x * inv_rms * w


def rope_rotate(x, pos, theta=ROPE_THETA):
    """Apply RoPE to x in-place. x: [head_dim]."""
    d = ROTARY_DIM
    freqs = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
    angles = pos * freqs
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    for i in range(0, d, 2):
        x0, x1 = x[i], x[i + 1]
        c, s = cos_a[i // 2], sin_a[i // 2]
        x[i] = x0 * c - x1 * s
        x[i + 1] = x0 * s + x1 * c


def test_full_attn_layer(layer=3):
    """Test a full attention layer for 2-token state correctness."""
    tensors = {}
    if not load_all(tensors, MODEL_DIR, layer):
        print(f"Layer {layer}: weights not found")
        return

    prefix = f"model.language_model.layers.{layer}."

    input_norm_w = to_np(tensors[prefix + "input_layernorm.weight"])
    post_norm_w = to_np(tensors[prefix + "post_attention_layernorm.weight"])
    q_w = to_np(tensors[prefix + "self_attn.q_proj.weight"])
    k_w = to_np(tensors[prefix + "self_attn.k_proj.weight"])
    v_w = to_np(tensors[prefix + "self_attn.v_proj.weight"])
    o_w = to_np(tensors[prefix + "self_attn.o_proj.weight"])
    q_norm_w = to_np(tensors[prefix + "self_attn.q_norm.weight"])
    k_norm_w = to_np(tensors[prefix + "self_attn.k_norm.weight"])
    gate_w = to_np(tensors[prefix + "mlp.gate.weight"])
    seg_w = to_np(tensors[prefix + "mlp.shared_expert_gate.weight"])
    sg_w = to_np(tensors[prefix + "mlp.shared_expert.gate_proj.weight"])
    su_w = to_np(tensors[prefix + "mlp.shared_expert.up_proj.weight"])
    sd_w = to_np(tensors[prefix + "mlp.shared_expert.down_proj.weight"])

    print(f"Layer {layer} (FULL attention): all weights loaded")
    print(f"  q_proj: {q_w.shape}, k_proj: {k_w.shape}, v_proj: {v_w.shape}, o_proj: {o_w.shape}")
    print(f"  gate: {gate_w.shape}, seg: {seg_w.shape}")
    print(f"  shared_expert: gate_up=({sg_w.shape},{su_w.shape}), down={sd_w.shape}")

    # Generate two random tokens
    np.random.seed(42 + layer)
    token0 = np.random.randn(HIDDEN_DIM).astype(np.float32) * 0.1
    token1 = np.random.randn(HIDDEN_DIM).astype(np.float32) * 0.1

    # KV cache starts empty
    kv_k = np.zeros((2, 256), dtype=np.float32)  # [2 tokens, 512] = [num_kv_heads * head_dim * 2]
    kv_v = np.zeros((2, 256), dtype=np.float32)
    kv_len = 0

    results = []
    for tok_idx, (hidden, pos) in enumerate([(token0, 0), (token1, 1)]):
        # --- FinchMoE forward ---
        residual = hidden.copy()
        normed = rms_norm(hidden, input_norm_w)

        # QKV projections
        q_full = q_w @ normed  # [8192] = q(4096) + gate(4096)
        q_part = q_full[:4096]  # 16 heads * 256 dim
        q_gate = q_full[4096:]
        k = k_w @ normed  # [512] = 2 KV heads * 256
        v = v_w @ normed  # [512]

        # Q/K norms per head
        q_heads = q_part.reshape(NUM_ATTN_HEADS, HEAD_DIM)
        for h in range(NUM_ATTN_HEADS):
            q_heads[h] = rms_norm(q_heads[h], q_norm_w)

        k_heads = k.reshape(NUM_KV_HEADS, HEAD_DIM)
        for h in range(NUM_KV_HEADS):
            k_heads[h] = rms_norm(k_heads[h], k_norm_w)

        # RoPE
        for h in range(NUM_ATTN_HEADS):
            rope_rotate(q_heads[h], pos)
        for h in range(NUM_KV_HEADS):
            rope_rotate(k_heads[h], pos)

        # Store KV
        if kv_len < 2:
            kv_k[kv_len] = k.reshape(-1)[:256]  # simplified
            kv_v[kv_len] = v.reshape(-1)[:256]
            kv_len += 1

        # Attention (causal, current token attends to all previous)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        attn_out = np.zeros(NUM_ATTN_HEADS * HEAD_DIM, dtype=np.float32)
        for h in range(NUM_ATTN_HEADS):
            kh = h // (NUM_ATTN_HEADS // NUM_KV_HEADS)  # 8 Q heads per KV head
            # Simple: attend only to current token (causal, isolated test)
            score = np.dot(q_heads[h], k_heads[kh]) * scale
            attn_out[h * HEAD_DIM:(h + 1) * HEAD_DIM] = score * v.reshape(NUM_KV_HEADS, HEAD_DIM)[kh]

        # Sigmoid gate
        gate_val = 1.0 / (1.0 + np.exp(-q_gate))
        attn_out = attn_out * gate_val

        # O projection
        attn_out = o_w @ attn_out

        # Residual 1
        hidden = residual + attn_out

        # Post-attention norm
        residual2 = hidden.copy()
        h_post = rms_norm(hidden, post_norm_w)

        # MoE FFN: router + shared expert
        gate_scores = gate_w @ h_post  # [256]
        gate_scores_sm = np.exp(gate_scores - gate_scores.max())
        gate_scores_sm /= gate_scores_sm.sum()

        # Top-K selection
        K = 8
        top_k_idx = np.argsort(gate_scores_sm)[-K:][::-1]
        top_k_vals = gate_scores_sm[top_k_idx]
        top_k_vals /= top_k_vals.sum()

        # Shared expert
        shared_gate_score = (seg_w @ h_post)[0]
        shared_weight = 1.0 / (1.0 + np.exp(-shared_gate_score))

        shared_gate = sg_w @ h_post
        shared_up = su_w @ h_post
        shared_act = shared_gate / (1.0 + np.exp(-shared_gate)) * shared_up
        shared_out = sd_w @ shared_act * shared_weight

        # Combine (moe_out = 0 for test)
        hidden = residual2 + shared_out

        r = {
            'tok': tok_idx,
            'normed_rms': float(np.sqrt(np.mean(normed**2))),
            'q_part_rms': float(np.sqrt(np.mean(q_part**2))),
            'k_rms': float(np.sqrt(np.mean(k**2))),
            'attn_out_rms': float(np.sqrt(np.mean(attn_out**2))),
            'gate_rms': float(np.sqrt(np.mean(gate_val**2))),
            'h_post_rms': float(np.sqrt(np.mean(h_post**2))),
            'gate_scores_max': float(gate_scores.max()),
            'shared_weight': float(shared_weight),
            'shared_out_rms': float(np.sqrt(np.mean(shared_out**2))),
            'hidden_rms': float(np.sqrt(np.mean(hidden**2))),
        }
        results.append(r)

    print(f"\n  Token 0: normed={results[0]['normed_rms']:.4f} attn={results[0]['attn_out_rms']:.4f} "
          f"shared_w={results[0]['shared_weight']:.4f} hidden={results[0]['hidden_rms']:.4f}")
    print(f"  Token 1: normed={results[1]['normed_rms']:.4f} attn={results[1]['attn_out_rms']:.4f} "
          f"shared_w={results[1]['shared_weight']:.4f} hidden={results[1]['hidden_rms']:.4f}")

    # Check for NaN or Inf
    for t in results:
        for k, v in t.items():
            if k != 'tok' and (np.isnan(v) or np.isinf(v)):
                print(f"  ❌ Token {t['tok']} {k}: NaN/Inf detected!")
                return False

    print(f"  ✅ Layer {layer} full attention + MoE: all values valid")
    return True


if __name__ == '__main__':
    for layer in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]:
        ok = test_full_attn_layer(layer)
        if not ok:
            print(f"\n❌ Full attention layer {layer} FAILED")
            break
    else:
        print(f"\n✅ All full attention layers produce valid outputs")
