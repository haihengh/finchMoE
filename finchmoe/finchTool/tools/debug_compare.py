#!/usr/bin/env python3
"""
Debug script: Compare C tokenizer vs Python tokenizer, then check embeddings.
This is the first step in the layer-by-layer debugging pipeline.

Usage: python debug_compare.py
"""

import os
import sys
import json
import struct
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def bf16_to_f32(val_u16):
    """Convert bfloat16 uint16 to float32."""
    # BF16: 1 sign + 8 exponent + 7 mantissa -> FP32: 1 sign + 8 exponent + 23 mantissa
    val_u32 = val_u16 << 16
    return struct.unpack('f', struct.pack('I', val_u32))[0]

def f32_to_bf16(val_f32):
    """Convert float32 to bfloat16 uint16 (round to nearest even)."""
    val_bytes = struct.pack('f', val_f32)
    val_u32 = struct.unpack('I', val_bytes)[0]
    # Round: add half of the truncated mantissa bits
    return (val_u32 + 0x8000) >> 16


class WeightFile:
    """Minimal reader for model_weights.bin (matches C engine's WeightFile)."""

    def __init__(self, bin_path, json_path):
        with open(json_path, 'r') as f:
            self.manifest = json.load(f)

        self.data = open(bin_path, 'rb')
        self.mmap = memoryview  # just a file reader

    def get_tensor_info(self, name):
        """Get tensor metadata by name."""
        return self.manifest.get('tensors', {}).get(name)

    def get_tensor_data(self, name):
        """Read tensor data as numpy array."""
        info = self.get_tensor_info(name)
        if info is None:
            return None

        offset = info['offset']
        size = info['size']
        shape = tuple(info['shape'])
        dtype_str = info.get('dtype', 'U32')

        self.data.seek(offset)
        raw = self.data.read(size)

        if dtype_str == 'U32':
            return np.frombuffer(raw, dtype=np.uint32).reshape(shape)
        elif dtype_str == 'U16':
            return np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        elif dtype_str == 'F32':
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        else:
            raise ValueError(f"Unknown dtype: {dtype_str}")


def cpu_dequant_matvec_py(W, scales, biases, x, out_dim, in_dim, group_size, bits):
    """Python equivalent of C's cpu_dequant_matvec."""
    num_groups = in_dim // group_size
    vals_per_u32 = 4 if bits == 8 else 8
    packed_per_group = group_size // vals_per_u32
    packed_cols = in_dim // vals_per_u32
    shift = bits

    out = np.zeros(out_dim, dtype=np.float32)

    for row in range(out_dim):
        acc = 0.0
        for g in range(num_groups):
            scale = bf16_to_f32(int(scales[row, g]))
            bias = bf16_to_f32(int(biases[row, g]))
            base_packed = g * packed_per_group
            base_x = g * group_size

            for p in range(packed_per_group):
                packed = int(W[row, base_packed + p])
                x_base = base_x + p * vals_per_u32

                for n in range(vals_per_u32):
                    val = (packed >> (n * shift)) & ((1 << bits) - 1)
                    acc += (float(val) * scale + bias) * float(x[x_base + n])

        out[row] = acc

    return out


def cpu_rms_norm_py(x, w_bf16, eps=1e-6):
    """Python equivalent of C's cpu_rms_norm."""
    sum_sq = np.sum(x.astype(np.float64) ** 2)
    inv_rms = 1.0 / np.sqrt(sum_sq / len(x) + eps)

    out = np.zeros(len(x), dtype=np.float32)
    for i in range(len(x)):
        weight = bf16_to_f32(int(w_bf16[i]))
        out[i] = x[i] * inv_rms * weight
    return out


def embed_lookup_py(wf, token_id):
    """Python equivalent of C's embed_lookup."""
    W = wf.get_tensor_data("model.embed_tokens.weight")
    S = wf.get_tensor_data("model.embed_tokens.scales")
    B = wf.get_tensor_data("model.embed_tokens.biases")

    if W is None:
        print("ERROR: embedding tensors not found")
        return None

    HIDDEN_DIM = 2048
    num_groups = S.shape[1]  # 64
    packed_cols = W.shape[1]  # 512
    group_size = HIDDEN_DIM // num_groups  # 32
    packed_per_group = group_size // 8  # 4 (8 nibbles per uint32)

    w_row = W[token_id]
    s_row = S[token_id]
    b_row = B[token_id]

    out = np.zeros(HIDDEN_DIM, dtype=np.float32)

    for g in range(num_groups):
        scale = bf16_to_f32(int(s_row[g]))
        bias = bf16_to_f32(int(b_row[g]))

        for p in range(packed_per_group):
            packed = int(w_row[g * packed_per_group + p])
            base = g * group_size + p * 8

            for n in range(8):
                nibble = (packed >> (n * 4)) & 0xF
                out[base + n] = float(nibble) * scale + bias

    return out


def main():
    wf_path = os.path.join(PROJECT_ROOT, "finchmoe/model_weights.bin")
    wf_json = os.path.join(PROJECT_ROOT, "finchmoe/model_weights.json")

    print(f"Loading weights from: {wf_path}")

    with open(wf_json, 'r') as f:
        manifest = json.load(f)

    print(f"Manifest: {manifest['num_tensors']} tensors from {manifest['model']}")

    # Print all tensor names for sanity check
    tensor_names = list(manifest['tensors'].keys())
    print(f"\nTensor names ({len(tensor_names)}):")
    for name in sorted(tensor_names)[:20]:
        info = manifest['tensors'][name]
        print(f"  {name}: shape={info['shape']}, dtype={info.get('dtype','?')}, "
              f"offset={info['offset']}, size={info['size']}")

    # Check for key naming differences
    lm_head_keys = [k for k in tensor_names if 'lm_head' in k]
    embed_keys = [k for k in tensor_names if 'embed' in k]
    norm_keys = [k for k in tensor_names if 'norm' in k]

    print(f"\nlm_head keys: {lm_head_keys}")
    print(f"embed keys: {embed_keys}")
    print(f"Norm weight keys (first 10): {norm_keys[:10]}")

    # Check layer 0 keys
    layer0_keys = [k for k in tensor_names if 'layers.0' in k]
    print(f"\nLayer 0 keys ({len(layer0_keys)}):")
    for name in sorted(layer0_keys):
        print(f"  {name}")

    # Now load the weight file and test embedding lookup
    wf = WeightFile(wf_path, wf_json)

    # Test with token ID 0 (usually a special token)
    print("\n--- Embedding Lookup Test ---")
    for tid in [0, 1, 100, 1000, 10000]:
        emb = embed_lookup_py(wf, tid)
        if emb is not None:
            print(f"Token {tid}: embedding mean={emb.mean():.6f}, std={emb.std():.6f}, "
                  f"min={emb.min():.6f}, max={emb.max():.6f}")

    # Test RMSNorm with layer 0 input norm
    print("\n--- RMSNorm Test (Layer 0 input_norm) ---")
    input_norm_w_u16 = wf.get_tensor_data("model.layers.0.input_layernorm.weight")
    if input_norm_w_u16 is not None:
        input_norm_w = np.array([bf16_to_f32(int(v)) for v in input_norm_w_u16.flatten()], dtype=np.float32)
        print(f"input_norm weight: shape={input_norm_w.shape}, mean={input_norm_w.mean():.6f}, "
              f"min={input_norm_w.min():.6f}, max={input_norm_w.max():.6f}")

        # Check if norm weights are around 1.0 (RMSNorm) or 0.0 (Qwen3_5RMSNorm)
        if input_norm_w.mean() < 0.1:
            print("  ⚠️  WARNING: Norm weights near 0! Qwen3_5RMSNorm needs 1+weight.")
            print("  If C code doesn't add 1.0, norm output will be ~0 → garbage.")
        else:
            print("  ✓ Norm weights look reasonable (near 1.0)")

    # Check post-attn norm too
    post_norm_w_u16 = wf.get_tensor_data("model.layers.0.post_attention_layernorm.weight")
    if post_norm_w_u16 is not None:
        post_norm_w = np.array([bf16_to_f32(int(v)) for v in post_norm_w_u16.flatten()], dtype=np.float32)
        print(f"\npost_attn_norm weight: mean={post_norm_w.mean():.6f}, "
              f"min={post_norm_w.min():.6f}, max={post_norm_w.max():.6f}")

    # Check final norm
    final_norm_w_u16 = wf.get_tensor_data("model.norm.weight")
    if final_norm_w_u16 is not None:
        final_norm_w = np.array([bf16_to_f32(int(v)) for v in final_norm_w_u16.flatten()], dtype=np.float32)
        print(f"\nfinal_norm weight: mean={final_norm_w.mean():.6f}, "
              f"min={final_norm_w.min():.6f}, max={final_norm_w.max():.6f}")


if __name__ == "__main__":
    main()
