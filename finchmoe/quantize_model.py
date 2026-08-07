#!/usr/bin/env python3
"""
Quantize BF16 Qwen3.6 model to MLX-compatible 4-bit/8-bit safetensors.

Produces output identical in structure to mlx-community quantized models.
Then use the existing extract_weights.py + repack_experts.py pipeline.

Usage:
    python quantize_model.py --model ../models/Qwen3.6-35B-A3B-bf16 \\
                             --output ../models/Qwen3.6-35B-A3B-4bit-custom
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import defaultdict
import numpy as np
from safetensors.numpy import save_file


def bf16_encode(arr_f32):
    """Encode float32 array as bfloat16 uint16."""
    return (arr_f32.view(np.uint32) >> 16).astype(np.uint16)


def quantize_affine(weights_f32, bits, group_size):
    """
    MLX-style affine quantization: w_q = round((w - bias) / scale)

    Returns (packed_u32, scales_bf16, biases_bf16)
    packed_u32: [out_dim, in_dim * bits / 32] — 8 values per U32 for 4-bit
    """
    out_dim, in_dim = weights_f32.shape
    num_groups = in_dim // group_size
    max_val = (1 << bits) - 1
    values_per_u32 = 32 // bits

    w = weights_f32.reshape(out_dim, num_groups, group_size)

    # Per-group min (bias) and max → scale
    w_min = w.min(axis=2)
    w_max = w.max(axis=2)
    scales = np.maximum((w_max - w_min) / max_val, 1e-8)
    biases = w_min

    # Quantize and clamp
    q = np.round((w - biases[:, :, np.newaxis]) / scales[:, :, np.newaxis])
    q = np.clip(q, 0, max_val).astype(np.uint8)

    # Pack into U32
    packed_cols = in_dim // values_per_u32
    u32_per_group = group_size // values_per_u32
    packed = np.zeros((out_dim, packed_cols), dtype=np.uint32)

    for g in range(num_groups):
        for u in range(u32_per_group):
            u32_val = np.zeros(out_dim, dtype=np.uint32)
            for v in range(values_per_u32):
                u32_val |= q[:, g, u * values_per_u32 + v].astype(np.uint32) << (v * bits)
            packed[:, g * u32_per_group + u] = u32_val

    scales_bf16 = bf16_encode(scales.flatten()).reshape(out_dim, num_groups)
    biases_bf16 = bf16_encode(biases.flatten()).reshape(out_dim, num_groups)

    return packed, scales_bf16, biases_bf16


# Tensors that use 8-bit for better routing precision
EIGHT_BIT_PATTERNS = ['.mlp.gate.weight', '.mlp.shared_expert_gate.weight']


def should_use_8bit(name):
    return any(p in name for p in EIGHT_BIT_PATTERNS)


def main():
    parser = argparse.ArgumentParser(description='Quantize Qwen3.6 BF16 → MLX 4-bit')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    model_path = args.model
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    # Load index
    with open(os.path.join(model_path, 'model.safetensors.index.json')) as f:
        idx = json.load(f)
    weight_map = idx['weight_map']

    # Copy metadata files
    import shutil
    for fname in os.listdir(model_path):
        if not fname.endswith('.safetensors'):
            src = os.path.join(model_path, fname)
            dst = os.path.join(output_path, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

    # Group by output shard (distribute evenly across 4 shards)
    all_tensors = sorted(weight_map.keys())
    n = len(all_tensors)
    shard_size = (n + 3) // 4
    shards = [all_tensors[i * shard_size:(i + 1) * shard_size] for i in range(4)]

    # Read quantization config from original model if present
    qc = {}
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        qc = cfg.get('quantization_config', {})

    new_weight_map = {}
    total_start = time.time()

    for shard_idx, tensor_names in enumerate(shards):
        if not tensor_names:
            break

        shard_name = f"model-{shard_idx + 1:05d}-of-00004.safetensors"
        print(f"\nShard {shard_idx + 1}/4: {len(tensor_names)} tensors")

        # Collect unique source files for this shard
        src_files = set(weight_map[n] for n in tensor_names)

        # Cache opened files and their headers
        file_cache = {}
        for fname in src_files:
            fpath = os.path.join(model_path, fname)
            with open(fpath, 'rb') as f:
                hlen = struct.unpack('<Q', f.read(8))[0]
                file_cache[fname] = {
                    'path': fpath,
                    'header': json.loads(f.read(hlen)),
                    'data_start': 8 + hlen,
                }

        tensors_out = {}
        t0 = time.time()

        for i, name in enumerate(tensor_names):
            src_fname = weight_map[name]
            fc = file_cache[src_fname]
            hdr = fc['header']
            ds = fc['data_start']

            if name not in hdr:
                print(f"  WARNING: {name} not in {src_fname}, skipping")
                continue

            info = hdr[name]
            doff = info['data_offsets']
            shape = info['shape']
            dtype = info['dtype']
            size = doff[1] - doff[0]

            with open(fc['path'], 'rb') as f:
                f.seek(ds + doff[0])
                raw = f.read(size)

            # Parse BF16 or F32
            if dtype == 'BF16':
                arr_u16 = np.frombuffer(raw, dtype=np.uint16)
                arr_f32 = (arr_u16.astype(np.uint32) << 16).view(np.float32)
            elif dtype == 'F32':
                arr_f32 = np.frombuffer(raw, dtype=np.float32)
            else:
                tensors_out[name] = raw  # keep as-is
                new_weight_map[name] = shard_name
                continue

            # Quantize weight tensors
            if '.weight' in name and len(shape) >= 2:
                bits = 8 if should_use_8bit(name) else 4

                if len(shape) == 2:
                    out_dim, in_dim = shape
                    if in_dim % 64 == 0:
                        w = arr_f32.reshape(out_dim, in_dim)
                        packed, scales, biases = quantize_affine(w, bits, 64)

                        tensors_out[name] = packed
                        sname = name.replace('.weight', '.scales')
                        bname = name.replace('.weight', '.biases')
                        tensors_out[sname] = scales
                        tensors_out[bname] = biases
                        new_weight_map[name] = shard_name
                        new_weight_map[sname] = shard_name
                        new_weight_map[bname] = shard_name
                    else:
                        tensors_out[name] = raw  # keep as BF16
                        new_weight_map[name] = shard_name

                elif len(shape) == 3:
                    # Expert tensor: [n_experts, out_dim, in_packed]
                    n_exp, out_d, in_p = shape
                    w = arr_f32.reshape(n_exp, out_d, -1)
                    actual_in = w.shape[-1]
                    if actual_in % 64 == 0:
                        packed_all = []
                        scales_all = []
                        biases_all = []
                        for e in range(n_exp):
                            p, s, b = quantize_affine(w[e].reshape(out_d, actual_in), bits, 64)
                            packed_all.append(p)
                            scales_all.append(s)
                            biases_all.append(b)

                        tensors_out[name] = np.stack(packed_all)
                        sname = name.replace('.weight', '.scales')
                        bname = name.replace('.weight', '.biases')
                        tensors_out[sname] = np.stack(scales_all)
                        tensors_out[bname] = np.stack(biases_all)
                        new_weight_map[name] = shard_name
                        new_weight_map[sname] = shard_name
                        new_weight_map[bname] = shard_name
                    else:
                        tensors_out[name] = raw
                        new_weight_map[name] = shard_name
                else:
                    tensors_out[name] = raw
                    new_weight_map[name] = shard_name
            else:
                # Non-weight: keep as BF16
                tensors_out[name] = raw if isinstance(raw, bytes) else arr_f32.tobytes() if dtype == 'F32' else arr_u16.tobytes()
                new_weight_map[name] = shard_name

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(tensor_names)}]")

        # Write shard
        shard_path = os.path.join(output_path, shard_name)
        save_file(tensors_out, shard_path)
        sz = os.path.getsize(shard_path) / 1e9
        elapsed = time.time() - t0
        print(f"  Wrote {sz:.2f} GB in {elapsed:.0f}s ({sz / elapsed:.2f} GB/s)")

    # Write index
    index_out = {
        'metadata': {'total_size': 0},
        'weight_map': new_weight_map,
    }
    with open(os.path.join(output_path, 'model.safetensors.index.json'), 'w') as f:
        json.dump(index_out, f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"\nDone in {total_elapsed:.0f}s")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
