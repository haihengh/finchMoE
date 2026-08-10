#!/usr/bin/env python3
"""
Extract and quantize MTP expert weights from safetensors shards.

The MTP layer (layer 40) has 2 expert tensors in the official Qwen3.6-35B-A3B:
  - mtp.layers.0.mlp.experts.gate_up_proj  [256, 1024, 2048] BF16
  - mtp.layers.0.mlp.experts.down_proj     [256, 2048, 512]  BF16

These need to be quantized to 4-bit (matching the main model format) and packed
into a single layer_40.bin file with the standard expert layout:
  [gate_W][gate_S][gate_B][up_W][up_S][up_B][down_W][down_S][down_B]

Usage:
  python3 extract_mtp_experts.py [--bits 4] [--output packed_experts/layer_40.bin]
"""
import json, struct, os, sys
import numpy as np

# Paths
MTP_DIR = "../models/Qwen3.6-35B-A3B-MTP-essentials"
OUTPUT_DIR = "."
GROUP_SIZE = 64
BITS = 4

def bf16_to_f32(bf16_bytes):
    """Convert BF16 bytes to float32 numpy array."""
    arr = np.frombuffer(bf16_bytes, dtype=np.uint16)
    f32 = np.zeros(len(arr), dtype=np.float32)
    bits = arr.astype(np.uint32) << 16
    f32 = np.frombuffer(bits.tobytes(), dtype=np.float32).copy()
    return f32

def f32_to_bf16(f32_arr):
    """Convert float32 numpy array to BF16 bytes."""
    f32 = np.asarray(f32_arr, dtype=np.float32)
    bits = np.frombuffer(f32.tobytes(), dtype=np.uint32)
    return (bits >> 16).astype(np.uint16).tobytes()

def quantize_4bit_affine(weight_f32, group_size=64):
    """
    MLX-style affine 4-bit quantization.
    Returns (packed_uint32, scales_bf16, biases_bf16).
    packed_uint32 shape: [out_dim, in_dim // 8]
    scales/biases shape: [out_dim, in_dim // group_size]
    """
    out_dim, in_dim = weight_f32.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by group_size {group_size}"
    assert in_dim % 8 == 0, f"in_dim {in_dim} not divisible by 8"

    num_groups = in_dim // group_size
    packed = np.zeros((out_dim, in_dim // 8), dtype=np.uint32)
    scales = np.zeros((out_dim, num_groups), dtype=np.float32)
    biases = np.zeros((out_dim, num_groups), dtype=np.float32)

    for row in range(out_dim):
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            chunk = weight_f32[row, start:end].astype(np.float64)

            w_min = float(chunk.min())
            w_max = float(chunk.max())
            scale = (w_max - w_min) / 15.0 if w_max > w_min else 1.0
            bias = w_min

            # Quantize: q = round((w - bias) / scale), clamp to [0, 15]
            q = np.clip(np.round((weight_f32[row, start:end] - bias) / scale), 0, 15).astype(np.uint32)

            # Pack 8 x 4-bit values per uint32
            for i in range(8):
                word_start = i * 8
                word_end = word_start + 8
                word = 0
                for j in range(8):
                    word |= int(q[word_start + j]) << (j * 4)
                packed[row, g * 8 + i] = word

            scales[row, g] = scale
            biases[row, g] = bias

    return packed, scales, biases

def extract_tensor(safetensors_path, tensor_name):
    """Extract a single tensor from a safetensors file."""
    with open(safetensors_path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))

    if tensor_name not in header:
        raise KeyError(f"Tensor '{tensor_name}' not found in {safetensors_path}")

    info = header[tensor_name]
    offset = 8 + header_len + info['data_offsets'][0]
    length = info['data_offsets'][1] - info['data_offsets'][0]

    with open(safetensors_path, 'rb') as f:
        f.seek(offset)
        data = f.read(length)

    dtype = info['dtype']
    shape = info['shape']
    return data, dtype, shape

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract and quantize MTP expert weights')
    parser.add_argument('--bits', type=int, default=4, help='Quantization bits (default: 4)')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--mtp-dir', type=str, default=MTP_DIR, help='MTP safetensors directory')
    args = parser.parse_args()

    bits = args.bits
    mtp_dir = args.mtp_dir
    output_path = args.output or os.path.join(OUTPUT_DIR, f'packed_experts/layer_40.bin')

    print(f"Extracting MTP experts from: {mtp_dir}")
    print(f"Quantization: {bits}-bit affine (group_size={GROUP_SIZE})")
    print(f"Output: {output_path}")

    # Load MTP tensors
    gate_up_data, gate_up_dtype, gate_up_shape = extract_tensor(
        os.path.join(mtp_dir, 'model-00025-of-00026.safetensors'),
        'mtp.layers.0.mlp.experts.gate_up_proj')

    down_data, down_dtype, down_shape = extract_tensor(
        os.path.join(mtp_dir, 'model-00026-of-00026.safetensors'),
        'mtp.layers.0.mlp.experts.down_proj')

    print(f"\n  gate_up_proj: shape={gate_up_shape}, dtype={gate_up_dtype}")
    print(f"  down_proj:    shape={down_shape}, dtype={down_dtype}")

    # Convert BF16 to float32
    gate_up_f32 = bf16_to_f32(gate_up_data).reshape(gate_up_shape)
    down_f32 = bf16_to_f32(down_data).reshape(down_shape)

    # gate_up_proj shape: [256, 1024, 2048]
    # This is gate+up concatenated: first 512 is gate_proj, last 512 is up_proj
    num_experts, gate_up_dim, in_dim = gate_up_shape
    assert gate_up_dim == 1024, f"Expected gate_up_dim=1024, got {gate_up_dim}"
    gate_dim = gate_up_dim // 2  # 512
    up_dim = gate_up_dim // 2    # 512

    # Split gate_up into separate gate and up
    gate_w = gate_up_f32[:, :gate_dim, :].reshape(num_experts * gate_dim, in_dim)
    up_w = gate_up_f32[:, gate_dim:, :].reshape(num_experts * up_dim, in_dim)

    # down_proj shape: [256, 2048, 512]
    down_w = down_f32.reshape(num_experts * down_shape[1], down_shape[2])

    out_dim_gate = num_experts * gate_dim  # 256 * 512 = 131072
    out_dim_up = num_experts * gate_dim     # same
    out_dim_down = num_experts * down_shape[1]  # 256 * 2048 = 524288

    print(f"\n  gate_proj:  [{out_dim_gate}, {in_dim}] -> quantizing...")
    gate_packed, gate_scales, gate_biases = quantize_4bit_affine(gate_w, GROUP_SIZE)

    print(f"  up_proj:    [{out_dim_up}, {in_dim}] -> quantizing...")
    up_packed, up_scales, up_biases = quantize_4bit_affine(up_w, GROUP_SIZE)

    print(f"  down_proj:  [{out_dim_down}, {gate_dim}] -> quantizing...")
    down_packed, down_scales, down_biases = quantize_4bit_affine(down_w, GROUP_SIZE)

    # Compute expert size (same layout as main model)
    # Layout: gate_W, gate_S, gate_B, up_W, up_S, up_B, down_W, down_S, down_B
    gate_w_bytes = len(gate_packed.tobytes())
    gate_s_bytes = len(f32_to_bf16(gate_scales))
    gate_b_bytes = len(f32_to_bf16(gate_biases))
    up_w_bytes = len(up_packed.tobytes())
    up_s_bytes = len(f32_to_bf16(up_scales))
    up_b_bytes = len(f32_to_bf16(up_biases))
    down_w_bytes = len(down_packed.tobytes())
    down_s_bytes = len(f32_to_bf16(down_scales))
    down_b_bytes = len(f32_to_bf16(down_biases))

    # Offsets
    GATE_W_OFF = 0
    GATE_S_OFF = GATE_W_OFF + gate_w_bytes
    GATE_B_OFF = GATE_S_OFF + gate_s_bytes
    UP_W_OFF   = GATE_B_OFF + gate_b_bytes
    UP_S_OFF   = UP_W_OFF + up_w_bytes
    UP_B_OFF   = UP_S_OFF + up_s_bytes
    DOWN_W_OFF = UP_B_OFF + up_b_bytes
    DOWN_S_OFF = DOWN_W_OFF + down_w_bytes
    DOWN_B_OFF = DOWN_S_OFF + down_s_bytes
    TOTAL_SIZE = DOWN_B_OFF + down_b_bytes

    print(f"\n  Expert layout:")
    print(f"    gate_W:  {gate_w_bytes:>12,} bytes (offset {GATE_W_OFF})")
    print(f"    gate_S:  {gate_s_bytes:>12,} bytes (offset {GATE_S_OFF})")
    print(f"    gate_B:  {gate_b_bytes:>12,} bytes (offset {GATE_B_OFF})")
    print(f"    up_W:    {up_w_bytes:>12,} bytes (offset {UP_W_OFF})")
    print(f"    up_S:    {up_s_bytes:>12,} bytes (offset {UP_S_OFF})")
    print(f"    up_B:    {up_b_bytes:>12,} bytes (offset {UP_B_OFF})")
    print(f"    down_W:  {down_w_bytes:>12,} bytes (offset {DOWN_W_OFF})")
    print(f"    down_S:  {down_s_bytes:>12,} bytes (offset {DOWN_S_OFF})")
    print(f"    down_B:  {down_b_bytes:>12,} bytes (offset {DOWN_B_OFF})")
    print(f"    TOTAL:   {TOTAL_SIZE:>12,} bytes ({TOTAL_SIZE/1e6:.1f} MB)")

    # Each expert is TOTAL_SIZE / 256 bytes
    expert_size = TOTAL_SIZE // num_experts
    print(f"\n  Per-expert size: {expert_size:,} bytes")
    print(f"  Expected EXPERT_SIZE_4BIT: 1,769,472 bytes")

    # Write the layer file (all 256 experts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        for expert_idx in range(num_experts):
            # Gate W
            start_row = expert_idx * gate_dim
            end_row = start_row + gate_dim
            f.write(gate_packed[start_row:end_row].tobytes())
            f.write(f32_to_bf16(gate_scales[start_row:end_row]))
            f.write(f32_to_bf16(gate_biases[start_row:end_row]))
            # Up W
            f.write(up_packed[start_row:end_row].tobytes())
            f.write(f32_to_bf16(up_scales[start_row:end_row]))
            f.write(f32_to_bf16(up_biases[start_row:end_row]))
            # Down W
            start_row_d = expert_idx * down_shape[1]
            end_row_d = start_row_d + down_shape[1]
            f.write(down_packed[start_row_d:end_row_d].tobytes())
            f.write(f32_to_bf16(down_scales[start_row_d:end_row_d]))
            f.write(f32_to_bf16(down_biases[start_row_d:end_row_d]))

    actual_size = os.path.getsize(output_path)
    print(f"\n  Wrote {output_path}: {actual_size:,} bytes ({actual_size/1e6:.1f} MB)")
    print(f"  Expected: {num_experts} experts x {expert_size:,} = {num_experts * expert_size:,} bytes")
    assert actual_size == num_experts * expert_size, f"Size mismatch!"
    print("  ✅ Done!")

if __name__ == '__main__':
    main()
