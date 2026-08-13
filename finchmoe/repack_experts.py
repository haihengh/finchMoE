#!/usr/bin/env python3
"""Repack expert weights from scattered safetensors into contiguous per-layer binary files.

Creates one binary file per layer: packed_experts/layer_XX.bin
Each file = 256 experts x 1,769,472 bytes = ~432 MB
Expert E starts at byte offset E * 1,769,472

Within each expert block, 9 components packed in fixed order:
  gate_proj.weight, gate_proj.scales, gate_proj.biases,
  up_proj.weight,   up_proj.scales,   up_proj.biases,
  down_proj.weight,  down_proj.scales,  down_proj.biases

Usage:
    python repack_experts.py                    # repack all 40 layers
    python repack_experts.py --layers 0-4       # repack layers 0-4
    python repack_experts.py --layers 0,5,10    # repack specific layers
    python repack_experts.py --dry-run           # verify without writing
    python repack_experts.py --verify-only 0     # verify layer 0 against originals
"""

import argparse
import json
import os
import time
import sys
import numpy as np

# Component order and expected sizes (Qwen3.6-35B-A3B 4-bit)
COMPONENTS_4BIT = [
    {"name": "gate_proj.weight",  "offset": 0,        "size": 524288,  "dtype": "U32",  "shape": [512, 256]},
    {"name": "gate_proj.scales",  "offset": 524288,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "gate_proj.biases",  "offset": 557056,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.weight",    "offset": 589824,   "size": 524288,  "dtype": "U32",  "shape": [512, 256]},
    {"name": "up_proj.scales",    "offset": 1114112,  "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.biases",    "offset": 1146880,  "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "down_proj.weight",  "offset": 1179648,  "size": 524288,  "dtype": "U32",  "shape": [2048, 64]},
    {"name": "down_proj.scales",  "offset": 1703936,  "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
    {"name": "down_proj.biases",  "offset": 1736704,  "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
]

# 8-bit expert format: weights are 2x larger (4 values per uint32 instead of 8)
COMPONENTS_8BIT = [
    {"name": "gate_proj.weight",  "offset": 0,        "size": 1048576, "dtype": "U32",  "shape": [512, 512]},
    {"name": "gate_proj.scales",  "offset": 1048576,  "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "gate_proj.biases",  "offset": 1081344,  "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.weight",    "offset": 1114112,  "size": 1048576, "dtype": "U32",  "shape": [512, 512]},
    {"name": "up_proj.scales",    "offset": 2162688,  "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.biases",    "offset": 2195456,  "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "down_proj.weight",  "offset": 2228224,  "size": 1048576, "dtype": "U32",  "shape": [2048, 128]},
    {"name": "down_proj.scales",  "offset": 3276800,  "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
    {"name": "down_proj.biases",  "offset": 3309568,  "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
]

EXPERT_SIZE_4BIT = 1769472
EXPERT_SIZE_8BIT = 3342336
EXPERT_SIZE_1BIT = 589824
EXPERT_SIZE_2BIT = 983040

# Large-model variants (397B A17B: hidden=4096, intermediate=1024)
COMPONENTS_4BIT_LARGE = [
    {"name": "gate_proj.weight",  "offset": 0,        "size": 2097152, "dtype": "U32",  "shape": [1024, 512]},
    {"name": "gate_proj.scales",  "offset": 2097152,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
    {"name": "gate_proj.biases",  "offset": 2228224,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
    {"name": "up_proj.weight",    "offset": 2359296,  "size": 2097152, "dtype": "U32",  "shape": [1024, 512]},
    {"name": "up_proj.scales",    "offset": 4456448,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
    {"name": "up_proj.biases",    "offset": 4587520,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
    {"name": "down_proj.weight",  "offset": 4718592,  "size": 2097152, "dtype": "U32",  "shape": [4096, 256]},
    {"name": "down_proj.scales",  "offset": 6815744,  "size": 131072,  "dtype": "BF16", "shape": [4096, 16]},
    {"name": "down_proj.biases",  "offset": 6946816,  "size": 131072,  "dtype": "BF16", "shape": [4096, 16]},
]
EXPERT_SIZE_4BIT_LARGE = 7077888

NUM_EXPERTS = 256

# 1-bit expert format: 32 values per uint32
COMPONENTS_1BIT = [
    {"name": "gate_proj.weight",  "offset": 0,        "size": 131072,  "dtype": "U32",  "shape": [512, 64]},
    {"name": "gate_proj.scales",  "offset": 131072,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "gate_proj.biases",  "offset": 163840,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.weight",    "offset": 196608,   "size": 131072,  "dtype": "U32",  "shape": [512, 64]},
    {"name": "up_proj.scales",    "offset": 327680,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.biases",    "offset": 360448,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "down_proj.weight",  "offset": 393216,   "size": 131072,  "dtype": "U32",  "shape": [2048, 16]},
    {"name": "down_proj.scales",  "offset": 524288,   "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
    {"name": "down_proj.biases",  "offset": 557056,   "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
]

# 2-bit expert format: 16 values per uint32
# 3-bit expert format: 8 values packed per 24 bits (3 bytes), group_size=64.
# gate/up: [512, 2048] -> 2048*3/8 = 768 bytes/row -> 512*768 = 393216 bytes
# down:    [2048, 512] -> 512*3/8   = 192 bytes/row -> 2048*192 = 393216 bytes
EXPERT_SIZE_3BIT = 1376256
COMPONENTS_3BIT = [
    {"name": "gate_proj.weight",  "offset": 0,        "size": 393216,  "dtype": "U8",   "shape": [512, 768]},
    {"name": "gate_proj.scales",  "offset": 393216,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "gate_proj.biases",  "offset": 425984,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.weight",    "offset": 458752,   "size": 393216,  "dtype": "U8",   "shape": [512, 768]},
    {"name": "up_proj.scales",    "offset": 851968,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.biases",    "offset": 884736,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "down_proj.weight",  "offset": 917504,   "size": 393216,  "dtype": "U8",   "shape": [2048, 192]},
    {"name": "down_proj.scales",  "offset": 1310720,  "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
    {"name": "down_proj.biases",  "offset": 1343488,  "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
]

COMPONENTS_2BIT = [
    {"name": "gate_proj.weight",  "offset": 0,        "size": 262144,  "dtype": "U32",  "shape": [512, 128]},
    {"name": "gate_proj.scales",  "offset": 262144,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "gate_proj.biases",  "offset": 294912,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.weight",    "offset": 327680,   "size": 262144,  "dtype": "U32",  "shape": [512, 128]},
    {"name": "up_proj.scales",    "offset": 589824,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "up_proj.biases",    "offset": 622592,   "size": 32768,   "dtype": "BF16", "shape": [512, 32]},
    {"name": "down_proj.weight",  "offset": 655360,   "size": 262144,  "dtype": "U32",  "shape": [2048, 32]},
    {"name": "down_proj.scales",  "offset": 917504,   "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
    {"name": "down_proj.biases",  "offset": 950272,   "size": 32768,   "dtype": "BF16", "shape": [2048, 8]},
]
NUM_LAYERS = 40


def parse_layers(spec):
    """Parse layer specification like '0-4' or '0,5,10' or 'all'."""
    if spec is None or spec == 'all':
        return list(range(NUM_LAYERS))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_index(index_path):
    """Load expert_index.json and return expert_reads dict + model_path."""
    with open(index_path) as f:
        idx = json.load(f)
    return idx['expert_reads'], idx['model_path']


def verify_component_sizes(expert_reads, components):
    """Verify that component sizes in the index match expected sizes."""
    expected = {c['name']: c['size'] for c in components}
    for layer_key, comps in expert_reads.items():
        for comp_name, info in comps.items():
            if comp_name not in expected:
                print(f"WARNING: unknown component {comp_name} in layer {layer_key}")
                continue
            if info['expert_size'] != expected[comp_name]:
                print(f"MISMATCH: layer {layer_key}, {comp_name}: "
                      f"index says {info['expert_size']}, expected {expected[comp_name]}")
                return False
    print("Component sizes verified: all match expected layout")
    return True


def open_source_files(expert_reads, model_path, layers):
    """Open all needed safetensors files, return {filename: fd}."""
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"WARNING: layer {layer_idx} not found in expert_reads")
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info['file'])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source safetensors files")
    return fds


def bf16_encode(arr_f32):
    """Convert float32 -> bfloat16 (stored as uint16)."""
    return (arr_f32.view(np.uint32) >> 16).astype(np.uint16)


def bf16_decode(arr_u16):
    """Convert bfloat16 (uint16) -> float32."""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def pack_3bit(q_vals, group_size=64):
    """Pack quantized values (0..7) into 24-bit triplets (8 values per 3 bytes).

    q_vals: uint8 array of length divisible by 8.
    Returns uint8 array of length len(q_vals)*3/8.
    """
    n = len(q_vals)
    assert n % 8 == 0, f"3-bit packing requires multiple of 8 values, got {n}"
    q = q_vals.astype(np.uint64).reshape(-1, 8)
    acc = np.zeros(n // 8, dtype=np.uint64)
    for j in range(8):
        acc |= q[:, j] << (3 * j)
    b0 = (acc & 0xFF).astype(np.uint8)
    b1 = ((acc >> 8) & 0xFF).astype(np.uint8)
    b2 = ((acc >> 16) & 0xFF).astype(np.uint8)
    return np.stack([b0, b1, b2], axis=1).reshape(-1)


def quantize_affine_3bit(w_f32, group_size=64):
    """Per-group affine 3-bit quantization -> (packed_bytes, scales, biases)."""
    out_dim, in_dim = w_f32.shape
    assert in_dim % group_size == 0
    num_groups = in_dim // group_size
    w = w_f32.reshape(out_dim, num_groups, group_size)
    w_min = w.min(axis=2)
    w_max = w.max(axis=2)
    scales = np.maximum((w_max - w_min) / 7.0, 1e-8)
    biases = w_min
    q = np.round((w - biases[:, :, np.newaxis]) / scales[:, :, np.newaxis])
    q = np.clip(q, 0, 7).astype(np.uint8)
    packed = pack_3bit(q.reshape(-1))  # [out_dim * in_dim * 3/8]
    packed = packed.reshape(out_dim, in_dim * 3 // 8)
    return (packed,
            bf16_encode(scales.astype(np.float32).reshape(-1)).reshape(out_dim, num_groups),
            bf16_encode(biases.astype(np.float32).reshape(-1)).reshape(out_dim, num_groups))


def _dequant_packed_weights(packed_u32, scales_u16, biases_u16, in_dim, bits, group_size=64):
    """Dequant packed rows to float32 (raw BF16 scale/bias convention).

    Supports bits = 1, 2, 4 (values per uint32 = 32/bits).
    Vectorized: unpacks all value planes at once, then applies the
    per-group affine transform.
    """
    out_dim, packed_cols = packed_u32.shape
    num_groups = in_dim // group_size
    vpu = 32 // bits
    mask = (1 << bits) - 1
    w = np.zeros((out_dim, in_dim), dtype=np.float32)
    pk = packed_u32.astype(np.uint32)
    for n in range(vpu):
        w[:, n::vpu] = ((pk >> (bits * n)) & mask).astype(np.float32)
    for g in range(num_groups):
        s = bf16_decode(scales_u16[:, g]).astype(np.float32)[:, None]
        b = bf16_decode(biases_u16[:, g]).astype(np.float32)[:, None]
        w[:, g * group_size:(g + 1) * group_size] *= s
        w[:, g * group_size:(g + 1) * group_size] += b
    return w


def repack_all_3bit(expert_reads, model_path, output_dir, layers):
    """Repack experts at 3-bit: read 4-bit (or BF16) source experts from the
    original safetensors, dequant, requantize to 3-bit affine (group 64),
    and write packed_experts_3bit/layer_XX.bin."""
    os.makedirs(output_dir, exist_ok=True)
    fds = {}
    t0 = time.monotonic()
    for layer_idx in layers:
        key = str(layer_idx)
        if key not in expert_reads:
            print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
            continue
        li = expert_reads[key]
        out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
        fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(fd_out, NUM_EXPERTS * EXPERT_SIZE_3BIT)

        for e in range(NUM_EXPERTS):
            for comp in COMPONENTS_3BIT:
                name = comp['name']
                if name not in li:
                    print(f"  Layer {layer_idx} expert {e}: missing {name}")
                    continue
                info = li[name]
                if info['file'] not in fds:
                    fds[info['file']] = os.open(os.path.join(model_path, info['file']), os.O_RDONLY)
                src_fd = fds[info['file']]
                if name.endswith('.weight'):
                    # weights: dequant source (packed U32 or BF16) -> 3-bit;
                    # write the REQUANTIZED weight + its own scales/biases.
                    wraw = os.pread(src_fd, info['expert_size'], info['abs_offset'] + e * info['expert_stride'])
                    src_dtype = info.get('dtype', 'U32')
                    if src_dtype == 'U32':
                        wu32 = np.frombuffer(wraw, np.uint32).reshape(comp['shape'][0], -1)
                        sname = name[:-7] + '.scales'
                        sinfo = li[sname]
                        sraw = os.pread(src_fd, sinfo['expert_size'], sinfo['abs_offset'] + e * sinfo['expert_stride'])
                        su16 = np.frombuffer(sraw, np.uint16).reshape(comp['shape'][0], -1)
                        bname = name[:-7] + '.biases'
                        binfo = li[bname]
                        braw = os.pread(src_fd, binfo['expert_size'], binfo['abs_offset'] + e * binfo['expert_stride'])
                        bu16 = np.frombuffer(braw, np.uint16).reshape(comp['shape'][0], -1)
                        # Detect source packing: bits = 32 * row_u32 / (groups * group_size)
                        row_u32 = wu32.shape[1]
                        num_groups = su16.shape[1]
                        src_bits = (row_u32 * 32) // (num_groups * 64)
                        in_dim = num_groups * 64
                        w_f32 = _dequant_packed_weights(wu32, su16, bu16, in_dim, src_bits)
                    else:
                        wu16 = np.frombuffer(wraw, np.uint16)
                        w_f32 = (wu16.astype(np.uint32) << 16).view(np.float32).reshape(comp['shape'][0], -1)
                    packed, scales, biases = quantize_affine_3bit(w_f32)
                    os.pwrite(fd_out, packed.tobytes(), e * EXPERT_SIZE_3BIT + comp['offset'])
                    # write requantized scales/biases at their 3-bit offsets
                    soff = next(c['offset'] for c in COMPONENTS_3BIT if c['name'] == name[:-7] + '.scales')
                    boff = next(c['offset'] for c in COMPONENTS_3BIT if c['name'] == name[:-7] + '.biases')
                    os.pwrite(fd_out, scales.tobytes(), e * EXPERT_SIZE_3BIT + soff)
                    os.pwrite(fd_out, biases.tobytes(), e * EXPERT_SIZE_3BIT + boff)
                    continue
                # .scales/.biases components are written by the .weight branch above
                continue
        os.close(fd_out)
        print(f"  Layer {layer_idx:2d}: done ({time.monotonic() - t0:.1f}s elapsed)")
    for fd in fds.values():
        os.close(fd)
    print(f"3-bit repack complete -> {output_dir}")


def repack_layer(layer_idx, expert_reads, model_path, fds, output_dir, components, expert_size, dry_run=False, fp16_scales=False):
    """Repack all experts for one layer into a contiguous binary file.

    Returns (bytes_written, elapsed_seconds).
    """
    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return 0, 0.0

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
    layer_size = NUM_EXPERTS * expert_size

    if dry_run:
        # Just verify we can compute all offsets
        for expert_idx in range(NUM_EXPERTS):
            for comp in components:
                info = layer_info[comp['name']]
                src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
                dst_offset = expert_idx * expert_size + comp['offset']
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {layer_size:,} bytes to {out_path}")
        return layer_size, 0.0

    t0 = time.monotonic()

    # Pre-allocate output file with zeros
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, layer_size)

    bytes_written = 0

    # Build read plan: group reads by source file for better locality
    # Each entry: (src_fd, src_offset, dst_offset, size, needs_fp16_convert)
    read_plan = []
    for expert_idx in range(NUM_EXPERTS):
        for comp in components:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']
            # Some mlx-community models store scales/biases as FP16 with
            # dtype='BF16'. Our self-quantized models (incl. Qwen3.6-35B-A3B-4bit)
            # store genuine BF16 data with the same dtype label — converting
            # those corrupts the values (BF16 -0.0041 misread as FP16 -0.945).
            # Only convert when --fp16-scales is explicitly passed.
            is_scale_or_bias = ('scales' in comp['name'] or 'biases' in comp['name'])
            needs_fp16_convert = fp16_scales and is_scale_or_bias and info.get('dtype') == 'BF16'
            read_plan.append((src_fd, src_offset, dst_offset, comp['size'], needs_fp16_convert))

    # Sort by (src_fd, src_offset) for sequential read locality
    read_plan.sort(key=lambda x: (x[0], x[1]))

    # Execute reads and writes, converting FP16 → BF16 for scales/biases
    for src_fd, src_offset, dst_offset, size, needs_convert in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} "
                          f"at offset {src_offset}")
        if needs_convert:
            # FP16 (uint16) → BF16 (uint16): decode as float16, re-encode as bfloat16
            arr = np.frombuffer(data, dtype=np.uint16)
            # View as float16, convert to float32
            f16 = arr.view(np.float16).astype(np.float32)
            # Convert float32 to bfloat16: take upper 16 bits
            bf16 = (f16.view(np.uint32) >> 16).astype(np.uint16)
            data = bf16.tobytes()
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    elapsed = time.monotonic() - t0

    return bytes_written, elapsed


def verify_layer(layer_idx, expert_reads, model_path, fds, output_dir, components, expert_size, fp16_scales=False):
    """Read back expert 0 from packed file and compare to originals."""
    layer_key = str(layer_idx)
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)

    mismatches = 0
    for expert_idx in [0, 1, 127, 255]:  # spot check several experts
        for comp in components:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * expert_size + comp['offset']

            original = os.pread(src_fd, comp['size'], src_offset)
            packed = os.pread(fd_packed, comp['size'], dst_offset)

            # For scales/biases: source may be FP16 (dtype=BF16) or BF16 (dtype=U16)
            is_sb = ('scales' in comp['name'] or 'biases' in comp['name'])
            if is_sb:
                src_u16 = np.frombuffer(original, dtype=np.uint16)
                pck_u16 = np.frombuffer(packed, dtype=np.uint16)
                src_dtype = info.get('dtype', 'BF16')
                if fp16_scales and src_dtype == 'BF16':
                    # mlx-community model: source is FP16, packed is converted BF16
                    src_f32 = src_u16.view(np.float16).astype(np.float32)
                    pck_f32 = (pck_u16.astype(np.uint32) << 16).view(np.float32)
                else:
                    # Self-quantized model (default): both source and packed are BF16
                    src_f32 = (src_u16.astype(np.uint32) << 16).view(np.float32)
                    pck_f32 = (pck_u16.astype(np.uint32) << 16).view(np.float32)
                if not np.allclose(src_f32, pck_f32, rtol=1e-2, atol=1e-3, equal_nan=True):
                    md = np.max(np.abs(src_f32 - pck_f32))
                    if md > 1e-2:  # only report significant diffs
                        print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']} max_diff={md:.6e}")
                        mismatches += 1
            elif original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)

    if mismatches == 0:
        print(f"  Layer {layer_idx}: verification PASSED")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")

    return mismatches == 0


def write_layout(output_dir, components, expert_size):
    """Write layout.json describing the packed format."""
    layout = {
        "expert_size": expert_size,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "components": components,
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Repack expert weights into contiguous per-layer binary files")
    parser.add_argument('--index', default='expert_index.json',
                        help='Path to expert_index.json')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--bits', type=int, default=4, choices=[1, 2, 3, 4, 8],
                        help='Quantization bits: 1, 2, 4, or 8 (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify offsets without writing')
    parser.add_argument('--fp16-scales', action='store_true',
                        help='Convert scale/bias tensors from FP16 to BF16 '
                             '(needed for some mlx-community models whose '
                             'dtype="BF16" actually holds FP16 data). '
                             'DEFAULT OFF: our self-quantized models store '
                             'genuine BF16 data, and the conversion corrupts it.')
    parser.add_argument('--verify-only', type=int, default=None, metavar='LAYER',
                        help='Verify a specific layer against originals')
    args = parser.parse_args()

    # Select component layout based on bits
    if args.bits == 8:
        components = COMPONENTS_8BIT
        expert_size = EXPERT_SIZE_8BIT
        dirname = "packed_experts_8bit"
    elif args.bits == 3:
        components = COMPONENTS_3BIT
        expert_size = EXPERT_SIZE_3BIT
        dirname = "packed_experts_3bit"
    elif args.bits == 2:
        components = COMPONENTS_2BIT
        expert_size = EXPERT_SIZE_2BIT
        dirname = "packed_experts_2bit"
    elif args.bits == 1:
        components = COMPONENTS_1BIT
        expert_size = EXPERT_SIZE_1BIT
        dirname = "packed_experts_1bit"
    else:
        components = COMPONENTS_4BIT
        expert_size = EXPERT_SIZE_4BIT
        dirname = "packed_experts"

    print(f"Using {args.bits}-bit expert format (expert_size={expert_size} bytes, dir={dirname})")

    print("Loading expert index...")
    expert_reads, model_path = load_index(args.index)
    print(f"Model path: {model_path}")
    print(f"Layers in index: {len(expert_reads)}")
    output_dir = os.path.join(model_path, dirname)

    # Verify component sizes — auto-detect large model (397B) vs standard (35B)
    # (skipped for 3-bit: the source experts are 4-bit/BF16 and get
    # dequant->requantized, so sizes differ by design)
    if args.bits != 3 and not verify_component_sizes(expert_reads, components):
        # Try large-model variants
        if args.bits == 4:
            components = COMPONENTS_4BIT_LARGE; expert_size = EXPERT_SIZE_4BIT_LARGE
        elif args.bits == 8:
            # Use scaled 8-bit for large model
            components = [dict(c) for c in COMPONENTS_8BIT]
            for c in components: c['size'] *= 4  # 4x larger weights
            expert_size = EXPERT_SIZE_8BIT * 4
        else:
            print("ABORTING: large model only supported at 4-bit currently")
            sys.exit(1)
        if not verify_component_sizes(expert_reads, components):
            print("ABORTING: component size mismatch even with large-model layout")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine which layers to process
    if args.verify_only is not None:
        layers = [args.verify_only]
    else:
        layers = parse_layers(args.layers)

    print(f"Layers to process: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    if args.bits == 3:
        # 3-bit requires dequant->requant (source experts are 4-bit or BF16).
        # Bypasses the raw-copy repack below and the component-size check.
        repack_all_3bit(expert_reads, model_path, output_dir, layers)
        return

    layer_total_size = NUM_EXPERTS * expert_size

    if not args.dry_run and args.verify_only is None:
        total_bytes = len(layers) * layer_total_size
        print(f"Total data to write: {total_bytes / (1024**3):.1f} GB")

        # Check free disk space
        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = total_bytes / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < total_bytes:
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB but only {free_gb:.1f} GB free.")
            sys.exit(1)

    # Open source files
    fds = open_source_files(expert_reads, model_path, layers)

    if args.verify_only is not None:
        verify_layer(args.verify_only, expert_reads, model_path, fds, output_dir, components, expert_size,
                     fp16_scales=args.fp16_scales)
        for fd in fds.values():
            os.close(fd)
        return

    # Write layout.json
    write_layout(output_dir, components, expert_size)

    # Repack each layer
    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        t_layer = time.monotonic()
        bytes_written, elapsed = repack_layer(
            layer_idx, expert_reads, model_path, fds, output_dir, components, expert_size,
            dry_run=args.dry_run, fp16_scales=args.fp16_scales
        )
        total_written += bytes_written

        if not args.dry_run and bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float('inf')
            overall_elapsed = time.monotonic() - t_start
            overall_throughput = total_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
            eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
            print(f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                  f"({throughput:.1f} GB/s) | "
                  f"Total: {total_written/1024**3:.1f}/{len(layers)*layer_total_size/1024**3:.1f} GB "
                  f"({overall_throughput:.1f} GB/s avg) | "
                  f"ETA: {eta:.0f}s")

            # Verify this layer immediately
            if not verify_layer(layer_idx, expert_reads, model_path, fds, output_dir, components, expert_size,
                                fp16_scales=args.fp16_scales):
                print(f"ABORTING: verification failed for layer {layer_idx}")
                sys.exit(1)

    # Close source files
    for fd in fds.values():
        os.close(fd)

    # Final summary
    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\n{'='*60}")
        print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
        print(f"Time: {total_elapsed:.1f}s")
        print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
        print(f"Output: {output_dir}")
    elif args.dry_run:
        print(f"\nDRY RUN complete: {len(layers)} layers validated")


if __name__ == '__main__':
    main()
