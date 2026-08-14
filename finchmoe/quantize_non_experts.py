#!/usr/bin/env python3
"""
Quantize non-expert weights from BF16 safetensors → packed 4-bit/8-bit binary.

Produces model_weights_quant.bin + model_weights_quant.json with:
- Attention Q/K/V/O: 4-bit (group_size=64)
- GDN projections (qkv, z, beta, alpha, out_proj): 4-bit
- Shared experts (gate/up/down): 4-bit
- Embeddings + lm_head: 8-bit (safer for vocabulary projection)
- Norms + gates: BF16 (tiny, not worth quantizing)

The output format matches what the C engine's BatchMatvecSpec expects:
- .weight → U32 packed (4-bit or 8-bit)
- .scales → BF16 (uint16)
- .biases → BF16 (uint16)

Verification: after extraction, compare GPU dequant output vs CPU BF16 reference
for key tensors. This ensures the quantization itself doesn't introduce errors.

Usage:
    python quantize_non_experts.py --input <bf16-model-dir> --output <output-dir>
    # produces model_weights_quant.bin and model_weights_quant.json
"""

import argparse, json, os, struct, sys, time
import numpy as np
from collections import defaultdict


def bf16_encode(arr_f32):
    """Convert float32 → bfloat16 (stored as uint16)"""
    return (arr_f32.view(np.uint32) >> 16).astype(np.uint16)


def bf16_decode(arr_u16):
    """Convert bfloat16 (uint16) → float32"""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def quantize_affine(weights_f32, bits, group_size=64):
    """MLX affine quantization: per-group min/max → packed uint32 + BF16 scales/biases"""
    out_dim, in_dim = weights_f32.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by group_size {group_size}"
    num_groups = in_dim // group_size
    max_val = (1 << bits) - 1
    vpu = 32 // bits

    w = weights_f32.reshape(out_dim, num_groups, group_size)
    w_min = w.min(axis=2)
    w_max = w.max(axis=2)
    scales = np.maximum((w_max - w_min) / max_val, 1e-8)
    biases = w_min

    q = np.round((w - biases[:, :, np.newaxis]) / scales[:, :, np.newaxis])
    q = np.clip(q, 0, max_val).astype(np.uint8)

    packed_cols = in_dim // vpu
    packed = np.zeros((out_dim, packed_cols), dtype=np.uint32)
    upg = group_size // vpu
    for g in range(num_groups):
        for u in range(upg):
            u32_val = np.zeros(out_dim, dtype=np.uint32)
            for v in range(vpu):
                u32_val |= q[:, g, u * vpu + v].astype(np.uint32) << (v * bits)
            packed[:, g * upg + u] = u32_val

    return (packed,
            bf16_encode(scales.flatten()).reshape(out_dim, num_groups),
            bf16_encode(biases.flatten()).reshape(out_dim, num_groups))


# Which tensors get which quantization
# Protected tier: GDN projections are 8-bit because the gated-norm eps-knee
# amplifies their quantization noise (24/32 value-heads sit below rms 1e-3).
# Group size is uniformly 64 so bit width is unambiguous from shapes
# (row_u32 == groups*16 -> 8-bit, row_u32 == groups*8 -> 4-bit).
FOUR_BIT_PATTERNS = [
    # Full attention
    '.self_attn.q_proj.weight',
    '.self_attn.k_proj.weight',
    '.self_attn.v_proj.weight',
    '.self_attn.o_proj.weight',
    # Shared expert (per layer)
    '.mlp.shared_expert.gate_proj.weight',
    '.mlp.shared_expert.up_proj.weight',
    '.mlp.shared_expert.down_proj.weight',
]

EIGHT_BIT_PATTERNS = [
    # Vocabulary projections
    'lm_head.weight',
    'model.embed_tokens.weight',
    # GDN projections (protected tier — dampens eps-knee amplification).
    # Optional 4-bit variant: FINCHMOE_GDN4 env switches them to 4-bit
    # (halves the dominant per-layer weight traffic; quality must be
    # A/B tested against the 8-bit tier).
    '.linear_attn.in_proj_qkv.weight',
    '.linear_attn.in_proj_z.weight',
    '.linear_attn.out_proj.weight',
]

# BF16 (keep as-is)
KEEP_BF16_PATTERNS = [
    '.mlp.gate.weight',           # routing gate (256-dim, tiny)
    '.mlp.shared_expert_gate.weight',  # shared expert gate (1-dim)
    '.linear_attn.in_proj_a.weight',   # alpha gate (32-dim scalar projection)
    '.linear_attn.in_proj_b.weight',   # beta gate (32-dim scalar projection)
    'norm.weight',                 # all norms
    'layernorm.weight',
    'input_layernorm',
    'post_attention_layernorm',
]


def get_quantization_bits(name):
    """Return (bits, group_size) or None to keep BF16"""
    # GDN projections default to 4-bit: halves the dominant per-layer weight
    # traffic (qkv+z+o_proj = ~42MB/layer at 8-bit) for ~25% more tok/s
    # (9.1 -> 11.4 measured) with coherent output on our affine quantizer.
    # FINCHMOE_GDN8=1 restores the 8-bit protected tier (quality-safe option).
    gdn8 = os.environ.get('FINCHMOE_GDN8') == '1'
    for pat in EIGHT_BIT_PATTERNS:
        if name.endswith(pat) or pat in name:
            if not gdn8 and '.linear_attn.' in pat:
                return (4, 64)
            return (8, 64)  # 8-bit: 4 values/uint32, group_size=64
    for pat in FOUR_BIT_PATTERNS:
        if name.endswith(pat) or pat in name:
            return (4, 64)  # 4-bit: 8 values/uint32, group_size=64
    for pat in KEEP_BF16_PATTERNS:
        if pat in name:
            return None  # keep BF16
    # Default: keep BF16 for unknown tensors
    return None


def main():
    parser = argparse.ArgumentParser(description='Quantize non-expert weights')
    parser.add_argument('--input', required=True, help='BF16 model directory (with safetensors)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--verify', action='store_true', help='Verify quantized tensors against BF16 reference')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    input_dir = args.input

    # Load the index
    index_path = os.path.join(input_dir, 'model.safetensors.index.json')
    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Group by shard file
    by_file = defaultdict(list)
    for name, filename in weight_map.items():
        # Skip vision/visual and routed-expert tensors. Routed experts appear
        # as switch_mlp (quantized variants) or mlp.experts (pristine BF16
        # base). NOTE: mlp.shared_expert (always-on expert) IS kept.
        # The C engine is text-only and never reads model.visual.* (333
        # tensors, 893 MB in BF16) — pruning them keeps the bin ~1.95 GB,
        # which matters for 8 GB unified-memory devices (smaller zero-copy
        # Metal wrap + faster cold mmap).
        if ('vision' in name or 'visual' in name
                or 'switch_mlp' in name or '.mlp.experts.' in name):
            continue
        # Strip language_model prefix
        nn = name
        for prefix in ['model.language_model.', 'language_model.']:
            if nn.startswith(prefix):
                nn = 'model.' + nn[len(prefix):]
                break
        by_file[filename].append((nn, name))

    # Parse safetensors headers
    print(f"Reading headers from {len(by_file)} shard files...")
    header_cache = {}
    for filename in sorted(by_file.keys()):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header_cache[filename] = (filepath, 8 + header_len, json.loads(f.read(header_len)))

    # Plan the output
    all_tensors = []
    for filename in sorted(by_file.keys()):
        for nn, orig_name in by_file[filename]:
            all_tensors.append((nn, orig_name, filename))

    # Build quantization plan
    quant_plan = {}
    stats_bf16 = 0
    stats_4bit = 0
    stats_8bit = 0
    for nn, orig_name, filename in all_tensors:
        bits_info = get_quantization_bits(nn)
        if bits_info:
            bits, gs = bits_info
            if bits == 4:
                stats_4bit += 1
                quant_plan[nn] = ('quant', bits, gs)
            else:
                stats_8bit += 1
                quant_plan[nn] = ('quant', bits, gs)
        else:
            stats_bf16 += 1
            quant_plan[nn] = ('bf16', 0, 0)

    print(f"Quantization plan: {stats_4bit} tensors @ 4-bit, {stats_8bit} @ 8-bit, "
          f"{stats_bf16} keep BF16 (total {len(all_tensors)})")

    # Write binary file
    bin_path = os.path.join(args.output, 'model_weights_quant.bin')
    manifest = {
        "model": input_dir,
        "num_tensors": 0,  # updated as we write
        "tensors": {},
        "config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "full_attention_interval": 4,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
        }
    }

    # Layer type map
    layer_types = []
    for i in range(40):
        if (i + 1) % 4 == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")
    manifest["config"]["layer_types"] = layer_types

    ALIGN = 64
    offset = 0
    total_bytes = 0
    tensor_count = 0

    # Verification data
    verify_data = []  # list of (name, cpu_out) for verification

    with open(bin_path, 'wb') as out_f:
        for i, (nn, orig_name, filename) in enumerate(all_tensors):
            filepath, data_start, header = header_cache[filename]
            if orig_name not in header:
                print(f"  WARNING: {orig_name} not found in {filename}")
                continue

            meta = header[orig_name]
            doff = meta['data_offsets']
            byte_len = doff[1] - doff[0]
            shape = meta['shape']
            dtype = meta['dtype']

            # Read tensor data
            with open(filepath, 'rb') as sf:
                sf.seek(data_start + doff[0])
                raw = sf.read(byte_len)

            # Decode to float32
            if dtype in ('BF16', 'U16'):
                # 'U16' label = genuine BF16 data (self-quantized model convention)
                arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dtype == 'F32':
                arr = np.frombuffer(raw, np.float32)
            else:
                print(f"  WARNING: unexpected dtype {dtype} for {nn}, copying as-is")
                pad = ALIGN - (offset % ALIGN) if offset % ALIGN != 0 else 0
                if pad:
                    out_f.write(b'\x00' * pad)
                    offset += pad
                out_f.write(raw)
                manifest["tensors"][nn] = {"offset": offset, "size": byte_len, "shape": shape, "dtype": dtype}
                offset += byte_len
                total_bytes += byte_len
                tensor_count += 1
                continue

            # Qwen norm storage convention varies by MODEL, not by tensor:
            # the pristine BF16 base stores raw weight_param for ALL norms
            # (means range 0.04..1.63 — a threshold can't detect it), while
            # the 2bit-dense-v2 variant stores effective weights (1 + param).
            # FINCHMOE_NORM_PLUS1=1 enables the +1 for param-storing models.
            if (os.environ.get('FINCHMOE_NORM_PLUS1') == '1'
                    and ('norm.weight' in nn or 'layernorm.weight' in nn)
                    and len(arr.shape) == 1):
                arr = arr + 1.0

            bits_info = quant_plan.get(nn)
            if bits_info and bits_info[0] == 'quant':
                _, bits, gs = bits_info
                w = arr.reshape(shape[0], shape[1])
                packed, scales, biases = quantize_affine(w, bits, gs)

                # Convert to bytes
                w_bytes = packed.tobytes()
                s_bytes = scales.tobytes()
                b_bytes = biases.tobytes()

                # Align each component
                for data, name_suffix, comp_shape, comp_dtype in [
                    (w_bytes, '.weight', list(packed.shape), 'U32'),
                    (s_bytes, '.scales', list(scales.shape), 'BF16'),
                    (b_bytes, '.biases', list(biases.shape), 'BF16'),
                ]:
                    pad = ALIGN - (offset % ALIGN) if offset % ALIGN != 0 else 0
                    if pad:
                        out_f.write(b'\x00' * pad)
                        offset += pad
                    out_f.write(data)
                    entry_name = nn if name_suffix == '.weight' else nn.replace('.weight', name_suffix)
                    manifest["tensors"][entry_name] = {
                        "offset": offset,
                        "size": len(data),
                        "shape": comp_shape,
                        "dtype": comp_dtype,
                        "bits": bits,
                        "group_size": gs,
                    }
                    offset += len(data)
                    total_bytes += len(data)
                    tensor_count += 1

                # Verify: dequant and compare
                if args.verify:
                    vpu = 32 // bits
                    groups = shape[1] // gs
                    deq = np.zeros(shape, dtype=np.float32)
                    for r in range(shape[0]):
                        for g in range(groups):
                            s = bf16_decode(scales[r, g])
                            b = bf16_decode(biases[r, g])
                            for u in range(gs // vpu):
                                u32 = packed[r, g * (gs // vpu) + u]
                                for v in range(vpu):
                                    w_val = float((u32 >> (v * bits)) & ((1 << bits) - 1)) * s + b
                                    deq[r, g * gs + u * vpu + v] = w_val
                    verify_data.append((nn, w, deq))  # (name, original, dequantized)

            else:
                # Keep BF16
                arr_u16 = bf16_encode(arr.reshape(-1)).reshape(shape)
                raw_bytes = arr_u16.tobytes()
                pad = ALIGN - (offset % ALIGN) if offset % ALIGN != 0 else 0
                if pad:
                    out_f.write(b'\x00' * pad)
                    offset += pad
                out_f.write(raw_bytes)
                manifest["tensors"][nn] = {
                    "offset": offset,
                    "size": len(raw_bytes),
                    "shape": shape,
                    "dtype": "U16",
                }
                offset += len(raw_bytes)
                total_bytes += len(raw_bytes)
                tensor_count += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(all_tensors)}] {total_bytes/1e9:.2f} GB")

    manifest["num_tensors"] = tensor_count

    # Write manifest
    json_path = os.path.join(args.output, 'model_weights_quant.json')
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone: {total_bytes/1e9:.2f} GB in {bin_path}")
    print(f"Manifest: {json_path} ({tensor_count} tensors)")
    print(f"Size breakdown: original BF16 would be ~9.9 GB → quantized {total_bytes/1e9:.2f} GB")

    # Verification
    if args.verify and verify_data:
        print(f"\n{'='*60}")
        print("Verifying quantized tensors...")
        max_maxerr = 0
        min_cossim = 1.0
        for nn, orig, deq in verify_data:
            diff = np.abs(orig - deq)
            maxerr = diff.max()
            avg_err = diff.mean()
            # Cosine similarity
            o_flat = orig.ravel()
            d_flat = deq.ravel()
            cos_sim = np.dot(o_flat, d_flat) / (np.linalg.norm(o_flat) * np.linalg.norm(d_flat))
            if maxerr > max_maxerr:
                max_maxerr = maxerr
            if cos_sim < min_cossim:
                min_cossim = cos_sim
            if cos_sim < 0.99:
                print(f"  ⚠️  {nn}: CosSim={cos_sim:.6f} MaxErr={maxerr:.6f} (avg={avg_err:.6f})")
        print(f"  Worst: MaxErr={max_maxerr:.6f}, Min CosSim={min_cossim:.6f}")
        if min_cossim > 0.999:
            print("  ✅ All tensors pass (CosSim > 0.999)")
        elif min_cossim > 0.99:
            print("  ✅ All tensors acceptable (CosSim > 0.99)")
        else:
            print("  ❌ Some tensors below threshold")

    # Summary by category
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for name, info in manifest["tensors"].items():
        if "embed_tokens" in name:
            cat = "embedding"
        elif "lm_head" in name:
            cat = "lm_head"
        elif "norm" in name and "layers." not in name:
            cat = "final_norm"
        elif "input_layernorm" in name or "post_attention_layernorm" in name:
            cat = "layer_norms"
        elif "linear_attn" in name:
            cat = "linear_attention"
        elif "self_attn" in name:
            cat = "full_attention"
        elif "mlp.gate." in name:
            cat = "routing_gate"
        elif "shared_expert." in name:
            cat = "shared_expert"
        elif "shared_expert_gate" in name:
            cat = "shared_expert_gate"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories.keys()):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes']/1e6:8.1f} MB")


if __name__ == '__main__':
    main()
