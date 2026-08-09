#!/usr/bin/env python3
"""
Quantize BF16 Qwen3.6 model to MLX-compatible 4-bit/8-bit safetensors.
Usage: python quantize_model.py --model <bf16-dir> --output <output-dir>
"""

import argparse, json, os, struct, sys, time, shutil
from collections import defaultdict
import numpy as np
from safetensors.numpy import save_file


def bf16_encode(arr_f32):
    return (arr_f32.view(np.uint32) >> 16).astype(np.uint16)


EXPERT_GROUP_SIZE = 64  # Standard group size

def quantize_affine(weights_f32, bits, group_size=64):
    """MLX affine quant: w_q = round((w - min)/scale), scale = (max-min)/(2^bits-1)"""
    out_dim, in_dim = weights_f32.shape
    num_groups = in_dim // group_size
    max_val = (1 << bits) - 1
    vpu = 32 // bits  # values per U32

    w = weights_f32.reshape(out_dim, num_groups, group_size)
    w_min = w.min(axis=2)
    w_max = w.max(axis=2)
    scales = np.maximum((w_max - w_min) / max_val, 1e-8)
    biases = w_min

    q = np.round((w - biases[:, :, np.newaxis]) / scales[:, :, np.newaxis])
    q = np.clip(q, 0, max_val).astype(np.uint8)

    packed_cols = in_dim // vpu
    packed = np.zeros((out_dim, packed_cols), dtype=np.uint32)
    upg = group_size // vpu  # U32 per group
    for g in range(num_groups):
        for u in range(upg):
            u32_val = np.zeros(out_dim, dtype=np.uint32)
            for v in range(vpu):
                u32_val |= q[:, g, u * vpu + v].astype(np.uint32) << (v * bits)
            packed[:, g * upg + u] = u32_val

    return (packed,
            bf16_encode(scales.flatten()).reshape(out_dim, num_groups),
            bf16_encode(biases.flatten()).reshape(out_dim, num_groups))


EIGHT_BIT_DENSE = [
    # 8-bit: embedding & lm_head — quality-critical, 8-bit is safe
    'lm_head.weight',
    'embed_tokens.weight',
]

FOUR_BIT_DENSE = [
    # 4-bit: attention/GDN projections — large, 4-bit is sufficient
    '.linear_attn.in_proj_qkv.weight',
    '.linear_attn.in_proj_z.weight',
    '.linear_attn.in_proj_a.weight',
    '.linear_attn.in_proj_b.weight',
    '.linear_attn.out_proj.weight',
    '.self_attn.q_proj.weight',
    '.self_attn.k_proj.weight',
    '.self_attn.v_proj.weight',
    '.self_attn.o_proj.weight',
    # 4-bit: shared expert FFN
    '.mlp.shared_expert.gate_proj.weight',
    '.mlp.shared_expert.up_proj.weight',
    '.mlp.shared_expert.down_proj.weight',
]

# Routed experts: 4-bit (standard)
INT4_EXPERTS = ['.mlp.experts.gate_up_proj', '.mlp.experts.down_proj']

# Keep as BF16: tiny tensors where quantization doesn't save meaningful space
KEEP_BF16 = [
    '.mlp.gate.weight',              # routing gate (0.04 GB)
    '.mlp.shared_expert_gate.weight', # shared expert gate (0.0002 GB)
    # Norm weights are automatically kept BF16 (detected by 'norm.weight' in name)
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load source index
    with open(os.path.join(args.model, 'model.safetensors.index.json')) as f:
        src_idx = json.load(f)
    src_map = src_idx['weight_map']  # original_name → file

    # Normalize names: strip 'language_model.' prefix, skip vision
    tensor_map = {}  # normalized_name → (source_file, original_name)
    for orig_name, fname in src_map.items():
        if 'visual' in orig_name or 'vision_tower' in orig_name:
            continue
        nn = orig_name
        for prefix in ['model.language_model.', 'language_model.']:
            if nn.startswith(prefix):
                nn = 'model.' + nn[len(prefix):]
                break
        tensor_map[nn] = (fname, orig_name)

    # Copy metadata
    for fn in os.listdir(args.model):
        if not fn.endswith('.safetensors'):
            s, d = os.path.join(args.model, fn), os.path.join(args.output, fn)
            if os.path.isfile(s):
                shutil.copy2(s, d)

    # Distribute across 4 shards
    names = sorted(tensor_map.keys())
    n = len(names)
    ss = (n + 3) // 4
    print(f"{n} tensors → 4 shards of ~{ss} each")

    # Pre-load source file headers (small)
    src_files = set(fname for fname, _ in tensor_map.values())
    headers = {}
    for fname in src_files:
        fpath = os.path.join(args.model, fname)
        with open(fpath, 'rb') as f:
            hl = struct.unpack('<Q', f.read(8))[0]
            headers[fname] = (fpath, 8 + hl, json.loads(f.read(hl)))

    new_map = {}
    t0 = time.time()

    for si in range(4):
        batch = names[si * ss:(si + 1) * ss]
        if not batch:
            break
        sn = f"model-{si + 1:05d}-of-00004.safetensors"
        sp = os.path.join(args.output, sn)
        print(f"Shard {si + 1}/4: {len(batch)} tensors → {sn}")

        out = {}
        for i, nn in enumerate(batch):
            src_fn, orig_name = tensor_map[nn]
            fpath, ds, hdr = headers[src_fn]
            if orig_name not in hdr:
                continue
            info = hdr[orig_name]
            doff, shape, dtype = info['data_offsets'], info['shape'], info['dtype']
            size = doff[1] - doff[0]

            with open(fpath, 'rb') as f:
                f.seek(ds + doff[0])
                raw = f.read(size)

            # Dequant from source
            if dtype == 'BF16':
                arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dtype == 'F32':
                arr = np.frombuffer(raw, np.float32)
            else:
                out[nn] = np.frombuffer(raw, np.uint8)
                new_map[nn] = sn
                continue

            # Qwen3_5RMSNorm: effective weight = 1 + weight_param (~0→~1)
            if 'norm.weight' in nn or 'layernorm.weight' in nn:
                arr = arr + 1.0

            # Quantize weight tensors; handle fused expert projections
            # Original model has fused gate_up_proj [n_exp, 2*intermediate, hidden]
            # We split into gate_proj [n_exp, intermediate, hidden] + up_proj [same]
            is_expert = '.mlp.experts.' in nn
            is_weight = '.weight' in nn or is_expert

            # Skip quantization for keep-BF16 tensors (tiny, not worth quantizing)
            keep_bf16 = any(p in nn for p in KEEP_BF16) or 'norm.weight' in nn or 'layernorm.weight' in nn

            if is_weight and len(shape) >= 2 and shape[-1] % 64 == 0 and not keep_bf16:
                # Determine quantization bits based on tensor category
                if any(p in nn for p in EIGHT_BIT_DENSE):
                    bits = 8
                elif any(p in nn for p in FOUR_BIT_DENSE):
                    bits = 4
                elif any(p in nn for p in INT4_EXPERTS):
                    bits = 4  # routed experts: 4-bit
                else:
                    bits = 4  # default: 4-bit for unknown weight tensors
                if len(shape) == 2:
                    packed, scales, biases = quantize_affine(arr.reshape(shape[0], shape[1]), bits)
                    out[nn] = packed
                    snn = nn.replace('.weight', '.scales')
                    bnn = nn.replace('.weight', '.biases')
                    out[snn], out[bnn] = scales, biases
                    new_map[nn] = new_map[snn] = new_map[bnn] = sn
                elif len(shape) == 3:
                    n_exp, out_d, in_p = shape
                    w = arr.reshape(n_exp, out_d, -1)
                    actual_in = w.shape[-1]

                    # Handle fused gate_up_proj: split into gate_proj + up_proj
                    if '.mlp.experts.gate_up_proj' in nn:
                        half = out_d // 2  # 1024 → 512 each
                        w_gate = w[:, :half, :]
                        w_up = w[:, half:, :]
                        pl_g, sl_g, bl_g = [], [], []
                        pl_u, sl_u, bl_u = [], [], []
                        for e in range(n_exp):
                            pg, sg, bg = quantize_affine(w_gate[e].reshape(half, actual_in), bits, EXPERT_GROUP_SIZE)
                            pu, su, bu = quantize_affine(w_up[e].reshape(half, actual_in), bits, EXPERT_GROUP_SIZE)
                            pl_g.append(pg); sl_g.append(sg); bl_g.append(bg)
                            pl_u.append(pu); sl_u.append(su); bl_u.append(bu)
                        # Store as split names: gate_proj + up_proj
                        gname = nn.replace('.mlp.experts.gate_up_proj', '.mlp.switch_mlp.gate_proj.weight')
                        uname = nn.replace('.mlp.experts.gate_up_proj', '.mlp.switch_mlp.up_proj.weight')
                        out[gname] = np.stack(pl_g)
                        out[uname] = np.stack(pl_u)
                        gsn = gname.replace('.weight', '.scales')
                        gbn = gname.replace('.weight', '.biases')
                        usn = uname.replace('.weight', '.scales')
                        ubn = uname.replace('.weight', '.biases')
                        out[gsn], out[gbn] = np.stack(sl_g), np.stack(bl_g)
                        out[usn], out[ubn] = np.stack(sl_u), np.stack(bl_u)
                        for n in [gname, uname, gsn, gbn, usn, ubn]:
                            new_map[n] = sn

                    # Handle down_proj (kept under switch_mlp name)
                    elif '.mlp.experts.down_proj' in nn:
                        dname = nn.replace('.mlp.experts.down_proj', '.mlp.switch_mlp.down_proj.weight')
                        pl, sl, bl = [], [], []
                        for e in range(n_exp):
                            p, s, b = quantize_affine(w[e].reshape(out_d, actual_in), bits, EXPERT_GROUP_SIZE)
                            pl.append(p); sl.append(s); bl.append(b)
                        out[dname] = np.stack(pl)
                        dsn = dname.replace('.weight', '.scales')
                        dbn = dname.replace('.weight', '.biases')
                        out[dsn], out[dbn] = np.stack(sl), np.stack(bl)
                        for n in [dname, dsn, dbn]:
                            new_map[n] = sn

                    else:
                        pl, sl, bl = [], [], []
                        for e in range(n_exp):
                            p, s, b = quantize_affine(w[e].reshape(out_d, actual_in), bits, EXPERT_GROUP_SIZE)
                            pl.append(p); sl.append(s); bl.append(b)
                        out[nn] = np.stack(pl)
                        snn = nn.replace('.weight', '.scales')
                        bnn = nn.replace('.weight', '.biases')
                        out[snn], out[bnn] = np.stack(sl), np.stack(bl)
                        new_map[nn] = new_map[snn] = new_map[bnn] = sn
            else:
                # Keep as BF16
                arr_u16 = (arr.view(np.uint32) >> 16).astype(np.uint16)
                # Reshape to match original shape
                out[nn] = arr_u16.reshape(shape) if len(shape) > 1 else arr_u16
                new_map[nn] = sn

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(batch)}]")

        save_file(out, sp)
        sz = os.path.getsize(sp) / 1e9
        del out  # free memory before next shard
        print(f"  {sz:.2f} GB in {time.time() - t0:.0f}s")

    # Write index
    with open(os.path.join(args.output, 'model.safetensors.index.json'), 'w') as f:
        json.dump({'metadata': {'total_size': 0}, 'weight_map': new_map}, f, indent=2)

    print(f"\nDone in {time.time() - t0:.0f}s → {args.output}")


if __name__ == '__main__':
    main()
