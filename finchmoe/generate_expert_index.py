#!/usr/bin/env python3
"""
generate_expert_index.py — Build expert_index.json for Qwen3.6-35B-A3B-4bit

Reads the model's model.safetensors.index.json and constructs a mapping
of (layer, component) → {file, offset, stride, size} for all expert tensors.

Output: expert_index.json with model_path and expert_reads.
"""

import json
import os
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Generate expert index for repack_experts.py')
    parser.add_argument('--model', type=str,
                        default='../models/Qwen3.6-35B-A3B-4bit',
                        help='Path to model directory')
    parser.add_argument('--output', type=str, default='expert_index.json',
                        help='Output JSON file path')
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    index_path = os.path.join(model_path, 'model.safetensors.index.json')

    if not os.path.exists(index_path):
        print(f"ERROR: {index_path} not found")
        return 1

    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Group expert tensors by layer and component
    # Expected pattern: language_model.model.layers.{L}.mlp.switch_mlp.{COMP}.{weight,scales,biases}
    expert_tensors = defaultdict(lambda: defaultdict(dict))

    for tensor_name, filename in weight_map.items():
        # Routed experts appear as either switch_mlp (quantized variants) or
        # mlp.experts (pristine BF16 base, fused gate_up_proj). Skip shared_expert.
        is_switch = '.switch_mlp.' in tensor_name
        is_experts = '.mlp.experts.' in tensor_name
        if not is_switch and not is_experts:
            continue

        # Handle MTP tensors: map to layer 40 (special MTP layer index)
        is_mtp = tensor_name.startswith('mtp.') or '.mtp.' in tensor_name

        parts = tensor_name.split('.')
        layer_idx = None
        if is_mtp:
            # MTP tensors: mtp.layers.0.mlp.switch_mlp.* → layer 40
            layer_idx = 40
        else:
            # Main model: model.layers.N.mlp.* or model.language_model.layers.N.mlp.*
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts) and i > 0:
                    prev = parts[i - 1]
                    if prev == 'model' or (prev == 'language_model' and i >= 2 and parts[i - 2] == 'model'):
                        try:
                            layer_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                        break

        if layer_idx is None:
            continue

        # Extract component name: e.g. "gate_proj.weight".
        # The pristine BF16 base uses bare tensor names (no .weight/.scales/
        # .biases suffixes) — those are weights.
        if 'gate_up_proj' in tensor_name:
            comp = 'fused_gate_up.weight'
        elif 'gate_proj' in tensor_name:
            comp = 'gate_proj.' + ('weight' if '.weight' in tensor_name else
                                    'scales' if '.scales' in tensor_name else
                                    'biases' if '.biases' in tensor_name else 'weight')
        elif 'up_proj' in tensor_name:
            comp = 'up_proj.' + ('weight' if '.weight' in tensor_name else
                                  'scales' if '.scales' in tensor_name else
                                  'biases' if '.biases' in tensor_name else 'weight')
        elif 'down_proj' in tensor_name:
            comp = 'down_proj.' + ('weight' if '.weight' in tensor_name else
                                    'scales' if '.scales' in tensor_name else
                                    'biases' if '.biases' in tensor_name else 'weight')
        else:
            continue

        expert_tensors[layer_idx][comp] = {
            'file': filename,
            'tensor_name': tensor_name,
        }

    # Now we need to read the actual safetensors headers to get offsets and sizes
    import struct

    file_headers = {}
    file_data_starts = {}  # filename -> data section start offset (8 + header_len)
    for filename in sorted(set(
        info['file'] for layer_data in expert_tensors.values()
        for info in layer_data.values()
    )):
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                header_len = struct.unpack('<Q', f.read(8))[0]
                file_headers[filename] = json.loads(f.read(header_len))
                file_data_starts[filename] = 8 + header_len

    # Build expert_reads
    expert_reads = {}
    for layer_idx in sorted(expert_tensors.keys()):
        layer_key = str(layer_idx)
        expert_reads[layer_key] = {}

        for comp_name in ['gate_proj.weight', 'gate_proj.scales', 'gate_proj.biases',
                           'up_proj.weight', 'up_proj.scales', 'up_proj.biases',
                           'down_proj.weight', 'down_proj.scales', 'down_proj.biases',
                           'fused_gate_up.weight']:
            if comp_name not in expert_tensors[layer_idx]:
                # fused_gate_up is optional (quantized models have separate tensors)
                if comp_name != 'fused_gate_up.weight':
                    print(f"WARNING: layer {layer_idx} missing component {comp_name}")
                continue

            info = expert_tensors[layer_idx][comp_name]
            filename = info['file']
            tensor_name = info['tensor_name']

            if filename not in file_headers:
                print(f"WARNING: file {filename} not found for layer {layer_idx}")
                continue

            header = file_headers[filename]
            if tensor_name not in header:
                print(f"WARNING: tensor {tensor_name} not in {filename}")
                continue

            meta = header[tensor_name]
            shape = meta['shape']
            dtype = meta['dtype']
            data_offsets = meta['data_offsets']
            # data_offsets are relative to the data section (after header).
            # os.pread needs absolute file offsets, so add the data section start.
            ds = file_data_starts[filename]
            byte_start = ds + data_offsets[0]
            byte_end = ds + data_offsets[1]

            num_experts = shape[0]
            total_size = byte_end - byte_start

            if comp_name == 'fused_gate_up.weight':
                # [num_experts, 1024, 2048] BF16 fused — split rows 0-511 (gate)
                # and 512-1023 (up). Per-expert full stride = 1024*2048*2 bytes;
                # per-half size = 512*2048*2 bytes.
                full_stride = total_size // num_experts
                half_size = full_stride // 2
                for half, name in [(0, 'gate_proj.weight'), (half_size, 'up_proj.weight')]:
                    expert_reads[layer_key][name] = {
                        'file': filename,
                        'abs_offset': byte_start + half,
                        'expert_stride': full_stride,
                        'expert_size': half_size,
                        'total_size': total_size // 2,
                        'shape': [num_experts, 512, 2048],
                        'dtype': dtype,
                    }
                continue

            expert_reads[layer_key][comp_name] = {
                'file': filename,
                'abs_offset': byte_start,
                'expert_stride': total_size // num_experts,
                'expert_size': total_size // num_experts,
                'total_size': total_size,
                'shape': shape,
                'dtype': dtype,
            }

    output = {
        'model_path': model_path,
        'expert_reads': expert_reads,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Generated {args.output}")
    print(f"  Model: {model_path}")
    print(f"  Layers: {len(expert_reads)}")
    total_experts = sum(1 for l in expert_reads.values() for _ in l.values())
    print(f"  Expert components: {total_experts}")
    return 0


if __name__ == '__main__':
    exit(main())
