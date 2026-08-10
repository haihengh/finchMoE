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
        # Only process switch_mlp (routed experts), not shared_expert
        if '.switch_mlp.' not in tensor_name:
            continue

        # Handle MTP tensors: map to layer 40 (special MTP layer index)
        is_mtp = tensor_name.startswith('mtp.') or '.mtp.' in tensor_name

        parts = tensor_name.split('.')
        layer_idx = None
        if is_mtp:
            # MTP tensors: mtp.layers.0.mlp.switch_mlp.* → layer 40
            layer_idx = 40
        else:
            # Main model: model.layers.N.mlp.switch_mlp.*
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts) and i > 0 and parts[i-1] == 'model':
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                    break

        if layer_idx is None:
            continue

        # Extract component name: e.g. "gate_proj.weight"
        if 'gate_proj' in tensor_name:
            comp = 'gate_proj.' + ('weight' if '.weight' in tensor_name else
                                    'scales' if '.scales' in tensor_name else 'biases')
        elif 'up_proj' in tensor_name:
            comp = 'up_proj.' + ('weight' if '.weight' in tensor_name else
                                  'scales' if '.scales' in tensor_name else 'biases')
        elif 'down_proj' in tensor_name:
            comp = 'down_proj.' + ('weight' if '.weight' in tensor_name else
                                    'scales' if '.scales' in tensor_name else 'biases')
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
                           'down_proj.weight', 'down_proj.scales', 'down_proj.biases']:
            if comp_name not in expert_tensors[layer_idx]:
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

            # For expert tensors, shape is [num_experts, out_dim, in_dim_packed]
            # expert_stride = size of one expert's worth of this component
            num_experts = shape[0]
            expert_size = (byte_end - byte_start) // num_experts
            total_size = byte_end - byte_start

            expert_reads[layer_key][comp_name] = {
                'file': filename,
                'abs_offset': byte_start,
                'expert_stride': expert_size,
                'expert_size': expert_size,
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
