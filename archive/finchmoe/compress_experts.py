#!/usr/bin/env python3
"""Compress expert weights with LZ4 for faster SSD streaming."""
import argparse, os, struct, json, time
import lz4.frame  # pip install lz4

EXPERT_SIZE_8BIT = 3342336
NUM_EXPERTS = 256
NUM_LAYERS = 40

def compress_layer(src_dir, dst_dir, layer_idx):
    src_path = os.path.join(src_dir, f"layer_{layer_idx:02d}.bin")
    dst_path = os.path.join(dst_dir, f"layer_{layer_idx:02d}.bin")

    with open(src_path, 'rb') as f:
        data = f.read()

    index = []
    compressed_blobs = []
    offset = 0

    for e in range(NUM_EXPERTS):
        expert_data = data[e * EXPERT_SIZE_8BIT : (e+1) * EXPERT_SIZE_8BIT]
        compressed = lz4.frame.compress(expert_data, compression_level=1)
        index.append((offset, len(compressed)))
        compressed_blobs.append(compressed)
        offset += len(compressed)

    with open(dst_path, 'wb') as f:
        # Write header: 256 entries × (uint64 offset + uint32 size) = 12 bytes each
        for off, size in index:
            f.write(struct.pack('<QI', off, size))
        # Write compressed blobs
        for blob in compressed_blobs:
            f.write(blob)

    total_compressed = sum(len(b) for b in compressed_blobs)
    original = NUM_EXPERTS * EXPERT_SIZE_8BIT
    ratio = total_compressed / original * 100
    return original, total_compressed, ratio

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='models/Qwen3.6-35B-A3B-8bit-custom/packed_experts_8bit')
    p.add_argument('--dst', default='models/Qwen3.6-35B-A3B-8bit-custom/packed_experts_lz4')
    p.add_argument('--layers', type=int, default=NUM_LAYERS)
    args = p.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    t0 = time.time()
    total_orig = 0
    total_comp = 0

    for layer in range(args.layers):
        orig, comp, ratio = compress_layer(args.src, args.dst, layer)
        total_orig += orig
        total_comp += comp
        print(f"Layer {layer:2d}: {orig/1e6:.1f}MB → {comp/1e6:.1f}MB ({ratio:.1f}%)")

    elapsed = time.time() - t0
    print(f"\nTotal: {total_orig/1e9:.1f}GB → {total_comp/1e9:.1f}GB ({total_comp/total_orig*100:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({total_orig/elapsed/1e9:.1f} GB/s uncompressed)")

if __name__ == '__main__':
    main()
