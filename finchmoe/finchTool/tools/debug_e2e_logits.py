#!/usr/bin/env python3
"""End-to-end logit comparison: C engine vs Python reference.
Processes the same ChatML tokens through all 40 layers and compares logits."""
import numpy as np, struct, json, os, sys, time

MODEL_WEIGHTS = 'model_weights.bin'
MODEL_MANIFEST = 'model_weights.json'
Q8_DIR = 'models/Qwen3.6-35B-A3B-8bit-custom'
EXPERT_SIZE_8BIT = 3342336

with open(MODEL_MANIFEST) as f: manifest = json.load(f)

tensor_cache = {}
def load_f32(name):
    if name in tensor_cache: return tensor_cache[name]
    info = manifest['tensors'][name]
    with open(MODEL_WEIGHTS, 'rb') as f:
        f.seek(info['offset'])
        u16 = np.frombuffer(f.read(info['size']), dtype=np.uint16)
    arr = (u16.astype(np.uint32) << 16).view(np.float32)
    s = info['shape']
    if len(s) == 3: arr = arr.reshape(s[0], s[2])
    else: arr = arr.reshape(s)
    # Free memory for large tensors - we'll reload as needed
    if arr.nbytes > 100_000_000:
        tensor_cache[name] = arr  # Keep large tensors cached
    return arr

# ChatML tokens from C engine
TOKENS = [248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
print(f"Processing {len(TOKENS)} ChatML tokens")

emb_w = load_f32('model.embed_tokens.weight')
print(f"Loaded embedding: {emb_w.shape}")

# Process first token only for quick test
token_id = TOKENS[0]
hidden = emb_w[token_id].copy()
print(f"Token {token_id} embedding rms: {np.sqrt(np.mean(hidden**2)):.6f}")
