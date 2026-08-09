# FinchMoE Design Document

## 1. Overview

FinchMoE is a C/Metal inference engine for **Qwen 3.6 35B A3B** on Apple Silicon. It streams quantized expert weights from SSD through Metal compute kernels, targeting 12 tok/s on M4 and 3-5 tok/s on iPhone.

**Status (2026-08-09): 8.3 tok/s on M4 16GB, 21 GB model, coherent output.**

## 2. Model Architecture

```
Qwen 3.6 35B A3B: 35B total parameters, 3B active per token
40 layers: 30× GatedDeltaNet (linear attention) + 10× full attention
Full attention at layers 3,7,11,15,19,23,27,31,35,39
256 experts, 8 active per token + 1 shared expert
Hidden dim: 2048, MoE intermediate: 512
Vocab: 248,320, Head dim: 256, 16Q/2KV GQA
RoPE theta: 10M, partial rotary: 0.25
```

## 3. Model Sizes & Quantization

| Model | Safetensors | Experts | Total | Speed (K=2) | Quality |
|-------|-------------|---------|-------|-------------|---------|
| BF16 (source) | 67 GB | — | 67 GB | — | Reference |
| 4-bit-dense | 19 GB | 17 GB | 36 GB | 7.2 tok/s | ~1-2% loss |
| **2-bit-dense (active)** | 11 GB | 9.4 GB | **21 GB** | **8.3 tok/s** | ~5% loss |

### Quantization Strategy

| Component | Format | Size | Reasoning |
|-----------|--------|------|-----------|
| Embeddings + lm_head | 8-bit | 1.1 GB | Vocabulary projections — near-lossless at 8-bit |
| Attention/GDN projections | 4-bit | 0.7 GB | Large tensors, 4-bit sufficient |
| Shared expert | 4-bit | 0.07 GB | Always active, small |
| Routed experts | 2-bit | 9.4 GB | MoE is quantization-tolerant |
| Norms + routing gate | BF16 | 0.0005 GB | Tiny, not worth quantizing |

## 4. Speed Analysis

### Measured Progression (M4 16GB, external TB4 NVMe)

| Step | expert_io | total_layer | tok/s | What changed |
|------|-----------|-------------|-------|-------------|
| BF16 baseline (K=4) | 2.04 ms | 4.93 ms | 5.08 | Starting point |
| + Fused expert kernel | 1.92 ms | 4.37 ms | 5.72 | gate+up+swiglu in 1 Metal dispatch |
| + 4-bit experts | 1.33 ms | 3.81 ms | 6.55 | Halved I/O volume |
| + Dense quantization | 0.59 ms | 2.83 ms | 7.20 | Embeddings 8-bit, attention 4-bit |
| + 2-bit experts (K=4) | 0.38 ms | 2.74 ms | 7.46 | Halved again |
| **+ K=2 default** | **0.21 ms** | **2.30 ms** | **8.27** | Half the experts per layer |

### Per-Layer Timing Breakdown (2-bit-dense, K=2)

| Phase | ms | % | What |
|-------|----|---|------|
| cmd1_wait | 0.87 | 38% | GPU: 5 GDN dispatches (conv1d→norm→decay→recur→gate) |
| cmd3_encode | 0.67 | 29% | CPU: encode expert dispatches into Metal cmd buffer |
| cmd2_wait | 0.49 | 21% | GPU: o_proj + residual + norm + routing + shared gate/up |
| expert_io | 0.21 | 9% | memcpy 2 experts × 0.96 MB from page cache |
| other | 0.07 | 3% | CPU attention, submit, routing |
| **Total** | **2.30** | | × 40 layers = 92 ms/token = **10.9 tok/s theoretical** |

Actual throughput (~8.3 tok/s) is lower due to prefill overhead, GPU synchronization, and token sampling.

## 5. Inference Pipeline

```
CMD3(prev) → CMD1: attention projections + GDN     [GPU, 5 dispatches]
           → CPU: flush results                     [CPU]
           → CMD2: o_proj + norm + routing + shared [GPU, ~6 dispatches]
           → CPU: softmax + topK routing            [CPU]
           → I/O: pread K experts from SSD          [parallel, 4 threads]
           → CMD3: expert forward + combine + norm  [GPU, DEFERRED commit]
```

Key design decisions:
1. **Serial GPU→SSD→GPU** — overlapping causes GPU latency spikes on Apple Silicon
2. **Trust OS page cache** — no custom expert cache (every attempt was slower)
3. **FMA dequant** — `fma(nibble, scale*x, bias*x)` uses GPU fused multiply-add
4. **Deferred CMD3** — async submit, GPU executes while CPU prepares next layer
5. **mmap model weights** — zero-copy access to non-expert BF16 tensors

## 6. Memory Layout

```
┌─────────────────────────────────────┐
│ model_weights.bin (mmap'd, 1.9 GB)  │  ← BF16/quantized non-expert weights
├─────────────────────────────────────┤
│ GPU state buffers (~0.3 GB)          │
│  - Delta-net recurrent (30L × 2MB)  │
│  - KV caches (10 attn layers)        │
│  - Expert compute scratch            │
├─────────────────────────────────────┤
│ OS page cache (~14 GB headroom)      │  ← Expert weight LRU caching
└─────────────────────────────────────┘
```

Runtime RAM: ~2 GB engine + page cache. On 16GB Mac, ~14 GB available for caching expert weights. On 8GB M1, ~3.8 GB used with zero swap.

## 7. Metal Kernels

| Kernel | Bits | Purpose |
|--------|------|---------|
| `dequant_matvec_2bit` | 2 | 16 values/uint32, expert forward |
| `dequant_matvec_4bit_v3` | 4 | 8 values/uint32, SIMD-stride + shared memory |
| `dequant_matvec_8bit` | 8 | 4 values/uint32, for INT8 experts |
| `fused_gate_up_swiglu` | 4 | gate+up+swiglu in 1 dispatch |
| `fused_gate_up_swiglu_8bit` | 8 | Same for 8-bit |
| `fused_gdn_core` | — | decay+beta+recur+gated_norm in 1 dispatch (optional) |
| `gated_delta_net_step` | — | Single GDN recurrence step |
| `swiglu_fused` | — | SiLU(gate) × up |
| `moe_combine_residual` | — | Weighted sum + shared gate + residual |
| `gemv_bf16` | — | Raw BF16 matvec (vectorized, SIMD reduction) |
| `rms_norm` / `rms_norm_apply` | — | Two-pass RMS normalization |

## 8. GatedDeltaNet (Linear Attention)

30 of 40 layers use delta-rule recurrence instead of self-attention:

```
Q, K, V, Z, A, B = projections(x)
Q = RMS_norm(Q) × 1/d_model    # Per-head, then scale
K = RMS_norm(K) × 1/sqrt(d)    # Per-head, then scale
QKV = SiLU(conv1d(QKV, kernel=4))  # Depthwise conv

g = exp(-exp(A_log) × softplus(A + dt_bias))  # Per-head decay
β = sigmoid(B)                                  # Per-head gate

S = S × g                        # Decay state
kv = S @ K                        # Predict V from state
Δ = (V - kv) × β                 # Error signal
S += Δ ⊗ K                       # Update state (outer product)
O = S @ Q                         # Read output

output = RMS_norm(O) × SiLU(Z) × weight  # Gated output norm
final = out_proj(output)                    # Project to hidden dim
```

State per layer: [32 v-heads, 128 value-dim, 128 key-dim] = 2.1 MB
Total GDN state: 30 layers × 2.1 MB = 63 MB

## 9. Full Attention

10 layers (every 4th) use standard GQA with KV cache:

```
Q_full = q_proj(x)    # [16 heads × 2 × 256] = Q + output gate
K, V = k_proj(x), v_proj(x)  # [2 KV heads × 256]
Q = RMS_norm(Q) × q_weight, K = RMS_norm(K) × k_weight
Q, K = RoPE(Q, K, position)
scores = Q @ K^T / sqrt(256)
attn = softmax(scores) @ V
output = o_proj(attn × sigmoid(gate))
```

## 10. MoE Routing & Expert Forward

```
gate_logits = gate_proj(hidden)     # [256]
probs = softmax(gate_logits)
top_k_idx, top_k_w = top_k(probs, K=8)  # model: 8, engine default: K=2
top_k_w = normalize(top_k_w)             # renormalize to sum=1

# Shared expert (always active)
shared_gate = sigmoid(shared_gate_proj(hidden))  # scalar
shared_out = shared_gate × SwiGLU(hidden)       # gate_proj→SiLU×up_proj→down_proj

# Routed experts (streamed from SSD)
for i in top_k_idx[:K]:
    expert_out += top_k_w[i] × SwiGLU_expert_i(hidden)

hidden = residual + expert_out + shared_out
```

Expert SwiGLU per expert: gate_proj(x) → SiLU(gate) × up_proj(x) → down_proj(act)

## 11. Hardware Targets

| Device | RAM | Speed (measured/est.) |
|--------|-----|----------------------|
| M4 Mac mini 16GB | 16 GB | **8.3 tok/s** (2-bit, K=2) |
| M1 Mac mini 8GB | 8 GB | **5.4 tok/s** (4-bit, K=4) |
| iPhone A18 Pro | 6-8 GB | **3-5 tok/s** (estimated) |

Both M1 and M4 tested with Samsung 990 Plus NVMe in Thunderbolt 4 enclosure (2800 MB/s read, faster than internal SSD).

## 12. Key Design Decisions

1. **4-bit → 2-bit experts**: Halved I/O volume (17GB → 9.4GB). Quality impact ~5% PPL — acceptable because MoE is quantization-tolerant (8/256 fire, errors cancel via weighted sum).

2. **Dense weight quantization**: Embeddings/lm_head at 8-bit, attention/GDN/shared at 4-bit. Saved 15GB (22GB → 11GB safetensors) with <1% quality impact.

3. **K=2 default**: Halves expert I/O vs model's K=8. Quality impact minimal on short prompts. Use `-k 4` or `-k 8` for complex tasks.

4. **Fused expert kernel**: Combined gate+up+swiglu into 1 Metal dispatch (was 3). 12.5% speedup.

5. **No custom expert cache**: OS page cache outperforms every custom scheme tested (Metal LRU, malloc cache, LZ4).

6. **mmap model weights**: Zero-copy access to non-expert tensors via `model_weights.bin`.

## 13. Optimization Roadmap

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 1 | Batched expert encoders | Medium | ~0.15ms/layer (cmd3_encode) |
| 2 | Fuse conv1d+rms_norm GDN dispatches | Medium | ~0.10ms/layer (cmd1_wait) |
| 3 | KV cache quantization (Q8_0) | Low | Saves 2.5 GB at 256K context |
| 4 | MTP speculative decoding | Medium | 1.5× speedup |
| 5 | iPhone port | High | Target deployment |
