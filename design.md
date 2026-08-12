# FinchMoE Design Document

## 1. Overview

FinchMoE is a C/Metal inference engine for **Qwen 3.6 35B A3B** on Apple Silicon. It streams quantized expert weights from SSD through Metal compute kernels.

**Current (2026-08-11): 3.88 tok/s on M4 16GB, ~6 GB RAM, 4-bit experts + BF16 non-experts.**
**Target: 12 tok/s on M4, ~2 GB RAM, matching turbo-fieldfare's efficiency.**

## 2. Model Architecture

```
Qwen 3.6 35B A3B: 35B total parameters, 3B active per token
40 layers: 30× GatedDeltaNet (linear attention) + 10× full attention
Full attention at layers 3,7,11,15,19,23,27,31,35,39
256 experts/layer, 8 active per token + 1 shared expert (always active)
Hidden dim: 2048, MoE intermediate: 512, Shared intermediate: 512
Vocab: 248,320, Head dim: 256, 16 Q-heads, 2 KV-heads (8:1 GQA)
RoPE: theta=10M, partial rotary 0.25 (64 of 256 dims)
GDN: 32 V-heads, 16 K-heads, key_dim=128, value_dim=128, conv kernel=4
```

## 3. Weight Inventory (Current Production)

### Non-Expert Weights: model_weights.bin (mmap'd, 4.96 GB)

| Component | Tensors | Size | Format |
|-----------|---------|------|--------|
| Embeddings | 1 | 0.97 GB | BF16 (uint16) |
| lm_head | 1 | 0.97 GB | BF16 (uint16) |
| Attention Q/K/V/O (full attn layers) | 40 | 0.52 GB | BF16 |
| GDN projections (linear attn layers) | 300 | 2.10 GB | BF16 |
| Shared experts (gate/up/down × 40) | 120 | 0.26 GB | BF16 |
| Routing gates (per-layer) | 40 | 0.04 GB | BF16 |
| Norms (input + post-attn × 80) | 80 | 0.01 GB | BF16 |
| MTP head (optional) | 19 | 0.05 GB | BF16 |
| **Total** | **632** | **4.96 GB** | All BF16 |

**Critical fact**: Non-expert weights are unquantized. turbo-fieldfare's equivalent is 1.35 GB (4-bit embeddings, 8-bit router, 4-bit attention). Our common model is **3.7× larger**. Quantizing to 4-bit/8-bit would save ~3.3 GB.

### Expert Weights: packed_experts/layer_NN.bin (per-layer files, 17 GB total)

256 experts × 40 layers. Each expert:
- gate_proj: [512, 256] U32 packed (4-bit, 8 vals/uint32)
- gate scales/biases: [512, 32] BF16 each
- up_proj: [512, 256] U32 packed
- up scales/biases: [512, 32] BF16 each
- down_proj: [2048, 64] U32 packed
- down scales/biases: [2048, 8] BF16 each
- **Total per expert**: 1,769,472 bytes (4-bit), 983,040 bytes (2-bit)

Streamed on demand via `pread` with F_NOCACHE for cold reads, OS page cache for warm.

## 4. Inference Pipeline (Per Token)

```
┌─────────────────────────────────────────────────────────────────┐
│ CMD1: Attention Projections (GPU, 1 cmd buffer, 3-9 encoders)   │
│   Full attn: Q[16384], K[512], V[512]  (3 BF16 matvecs)        │
│   Linear:    QKV[12288], Z[4096], β[32], α[32] (4 matvecs)     │
│   + GPU linear attn fusion: conv1d→norm→decay→recur→gate (5)   │
│   Commit + Wait                                                  │
├─────────────────────────────────────────────────────────────────┤
│ CPU: Attention Compute                                           │
│   Full: RMSNorm(Q,K) + RoPE + KV cache update + GPU scores      │
│   Linear: Conv1d + RMSNorm(q,k) + GDN recurrence + gated_norm   │
│   (~10 ms)                                                       │
├─────────────────────────────────────────────────────────────────┤
│ CMD2: O-Projection + Residual + Norm + Routing (GPU, 8-12 enc)  │
│   o_proj → residual_add → rms_norm → gate/routing/shared projs  │
│   + optional GPU attention (scores+softmax+values+sigmoid)      │
│   Commit + Wait                                                  │
├─────────────────────────────────────────────────────────────────┤
│ CPU: Routing + Parallel I/O                                      │
│   softmax(gate_scores[256]) → topK → normalize_weights          │
│   Parallel pread: K experts from SSD (GCD dispatch_group)        │
│   (~45 ms, dominated by SSD random read IOPS)                   │
├─────────────────────────────────────────────────────────────────┤
│ CMD3: Expert Forwards + Combine (GPU, K×2+4 enc, ASYNC commit)  │
│   Per expert: fused_gate_up_swiglu + down_proj                  │
│   Shared expert: swiglu + down_proj + gate weighting             │
│   GPU-side combine: weighted_sum + residual + norm → next layer │
│   NO wait — runs concurrently with next layer's CMD1            │
└─────────────────────────────────────────────────────────────────┘
```

**Deferred CMD3**: Expert compute is launched async. At the start of the next layer, finalize_deferred() waits for GPU completion and reads back the combined hidden state. The GPU queue serializes CMD3(layer N-1) → CMD1(layer N), so no explicit synchronization is needed.

**GPU-side combine**: For non-last layers, CMD3 computes weighted sum of expert outputs + shared expert + residual → rms_norm → buf_input. This eliminates the CPU round-trip (0.83 ms/layer saved).

## 5. Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ model_weights.bin (mmap'd, 4.96 GB)                             │
│   → Wrapped as Metal buffer (zero-copy, same physical pages)    │
│   → Read directly by GPU kernels (gemv_bf16 for BF16 weights)   │
├─────────────────────────────────────────────────────────────────┤
│ packed_experts/layer_NN.bin (mmap'd, 17 GB virtual)             │
│   → Tiered I/O: F_NOCACHE for cold, page cache for warm         │
│   → 8 parallel pread threads via GCD                             │
├─────────────────────────────────────────────────────────────────┤
│ GPU Buffers (Metal, ~449 MB allocated)                          │
│   → 10 KV caches: 16.8 MB each (2 KV heads × 256 dim × 8192)   │
│   → 30 GDN states: 2.1 MB each (32×128×128 floats)              │
│   → 16 expert data slots: 2 MB each (2MB-aligned DMA)           │
│   → Batch output slots: 16 KB each (16 slots)                   │
│   → lm_head output: 970 KB (VOCAB_SIZE floats)                  │
├─────────────────────────────────────────────────────────────────┤
│ CPU Scratch (~2 MB)                                             │
│   → Per-layer: normed, residual, attn_proj, h_post, h_mid, etc │
└─────────────────────────────────────────────────────────────────┘

Physical RAM: ~6 GB (GPU buffers always resident, mmap pages faulted in on demand)
After non-expert quantization: ~2.5 GB (1.7 GB weight file + 0.45 GB GPU + scratch)
```

## 6. GPU Kernels

| Kernel | Purpose | Grid | Threads | Notes |
|--------|---------|------|---------|-------|
| `gemv_bf16_x2` | BF16 matvec, 2 rows/tg | out_dim/2 TG | 256 | 128-bit uint4 loads, interleaved |
| `dequant_matvec_4bit_v3` | 4-bit affine dequant matvec | out_dim/8 TG | 256 | 8 rows/tg, shared-mem x-cache |
| `dequant_matvec_2bit` | 2-bit affine dequant | out_dim/8 TG | 256 | 16 vals/uint32 |
| `dequant_matvec_1bit` | 1-bit affine dequant | out_dim/32 TG | 256 | 32 vals/uint32 |
| `dequant_matvec_8bit` | 8-bit affine dequant | out_dim/4 TG | 256 | 4 vals/uint32 |
| `fused_gate_up_swiglu` | Gate+up+SiLU fused | out_dim/256 TG | 256 | 1 dispatch instead of 3 |
| `swiglu_fused` | SiLU(gate)×up | out_dim/256 TG | 256 | Downstream of gate+up |
| `moe_combine_residual` | Expert combine+residual+norm | dim/256 TG | 256 | GPU-side combine |
| `gated_delta_net_step` | GDN recurrence | 32 TG | 128 | One per V-head |
| `conv1d_step` | Depthwise conv + SiLU | dim/256 TG | 256 | Kernel=4 |
| `rms_norm_sum_sq` | Sum of squares reduce | 1 TG | 256 | Pre-reduction |
| `rms_norm_apply_bf16` | Apply norm with BF16 weights | dim/256 TG | 256 | Uses pre-computed sum_sq |
| `rms_norm_qk` | Per-head Q/K norm | 16 TG | 128 | Linear attention |
| `compute_decay_beta` | g_decay + beta_gate | 1 TG | 32 | GDN parameters |
| `gated_rms_norm` | norm(out) ⊙ silu(z) × w | 32 TG | 128 | GDN output norm |
| `attn_scores_batched` | Q @ K^T | seq×16 TG | 256 | Batched GQA |
| `attn_softmax_batched` | Softmax over seq | 16 TG | 256 | Per-head |
| `attn_values_batched` | Softmax @ V | 16×256 TG | 256 | Weighted sum |

All kernels compiled at runtime from `shaders.metal`. Verified with finchTool: CosSim=1.0, MaxDiff<1e-4 against CPU reference.

## 7. Per-Layer Timing (4-bit, K=8, warm cache, M4 16GB)

| Phase | ms | Bound By |
|-------|-----|----------|
| CMD1 (attn projections) | 90 | GPU dispatch overhead (5-9 encoders) |
| CPU attention compute | 10 | Memory bandwidth |
| CMD2 (o_proj+norm+routing) | 60 | GPU dispatch (8-12 encoders) |
| CPU routing + I/O | 45 | SSD random read IOPS |
| CMD3 (expert forwards, async) | 30 | GPU compute |
| CMD3 deferred wait/readback | 15 | GPU sync + CPU copy |
| Overhead | 10 | State management |
| **Total** | **260** | → 3.85 tok/s |

## 8. Key Design Decisions

1. **BF16 non-experts**: Currently kept as BF16 because CPU BLAS is 400× faster than CPU 4-bit dequant, and GPU path had a hang bug for large tensors. turbo-fieldfare proves GPU 4-bit dequant works at scale — this is the #1 optimization priority.

2. **GPU-side combine (CMD3)**: Eliminates CPU expert-output readback per layer. Saved ~0.83 ms/layer. turbo-fieldfare does the same.

3. **Deferred CMD3**: Expert compute launched async, overlaps with next layer's attention projections. GPU queue serialization ensures correctness without explicit fences.

4. **K=8 default**: Model trained with 8 experts/token. K=2 was a performance shortcut that produced garbage. Quality > speed.

5. **GPU lm_head**: gemv_bf16_x2 kernel reads BF16 weights directly from Metal buffer. Avoids 2 GB F32 cache allocation that caused OOM crashes.

6. **Fused GDN pipeline**: Optional GPU path fuses conv1d → norm → decay → recurrence → gated_norm into CMD1 (5 extra encoders). Falls back to CPU for correctness when needed.

7. **No custom expert cache**: OS page cache outperforms custom LRU/LFU schemes for our workload. turbo-fieldfare disagrees (uses LFU with 16 slots/layer) — worth revisiting.

## 9. Optimization History

| Step | ms/layer | tok/s | What |
|------|----------|-------|------|
| Baseline (8-bit, K=4) | 4.93 | 5.1 | Starting point |
| Fused expert kernel | 4.37 | 5.7 | gate+up+swiglu in 1 dispatch |
| 4-bit experts | 3.81 | 6.6 | Halved I/O volume |
| 2-bit experts (K=4) | 2.74 | 7.5 | Halved again |
| **K=2 default** | **2.30** | **8.3** | Wrong — produces garbage |
| **K=8 fix** | **6.50** | **3.9** | Correct, 2.1× slower |

The optimization history reveals a painful truth: the 8.3 tok/s headline number was achieved by silently breaking the model. Real speed at K=8 is 3.9 tok/s. The path to 12 tok/s requires genuine optimization, not parameter shortcuts.

## 10. Future: Road to 12 tok/s @ 2 GB

See [`finchmoe/OPTIMIZATION_PLAN.md`](finchmoe/OPTIMIZATION_PLAN.md) for the complete phased plan.

**Phase 1**: Non-expert quantization (4.96 GB → 1.7 GB, ~4 tok/s) — CRITICAL PATH  
**Phase 2**: 2-bit experts default (I/O halved, ~6 tok/s) — WORKING, needs validation  
**Phase 3**: GPU pipeline (fuse CMD1+CMD2, ICB, MTP → ~9-12 tok/s)  
**Phase 4**: Advanced (single-kernel multi-expert, KV FP16 → beyond 12 tok/s)

## 11. Hardware Targets

| Device | RAM | Current Speed | Target Speed |
|--------|-----|---------------|-------------|
| M4 Mac mini 16GB | 16 GB | 3.9 tok/s (4-bit, K=8) | **12 tok/s** |
| M1 Mac mini 8GB | 8 GB | TBD | 6-8 tok/s (estimated) |
| iPhone A18 Pro | 6-8 GB | Not yet ported | 3-5 tok/s (target) |

All measurements with Samsung 990 Plus NVMe in Thunderbolt 4 enclosure (~2.8 GB/s sequential, ~800 MB/s random).
