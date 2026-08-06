# FinchMoE Design Document

## 1. Overview

FinchMoE is a C/Metal inference engine purpose-built for **Qwen 3.6 35B A3B** on Apple Silicon. It streams 4-bit quantized expert weights from SSD through a custom Metal compute pipeline, targeting ≥3.5 tok/s at ≤3 GB RAM.

The engine is forked from [flash-moe](flash-moe/), a proven C/Metal engine that runs Qwen3.5-397B-A17B at 4.36 tok/s on M3 Max (48GB). We adapt its architecture for the smaller Qwen 3.6 35B A3B model, targeting lower-RAM devices (16GB Mac mini → iPhone).

## 2. Why Not finchmoe_engine?

The original `finchmoe_engine` (archived) was built in C++/ObjC++ for a pre-release Qwen MoE specification that never shipped. The published Qwen 3.6 architecture is fundamentally different:

| Aspect | finchmoe_engine target | Actual Qwen 3.6 |
|---|---|---|
| Attention | 100% full attention | 75% GatedDeltaNet, 25% full attention |
| Expert layout | Fused gate_up_proj | Separate gate_proj + up_proj + down_proj |
| Hidden dim | Different | 2048 |
| Quantization | Custom scheme | MLX affine INT4 group-64 |

flash-moe already targets the correct model family (`qwen3_5_moe`) and implements all required operations (GatedDeltaNet, MoE routing, SSD streaming, Metal dequant). Adaptation requires dimension changes, not architectural redesign.

## 3. Hardware Targets

### Development: M4 Mac mini 16GB

| Resource | Available | Engine Usage | Headroom |
|---|---|---|---|
| RAM | 16 GB unified | ~1.6 GB | ~14.4 GB for OS + page cache |
| SSD | External 1.8TB | ~19 GB model | Plenty |
| GPU | M4 (10-core?) | Metal compute | TBD |
| Memory bandwidth | ~120 GB/s (est.) | — | — |

The larger page cache (14.4 GB vs 7-8 GB on 397B model) means higher expert cache hit rates, potentially compensating for slower SSD vs the M3 Max.

### Target: iPhone (A-series)

- Much tighter RAM (6-8 GB)
- No swap — must fit everything in physical memory
- Slower SSD, less bandwidth
- Likely needs model shrinking (3-bit? fewer experts?) or aggressive preloading

## 4. Inference Pipeline

Adapted from flash-moe's proven per-layer pipeline:

```
CMD3(prev) → CMD1: attention projections + delta-net  [GPU]
           → CPU: flush results                        [CPU]
           → CMD2: o_proj + norm + routing + shared    [GPU]
           → CPU: softmax + topK routing               [CPU]
           → I/O: parallel pread K=8 experts           [SSD]
           → CMD3: expert forward + combine + norm     [GPU, DEFERRED]
```

### Key Design Decisions (from flash-moe experiments)

1. **Serial GPU → SSD → GPU** — On Apple Silicon, SSD DMA and GPU compute share the memory controller. Overlapping them causes GPU latency spikes. Serial pipeline is hardware-optimal.

2. **No custom expert cache** — The OS page cache outperforms every custom caching scheme tested (Metal LRU, malloc cache, LZ4 compressed cache). "Trust the OS."

3. **FMA dequant** — Rearranging `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)` gives +12% throughput by using the GPU's fused multiply-add unit.

4. **Deferred CMD3** — Submit expert compute without waiting. GPU executes while CPU prepares next layer.

5. **Accelerate BLAS for GatedDeltaNet** — `cblas_sgemv` + `cblas_sger` for the recurrent state update is 64% faster than scalar code.

## 5. Component Design

### 5.1 Model Loading

Two-phase extraction from MLX 4-bit safetensors:

**Phase 1: Non-expert weights** (`extract_weights.py`)
- Extract all non-expert tensors (embeddings, norms, attention, linear attention, shared expert, router)
- Pack into single `model_weights.bin` (est. ~1.4 GB for Qwen3.6)
- mmap'd at startup, read-only, zero-copy
- Manifest in `model_weights.json`

**Phase 2: Expert weights** (`repack_experts.py`)
- Extract 256 experts × 40 layers = 10,240 experts
- Each expert: 1,769,472 bytes at 4-bit (gate_proj + up_proj + down_proj with scales/biases)
- Write 40 contiguous layer files in `packed_experts/layer_XX.bin`
- Per-layer file size: 256 × 1.69 MB ≈ 432 MB
- Total: 40 × 432 MB ≈ 16.9 GB

Expert layout (same as flash-moe):
```
[gate_proj.weight (U32)] [gate_proj.scales (BF16)] [gate_proj.biases (BF16)]
[up_proj.weight   (U32)] [up_proj.scales   (BF16)] [up_proj.biases   (BF16)]
[down_proj.weight (U32)] [down_proj.scales (BF16)] [down_proj.biases (BF16)]
```

### 5.2 Memory Layout

```
┌─────────────────────────────────────┐
│ model_weights.bin (mmap'd, ~1.4 GB) │  ← Read-only, OS-managed
├─────────────────────────────────────┤
│ Metal scratch buffers (~100 MB)     │  ← GPU-accessible
│  - buf_input [HIDDEN_DIM]           │
│  - buf_expert_data [EXPERT_SIZE]    │
│  - buf_expert_input [HIDDEN_DIM]    │
│  - buf_expert_gate [MOE_INTER]      │
│  - buf_expert_up [MOE_INTER]        │
│  - buf_expert_act [MOE_INTER]       │
│  - buf_expert_out [HIDDEN_DIM]      │
│  - KV cache (10 attn layers)        │
│  - GDN recurrent states (30 layers) │
├─────────────────────────────────────┤
│ OS page cache (~14 GB available)    │  ← Expert LRU caching
└─────────────────────────────────────┘
```

Total engine RAM: ~1.6 GB (vs 6 GB for the 397B model)

### 5.3 Metal Shaders

Reused from flash-moe with dimension updates:

| Kernel | Purpose | Changes Needed |
|---|---|---|
| `dequant_matvec_4bit` | 4-bit dequant + matvec (FMA optimized) | Update tile sizes for 2048-dim |
| `swiglu_fused` | SiLU gating × up projection | Dimension updates |
| `rms_norm` | Two-pass sum-of-squares + apply | None (generic) |
| `rms_norm_apply` | Apply pre-computed scale | None (generic) |
| `batched_attention` | Q@K^T + softmax + scores@V | Head count update (16Q, 2KV) |
| `rope_fused` | Rotary embeddings with Q deinterleave | Head dim unchanged (256) |
| `moe_combine_residual` | Weighted sum + residual + sigmoid gate | Expert count update |

### 5.4 GatedDeltaNet (Linear Attention)

Per-token recurrence using delta rule:

```
Q, K, V, Z, A, B = projections(x)
Q = L2_normalize(Q)
K = L2_normalize(K)
V = SiLU(conv1d(V))
Z = SiLU(conv1d(Z))

# Recurrent state update (via Accelerate BLAS)
S_t = A * S_{t-1} + B * K_t^T @ V_t   # cblas_sger

# Output
o_t = Z * S_t @ Q_t / (A_log + dt_bias)  # cblas_sgemv
```

State per layer: [32 heads, 128 key_dim, 128 value_dim] = 2.1 MB BF16
Total state (30 GDN layers): ~63 MB

### 5.5 Full Attention

Standard GQA with KV cache (only on 10 full-attention layers):

```python
Q, K, V = projections(x)
Q, K = apply_rotary_embeddings(Q, K)
scores = Q @ K^T / sqrt(head_dim)       # GPU batched
attn = softmax(scores) @ V               # GPU batched
output = o_proj(attn) * sigmoid(gate)    # Output gate
```

KV cache per layer (FP16): 2 × 2 heads × 256 head_dim × seq_len × 2 bytes
At 4096 tokens: ~8 MB per layer, ~80 MB total

### 5.6 MoE Routing

```python
router_logits = gate_proj(hidden)        # [256]
probs = softmax(router_logits)
top_k_indices, top_k_weights = top_k(probs, k=8)
# Normalize top-k weights
top_k_weights = softmax(top_k_weights)

# Shared expert (always active)
shared_out = shared_expert(hidden) * sigmoid(shared_gate(hidden))

# Routed experts (streamed from SSD on demand)
expert_out = sum(top_k_weights[i] * expert_i(hidden) for i in top_k_indices)

output = expert_out + shared_out + residual
```

## 6. Dimension Adaptation from flash-moe

All changes from the Qwen3.5-397B baseline:

```c
// infer.m constant changes
#define HIDDEN_DIM          2048    // was 4096
#define NUM_LAYERS          40      // was 60
#define NUM_ATTN_HEADS      16      // was 32
#define NUM_KV_HEADS        2       // unchanged
#define HEAD_DIM            256     // unchanged
#define VOCAB_SIZE          248320  // unchanged
#define NUM_EXPERTS         256     // was 512
#define NUM_EXPERTS_PER_TOK 8       // was 10
#define MOE_INTERMEDIATE    512     // was 1024
#define SHARED_INTERMEDIATE 512     // was 1024
#define LINEAR_NUM_V_HEADS  32      // was 64
#define LINEAR_NUM_K_HEADS  16      // unchanged
#define LINEAR_KEY_DIM      128     // unchanged
#define LINEAR_VALUE_DIM    128     // unchanged
// Derived:
// LINEAR_TOTAL_KEY    = 16 * 128 = 2048 (was 2048, unchanged!)
// LINEAR_TOTAL_VALUE  = 32 * 128 = 4096 (was 8192)
// LINEAR_CONV_DIM     = 2048*2 + 4096 = 8192 (was 12288)

#define EXPERT_SIZE         1769472  // was 7077888 (~1.69 MB vs ~6.75 MB)
```

## 7. Search Integration (Future)

Plan to integrate internet search directly into the inference loop, inspired by [llm-search](https://github.com/...):

- Tool definitions injected into the system prompt
- Engine intercepts `<tool_call>` tokens in generated output
- Executes SearXNG queries and `fetch_page` operations
- Injects results back into the context
- Self-hosted, no API keys required

Unlike llm-search's middleware approach, FinchMoE will handle tool calling natively in the C engine — no Python proxy needed.

## 8. Performance Model

Estimated performance on M4 Mac mini 16GB:

| Component | Time (est.) | Notes |
|---|---|---|
| CMD1: attn + delta-net | ~0.6 ms | Half the hidden dim of 397B |
| CPU: flush | ~0.01 ms | Unchanged |
| CMD2: o_proj + norm + routing | ~0.3 ms | Smaller matrices |
| CPU: softmax + topK | ~0.003 ms | 256 experts vs 512 |
| I/O: pread K=8 experts | ~0.8 ms | 8 × 0.94 MB = 7.5 MB read |
| CMD3: expert compute | ~0.02 ms | Deferred |
| **Per layer** | **~1.7 ms** | |
| **Per token (40 layers)** | **~68 ms** | **~14.7 tok/s theoretical** |

Real-world will be lower due to:
- External SSD latency (not internal NVMe)
- Page cache misses requiring actual SSD reads
- M4 GPU being slower than M3 Max GPU

Conservative estimate: **5-10 tok/s** — well above the 3.5 tok/s target.

## 9. Appendix: flash-moe Experiments Reference

Key findings from flash-moe's 58 experiments that inform our design:

| Finding | Impact |
|---|---|
| FMA dequant kernel | +12% tok/s |
| Trust OS page cache (vs custom LRU) | +38% tok/s |
| BLAS delta-net (Accelerate) | +64% attention speed |
| GPU combine+norm in CMD3 | Pipeline-critical |
| SSD DMA + GPU overlap impossible | Serial pipeline optimal |
| LZ4 expert compression | -13% (decompress overhead) |
| mmap expert files | -5× (page fault overhead) |
| MTP speculative decoding | Break-even (MoE I/O scales per-token) |
