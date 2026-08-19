# FinchMoE Inference Engine — Comprehensive Analysis

> **CURRENT STATUS (2026-08-18)** — this document's architecture analysis
> remains accurate; the state below reflects the latest commits.
>
> | Item | State |
> |---|---|
> | Weights | 1.95 GB: 4-bit GDN tier (qkv/z/out_proj), Q/K/V/O 4-bit, embed/lm_head 8-bit, beta/alpha BF16, **3-bit experts (default)** |
> | Speed | 16-22 tok/s decode (M4, K=8, warm cache — 2026-08-14 reruns); chunked batched prefill: 90-token 6.2-7.0s, 883-token 51s (2.1× vs per-token). M1 mini 8 GB: ~4.1 tok/s. **GGUF mode: chunked prefill default ON (13-token TTFT ~1.0-1.2s, 90-token ~5.5s), decode ~1.06 tok/s** |
> | RAM | 2.9 GB peak GPU + 0.34 GB CPU KV @ 8k; 256k context = 10.7 GB CPU KV (does not fit 16 GB) |
> | Correctness | Engine bit-exact vs numpy GDN reference (CosSim 1.000000/stage); kernels 1.0 vs CPU; GGUF 90-token chunked vs per-token bitwise (cos 1.000000 after the sl≥32 attention fix) |
> | Command pipeline | CMD1+CMD2 fused (one round trip/layer); GDN fully fused (conv+qk-norm+decay+delta+gated in one kernel); residual+norm fused; routing batch fused; GGUF: batched CMD3 (one CB + 21 dispatches/layer), expert pread dedup, deferred CMD3 overlap |
> | Expert formats | `-3` default / `-4` / `-2` / `-8`; packed_experts dirs per format; GGUF Q4_K/Q6_K via `--gguf FILE` |
> | Server | OpenAI-compatible SSE on `-R PORT`; `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`; chat TUI client (`make chat`); multi-turn sessions re-enabled (rollback + think ban, 2026-08-16) |
> | Known issues | (1) prefill GPU+IO co-bound — S6 decomposition: kernel-CB dispatch ~0.26 ms each + queue-drain wake tax (0.07@0.1ms gap → 1.4@3ms) + per-16KB page-walk cost on file-backed reads; the expert preads are IOMMU priming; (2) long-form essay drift ~100-250 tokens (quant stability limit; 8-bit GDN tier drifts later); (3) MTP head inherently weak (cos 0.3-0.8, 0% acceptance) — not shippable |

## 1. Model Architecture

### Qwen 3.6 35B A3B (Mixture of Experts)

| Parameter | Value |
|-----------|-------|
| Total parameters | 35B |
| Active parameters per token | ~3B (A3B) |
| Hidden dimension | 2048 |
| Vocabulary | 248,320 tokens |
| Layers | 40 total |
| Full attention layers | 10 (every 4th: 3, 7, 11, …, 39) |
| Linear attention layers | 30 (GatedDeltaNet) |
| Attention heads | 16 Q heads, 2 KV heads, head_dim=256 |
| Experts per layer | 256 |
| **Active experts per token** | **8** (8 × 512 intermediate = 4096, plus shared expert) |
| Shared expert | 1 per layer (always active, 512 intermediate) |
| GDN key/value heads | 16 K heads / 32 V heads, key_dim=128, value_dim=128 |
| Conv1d kernel size | 4 |
| RoPE | theta=10M, partial rotary 0.25 (64 of 256 dims) |

### Layer Structure (each of 40)

```
input_norm(hidden) → attention_block → h_mid (residual + attn_out)
    → post_attn_norm(h_mid) → h_post
    → expert_routing(h_post) → top-K experts + shared expert
    → combine → hidden' (output)
```

### Weight Inventory (4-bit model, current production)

| Component | Tensors | Raw Size | Format | Stored In |
|-----------|---------|----------|--------|-----------|
| Embeddings | 1 | 0.97 GB | BF16 | model_weights.bin |
| lm_head | 1 | 0.97 GB | BF16 | model_weights.bin |
| Attention Q/K/V/O + GDN | 377 | 2.62 GB | BF16 | model_weights.bin |
| Layer norms (input + post-attn) | 80 | 0.00 GB | BF16 | model_weights.bin |
| Routing gates (per-layer) | 40 | 0.04 GB | BF16 | model_weights.bin |
| Shared expert (gate/up/down) | 120 | 0.26 GB | BF16 | model_weights.bin |
| **Non-expert subtotal** | **632** | **4.96 GB** | BF16 | model_weights.bin |
| Routed experts (256 × 40) | 10,240 | ~14 GB | **4-bit packed** | packed_experts/layer_NN.bin |
| **Total on disk** | — | **~19 GB** | mixed | — |

**Key fact**: Non-experts are 4.96 GB (26% of total). Experts are ~14 GB (74%). But at inference, only K=8 experts per layer are needed, so per-token I/O is minimal (~13.5 MB for all 40 layers).

---

## 2. Weight Conversion Pipeline

### Source: MLX-format safetensors (4-bit model)

```
Qwen3.6-35B-A3B-4bit/
  model-00001-of-00004.safetensors  (4.9 GB)
  model-00002-of-00004.safetensors  (3.3 GB)
  model-00003-of-00004.safetensors  (3.1 GB)
  model-00004-of-00004.safetensors  (3.0 GB)
  model.safetensors.index.json
```

### Step 1: Extract non-expert weights → model_weights.bin

**Script**: `extract_weights.py`

```bash
python extract_weights.py \
  --model ../models/Qwen3.6-35B-A3B-4bit \
  --output .
```

**What it does**:
1. Parse `model.safetensors.index.json` to get file→tensor mapping
2. Filter out `switch_mlp.*` (routed expert) tensors — experts handled separately
3. Filter out `vision_tower.*` tensors
4. For each remaining tensor:
   - Read raw bytes from safetensors file
   - If scales/biases with dtype BF16: convert FP16→BF16 (MLX stores these as FP16)
   - Write to binary file at 64-byte aligned offsets
5. Produce `model_weights.bin` (4.96 GB) + `model_weights.json` manifest

**Format**: All non-expert tensors stored as raw BF16 (`uint16`). No quantization — the source model already has non-experts in BF16. The "4-bit" in the model name refers only to expert weights.

**Manifest structure** (`model_weights.json`):
```json
{
  "model": "../models/Qwen3.6-35B-A3B-4bit",
  "config": { "hidden_size": 2048, "num_experts": 256, ... },
  "tensors": {
    "lm_head.weight": { "offset": 0, "size": 1017118720, "shape": [248320, 2048], "dtype": "U16" },
    "model.embed_tokens.weight": { "offset": 1017118720, "size": 1017118720, ... },
    "model.layers.0.self_attn.q_proj.weight": { ... },
    ...
  }
}
```

### Step 2: Repack expert weights → packed_experts/layer_NN.bin

**Scripts**: `generate_expert_index.py` + `repack_experts.py`

```bash
python repack_experts.py
```

**What it does**:
1. `generate_expert_index.py`: Parse safetensors index, map each (layer, component) → file offset
2. `repack_experts.py`: For each of the 40 layers:
   - For each of the 256 experts:
     - Read gate_proj.weight, gate_proj.scales, gate_proj.biases
     - Read up_proj.weight, up_proj.scales, up_proj.biases
     - Read down_proj.weight, down_proj.scales, down_proj.biases
     - Write all 9 components contiguously → 1,769,472 bytes per expert
   - Write all 256 experts sequentially → `layer_NN.bin` (452 MB)

**Expert binary layout** (4-bit, per expert):
```
Offset      Size        Component
0           524,288     gate_proj.weight   [512, 256] U32 packed (512 × 256 / 32 × 4 = 8 vals/U32)
524,288     32,768      gate_proj.scales   [512, 32] BF16
557,056     32,768      gate_proj.biases   [512, 32] BF16
589,824     524,288     up_proj.weight     [512, 256] U32 packed
1,114,112   32,768      up_proj.scales     [512, 32] BF16
1,146,880   32,768      up_proj.biases     [512, 32] BF16
1,179,648   524,288     down_proj.weight   [2048, 64] U32 packed
1,703,936   32,768      down_proj.scales   [2048, 8] BF16
1,736,704   32,768      down_proj.biases   [2048, 8] BF16
─────────────────
TOTAL: 1,769,472 bytes
```

**Quantization format** (MLX affine):
- `w_q = round((w_f32 - min) / scale)`, clamped to [0, 2^bits-1]
- `scale = (max - min) / (2^bits - 1)`, `bias = min`
- Packed: 8 values per uint32 (4-bit), group_size=64
- Scales/biases stored as BF16 to save space

**Supported bit widths**:
| Bits | Expert Size | Disk (40 layers) |
|------|-------------|------------------|
| 1-bit | 589,824 B | 5.6 GB |
| 2-bit | 983,040 B | 9.4 GB |
| 4-bit | 1,769,472 B | 17 GB |
| 8-bit | 3,342,336 B | 31 GB |

### Step 3: Export vocabulary → vocab.bin

**Script**: `export_tokenizer.py`

Reads tokenizer.json from the model directory and produces a binary vocabulary file containing token strings in a compact format. Also exports BPE merge rules for tokenization.

---

## 3. Inference Engine Architecture

### 3.1 Memory Layout at Runtime

```
┌─────────────────────────────────────────────────┐
│  model_weights.bin (mmap'd, 4.96 GB)            │
│  ├─ Metal buffer wrap (zero-copy, same pages)   │
│  └─ Read directly by GPU kernels (BF16→float)   │
├─────────────────────────────────────────────────┤
│  packed_experts/layer_*.bin (mmap'd, 17 GB)     │
│  ├─ Tiered I/O: F_NOCACHE for first read        │
│  ├─ OS page cache for repeated experts          │
│  └─ LRU cache / malloc cache (optional)          │
├─────────────────────────────────────────────────┤
│  GPU Buffers (Metal, ~450 MB allocated)         │
│  ├─ 10 KV caches (2 × 256 × 8192 × 4 = 16.8 MB │
│  ├─ 30 GDN states (32×128×128 × 4 = 2.1 MB each)│
│  ├─ 16 expert data slots (2 MB each, 2MB-align)│
│  ├─ Batch output slots, scratch buffers        │
│  └─ lm_head output (970 KB)                     │
├─────────────────────────────────────────────────┤
│  CPU Scratch (~2 MB)                            │
│  ├─ Per-layer: normed, residual, attn_proj,     │
│  │   h_post, h_mid, gate_scores, moe_out, ...   │
│  └─ Attention: q_proj_out, k_out, v_out, ...    │
└─────────────────────────────────────────────────┘
```

### 3.2 GPU Pipeline (Metal Shaders)

All GPU compute uses Metal Shading Language. Kernels compiled at runtime from `shaders.metal`.

**Core kernels**:

| Kernel | Purpose | Grid | Threads | Notes |
|--------|---------|------|---------|-------|
| `dequant_matvec_4bit_v3` | 4-bit dequant + matvec | out_dim/8 TG | 256 | 8 rows/tg, tiled, vector loads |
| `dequant_matvec_2bit` | 2-bit dequant + matvec | out_dim/8 TG | 256 | 16 vals/uint32 |
| `dequant_matvec_1bit` | 1-bit dequant + matvec | out_dim/32 TG | 256 | 32 vals/uint32 |
| `dequant_matvec_8bit` | 8-bit dequant + matvec | out_dim/4 TG | 256 | 4 vals/uint32 |
| `gemv_bf16` | Raw BF16 matvec | out_dim TG | 256 | No dequant, interleaved CHUNK=8 |
| `gemv_bf16_x2` | 2-row BF16 matvec | out_dim/2 TG | 256 | 128-bit uint4 loads |
| `swiglu_fused` | gate+up+swiglu fused | out_dim/256 TG | 256 | Single-pass fused |
| `fused_gate_up_swiglu_2x` | 2 experts in 1 dispatch | out_dim/256 TG | 256 | Both gate+up+swiglu at once |
| `moe_combine_residual` | Combine + residual + norm | dim/256 TG | 256 | GPU-side combine |
| `rms_norm_sum_sq` | Sum of squares reduction | 1 TG | 256 | Pre-reduction |
| `rms_norm_apply_bf16` | Apply norm to vector | dim/256 TG | 256 | Uses pre-computed sum_sq |
| `gated_delta_net_step` | GDN recurrence | 32 TG | 128 | One per V-head |
| `conv1d_step` | Depthwise conv1d | (8192+255)/256 | 256 | SiLU activation |
| `attn_scores_batched` | Q @ K^T scores | seq_len × 16 TG | 256 | Batched for all heads |
| `attn_softmax_batched` | Softmax over seq | 16 TG | 256 | Per-head |
| `attn_values_batched` | Softmax @ V | 16 × 256 TG | 256 | Weighted sum |

### 3.3 Per-Layer Forward Pass (3 Command Buffers)

The core of the engine is `fused_layer_forward()`. Each transformer layer processes one token through three Metal command buffers:

#### CMD1: Attention Input Projections
```
┌─────────────────────────────────────────────────┐
│  GPU: 3-4 batch matvecs (1 command buffer)       │
│                                                   │
│  Full attention (3 specs):                        │
│    q_proj: [16384] = Q_w[16384,2048] @ x[2048]  │
│    k_proj: [512]   = K_w[512,2048]  @ x[2048]   │
│    v_proj: [512]   = V_w[512,2048]  @ x[2048]   │
│                                                   │
│  Linear attention (4 specs):                      │
│    qkv:  [12288] = QKV_w[12288,2048] @ x[2048]  │
│    z:    [4096]  = Z_w[4096,2048]    @ x[2048]  │
│    beta: [32]    = B_w[32,2048]      @ x[2048]  │
│    alpha:[32]    = A_w[32,2048]      @ x[2048]  │
│                                                   │
│  GPU Linear Attention (optional, 5 more encoders):│
│    L1: conv1d_step (SiLU activation)              │
│    L2: rms_norm_qk (q,k normalization)            │
│    L3: compute_decay_beta (g_decay, beta_gate)    │
│    L4: gated_delta_net_step (SSM update + output) │
│    L5: gated_rms_norm (norm + gate apply)         │
│                                                   │
│  Commit + Wait                                    │
└─────────────────────────────────────────────────┘
```

**GPU kernels used**: `gemv_bf16_x2` or `gemv_bf16` (BF16 weights), `dequant_matvec_4bit_v3` (quantized weights)

**Time**: ~0.87 ms (dominated by kernel dispatch overhead for 5-9 encoders)

#### CPU: Attention Compute
```
┌─────────────────────────────────────────────────┐
│  Full attention:                                  │
│    1. Split q_proj_out → q (4096) + q_gate (4096)│
│    2. RMSNorm on Q heads, K heads                │
│    3. Apply RoPE to Q and K                       │
│    4. Update KV cache (CPU + GPU mirror)         │
│    5. GPU: attn_scores + softmax + values → attn  │
│       (fused into CMD2 for seq ≥ 32)              │
│    6. Sigmoid gate → gated attention output       │
│                                                   │
│  Linear attention:                                │
│    1. Conv1d step (depthwise, kernel=4) → q,k,v  │
│    2. Update conv_state ring buffer               │
│    3. Split: q (2048), k (2048), v (4096)        │
│    4. RMSNorm on q and k heads                   │
│    5. Compute g_decay, beta_gate (softplus/sigmoid)│
│    6. GDN recurrence (GPU or CPU BLAS)            │
│       S *= g  →  kv = S @ k  →  delta = (v-kv)*β │
│       S += k ⊗ delta  →  out = S^T @ q           │
│    7. Gated RMS norm: norm(out) ⊙ silu(z) × w    │
│       (CPU fallback if GPU path unavailable)      │
└─────────────────────────────────────────────────┘
```

**Time**: ~0.1 ms (CPU, mostly memory-bound)

#### SPECULATIVE I/O (overlapped with CPU attention)
```
┌─────────────────────────────────────────────────┐
│  Predict experts from pre-attention hidden state  │
│  Dispatch async pread for cache misses            │
│  (DISABLED: only 41% hit rate, overhead > savings)│
└─────────────────────────────────────────────────┘
```

#### CMD2: O-Projection + Residual + Norm + Routing
```
┌─────────────────────────────────────────────────┐
│  GPU: 8-12 encoders (1 command buffer)            │
│                                                   │
│  Enc 1:   o_proj (gemv_bf16)                     │
│           batch_out[6] → buf_output (4096 floats) │
│  Enc 2:   residual_add                           │
│           buf_residual + buf_output → buf_h_mid   │
│  Enc 3:   rms_norm_sum_sq                        │
│           buf_h_mid → buf_sum_sq                  │
│  Enc 4:   rms_norm_apply_bf16                     │
│           buf_h_mid × w / rms → buf_input         │
│  Enc 5-8: routing + shared expert (4 matvecs)     │
│           gate: [256] + shared_gate: [512]        │
│           shared_up: [512] + shared_gate_score: [1]│
│                                                   │
│  With GPU attention (seq ≥ 32), adds 4 encoders: │
│  Enc A1-4: attn_scores + softmax + values + gate │
│             → buf_attn_out (o_proj reads from it) │
│                                                   │
│  Commit + Wait                                    │
└─────────────────────────────────────────────────┘
```

**GPU kernels used**: `gemv_bf16`, `residual_add`, `rms_norm_sum_sq`, `rms_norm_apply_bf16`, `dequant_matvec_4bit_v3` (routing), `attn_scores_batched`, `attn_softmax_batched`, `attn_values_batched`

**Time**: ~0.6 ms

#### CPU: Routing + Parallel I/O
```
┌─────────────────────────────────────────────────┐
│  1. cpu_softmax(gate_scores, 256)                 │
│  2. cpu_topk(scores, K=8) → expert_indices       │
│  3. cpu_normalize_weights(8)                      │
│  4. Parallel pread: 8 experts × 1.77 MB each     │
│     Using GCD dispatch_group, 4 threads           │
│     Cache-aware: skip pread for cached experts    │
│  5. Shared expert gate: sigmoid(score)            │
└─────────────────────────────────────────────────┘
```

**Time**: ~0.4 ms (I/O-bound, depends on SSD speed + page cache warmth)

#### CMD3: Expert Forwards + Combine (DEFERRED)
```
┌─────────────────────────────────────────────────┐
│  GPU: K×2 + 4 encoders (1 command buffer)         │
│                                                   │
│  For each expert k (K=8):                         │
│    Enc k_gate: fused_gate_up_swiglu               │
│      gate_proj + up_proj + SiLU(gate) × up       │
│      → buf_multi_expert_act[k] (512 floats)       │
│    Enc k_down: swiglu_fused + down_proj           │
│      buf_act[k] × down_proj_w → buf_out[k] (2048)│
│                                                   │
│  Shared expert (always active):                   │
│    Enc shared: swiglu + down_proj                 │
│      buf_shared_gate ⊙ SiLU → buf_shared_act      │
│      buf_shared_act × down_w → buf_shared_out     │
│                                                   │
│  GPU-side combine (for non-last layers):           │
│    Enc combine: moe_combine_residual               │
│      weighted_sum(all expert outputs)              │
│      + shared_out × shared_gate_score             │
│      + buf_h_mid (residual)                       │
│      → buf_moe_hidden                             │
│    Enc rms_norm: buf_moe_hidden → buf_input       │
│      (using next layer's input_norm weights)       │
│                                                   │
│  Commit (ASYNC — NO WAIT)                         │
│  ↓ Next layer starts immediately                  │
│  ↓ CMD3(N-1) finishes before CMD1(N) on GPU queue│
│  ↓ At start of next layer: finalize_deferred()    │
│    reads buf_moe_hidden → computes hidden         │
└─────────────────────────────────────────────────┘
```

**GPU kernels used**: `fused_gate_up_swiglu`, `swiglu_fused` + `dequant_matvec_4bit_v3` (down_proj), `moe_combine_residual`, `rms_norm_sum_sq`, `rms_norm_apply_bf16`

**Time**: ~0.5 ms (expert compute) + 0 ms (combine runs async)

### 3.4 Per-Token Timing Budget (4-bit experts, K=8)

Measured from the 400-token run (avg ~260 ms/token = 3.85 tok/s):

| Phase | ms | % | What |
|-------|----|---|------|
| CMD1 (attn projections) | ~90 | 35% | 3-4 BF16 matvecs + optional GPU linear attn |
| CPU attention compute | ~10 | 4% | Conv1d/GDN or full attn (mostly memory-bound) |
| CMD2 (o_proj + norm + routing) | ~60 | 23% | 8 encoders (4 matvecs + 3 norm ops + 1 o_proj) |
| CPU routing + parallel I/O | ~45 | 17% | Softmax + top-K + 8 × 1.77 MB pread from SSD |
| CMD3 (expert forwards, ASYNC) | ~30 | 12% | K×2 + shared + combine (all GPU) |
| CMD3 deferred completion | ~15 | 6% | Wait + readback at start of next layer |
| Overhead (sync, misc) | ~10 | 4% | State management, combine output readback |
| **Total per layer** | **~260** | **100%** | |

Per layer, averaged over 40 layers, 400 tokens.

### 3.5 Token Generation Loop

```c
// --- Prefill (batch of prompt tokens) ---
for each prompt token t:
    embed_lookup(token) → hidden[2048]
    for each layer 0..39:
        fused_layer_forward(layer, hidden, ...)
        if not last token: discard_deferred_experts()  // skip readback
    if last token: complete_deferred_experts()          // read back

// --- Generation (autoregressive loop) ---
for gen = 1..max_tokens:
    // 1. Final norm + lm_head
    rms_norm(hidden, final_norm_w) → normed
    lm_head_forward(wf, normed, logits)  // GPU: gemv_bf16_x2, 124K TG
    
    // 2. Sample next token
    cpu_sample_temp(logits, temp, top_k)  // rep penalty, softmax, sampling
    → next_token
    
    // 3. Decode + output
    decode_token(vocab, next_token) → string
    
    // 4. Embed + forward pass
    embed_lookup(next_token) → hidden
    for each layer 0..39:
        fused_layer_forward(layer, hidden, ...)
    complete_deferred_experts()
    pos++
```

**Key optimizations**:
1. **GPU-side combine** (CMD3): Combines expert outputs + residual + norm on GPU, eliminating 0.83 ms of CPU readback per layer
2. **Async CMD3**: Expert computation runs concurrently with next layer's attention
3. **Batched encoders**: All K experts in 1 command buffer (not K separate buffers)
4. **Preallocated scratch**: ~1200 malloc/free per token eliminated via static buffers
5. **Zero-copy weights**: mmap'd weight file wrapped as Metal buffer, read directly by GPU
6. **Tiered I/O**: F_NOCACHE for first expert read, page cache for repeated reads

---

## 4. Performance Characteristics

### 4.1 Speed vs Expert Count

| K | tok/s | Quality | Note |
|---|-------|---------|------|
| 2 | 8.3 | ❌ Garbage | Only 25% of expert compute |
| 4 | 7.5 | ⚠️ Unknown | Not tested recently |
| 8 | 3.9 | ✅ Coherent | Model's training configuration |

**Why K matters**: Each expert requires I/O (1.77 MB pread) + GPU compute (two dispatches). At K=8, I/O is 14 MB/layer × 40 = 560 MB per token. SSD bandwidth (~800 MB/s for random reads on TB4 NVMe) limits TG speed.

### 4.2 Expert Bit Width vs Speed

| Bits | Expert Size | I/O Time (8 experts) | Speed Impact |
|------|-------------|---------------------|--------------|
| 1-bit | 590 KB | ~5 ms | Fastest, degraded quality |
| 2-bit | 983 KB | ~8 ms | 7-8 tok/s, acceptable quality |
| 4-bit | 1.77 MB | ~15 ms | 4-5 tok/s, reference quality |
| 8-bit | 3.34 MB | ~30 ms | Slower, higher quality |

### 4.3 Memory Requirements

| Component | Size | Type |
|-----------|------|------|
| Weight mmap | 4.96 GB | Virtual (demand-paged) |
| Expert mmaps (40 layers) | 17 GB | Virtual (demand-paged) |
| GPU buffers | 449 MB | Physical (Metal) |
| CPU scratch | 2 MB | Physical (malloc) |
| Page cache (warm) | ~5 GB | Physical (OS-managed) |
| **Physical RAM (warm)** | **~6 GB** | |
| **Physical RAM (cold)** | **~0.5 GB** | (just GPU + CPU scratch) |

Apple Silicon unified memory means the mmap'd files consume only virtual address space, not physical RAM. Physical RAM usage is ~450 MB for GPU buffers + expert data that's been faulted in by pread.

### 4.4 SSD Bottleneck Analysis

The primary bottleneck is expert I/O. Each token requires reading K=8 experts × 40 layers = 320 expert reads. At 1.77 MB each (4-bit) = 566 MB per token. At 3.88 tok/s, sustained SSD bandwidth is **2.2 GB/s**.

**Samsung 990 Plus NVMe (TB4 enclosure)**: ~2.8 GB/s sequential, ~800 MB/s random (QD1, 4K).

The random-read nature of expert access (different experts per layer per token) means we're limited by SSD random read IOPS, not sequential bandwidth. With K=8, we hit ~320 reads per token, each 1.77 MB. At 3.88 tok/s, that's ~1240 reads/s. The drive's random read capability (~500-1000 IOPS for 1.77 MB chunks) means I/O is the bottleneck.

**Optimization opportunities**:
- Reduce expert bit width (2-bit halves I/O volume)
- Increase expert reuse (cache hits avoid I/O)
- Batch preads (use dispatch_group for parallel reads)

### 4.5 Best Configuration Found

```bash
./infer \
  -t 400 \                   # max tokens
  -k 8 \                     # experts per token (MUST be 8)
  --temperature 0.7 \        # avoid greedy attractor states
  --rep-penalty 1.15         # prevent token repetition loops
```

---

## 5. GPU Kernel Dispatch Summary

### 5.1 Per-Layer Dispatches (CMD1 + CMD2 + CMD3)

| Command Buffer | Encoders | Kernels | Commitment |
|---------------|----------|---------|------------|
| CMD1 | 3-9 | gemv_bf16_x2, conv1d_step, rms_norm_qk, compute_decay_beta, gated_delta_net_step, gated_rms_norm | Commit + Wait |
| CMD2 | 8-12 | gemv_bf16, residual_add, rms_norm_sum, rms_norm_apply_bf16, dequant_matvec_4bit_v3, attn_scores/softmax/values | Commit + Wait |
| CMD3 | K×2+4 | fused_gate_up_swiglu, swiglu_fused, dequant_matvec_4bit_v3, moe_combine_residual | **Commit only** (async) |
| **Total** | **17-29** | | 2 waits, 3 commits |

### 5.2 Per-Layer GPU Time Breakdown

| Component | ms | Note |
|-----------|-----|------|
| CMD1 encode + commit + wait | 87.4 | Dominated by GPU dispatch overhead |
| CMD2 encode + commit + wait | 62.4 | 8 encoders (o_proj + residual + norm + routing) |
| CMD3 encode | 46.6 | K×2 + shared + combine (async) |
| CMD3 deferred wait + finalize | 15.8 | Wait for GPU + CPU combine readback |
| CPU attention + routing + I/O | 48.2 | Softmax, top-K, parallel pread |
| **Total** | **260.4** | |

(Timing from 4-bit experts, K=8, warm page cache, M4 Mac mini 16GB)

---

## 6. Optimization Opportunities

### 6.1 High-Impact, Low-Risk

| # | Optimization | Est. Gain | Complexity | Approach |
|---|-------------|-----------|------------|----------|
| 1 | **2-bit experts** | +2 tok/s | Low | Already supported (`--2bit`). I/O volume halved. Slight quality tradeoff. |
| 2 | **Pre-warm expert cache** | +1-2 tok/s | Low | Prefetch experts for common prompts. First 50 tokens are cold (SSD reads). After warmup, 5-6 tok/s. |
| 3 | **Increase GPU KV sequence** | Free | Low | `--gpu-kv-seq 16384` (from 8192). More GPU attention = less CPU overhead for long sequences. |
| 4 | **Fuse CMD1+CMD2** | +0.5 tok/s | Medium | Eliminate 1 commit+wait per layer (~5 ms). Requires combining attention projections with o_proj+norms into single command buffer. |

### 6.2 Medium-Impact, Medium-Risk

| # | Optimization | Est. Gain | Complexity | Approach |
|---|-------------|-----------|------------|----------|
| 5 | **MTP speculative decoding** | +1.5-2× TG | High | Infrastructure complete, latent bugs. Predicts 1 future token from MTP head, main model verifies. 50% acceptance → 2× effective TG. |
| 6 | **ICB (Indirect Command Buffers)** | -15 ms in CMD3 | High | Metal ICB for K expert dispatches in CMD3. Currently each expert has per-encoder overhead. ICB amortizes encoding cost. |
| 7 | **Single-kernel multi-expert** | -10 ms in CMD3 | High | Process all K experts in one kernel dispatch. GPU occupancy goes up. Buffer binding complexity. |
| 8 | **KV cache FP16** | -448 MB RAM | Medium | Currently FP32. Half precision halves KV cache memory, enabling longer context or larger GPU KV. |

### 6.3 Low-Impact or High-Risk

| # | Optimization | Est. Gain | Complexity | Risk |
|---|-------------|-----------|------------|------|
| 9 | Non-expert quantization | +2-3 GB saved | High | CPU dequant 400× slower; GPU path "hangs" for large tensors per quantize_model.py comments |
| 10 | Expert prediction/prefetch | ~1 tok/s | Medium | 41% hit rate, validation overhead > savings. Needs better predictor. |
| 11 | Batched GPU prefill | 5-10× PP | High | Current prefill is CPU-only (embeddings pre-computed). GPU batching needed for production. |
| 12 | GDN chunked prefill | 2-3× PP | Medium | Recurrent state processed sequentially. Chunking enables parallelism. |

### 6.4 Quality Improvements

| # | Optimization | Impact | Approach |
|---|-------------|--------|----------|
| 13 | **3-bit experts** | Quality↑ Size↓ | Sweet spot between 2-bit (degraded) and 4-bit (reference). 75% of 4-bit I/O. |
| 14 | **Per-layer mixed precision** | Quality↑ | Early layers (more important) at 4-bit, later layers at 2-bit. |
| 15 | **Outlier-aware quantization** | Quality↑ | Handle activation outliers with per-channel scaling. |
| 16 | **Calibration dataset** | Quality↑ | Use representative data (not random) for quantization scale calibration. |

---

## 7. Key Design Decisions & Tradeoffs

### Why BF16 non-experts (not quantized)?

- **CPU**: `cblas_sgemv` on BF16→F32 converted weights runs at ~0.01 ms for a [2048, 2048] matmul. The 4-bit dequant path on CPU is scalar and takes ~4.2 ms (400× slower). 
- **GPU**: Dequant kernels work perfectly for expert shapes ([512, 2048], [2048, 512]) but "hang" for larger non-expert tensors ([2048, 2048], [4096, 2048], [248320, 2048]). The root cause is unknown — possibly Metal driver limits on total threadgroups or buffer binding overhead.
- **lm_head**: Uses GPU `gemv_bf16_x2` to avoid the 2 GB F32 cache allocation. Reads BF16 weights directly from Metal buffer (zero-copy).

### Why K=8 (not configurable)?

The model was trained with 8 experts/token. At K<8, the missing expert outputs create a systematic error in the residual stream at every layer. Unlike some MoE models that tolerate fewer active experts at inference time, Qwen 3.6 35B's routing is tightly coupled to K=8. Even K=4 produces degraded output.

### Why GPU-side combine?

Eliminating the CPU round-trip for expert output readback saves ~0.83 ms per layer (42% of per-layer time). The GPU combines K expert outputs + shared expert + residual directly, writes the result to GPU memory, and the next layer reads it from GPU memory. Only at the very end (before lm_head) does the result come back to CPU.

### Why deferred CMD3?

Expert GPU compute is launched async. While it runs, the next layer starts CMD1. The GPU queue serializes CMD3(layer N-1) then CMD1(layer N). This overlaps expert compute with attention projections, saving ~15 ms per layer (the CMD3 execution time).

---

## 8. Known Issues & Limitations

1. **Non-expert quantization blocked**: CPU 400× slower, GPU path hangs. Root cause unknown.
2. **MTP speculative decoding latent**: Infrastructure complete but disabled (g_use_mtp=0). Had 50% acceptance rate but quality issues.
3. **Expert prediction disabled**: 41% hit rate, validation overhead exceeds savings.
4. **No continuous batching**: Single-request processing. Multi-request batching would improve throughput.
5. **CPU-only prefill**: Prompt processing is slow (3.4s for 4 tokens). GPU batching needed.
6. **Temporal drift at T=0**: Slight numeric drift accumulates over ~270 tokens with greedy decoding. Fixed with T≥0.1 + rep_penalty. Root cause likely in GDN recurrence precision mismatch.
7. **Weight file too large for iPhone**: 4.96 GB non-expert + 17 GB experts. iPhone target needs ≤2 GB total. Requires non-expert quantization (blocked) + 2-bit experts (9.4 GB). Total with both: ~4.2 GB — still too large. Need 1-bit experts (5.6 GB) + quantized non-experts (~1.5 GB) = ~7 GB. Further optimization needed.
