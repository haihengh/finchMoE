# FinchMoE — Research & Architecture Reference

> Generated 2026-08-06. All findings from analyzing three reference projects and the Qwen 3.6 architecture.

## Project Goals

1. **Tailor-made inference engine** for Qwen MoE models on Apple Silicon
2. **Built-in internet search** — native in the generation loop, not middleware
3. **≤3 GB RAM** — targeting eventual iPhone deployment
4. **≥3.5 tok/s decode** — matching turbo-fieldfare's M4 mini performance
5. **Single binary** — no Python, no Docker, no separate servers

## Architecture Decision

**Base: flash-moe** (not finchmoe_engine). Reason: flash-moe already runs Qwen3.5-MoE architecture with GatedDeltaNet, 4-bit quantization, and SSD expert streaming. finchmoe_engine targets a non-existent early MoE spec and lacks all key features needed.

## Reference Projects

### turbo-fieldfare (THE BENCHMARK)
- **Author**: Andrey Mikhaylov (drumih)
- **Stack**: Swift 6.2 + Metal 4, macOS 26+
- **Model**: Gemma 4 26B-A4B IT (A4B = ~4B active per token)
- **Performance**: 3.5 tok/s on M4 Mac mini 16GB, 5.1-6.3 tok/s on M2 Air 8GB
- **RAM**: ~2 GB (common weights 1.35 GB mmap'd + 16-slot LFU expert cache + FP16 KV cache ~305 MB)
- **Key techniques**:
  - MLX affine INT4 quantization (group-64, bf16 scale+bias) everywhere + INT8 router
  - GPU router (INT8 affine GEMV + top-8 select kernel) — CPU only reads 8 IDs
  - cb1/io/cb2 three-phase pipeline: cb1(GPU attention+router) → io(parallel pread) → cb2(GPU MoE+combine)
  - Shared-expert GPU work overlaps expert I/O reads
  - Fused Metal kernels: QKV GEMV, QKV epilogue, post-attn setup, layer tail, greedy head
  - Persistent-workgroup fused MoE kernels (gate·up+GeGLU, down+weighted reduce)
  - Split-KV flash decode attention
  - Chunked prefill (≤128 tokens) with staged affine MPP (Metal Performance Primitives)
  - mmap'd resident weights with `bytesNoCopy` — zero heap copy
  - 2 MiB-aligned expert slots with `posix_memalign` + `makeBuffer(bytesNoCopy:)`
  - F_RDADVISE off by default (experiments showed it can slow long decodes)
  - MADV_DONTNEED KV cache reset
  - 103 audited experiments in docs/experiments/
- **Model format**: `.gturbo` directory with `model_weights.bin` (mmap'd) + `packed_experts/layer_XX.bin` (128 blobs/layer)
- **Streaming installer**: Downloads HF byte ranges directly into final layout, max 512KB scratch

### flash-moe (THE STARTING POINT)
- **Stack**: Pure C + Objective-C + Metal, ~7200 lines in infer.m + ~1300 lines shaders.metal
- **Model**: Qwen3.5-397B-A17B (same model_type: qwen3_5_moe as Qwen 3.6)
- **Performance**: 4.36 tok/s on M3 Max 48GB, 209GB model on SSD
- **RAM**: ~6 GB (non-expert weights 5.5GB mmap'd + ~200MB scratch)
- **Key techniques**:
  - FMA dequant kernel: `fma(nibble, scale*x, bias*x)` = +12% throughput
  - "Trust the OS" page cache — no custom expert cache; OS page cache ~71% hit rate
  - Every custom caching approach tested was SLOWER (Metal LRU, malloc cache, LZ4)
  - Parallel pread() via persistent pthread pool + GCD dispatch groups
  - GatedDeltaNet via Accelerate BLAS (cblas_sscal/sgemv/sger) — +64% attention
  - Deferred GPU expert compute (CMD3 submitted without wait)
  - GPU-side combine+residual+norm in CMD3 eliminates CPU round-trip
  - 2MB-aligned DMA buffers = 3.6x faster DMA
  - **Critical finding**: On Apple Silicon, SSD DMA and GPU compute CANNOT be profitably overlapped — serial GPU→SSD→GPU is hardware-optimal
- **Architecture**: 60 layers (45 GatedDeltaNet + 15 full attention), 512 experts, K=4 active
- **58-experiment log** in results.tsv covering: FMA dequant, LZ4 compression (-13%), routing prediction (-18%), F_RDADVISE (0%), mmap disaster (-5x), MTP analysis (break-even for MoE)
- **Tool calling**: At HTTP server layer, not in engine. chat.m is an HTTP client.
- **Build**: `make` in metal_infer/, produces `infer`, `chat`, `main`

### omlx (SUPPLEMENTAL OPTIMIZATIONS)
- **Stack**: Python + MLX + nanobind Metal kernels
- **Purpose**: General-purpose LLM serving with continuous batching
- **Relevant for us**: Qwen3.5-specific Metal kernels in `omlx/custom_kernels/qwen35_prefill/`:
  - `qwen35_q{2,4,5,6,8}_affine_qmm_t` — quantized matmul tile variants
  - `qwen35_fa256_attention` — steel-attention FA for head_dim=256
  - `qwen35_moe_weighted_sum` — fused scatter-free MoE combine
  - `gdn.py` — GatedDeltaNet blocked-seq kernel (2x faster than stock at 16k)
  - `qwen35_moe_gate_up.py` — **fuses gate+up expert projections** (one gather per MoE layer)
- **oQ quantization**: Data-driven mixed precision; Qwen3.5-35B-A3B benchmarks show 3-bit preserves 85% MMLU (vs 14% plain)

### llm-search (SEARCH INTEGRATION REFERENCE)
- **Stack**: Python FastAPI middleware
- **Approach**: Auto-injects web_search + fetch_page tools into OpenAI-compatible LLM
- **Search**: SearXNG (self-hosted, no API key, aggregates Google/Bing/DDG) + optional Brave/SerpAPI
- **Tool loop**: Intercepts tool calls server-side, executes searches, feeds results back, loops until answer
- **Model requirements**: MUST support OpenAI function calling (emit `tool_calls`)
- **Tested compatible**: qwythos-9b-claude-mythos-5-1m, qwen3.6-27b-claude-mythos-distilled-mtp, qwopus3.6-27b-v2-mtp, gemma-4-31b-it-qat
- **Tested NOT working**: qwen3.6-35b-a3b (MoE — loops without answering)
- **For our engine**: We'll build search directly into the generation loop, not as middleware

## Qwen 3.6 35B A3B Architecture

### Model Config (from config.json)
| Field | Value |
|-------|-------|
| model_type | qwen3_5_moe |
| architectures | Qwen3_5MoeForConditionalGeneration |
| hidden_size | 2048 |
| num_hidden_layers | 40 |
| num_attention_heads | 16 |
| num_key_value_heads | 2 |
| head_dim | 256 (explicit, NOT hidden/heads!) |
| vocab_size | 248320 |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 512 |
| shared_expert_intermediate_size | 512 |
| rms_norm_eps | 1e-6 |
| rope_theta | 10,000,000 |
| max_position_embeddings | 262,144 |
| partial_rotary_factor | 0.25 |
| mrope_section | [11, 11, 10] |
| mrope_interleaved | true |
| full_attention_interval | 4 |
| mamba_ssm_dtype | float32 |

### Layer Pattern (40 layers, 0-indexed)
```
3× linear_attention → 1× full_attention — repeated 10×
Full attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 (10 total)
Linear attention layers: all others (30 total)
```

### Full Attention Layers (10 layers)
- q_proj [8192, 2048] — DOUBLED: Q[4096] + output-gate[4096] fused
- k_proj [512, 2048], v_proj [512, 2048]
- o_proj [2048, 4096]
- GQA: 16 Q-heads × 256, 2 KV-heads × 256
- QK-norm: Qwen3_5RMSNorm = rms_norm(x) * (1 + weight), init=0
- Partial MRoPE: only 64 of 256 dims rotated, mrope_section [11,11,10] interleaved
- Attention output gate: attn_out *= sigmoid(gate) where gate = second-half of q_proj
- KV cache only on these 10 layers
- Sliding window: 32768

### GatedDeltaNet Layers (30 layers)
```
Tensors:
  in_proj_qkv [8192, 2048]     → Q[16×128], K[16×128], V[32×128]
  in_proj_z   [2048, 2048]     → output gate
  in_proj_a   [32, 2048]       → per-V-head decay offset
  in_proj_b   [32, 2048]       → per-V-head beta (write gate)
  conv1d      [8192, 1, 4]     → depthwise causal conv, groups=8192
  A_log       [32]             → log-decay base (fp32)
  dt_bias     [32]             → delta-t bias (fp32)
  norm        [128]            → RMSNormGated weight
  out_proj    [2048, 4096]     → 32×128 → hidden

Delta-rule recurrence (per token):
  q,k,v = split(conv1d(in_proj_qkv(x)))
  q,k = L2_normalize(q), L2_normalize(k); q *= 1/sqrt(128)
  g = -exp(A_log) * softplus(a + dt_bias)
  S_t = S_{t-1} * exp(g) + outer(k, (v - S_{t-1}@k) * sigmoid(b))
  out = S_t @ q
  out = rms_norm(out) * weight * silu(z)
  out = out_proj(out)
```

State per layer: conv_ring [8192×4] + recurrent [32×128×128] f32 ≈ 2.1 MB
30 layers × 2.1 MB ≈ 63 MB per sequence

### MoE (all 40 layers)
- 256 routed experts (top-8) + 1 shared expert
- FUSED gate_up_proj [256, 1024, 2048] (not separate gate/up!)
- down_proj [256, 2048, 512]
- Shared expert: gate_proj/up_proj [512, 2048], down_proj [2048, 512]
- Shared expert gate: mlp.shared_expert_gate.weight [1, 2048] (scalar sigmoid)
- Expert tensor names lack .weight suffix

### MTP (Multi-Token Prediction)
- 1 MTP layer with self_attn + 256-expert MoE
- Shares lm_head with main model
- Skip for v1 (flash-moe analysis: break-even for MoE SSD streaming)

### Vision Tower
- 27 ViT blocks, skip for text-only v1

## Quantization Format

MLX affine INT4 (group-64, bf16 scale+bias):
- 4-bit unsigned nibbles packed into uint32 (8 values per uint32)
- Per group of 64: scale (bf16) + bias (bf16) = 4 bytes metadata
- Dequant: w[i] = uint4[i] * scale[i/64] + bias[i/64]
- Overhead: 12.5% vs raw INT4
- Expert size (4-bit): ~2.1 MB per expert (vs ~6.3 MB bf16)

## Key flash-moe → Qwen 3.6 Changes

| Constant | flash-moe (397B) | Qwen 3.6 (35B) |
|----------|-----------------|----------------|
| HIDDEN_DIM | 4096 | 2048 |
| NUM_LAYERS | 60 | 40 |
| NUM_ATTN_HEADS | 32 | 16 |
| NUM_KV_HEADS | 2 | 2 |
| HEAD_DIM | 256 | 256 |
| NUM_EXPERTS | 512 | 256 |
| MOE_INTERMEDIATE | 1024 | 512 |
| SHARED_INTERMEDIATE | 1024 | 512 |
| LINEAR_NUM_V_HEADS | 64 | 32 |
| LINEAR_NUM_K_HEADS | 16 | 16 |
| LINEAR_KEY_DIM | 128 | 128 |
| LINEAR_VALUE_DIM | 128 | 128 |
| LINEAR_CONV_DIM | 12288 | 8192 |
| EXPERT_SIZE | 7,077,888 | ~2,160,000 |
| Expert layout | gate\|up\|down | gate_up\|down (fused) |

## Model Download

- **Source**: mlx-community/Qwen3.6-35B-A3B-4bit
- **Size**: ~20 GB (4 shards × ~5 GB)
- **Location**: /Volumes/samsung 2t/code/finchMoE/models/Qwen3.6-35B-A3B-4bit/
- **Status**: ~50% downloaded (needs resume after restart)

## Build Fixes Applied to finchmoe_engine

1. norm_kernels.metal: uint2 → uint for 1D grid (macOS 26 Metal compiler)
2. rope_kernels.metal: half → hd2 (half is reserved Metal type)
3. finchmoe_types.h: rope_scaling_type[8] → [9] (string too long)
4. 5 header files: removed #ifdef __OBJC__ MTL type aliases → always void*
5. CMakeLists.txt: no Metal compilation step — manual metallib build required
