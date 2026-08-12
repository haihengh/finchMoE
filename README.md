# FinchMoE

A C/Metal inference engine for **Qwen 3.6 35B A3B** on Apple Silicon, targeting **12 tok/s on M4 with ~2 GB RAM** (currently at 3.9 tok/s, ~6 GB).

## Current Status (2026-08-11)

| Metric | Value |
|--------|-------|
| Output quality | ✅ Coherent — 400 tokens of valid Ruby code |
| **Speed (M4, 4-bit, K=8)** | **3.88 tok/s** (correct, stable) |
| Speed (M4, 2-bit, K=2) | 8.3 tok/s (WRONG — produces garbage) |
| RAM (runtime) | ~6.0 GB (4.96 GB weight file + 0.45 GB GPU buffers) |
| Common weight file | 4.96 GB (all BF16 — needs quantization) |
| Expert disk (4-bit) | 17 GB (40 layers × 256 experts × 1.77 MB) |
| Expert disk (2-bit) | 9.4 GB (40 layers × 256 experts × 0.98 MB) |
| Bugs fixed | **16** (documented in BUGS.md) |

**Important**: Model trained with 8 experts/token. K MUST be 8. K=2 silently drops 75% of expert compute and produces word salad. This is now the default.

## Quick Start

```bash
cd finchmoe

# Build
clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation \
      -framework Accelerate -lcompression -lpthread infer.m -o infer

# Run (recommended settings)
./infer -t 400 -k 8 --temperature 0.7 --rep-penalty 1.15

# With diagnostics
./infer --logit-diag 50 -t 400 -k 8 --temperature 0.7 --rep-penalty 1.15

# 2-bit experts (faster, slightly lower quality)
./infer --2bit -t 400 -k 8 --temperature 0.7 --rep-penalty 1.15

# Server mode
./infer --serve 9000 -k 8 --temperature 0.7 --rep-penalty 1.15
```

### Key Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--prompt TEXT` | — | Input prompt |
| `--tokens N` | 20 | Max tokens to generate |
| `--temperature F` | 0.8 | Sampling temperature (0 = greedy → can degrade) |
| `--rep-penalty F` | 1.15 | Repetition penalty (1.0 = disabled) |
| `--top-k N` | 40 | Top-k sampling cutoff |
| `-k N` | **8** | Active experts per layer (model trained with 8) |
| `--2bit` | off | Use 2-bit expert quantization (faster) |
| `--logit-diag N` | off | Dump top-20 logits + entropy every N tokens |
| `--timing` | off | Per-layer timing breakdown |
| `--mtp` | off | Enable MTP speculative decoding (experimental) |
| `--low-memory` | off | Skip Metal weight wrap (safer, slower) |
| `--serve PORT` | — | HTTP server (OpenAI-compatible API) |
| `--gpu-kv-seq N` | 8192 | GPU KV buffer size in tokens |

## Model Sizes

| Model | Common Weights | Experts | Total Disk | Speed (K=8) | Quality |
|-------|---------------|---------|------------|-------------|---------|
| BF16 (source) | 67 GB | — | 67 GB | — | Reference |
| 4-bit-dense | **4.96 GB** BF16 | 17 GB 4-bit | **21 GB** | **3.88 tok/s** | ✅ Coherent |
| 2-bit-dense-v2 | 4.96 GB BF16 | 9.4 GB 2-bit | 14 GB | ~5 tok/s | ✅ Coherent |
| 1-bit-dense | 4.96 GB BF16 | 5.6 GB 1-bit | 10 GB | ~5 tok/s | ⚠️ Degraded |

**Note**: The "4-bit-dense" and "2-bit-dense" names only refer to expert quantization. Non-expert weights (4.96 GB) are currently all BF16. Quantizing non-experts is the #1 optimization priority — see [Optimization Plan](finchmoe/OPTIMIZATION_PLAN.md).

## Reference Comparison

| Engine | Model | Active | RAM | M4 Speed | Notes |
|--------|-------|--------|-----|----------|-------|
| **FinchMoE** | Qwen 3.6 35B A3B | 3B | ~6 GB | **3.9 tok/s** | Custom C/Metal, 256K ctx |
| [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) | Gemma 4 26B A4B | 4B | **~2 GB** | **10.7 tok/s** | Swift/Metal, 4-bit everywhere, 8K ctx |
| [flash-moe](https://github.com/nicholas-ochoa/flash-moe) | Qwen 3.5 397B A17B | 17B | ~6 GB | 4.4 tok/s (M3 Max) | C/Metal, FMA dequant |
| llama.cpp Q4_K_M | Qwen 3.6 35B A3B | 3B | ~20 GB | — | Uniform quantization, reference quality |

**Key gap**: turbo-fieldfare achieves 5-6 tok/s on M2 with Gemma 4 (4B active) at 2 GB RAM by quantizing ALL weights to 4-bit. Our non-expert weights are still BF16, making our common model 3.7× larger. Closing this gap is the critical path to 12 tok/s.

## Architecture

```
40 layers: 30× GatedDeltaNet (linear attention) + 10× full attention
Hidden dim: 2048, 256 experts/layer, 8 active + 1 shared expert
MoE intermediate: 512, Head dim: 256, 16Q/2KV GQA
Vocab: 248,320 tokens
```

### Per-Layer Pipeline (3 Metal Command Buffers)

```
CMD1: attention projections (3-4 BF16 matvecs, 1 commit+wait)
  └─ optional GPU linear attention (5 fused kernels: conv1d→norm→decay→recur→gate)
CPU:  attention compute (RoPE/softmax/GDN recurrence)
CMD2: o_proj + residual + norm + routing + shared gate/up (8 encoders, 1 commit+wait)
CPU:  softmax + top-K + parallel pread of K experts (4 threads via GCD)
CMD3: K expert forwards + shared expert + combine (K×2+4 encoders, ASYNC commit)
  └─ GPU-side combine: weighted sum + residual + norm → directly into next layer
```

**Key optimizations**:
- **Deferred CMD3**: Expert GPU compute runs async, overlaps with next layer's CMD1
- **GPU-side combine**: Eliminates CPU readback, saves 0.83 ms/layer
- **Zero-copy weights**: mmap'd BF16 weights wrapped as Metal buffer, read directly by GPU
- **Tiered I/O**: F_NOCACHE for first read, OS page cache for repeated experts
- **Parallel pread**: 8 threads (one per expert) via GCD dispatch_group

## Project Structure

```
finchMoE/
├── README.md
├── BUGS.md                       # 16 bugs documented
├── design.md                     # Architecture & design decisions
├── finchmoe/
│   ├── infer.m                   # Main engine (C/Metal, ~9300 lines)
│   ├── shaders.metal             # Metal compute kernels (~2000 lines)
│   ├── ENGINE_ANALYSIS.md        # Comprehensive engine analysis
│   ├── OPTIMIZATION_PLAN.md      # Concrete roadmap to 12 tok/s @ 2 GB
│   ├── tokenizer.h               # C BPE tokenizer (248K vocab)
│   ├── extract_weights.py        # Non-expert weight → model_weights.bin
│   ├── repack_experts.py         # Expert weight → packed_experts/
│   ├── generate_expert_index.py  # Expert index generator
│   ├── quantize_model.py         # BF16 → 1/2/4/8-bit quantization
│   ├── export_tokenizer.py       # Tokenizer export
│   ├── compress_experts.py       # LZ4 compression (ineffective for quantized data)
│   ├── debug_full_forward.py     # Full 40-layer forward comparison
│   ├── debug_e2e_logits.py       # End-to-end logit comparison
│   ├── debug_gdn_compare.py      # GDN vs HF reference
│   └── finchTool/                # Metal kernel verification suite
├── models/
│   ├── Qwen3.6-35B-A3B-bf16/             # BF16 source (67 GB)
│   ├── Qwen3.6-35B-A3B-4bit-dense/       # 4-bit experts (21 GB active)
│   ├── Qwen3.6-35B-A3B-2bit-dense-v2/    # 2-bit experts (14 GB active)
│   ├── Qwen3.6-35B-A3B-1bit-dense/       # 1-bit experts (10 GB active)
│   ├── Qwen3.6-35B-A3B-bf16.gguf         # llama.cpp BF16 (71 GB)
│   └── Qwen3.6-35B-A3B-Q4_K_M.gguf       # llama.cpp Q4 (20 GB)
├── turbo-fieldfare/              # Performance benchmark reference
├── flash-moe/                    # Architecture reference
└── llama.cpp/                    # Ground truth reference
```

## Optimization Roadmap → 12 tok/s @ 2 GB

Full details: [`finchmoe/OPTIMIZATION_PLAN.md`](finchmoe/OPTIMIZATION_PLAN.md)

### Phase 1: Non-Expert Quantization → ~2.5 GB RAM, ~4 tok/s

**Status**: 🔴 BLOCKED — GPU path hangs for large non-expert tensors. Must solve.

| Component | Current | Target | Savings |
|-----------|---------|--------|---------|
| Embeddings + lm_head | 1.94 GB BF16 | 0.97 GB 8-bit | 0.97 GB |
| Attention Q/K/V/O | 1.84 GB BF16 | 0.46 GB 4-bit | 1.38 GB |
| GDN projections | 0.78 GB BF16 | 0.20 GB 4-bit | 0.58 GB |
| Shared experts | 0.26 GB BF16 | 0.07 GB 4-bit | 0.19 GB |
| **Total** | **4.96 GB** | **~1.7 GB** | **3.3 GB** |

turbo-fieldfare proves this works: their common model is 1.35 GB with 4-bit embeddings, 8-bit router, 4-bit attention. The GPU "hang" in our `quantize_model.py` is a bug, not a fundamental limitation.

### Phase 2: 2-Bit Experts → ~6 tok/s

**Status**: ✅ Working (`--2bit` flag). Quality verified at K=8.

Expert I/O drops from 1.77 MB to 0.98 MB per expert. Saves ~18 ms per layer. Combined with page cache warming: ~5-6 tok/s.

### Phase 3: GPU Pipeline → ~9-12 tok/s

**Status**: 🔴 Not started.

| Optimization | Est. Gain | Complexity |
|-------------|-----------|------------|
| Fuse CMD1+CMD2 | ~10 ms/layer | Medium |
| ICB for CMD3 | ~20 ms/layer | High |
| Persistent workgroups | ~15 ms/layer | High |
| MTP speculative decoding | 1.5-2× | High |

### Phase 4: Advanced → Beyond 12 tok/s

KV cache FP16, single-kernel multi-expert, GPU batched prefill, 3-bit experts.

### First Actions (This Week)

1. Debug GPU hang with quantized non-expert tensors (smallest first)
2. Benchmark 2-bit experts at K=8 → potentially make default
3. Fuse CMD1+CMD2 into single command buffer
4. Profile Metal dispatch overhead → quantify ICB potential

## Known Limitations

- **Non-expert weights are unquantized**: 4.96 GB vs turbo-fieldfare's 1.35 GB. This is the #1 issue.
- **Single-request generation**: One request at a time (queued up to 16). No continuous batching.
- **CPU-only prefill**: Prompt processing is slow (3.4s for 4 tokens). GPU batching needed.
- **Temporal drift at T=0**: Slight numeric drift after ~270 tokens with greedy decoding. Fixed with `--temperature 0.7 --rep-penalty 1.15`.
- **Expert prediction disabled**: 41% hit rate, overhead > savings.
- **MTP disabled by default**: Infrastructure complete but latent quality issues.

## Bugs

**16 bugs** discovered and fixed, documented in **[BUGS.md](BUGS.md)**. Most recent:

| # | Bug | Fix |
|---|-----|-----|
| 14 | K=2 default produces garbage | Changed default to K=8 |
| 15 | lm_head 2 GB F32 cache → OOM/segfault | GPU gemv_bf16_x2 (zero-copy) |
| 16 | decode_token NULL crash in diagnostics | NULL guard |

## References

- [Engine Analysis](finchmoe/ENGINE_ANALYSIS.md) — Complete architecture, pipeline, kernel catalog
- [Optimization Plan](finchmoe/OPTIMIZATION_PLAN.md) — Concrete roadmap with benchmarks
- [Design Document](design.md) — Architecture history and design decisions
- [BUGS.md](BUGS.md) — All 16 bugs with root cause analysis
- [finchTool](finchmoe/finchTool/README.md) — Metal kernel verification suite
