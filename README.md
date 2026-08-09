# FinchMoE

A C/Metal inference engine for **Qwen 3.6 35B A3B** on Apple Silicon, targeting 12 tok/s on M4 and 3-5 tok/s on iPhone.

## Current Status

| Metric | Value |
|---|---|
| Output quality | ✅ Coherent — "Hello! How can I help you today?" |
| Speed (M4, 2-bit, K=2) | **8.3 tok/s** |
| Speed (M4, 2-bit, K=4) | **7.5 tok/s** |
| Speed (M4, 1-bit, K=2) | **8.1 tok/s** (degraded quality) |
| RAM (runtime, 1K ctx) | ~400 MB |
| RAM (runtime, 60K ctx) | ~3.3 GB |
| Disk (2-bit-dense, active) | **21 GB** |
| Disk (1-bit-dense) | **13.2 GB** |

## Model Size & Quality Tradeoff

| Model | Safetensors | Experts | Total | Speed (K=2) | Quality |
|-------|-------------|---------|-------|-------------|---------|
| BF16 (source) | 67 GB | — | 67 GB | — | 0% (reference) |
| 2-bit-dense ✅ | 11 GB | 9.4 GB | **21 GB** | 8.3 tok/s | ~5% |
| 4-bit-dense | 19 GB | 17 GB | **36 GB** | 7.2 tok/s | ~1-2% |
| 1-bit-dense | 7.6 GB | 5.6 GB | **13.2 GB** | 8.1 tok/s | ~10-15% (degraded) |

### Quantization Strategy

| Component | Format | Size | Why |
|-----------|--------|------|-----|
| Embeddings + lm_head | 8-bit | 1.1 GB | Near-lossless, vocabulary-critical |
| Attention/GDN projections | 4-bit | 0.7 GB | Large, 4-bit sufficient |
| Shared expert | 4-bit | 0.07 GB | Always active, small |
| Routed experts | 2-bit (→1-bit) | 9.4 GB | MoE-tolerant: 8/256 fire, errors cancel |
| Norms + routing gate | BF16 | 0.0005 GB | Tiny, not worth quantizing |

## Comparison Models

| Model | Arch | Active | Size | TG (M4) | Notes |
|-------|------|--------|------|----------|-------|
| **FinchMoE 2-bit** | MoE 35B A3B | 3B | 21 GB | **8.3 tok/s** | Custom C/Metal, 256K context |
| [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) | MoE 26B A4B | 4B | 13 GB | **10.7 tok/s** | Swift/Metal, better poetry, 8K max context |
| [Ternary-Bonsai-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) | Dense 27B | 27B | 7.1 GB | 0.008 tok/s ❌ | 27B dense = 9× more compute |
| [Bonsai-27B-1bit](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit) | Dense 27B | 27B | 4.8 GB | ~1-3 tok/s (est.) | Runs on iPhone via LM Studio ✅ |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | MoE ~200B A13B | 13B | ~160 GB BF16 | TBD | FP4 experts, hash routing, MTP, 1M ctx |

**Key insight**: Active parameter count dominates speed, not model size. DeepSeek-V4 has 13B active (4× Qwen's 3B) but uses FP4 native quantization and hash-based routing for efficiency.

### Benchmarks (M4 Mac mini 16GB, Samsung 990 Plus NVMe TB4)

| Model | Size | RAM | TG (1K) | TG (256K) | KV (256K) | Quality |
|-------|------|-----|---------|-----------|-----------|---------|
| **FinchMoE 2-bit** | 21 GB | ~400 MB | **8.3 tok/s** | ~5-8 tok/s (est.) | 5.2 GB | "You are a helpful assistant." |
| **FinchMoE 1-bit** | 13.2 GB | ~350 MB | **8.1 tok/s** | ~5-8 tok/s (est.) | 5.2 GB | "You are a helpful assistant." (degraded) |
| **turbo-fieldfare** | 13 GB | ~3 GB | **10.7 tok/s** | N/A (63 GB KV) | 63 GB | "The salt-crust clings..." |

**PP (prompt processing)**: 3.6 tok/s at 1K. Estimated ~0.5-1 tok/s at 60K (full attention O(n²) on CPU).
**TG at 256K**: Estimated from attention scaling. 30 GDN layers unaffected by context length.
**KV at 256K (FP16)**: FinchMoE 5.2 GB (10 attn layers × 2 KV heads). turbo-fieldfare 63 GB (30 layers × 8 heads).

### KV Cache Memory at Context Lengths

| Model | Attn Layers | KV Heads | 1K | 50K | 256K |
|-------|------------|----------|-----|------|-------|
| **FinchMoE** | 10/40 | 2 | 0.02 GB | 1.0 GB | 5.2 GB |
| **turbo-fieldfare** | 30/30 | 8 | 0.3 GB | 12.3 GB | 62.9 GB |

FinchMoE's GatedDeltaNet layers (30/40) use fixed 2.1MB recurrent state — no KV growth. Only the 10 full-attention layers need caching.

## Optimization History

### Speed Progression

| Step | expert_io | total_layer | tok/s | What |
|------|-----------|-------------|-------|------|
| Baseline (8-bit, K=4) | 2.04ms | 4.93ms | 5.1 | Starting point |
| Fused expert kernel | 1.92ms | 4.37ms | 5.7 | gate+up+swiglu → 1 dispatch |
| 4-bit experts | 1.33ms | 3.81ms | 6.6 | Half I/O volume |
| Dense quantization | 0.59ms | 2.83ms | 7.2 | Embeds 8-bit, attn 4-bit |
| 2-bit experts (K=4) | 0.38ms | 2.74ms | 7.5 | Half I/O again |
| **K=2 default** | **0.21ms** | **2.30ms** | **8.3** | Half experts/layer |
| 1-bit experts | 0.13ms | 2.33ms | 8.1 | Diminishing returns |

### What Didn't Work

| Attempt | Result | Why |
|---------|--------|-----|
| Batched Metal encoders | Slower | Metal cost is per-dispatch, not per-encoder |
| Multi-expert 2x kernel | Slower | Buffer binding overhead > dispatch savings |
| Spatial expert prediction | 1.3% hit rate | Adjacent layers pick different experts |
| Temporal expert prediction | 41% hit rate | Validation overhead > savings |
| mmap zero-copy Metal buffers | OOM | 17GB Metal buffers exceed 16GB RAM |
| LZ4 expert compression | 6% ratio | Quantized data near-random |

### Current Per-Layer Timing (2-bit, K=2)

| Phase | ms | % | Stuck Because |
|-------|----|---|---------------|
| cmd1_wait (GDN GPU) | 0.87 | 38% | 5 Metal dispatches per GDN layer |
| cmd3_encode (expert dispatch) | 0.67 | 29% | Per-dispatch Metal driver cost |
| cmd2_wait (routing GPU) | 0.49 | 21% | 6 dispatches for o_proj+routing+shared |
| expert_io | 0.21 | 9% | ✅ Fully optimized (RAM bandwidth limited) |
| **Total** | **2.30** | | **× 40 layers = 92ms = 10.9 tok/s theoretical** |

## Future Implementation Plan

### P1: TG Speed (12+ tok/s)

| # | Feature | Gain | Effort |
|---|---------|------|--------|
| A | ICBs (Indirect Command Buffers) | cmd3: 0.67→0.05ms | High |
| B | Double-buffered async encoding | Hide CPU wait | High |
| C | Single-kernel multi-expert MoE | K dispatches → 1 | High |
| D | **MTP speculative decoding** | 1.5-2× TG | High | 
|   | → Target model: Qwen 3.5 397B-A17B (has `mtp_num_hidden_layers: 1`) | | |
|   | → Draft head predicts 2 tokens/forward pass, main model verifies | | |
|   | → At 70% acceptance: effective 1.7× speedup | | |

### P2: Long Context

| # | Feature | Gain | Effort |
|---|---------|------|--------|
| D | KV cache FP16 | 2× capacity | Medium |
| E | Batched GPU prefill attention | 5-10× PP speed | High |
| F | GDN chunked prefill | 2-3× PP speed | Medium |
| G | `--gpu-kv-seq` bump (default 8K→match ctx) | TG at long ctx | Free |

### P3: Larger Models & Compression

| # | Feature | Gain | Effort |
|---|---------|------|--------|
| H | 3-bit experts | 21→~16 GB | Low |
| I | Mixed-precision experts | ~1-2% quality | Low |
| J | Completions API (`/v1/completions`) | Standard benchmarks | Low |
| K | **Qwen 3.5 397B-A17B** ✅ | Working at 3.2 tok/s | Done |
|   | → Separate binary `infer_397b.m`, 397b/ weights dir | | |
|   | → 68 GB packed experts, 5.1 GB model_weights, <1 GB RAM | | |
|   | → Fix: export_tokenizer fails for 397B tokenizer (use 35B vocab) | | |
| L | **DeepSeek-V4-Flash** (13B active MoE) | New model target | High |
|   | → 43 layers, 256 experts (6 active), sliding window attn | | |
|   | → FP4 native experts, hash routing, MTP, 1M context | | |
|   | → BF16 download: ~160 GB. More advanced but needs engine adaptation | | |

## Project Structure

```
finchMoE/
├── README.md
├── BUGS.md                    # 8 bugs documented
├── design.md                  # Architecture & design decisions
├── finchmoe/
│   ├── infer.m                # Main engine (C/Metal, ~8000 lines)
│   ├── shaders.metal          # Metal compute kernels (~1500 lines)
│   ├── tokenizer.h            # C BPE tokenizer (248K vocab)
│   ├── quantize_model.py      # BF16 → 1/2/4/8-bit quantization
│   ├── extract_weights.py     # Non-expert weight extraction
│   ├── repack_experts.py      # Expert weight repacking (1/2/4/8-bit)
│   ├── generate_expert_index.py
│   ├── compress_experts.py    # LZ4 compression tool
│   ├── debug_gdn_compare.py   # GDN vs HF reference
│   ├── debug_full_forward.py  # Full 40-layer forward test
│   ├── debug_layer_diff.py    # 4-probe per-layer differential
│   └── debug_e2e_logits.py    # End-to-end logit comparison
├── models/
│   ├── Qwen3.6-35B-A3B-bf16/          # Source (67 GB)
│   ├── Qwen3.6-35B-A3B-2bit-dense/    # Active (21 GB) ✅
│   ├── Qwen3.6-35B-A3B-4bit-dense/    # Higher quality (36 GB)
│   ├── Qwen3.6-35B-A3B-1bit-dense/    # Ultra-compact (13.2 GB)
│   ├── Ternary-Bonsai-27B-Q2_g64.gguf # Reference (7.1 GB)
│   └── Bonsai-27B-mlx-1bit/           # Reference (4.8 GB)
├── flash-moe/                 # Starting codebase (unmodified)
├── turbo-fieldfare/           # Benchmark reference
└── llama.cpp/                 # Ground truth reference
```

## Running the Engine

### Quick Start

```bash
cd finchmoe

# Build
clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation \
      -framework Accelerate -lcompression -lpthread infer.m -o infer

# Prepare model (one-time)
python3 quantize_model.py --model ../models/Qwen3.6-35B-A3B-bf16 \
    --output ../models/Qwen3.6-35B-A3B-2bit-dense
python3 generate_expert_index.py --model ../models/Qwen3.6-35B-A3B-2bit-dense
python3 repack_experts.py --index ../models/Qwen3.6-35B-A3B-2bit-dense/expert_index.json --bits 2
python3 extract_weights.py --model ../models/Qwen3.6-35B-A3B-2bit-dense --output .
python3 export_tokenizer.py ../models/Qwen3.6-35B-A3B-2bit-dense/tokenizer.json vocab.bin

# Run
./infer --model ../models/Qwen3.6-35B-A3B-2bit-dense --prompt "Hello" --tokens 50 --no-think --temp 0

# Server (OpenAI-compatible API)
./infer --model ../models/Qwen3.6-35B-A3B-2bit-dense --serve 9000
```

### Key Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--prompt TEXT` | — | Input prompt |
| `--tokens N` | 20 | Max tokens to generate |
| `--temp F` | 0.80 | Temperature (0 = greedy) |
| `--top-k N` | 40 | Top-k sampling |
| `-k N` | 2 | Active experts per layer (2=speed, 4=quality, 8=best) |
| `--no-think` | off | Disable thinking mode |
| `--timing` | off | Per-layer timing breakdown |
| `--predict` | off | Temporal expert prefetch |
| `--cpu-linear` | off | CPU GDN path (debugging) |
| `--cpu-experts` | off | CPU expert path (debugging) |
| `--serve PORT` | — | HTTP server (OpenAI-compatible API) |
| `--max-seq-len N` | 262144 | Max context length |
| `--gpu-kv-seq N` | 8192 | GPU KV buffer size in tokens |
| `--model PATH` | auto | Model directory |

### Server Mode

```bash
./infer --serve 9000

# Test
curl http://localhost:9000/health
curl -N -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":100,"stream":true}'
```

Endpoints: `POST /v1/chat/completions` (SSE streaming), `GET /v1/models`, `GET /health`.

### Context Window

| Flag | Default | Purpose |
|------|---------|---------|
| `--max-seq-len` | 262,144 (256K) | KV cache allocation (model's RoPE limit) |
| `--gpu-kv-seq` | 8,192 | GPU-accelerated attention cap. Past this → CPU fallback |

```bash
# 60K agentic context: GPU KV = 2.5 GB (fits in 16GB)
./infer --serve 9000 --gpu-kv-seq 60000
# 256K max context: KV = 10.5 GB FP32, 5.2 GB FP16
./infer --serve 9000 --gpu-kv-seq 256000 --max-seq-len 262144
```

30/40 layers use GatedDeltaNet with fixed 2.1MB state — no KV growth. Only 10 full-attention layers need caching.

## Known Limitations

- **Single-worker generation**: One request at a time (queued up to 16). Fine for personal/agentic use.
- **GPU context cap**: Attention falls to CPU above `--gpu-kv-seq` (default 8K). Bump for long contexts.
- **`/v1/completions` not implemented**: Only chat completions API. lm-eval needs completions for loglikelihood benchmarks.
- **Chat template applied in CLI**: As of Bug 8 fix, CLI now uses ChatML template for instruct behavior.

## Bugs & Debugging

8 bugs discovered and fixed during development, documented in **[BUGS.md](BUGS.md)**:
1. INT4 attention weights → catastrophic incoherence
2. FP16/BF16 format mismatch in MLX community models
3. Norm weight +1.0 adjustment missing
4. Compiler warnings hiding unused variable errors
5. Expert weight extraction namespace collision
6. Safetensors tensor naming inconsistency
7. switch_mlp weights excluded from extraction
8. **Shared expert Metal command buffer sync** (root cause of "Con Con Con" loop)
