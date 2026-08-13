# FinchMoE

A C/Metal inference engine for **Qwen 3.6 35B A3B** on Apple Silicon.

**Phase 1 targets MET (2026-08-13)**: 11.4-12.4 tok/s decode on M4, 1.95 GB weights, 2.25 GB peak GPU — see [finchmoe/OPTIMIZATION_PLAN.md](finchmoe/OPTIMIZATION_PLAN.md) for the full progress log.

## Current Status (2026-08-13)

| Metric | Value |
|--------|-------|
| Decode speed (M4, K=8, greedy) | **11.4-12.4 tok/s** |
| Prefill speed | ~12 tok/s (789-token prompt = 66 s — batched GPU prefill pending) |
| Weight file | **1.95 GB** (4-bit GDN tier + 3-bit experts, both default) |
| RAM (8k context) | ~2.8 GB total (2.25 GB peak GPU + 0.34 GB CPU KV) |
| Expert disk (3-bit) | 13 GB (40 layers × 256 × 1.31 MB) |
| Correctness | Bit-exact vs numpy GDN reference (CosSim 1.000000 per stage) |
| Bugs fixed | 16 + Phase 1 root causes (see finchmoe/PHASE1_NAN_ANALYSIS.md) |

Known issues (ranked): (1) typo'd/ambiguous prompts degrade into repetition
loops — traced to weights quantized from the marginal `2bit-dense-v2`
variant; fix = requant from pristine `Qwen3.6-35B-A3B-bf16`. (2) Prefill
not batched. (3) Server multi-turn session corruption after turn 1
(stateless fallback active). (4) MTP speculative decoding: harness runs,
draft math wrong (α=0%).

## Quick Start

```bash
cd finchmoe
make                                   # builds finchmoe-infer
make chat                              # builds the chat TUI client

# One-shot generation
./finchmoe-infer -t 300 -k 8 -e 0 -B 100 -P "Tell me a story about an LLM."

# Server (OpenAI-compatible, SSE streaming) + chat client
./finchmoe-infer -R 9000 -k 8 -e 0 -B 100     # terminal 1
./chat --show-think                            # terminal 2

# Cross-validation (expect CosSim 1.000000 for gated/o_proj/h_mid/h_post)
FINCHMOE_DUMP_STAGES=1 ./finchmoe-infer -t 1 -k 8 -e 0 -P "Explain what a MoE transformer is in one sentence."
FINCHMOE_REF_MANIFEST=quant_self/model_weights_quant.json FINCHMOE_REF_WEIGHTS=quant_self/model_weights_quant.bin python3 debug_gdn_reference.py /tmp/stage_dump.bin
```

### Key Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `-P TEXT` | — | Input prompt (chat template applied) |
| `-p FILE` | — | Prompt from token-id file |
| `-t N` | 20 | Max tokens to generate |
| `-e F` | 0.8 | Temperature (0 = greedy) |
| `--rep-penalty F` | 1.15 | Repetition penalty |
| `-k N` | 8 | Active experts per layer (model trained with 8) |
| `-3 / -4 / -2 / -8` | **-3** | Expert bit-width (3-bit is the default) |
| `-B N` | 2048 | Think budget: force `</think>` after N reasoning tokens |
| `-R PORT` | off | HTTP server (OpenAI-compatible: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`) |
| `-J` | off | MTP speculative decoding (experimental — currently slower, see issues) |
| `-N N` | 262144 | CPU KV context (256k does NOT fit 16 GB — use 16384-32768) |
| `-Q N` | 8192 | GPU KV pre-allocation |
| `--low-memory` | off | Skip Metal weight wrap |

## Model Sizes

| Configuration | Weights | Expert disk | Speed (K=8) | Notes |
|---------------|---------|-------------|-------------|-------|
| **Default (quant_self)** | **1.95 GB** (4-bit GDN, 8-bit embed/lm_head) | 13 GB 3-bit | **11.4-12.4 tok/s** | current production config |
| Protected tier | 2.45 GB (8-bit GDN, `FINCHMOE_GDN8=1`) | 13 GB 3-bit | ~9.1 tok/s | quality-safe fallback |
| BF16 (source) | 67 GB | — | — | reference; the intended requant base |
| 2bit-dense-v2 (legacy) | 4.96 GB BF16 | 9.4 GB 2-bit | ~5 tok/s | marginal quality — being replaced |

## Architecture

```
40 layers: 30× GatedDeltaNet (linear attention) + 10× full attention
Hidden dim: 2048, 256 experts/layer, 8 active + 1 shared expert
MoE intermediate: 512, head dim 256, 16Q/2KV GQA, vocab 248,320
```

### Per-Layer Pipeline (one command buffer for CMD1+CMD2, deferred CMD3)

```
CMD1+CMD2 fused (1 commit+wait):
  attention projections (4 matvecs) → conv1d+qk-norm+GDN+gated (ONE fused
  kernel) → o_proj → residual+rms-norm (ONE fused kernel) → routing batch
  (gate+sg+su+seg, ONE mixed-bits kernel)
CPU:  softmax + top-K + parallel pread of K experts (3-bit, 1.31 MB each)
CMD3 (ASYNC): K expert forwards + shared expert + combine → next layer
```

Key optimizations: deferred CMD3 overlaps the next layer; GPU-side combine
reads `buf_moe_hidden` directly as the residual (no CPU round trip);
fully fused GDN (no intermediate global round trips); zero-copy mmap'd
weights; parallel expert preads; 3-bit experts default (1.31 MB,
page-cache friendly).

## Reference Comparison

| Engine | Model | RAM | M4 Speed | Notes |
|--------|-------|-----|----------|-------|
| **FinchMoE** | Qwen 3.6 35B A3B | **~2.8 GB** | **11.4-12.4 tok/s** | custom C/Metal, bit-exact |
| turbo-fieldfare | Gemma 4 26B A4B | ~2 GB | 10.7 tok/s (est.) | Swift/Metal reference |
| llama.cpp Q4_K_M | Qwen 3.6 35B A3B | ~20 GB | — | reference quality; handles edge prompts well |

## Project Structure

```
finchMoE/
├── README.md
├── BUGS.md
├── design.md
├── finchmoe/
│   ├── infer.m                   # Main engine (C/Metal)
│   ├── shaders.metal             # Metal kernels (4/3/2/1/8-bit dequant, fused GDN, routing batch)
│   ├── chat.m                    # Chat TUI client
│   ├── ENGINE_ANALYSIS.md        # Comprehensive engine analysis (current-status header)
│   ├── OPTIMIZATION_PLAN.md      # Roadmap + progress log
│   ├── PHASE1_NAN_ANALYSIS.md    # Phase 1 NaN root causes + resolution
│   ├── extract_weights.py        # Non-expert weights → model_weights.bin (+ --fp16-scales)
│   ├── quantize_non_experts.py   # BF16 → 4/8-bit (4-bit GDN default; FINCHMOE_GDN8=1)
│   ├── repack_experts.py         # Expert repack (--bits 1/2/3/4/8; 3-bit requant path)
│   ├── generate_expert_index.py  # Expert index generator
│   ├── debug_gdn_reference.py    # Numpy GDN reference (stage cross-validation)
│   └── test_engine_path.m        # Standalone kernel verification suite
├── models/                       # Qwen3.6-35B-A3B variants + GGUF references
└── quant_self/                   # Current production weights (model_weights_quant.bin/.json)
```
