# FinchMoE

A C/Metal inference engine for **Qwen 3.6 35B A3B** on Apple Silicon.

**Phase 1 targets MET (2026-08-13)**: ~9-10 tok/s decode on M4 (3-bit experts, page-cache dependent), 1.95 GB weights — see [finchmoe/OPTIMIZATION_PLAN.md](finchmoe/OPTIMIZATION_PLAN.md) for the full progress log.

## Current Status (2026-08-14)

| Metric | Value |
|--------|-------|
| Decode speed (M4, K=8) | **~9-10 tok/s** (3-bit experts, page-cache dependent) |
| Prefill speed | **Chunked batched GPU prefill** (default `--prefill-chunk 8`): 90-token prompt 6.2s, 883-token 51s (**2.1×** vs per-token, 3-bit experts); logits bitwise-identical to the per-token path. Hot-set expert prefetch (build_hot_sets.py) is memory-adaptive — auto-disabled when the OS lacks page-cache headroom |
| Weight file | **1.95 GB** (4-bit GDN tier + 3-bit experts, both default) |
| RAM (8k context) | ~3.3 GB total (2.9 GB GPU peak + 0.34 GB CPU KV; +0.27 GB when the hot-set prefetch is active) |
| Expert disk (3-bit) | 14 GB (40 layers × 256 × 1.31 MiB) |
| Correctness | Bit-exact vs numpy GDN reference (CosSim 1.000000 per stage) |
| Bugs fixed | 20 + Phase 1 root causes (see [BUGS.md](BUGS.md) and [finchmoe/PHASE1_NAN_ANALYSIS.md](finchmoe/PHASE1_NAN_ANALYSIS.md)) |

Known issues (ranked): (1) intermittent ~1e-4…1e-2 run-to-run logit wobble
under page-cache starvation — predates the perf refactor, needs a
healthy-machine session to isolate. (2) prefill is expert-IO bound
(~5.4 ms/layer pread against an ~14 GB expert working set; the hot-set
prefetch needs RAM headroom to pay, the 5-10× tier needs a learned router
predictor — layer→layer expert carry-over is only 3.3%). (3) Server
multi-turn session corruption after turn 1 (stateless fallback active).
(4) MTP speculative decoding: harness runs, draft math wrong (α=0%).

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
FINCHMOE_REF_MANIFEST=quant_clean/model_weights_quant.json FINCHMOE_REF_WEIGHTS=quant_clean/model_weights_quant.bin python3 debug_gdn_reference.py /tmp/stage_dump.bin
```

### Key Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `-P TEXT` | — | Input prompt (chat template applied) |
| `-p FILE` | — | Prompt from token-id file |
| `-t N` | 20 | Max tokens to generate |
| `-e F` | 0.3 | Temperature (0 = greedy) |
| `--rep-penalty F` | 1.15 | Repetition penalty |
| `-k N` | 8 | Active experts per layer (model trained with 8) |
| `-3 / -4 / -2 / -8` | **-3** | Expert bit-width (3-bit is the default) |
| `-B N` | 2048 | Think budget: force `</think>` after N reasoning tokens |
| `-R PORT` | off | HTTP server (OpenAI-compatible: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`) |
| `-J` | off | MTP speculative decoding (experimental — currently slower, see issues) |
| `-N N` | 262144 | CPU KV context (256k does NOT fit 16 GB — use 16384-32768) |
| `-Q N` | 8192 | GPU KV pre-allocation |
| `--prefill-chunk N` | **8** | Chunked batched prefill (0 = per-token path; 8 = pooled+batched-attn sweet spot) |
| `--low-memory` | off | Skip Metal weight wrap |

## Model Sizes

| Configuration | Weights | Expert disk | Speed (K=8) | Notes |
|---------------|---------|-------------|-------------|-------|
| **Default (quant_clean)** | **1.95 GB** (4-bit GDN, 8-bit embed/lm_head) | 14 GB 3-bit | **~9-10 tok/s** | current production config; 11.4+ on a healthy page cache |
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

### Chunked Batched Prefill (default `--prefill-chunk 8`)

The prefill of a prompt runs as chunks of up to 8 positions through a
dedicated pipeline (bitwise-identical logits vs the per-token path):

```
Per layer, per chunk:
  cmdA (ONE CB, linear layers):  batched input-norm → batched qkv/z/a/b
    matvecs → M sequential fused GDN chains → batched out_proj →
    batched residual+norm → batched routing matmuls → commit+wait
  cmdA (full-attn layers):       batched q/k/v matvecs → wait → per-position
    CPU Q/K-norm+RoPE+KV-append (sl<32 or sl≥8192 CPU attention; the rest
    staged) → cmdB: 4 batched M-position GPU attention dispatches
    (scores/softmax/values/sigmoid) → batched o_proj → residual+norm →
    routing → commit+wait
  Phase B:                       CPU softmax+topK per position → hot-set
    prefetch hit/miss split (see below) → ONE batched pread of the cold
    misses into a 64-slot pool → M CMD3s committed back-to-back with ZERO
    backpressure (every buffer is per-position-disjoint)
```

Key properties: one commit+wait per linear layer (two for full-attn);
expert preads for the whole chunk issue as a single GCD batch; deferred
CMD3s pipeline across layers via queue order alone (no fences/events).
Hot-set expert prefetch: layer L+1's top-32 most-frequent experts
(`hot_sets.bin`, built by `build_hot_sets.py`) are prefetched into an
alternating 2×32-slot pool during layer L's compute — enabled only when
the OS has ≥1 GB strictly-free RAM (`FINCHMOE_PF_PREFETCH=1` forces it).

## Reference Comparison

| Engine | Model | RAM | M4 Speed | Notes |
|--------|-------|-----|----------|-------|
| **FinchMoE** | Qwen 3.6 35B A3B | **~3.3 GB** | **~9-10 tok/s** (11.4+ warm cache) | custom C/Metal, bit-exact |
| turbo-fieldfare | Gemma 4 26B A4B | ~2 GB | 10.7 tok/s (est.) | Swift/Metal reference |
| llama.cpp Q4_K_M | Qwen 3.6 35B A3B | ~20 GB | — | reference quality; handles edge prompts well |

## Documentation

| Document | What it covers |
|----------|----------------|
| [design.md](design.md) | Engine design: pipeline, key decisions, timing budget, future work |
| [BUGS.md](BUGS.md) | Chronological bug log (17+ fixed; 2026-08-07 → chunked-prefill era) |
| [finchmoe/OPTIMIZATION_PLAN.md](finchmoe/OPTIMIZATION_PLAN.md) | Roadmap + progress log (Phase 1 targets, prefill levers) |
| [finchmoe/ENGINE_ANALYSIS.md](finchmoe/ENGINE_ANALYSIS.md) | Comprehensive engine analysis (current-status header) |
| [finchmoe/PHASE1_NAN_ANALYSIS.md](finchmoe/PHASE1_NAN_ANALYSIS.md) | Phase 1 NaN root causes + resolution |
| [finchmoe/BUG_REPORT.md](finchmoe/BUG_REPORT.md) | Original degenerate-output bug report (historical) |
| [finchmoe/BUGS_DEEPSEEK.md](finchmoe/BUGS_DEEPSEEK.md) | DeepSeek-V4-Flash engine bug log (sibling project) |
| [finchmoe/design_deepseek.md](finchmoe/design_deepseek.md) | DeepSeek-V4-Flash engine design (sibling project) |

## Tools

### Build & quantization (finchmoe/)

| Tool | Purpose |
|------|---------|
| `Makefile` | Builds `finchmoe-infer` (+ `chat`, `extract`, `index`, `repack` targets) |
| `extract_weights.py` | Non-expert weights → model_weights.bin (safetensors → flat binary) |
| `quantize_non_experts.py` | BF16 → 4/8-bit non-experts (4-bit GDN default; `FINCHMOE_GDN8=1`) |
| `quantize_model.py` | Full-model MLX quantizer (incl. Qwen3_5RMSNorm +1.0 fix) |
| `repack_experts.py` | Expert repack `--bits 1/2/3/4/8` (3-bit requant path from BF16 source) |
| `generate_expert_index.py` | Expert tensor index (offset math, MTP-aware) |
| `compress_experts.py` | Expert compression variants |
| `export_tokenizer.py` | vocab.bin from the HF tokenizer (BPE merges) |
| `extract_mtp_experts.py` | MTP auxiliary-head expert extraction |
| `build_hot_sets.py` | Per-layer hot expert sets from `--collect-routing` logs → hot_sets.bin (prefill prefetch) |

### Debugging & verification (finchmoe/)

| Tool | Purpose |
|------|---------|
| `debug_compare.py` | Weight/manifest consistency checks + logits comparison helper |
| `debug_gdn_reference.py` | Numpy GDN reference (stage cross-validation, CosSim 1.000000) |
| `debug_e2e_logits.py` | End-to-end logits cross-check |
| `debug_full_forward.py` / `debug_layer_compare.py` / `debug_layer_diff.py` | Full-forward and per-layer reference diffs |
| `debug_full_attn_moe.py` / `debug_2token_gdn.py` | Attention/MoE and GDN targeted references |
| `debug_bf16_vs_4bit.py` / `debug_mlx_inference.py` | Format and MLX-loader verification |
| `verify_clean_rebuild.py` | CosSim verification of the clean-rebuild weights (3-bit experts, non-experts) |
| `bench_prefill.sh` | Prefill benchmark + bitwise parity matrix (chunk sizes × timings) |
| `finchTool/` | Standalone Metal kernel verification suite (matvec/attention kernels) |
| `test_engine_path.m` | Engine-path standalone test |

### Runtime diagnostics (engine flags & env vars)

| Flag / Env | Purpose |
|------------|---------|
| `--timing` / `FINCHMOE_PF_TIMING=1` | Per-phase timing (per-token path / chunked prefill path) |
| `--debug-layers` | Per-layer hidden-state statistics |
| `--compare-experts N` | GPU vs CPU expert outputs for layer N |
| `--dump-logits FILE` | First-token logits for cross-validation |
| `--collect-routing FILE` | Routing logs (layer, hidden, top-K, top-24) for predictor training |
| `FINCHMOE_DUMP_HIDDEN` / `FINCHMOE_PF_DUMP` / `FINCHMOE_DUMP_STAGES` | Per-layer hidden / chunked-stage dumps for parity forensics |
| `FINCHMOE_PF_PREFETCH=1` | Force-enable hot-set prefetch (bypasses the memory gate) |

## Project Layout

```
finchMoE/
├── README.md, design.md, BUGS.md
├── finchmoe/                     # engine + tools (see tables above)
│   ├── infer.m                   # main engine (C/Metal, ~15k lines)
│   ├── shaders.metal             # Metal kernels (4/3/2/1/8-bit dequant, fused GDN, routing, attention)
│   └── chat.m                    # chat TUI client
├── models/                       # Qwen3.6-35B-A3B variants + GGUF references
│   └── Qwen3.6-35B-A3B-bf16/     # pristine BF16 base (requant source, expert packs)
├── quant_clean/                  # current production weights (model_weights_quant.bin/.json)
└── quant_*/                      # quant experiments (self, 4gdn, 8gdn, visual, …)
```
