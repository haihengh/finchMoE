# FinchMoE

A C/Metal inference engine for **Qwen 3.6 35B A3B** on Apple Silicon.

## Goal

Run a capable model at reasonable speed on low-spec machines (Mac mini,
MacBook Neo) for agentic work and general chat, fully local and offline. The
binding constraint on such machines is RAM/VRAM, so model weights are stored
on SSD and streamed through the OS page cache instead of being held in
memory — letting low-spec machines run MoE models far larger than their RAM.
The trade-off is that the SSD becomes the bottleneck, so the model must stay
small enough to stream; Qwen 3.6 35B A3B is the sweet spot — a 35B-class
model with proven capability, at a size the SSD can feed.

**Phase 1 targets MET (2026-08-13)**: ~9-10 tok/s decode on M4 (3-bit experts, page-cache dependent), 1.95 GB weights — see [finchmoe/OPTIMIZATION_PLAN.md](finchmoe/OPTIMIZATION_PLAN.md) for the full progress log.

## Current Status (2026-08-14)

| Metric | Value |
|--------|-------|
| Decode speed (M4, K=8) | **16-22 tok/s warm cache** (n=6, 2026-08-14 reruns; 10.3 cold-restart, 8.8 first-run) — 3-bit experts, page-cache dependent |
| Decode speed (M1 mini, 8 GB) | **~4.1 tok/s**; chunk-8 prefill ~25 s is *slower* than per-token ~22.6 s on 8 GB (IO-bound) — full table in [M1 mini benchmark](#m1-mini-benchmark-2026-08-14) |
| Prefill speed | **Chunked batched GPU prefill** (default `--prefill-chunk 8`): 90-token prompt 6.2-7.0s, 883-token 51s (**2.1×** vs per-token, 3-bit experts); logits bitwise-identical to the per-token path. Hot-set expert prefetch (build_hot_sets.py) is memory-adaptive — measured 2026-08-14, **does not pay** (26% unique-expert coverage, pread_wait 6.2→6.3 ms), auto-gate stays |
| Weight file | **1.95 GB** (4-bit GDN tier + 3-bit experts, both default) |
| RAM (8k context) | ~3.3 GB total (2.9 GB GPU peak + 0.34 GB CPU KV; +0.27 GB when the hot-set prefetch is active) |
| Expert disk (3-bit) | 14 GB (40 layers × 256 × 1.31 MiB) |
| Correctness | Bit-exact vs numpy GDN reference (CosSim 1.000000 per stage) |
| Bugs fixed | 20 + Phase 1 root causes (see [BUGS.md](BUGS.md) and [finchmoe/PHASE1_NAN_ANALYSIS.md](finchmoe/PHASE1_NAN_ANALYSIS.md)) |
| Agent harnesses | Targets: **Kon** (coding) + **Hermes** (general agentic); ~1-2 min/turn overhead at ~58 ms/token prefill; native `tools` param pending — see [Agent Harness Targets](#agent-harness-targets) |

Known issues (ranked): (1) intermittent ~1e-4…1e-2 run-to-run logit wobble
under page-cache starvation — predates the perf refactor, needs a
healthy-machine session to isolate (post-restart parity battery still
pending as of 2026-08-14). (2) prefill is GPU+IO co-bound
(post-restart retest 2026-08-14: cmdA_wait 7.4 ms/layer + pread_wait
6.2 ms/layer against an ~14 GB expert working set; the static hot-set
prefetch was measured and does not pay — 26% unique-expert coverage —
the 5-10× tier needs a learned router predictor, layer→layer expert
carry-over is only 3.3%). (3) Server
multi-turn session corruption after turn 1 (stateless fallback active).
(4) MTP speculative decoding: forward math verified correct against a
pristine-BF16 numpy reference, but the model's MTP head is inherently weak
(cos 0.3-0.8, ~0% acceptance) — not shippable (see finchmoe/mtp_reference.py).

### M1 mini benchmark (2026-08-14)

Measured on an M1 mini (8 GB unified memory, 256 GB internal Apple SSD), running
the `finchmoe-m1/` deploy copy (prebuilt binary + quant_clean weights +
`bench.sh`). Same suite as the M4 reference:

| Metric | M1 mini (8 GB) | M4 reference | Ratio |
|--------|----------------|--------------|-------|
| Decode (K=8, 50 tokens) | **~4.1 tok/s** (3.79-4.29, n=3) | **16-22 tok/s** (n=6, warm cache; 10.3 cold, 2026-08-14 reruns) | 0.19-0.26× |
| Prefill 90 tok, per-token (chunk 0) | **~22.6 s** (22.4/22.7, n=2) | 4.9-5.1 s (n=2, warm; 10.6 cold) | 4.4-4.6× slower |
| Prefill 90 tok, chunked (chunk 8) | **~25 s** (24.9/25.2, n=2) | **1.9 s** (n=2, warm; 6.9 cold) | 12.9× slower |
| Expert `pread_wait` (chunked) | 33.0 ms/layer | 0.019 ms/layer (warm; 6.2 cold) | ~1700× |

Findings:

- **Chunked batched prefill does not pay on the M1** — it is ~11% *slower*
  than the per-token path (25.1 vs 22.6 s), whereas on the M4 the same flag is the
  2.1× win. Both paths are expert-IO-bound here: the 14 GB expert working set
  never fits the 8 GB machine's page cache (`pread_wait` 33 ms/layer vs 6.2 on
  the M4), so batching commit+waits buys nothing while the pf-pool adds overhead.
- **Hot-set prefetch does not pay on the M1 either**: forced run 24.9 s vs 25.2 s
  with the auto-gate on (unchanged `pread_wait`). The prefetched 32 experts/layer
  cover only part of the reads, and SSD-bound preads dominate regardless. The
  auto-gate enables prefetch only above 1 GB strictly-free RAM (it was active in
  the 3.6+ GB runs).
- **The bootstrap memory gate is live on 8 GB machines**: it refuses below
  3.0 GB available (free+inactive+purgeable+speculative) and warns below 7 GB.
  During the benchmark session available memory fluctuated 2.5-3.8 GB and 2 of 7
  suite invocations were refused and had to be rerun. The check runs before
  argument parsing, so `--low-memory` does not lower the hard floor — close
  memory-heavy apps first (the engine itself wants ~3.3 GB total, see the RAM
  row above, and Claude Code alone is ~2 GB).
- Quality on the copied deploy is intact: coherent output on the typo prompt
  (default sampling, temp 0.30/top-k 40); no obvious quality issues.

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

| Configuration | Weights | Expert disk | Speed (K=8, M4) | Quality loss vs BF16 | Notes |
|---------------|---------|-------------|-------------|----------------------|-------|
| **Default (quant_clean)** | **1.95 GB** (4-bit GDN, 8-bit embed/lm_head) | 14 GB 3-bit | **~10.3 tok/s** (post-restart retest 2026-08-14) | **Small**: 3-bit experts weight CosSim 0.966-0.979, 4-bit non-experts ≥0.995 (requant 2026-08-13), typo-prompt PASS | current production config; 11.4+ was the quant_self-era peak |
| Protected tier | 2.45 GB (8-bit GDN, `FINCHMOE_GDN8=1`) | 13 GB 3-bit | ~9.1 tok/s | **Minimal**: 8-bit non-experts near-lossless; experts as default tier | quality-safe fallback |
| BF16 (source) | 67 GB | — | — | None (reference) | reference; the intended requant base |
| 2bit-dense-v2 (legacy) | 4.96 GB BF16 | 9.4 GB 2-bit | ~5 tok/s | **Large**: edge prompts degrade into repetition loops (llama.cpp Q4_K_M of the clean base handles the same prompts) | marginal quality — being replaced |

Quality figures are weight-level CosSim from the requant validation plus spot
checks. End-to-end loss (e.g. HumanEval pass@1 per tier) is not yet published —
the `humaneval_m1/` harness exists to measure it.

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
| **FinchMoE** | Qwen 3.6 35B A3B | **~3.3 GB** | **~9-10.3 tok/s** (post-restart retest 2026-08-14) | custom C/Metal, bit-exact |
| turbo-fieldfare | Gemma 4 26B A4B | ~2 GB | 10.7 tok/s (est.) | Swift/Metal reference |
| llama.cpp Q4_K_M | Qwen 3.6 35B A3B | ~20 GB | — | reference quality; handles edge prompts well |

## Agent Harness Targets

The engine is a stateless OpenAI-compatible server with **no prefix cache**,
so a harness's fixed per-turn prompt is re-prefilled every turn at ~58 ms/token
(883-token prompt = 51 s). Harness selection is decided by absolute token
overhead, not feature lists:

| Harness | Fixed overhead/turn | Cost on FinchMoE | Fit |
|---------|--------------------|------------------|-----|
| **Kon** ([0xku/kon](https://github.com/0xku/kon)) | ~1,000 tokens (system prompt <270) | **~1 min** | ✅ Primary — coding: read/edit/write/bash/grep/find, exact-text edits, headless `-p` mode |
| **Hermes** (Nous `hermes-agent`) | ~1,300-2,000 tokens (memory ~1.3k + system ~800) | ~1.3-2 min | ✅ General agentic — XML-tag tool steering (matches the engine's `<tool_call>` text protocol), dynamic tool loading |
| OpenClaw | ~18,400 tokens (measured bare turn) | ~18 min; exceeds 8k context | ❌ Rejected — does not fit |

Both targets speak plain `/v1/chat/completions` and resend full history each
turn, matching the server's stateless default. Neither needs a bigger context
window: the 8k practical limit fits the one-file-at-a-time edit loop both
encourage. Known fit notes: keep the engine at temp 0.3 + `-B` think budget
(the model loops in the thinking phase otherwise), and cap harness-side
`AGENTS.md`/memory injections at ~500 tokens (~30 s of prefill each turn).

### Implementation Plan

**Phase A — API contract (server)** — verify and close the gaps a real
OpenAI-compatible client trips over:

- [ ] `/v1/models` advertises a stable model id + context window (harnesses
      probe this to size their token budgets)
- [ ] `usage` (prompt/completion tokens) present in streamed and non-streamed
      responses (Kon/Hermes count tokens for compaction decisions)
- [ ] Honor `stop` sequences from the request
- [ ] System/assistant/tool role messages routed through the Qwen chat
      template; over-context requests rejected with HTTP 400 instead of
      silent truncation
- [ ] Fix `-N` default (262144 → 16384): 256k KV is 10.7 GB fp32 and does not
      fit 16 GB — validate against free RAM at startup

**Phase B — tool calling** — the decisive feature:

- [ ] Native `tools` parameter on `/v1/chat/completions`: inject Qwen tool-call
      format, parse `<tool_call>` from the stream (parser already exists in
      `chat.m`), execute via a registered tool table, append `<tool_result>`,
      loop until the model answers
- [ ] Text tool protocol as the zero-engine-change fallback: a small sidecar
      running the `chat.m`-style loop over the plain endpoint (the path
      Hermes' XML steering works through today)

**Phase C — harness validation** — acceptance suite: config edit, SQL text
replacement, CREATE TABLE → IF-TABLE migration, 20-file TF repo:

- [ ] Kon: point at `-R 9000` (auth `none`, TLS-verify skip), set `context`
      8000 / `buffer_tokens` 1500; measure per-turn overhead (target
      ~1-1.5 min) and edit reliability (exact-text `edit` vs `write` fallback
      at 3-bit quality)
- [ ] Hermes: run `hermes prompt-size` to lock the budget, verify XML tool
      parsing against the engine's stream, run a multi-turn task
- [ ] Record turns/task and wall-clock per task; add a retry policy for the
      known logit wobble under page-cache starvation

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
├── quant_*/                      # quant experiments (self, 4gdn, 8gdn, visual, …)
└── finchmoe-m1/                  # M1 mini (8GB) deploy copy: binary + weight symlinks + bench.sh (untracked)
```
