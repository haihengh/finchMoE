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

**Phase 1 targets MET (2026-08-13)**: ~9-10 tok/s decode on M4 (3-bit experts, page-cache dependent), 1.95 GB weights — see [finchmoe/OPTIMIZATION_PLAN.md](finchmoe/OPTIMIZATION_PLAN.md) for the full progress log. Quantized weights: [huggingface.co/haihengh/Qwen3.6-35B-A3B-finchmoe-3bit](https://huggingface.co/haihengh/Qwen3.6-35B-A3B-finchmoe-3bit).

## Current Status (2026-08-19)

| Metric | Value |
|--------|-------|
| Decode speed (M4, K=8) | **16-22 tok/s warm cache** (n=6, 2026-08-14 reruns; 10.3 cold-restart, 8.8 first-run) — 3-bit experts, page-cache dependent |
| Decode speed (M1 mini, 8 GB) | **~4.1 tok/s**; chunk-8 prefill ~25 s is *slower* than per-token ~22.6 s on 8 GB (IO-bound) — full table in [M1 mini benchmark](#m1-mini-benchmark-2026-08-14) |
| Decode speed (M4 Pro, 24 GB) | **38.4-42.4 tok/s** (n=3, 2026-08-19; 200-tok 38.6-39.8); prefill 90-tok 1.21-1.23 s (chunk 8) / 2.2-2.3 s (per-token) — ~2× the M4 baseline, always warm — see [M4 Pro benchmark](#m4-pro-benchmark-2026-08-19) |
| Prefill speed | **Chunked batched GPU prefill** (default `--prefill-chunk 8`): 90-token prompt 6.2-7.0s, 883-token 51s (**2.1×** vs per-token, 3-bit experts); logits bitwise-identical to the per-token path. Hot-set expert prefetch (build_hot_sets.py) is memory-adaptive — measured 2026-08-14, **does not pay** (26% unique-expert coverage, pread_wait 6.2→6.3 ms), auto-gate stays |
| Weight file | **1.95 GB** (4-bit GDN tier + 3-bit experts, both default) |
| RAM (8k context) | ~3.3 GB total (2.9 GB GPU peak + 0.34 GB CPU KV; +0.27 GB when the hot-set prefetch is active) |
| Expert disk (3-bit) | 14 GB (40 layers × 256 × 1.31 MiB) |
| Correctness | Bit-exact vs numpy GDN reference (CosSim 1.000000 per stage) |
| Bugs fixed | 20 + Phase 1 root causes (see [BUGS.md](BUGS.md) and [finchmoe/PHASE1_NAN_ANALYSIS.md](finchmoe/PHASE1_NAN_ANALYSIS.md)) |
| Agent harnesses | Targets: **Kon** (coding) + **Hermes** (general agentic); ~1-2 min/turn overhead at ~58 ms/token prefill; native `tools` param pending — see [Agent Harness Targets](#agent-harness-targets) |

Known issues (ranked): (1) long-form generation drift — see the
"Quantization quality" section (the quant's ~150-250 token stability limit;
the engine state is proven clean). (2) prefill is GPU+IO co-bound
(post-restart retest 2026-08-14: cmdA_wait 7.4 ms/layer + pread_wait
6.2 ms/layer against an ~14 GB expert working set; the static hot-set
prefetch was measured and does not pay — 26% unique-expert coverage —
the 5-10× tier needs a learned router predictor, layer→layer expert
carry-over is only 3.3%). (3) ~~Server multi-turn~~ — FIXED 2026-08-14: truncated turns roll back to a
pre-turn snapshot + think re-entry ban (sessions accumulate history).
(4) MTP speculative decoding: forward math verified correct against a
pristine-BF16 numpy reference, but the model's MTP head is inherently weak
(cos 0.3-0.8, ~0% acceptance) — not shippable (see finchTool/tools/mtp_reference.py).
(5) Machine safety: two kernel panics from memory-heavy quant jobs (2026-08-15
logit_dump; 2026-08-20 two concurrent audits at 36 GB) — now enforced by the
heavy-job lock + memory guards, see [Machine safety](#machine-safety-2026-08-20).
The MTP weight files (`model_weights_mtp.bin` 4.96 GB + `packed_experts/layer_40.bin`
453 MB) are OPTIONAL — the engine skips them unless `--mtp` is passed, so they
can be deleted to reclaim ~5.4 GB of disk.

### GGUF mode (`--gguf FILE`) — Phase C (2026-08-18)

The engine loads llama.cpp GGUF files directly (zero-copy mmap + per-tensor
Metal wraps; Q4_K/Q6_K dequant on GPU). Targets the Q4_K_M tier of the model
(the "real long-form fix" candidate). Status:

| Metric | Value |
|--------|-------|
| Prefill (chunked, default) | 13-token prompt **~0.9-1.2 s** TTFT vs ~2.9 s per-token (2×); 90 tokens **~5.5 s** vs ~29 s (5×). Chunked GGUF prefill is the default. |
| Decode | **~5.5 tok/s warm / ~1.7-2.3 cold** (S7 parallel slab preads, 3.2×, bitwise; the native 3-bit path is 16-22 tok/s) |
| Correctness | Bitwise vs the pool-path reference at 13 tokens (cos 1.000000); 90-token chunked vs per-token cos 1.000000 (after the sl≥32 attention fix below); llama.cpp cross-validation cos 0.9998 / 0.9962 (1/13 tokens) and **90-token cos 0.9982, argmax MATCH, top10 10/10 (2026-08-18) — GGUF path closed for correctness** |
| RAM | weights stream from the 21.7 GB GGUF mmap; 256 MB expert pool + per-layer wraps |

**Phase C S6 findings (2026-08-18)** — the per-wait cost is fully decomposed
(`FINCHMOE_CBLAT` probes): a kernel-carrying command buffer costs ~0.26 ms of
dispatch overhead regardless of work (empty-CB round trip is 0.013 ms); the GPU
pays a queue-drain wake tax after idle gaps (0.07 ms@0.1 ms, 0.55 ms@1 ms,
1.4 ms@3 ms — the routing+pread gap before CMD3 is the biggest); file-backed
weight reads pay per-16KB DART page-walk costs (the 2 MB-aligned anonymous
expert pool walks free — the preads are IOMMU priming, not just copies).
Env-gated probes: `FINCHMOE_GGUF_GDN_GPU` (0=CPU chain, 1=fused chain CB,
2=fused+cmdA merged in ONE CB — bitwise, opt-in), `FINCHMOE_PF_PINGPONG`
(split-pool CMD3 groups — neutral), `FINCHMOE_GGUF_STAGE2` (aligned copies of
GPU-read tensors — helps kernels, costs expert-slab cache), `FINCHMOE_PF_NOPREAD`
(direct-mmap CMD3 — dead end: GPU faults on untouched pages ~0.5 ms each and
>100 MB no-copy wraps return wrong data).

**Fixed 2026-08-18 — GGUF attention at sl ≥ 32**: the per-token path signaled
GPU attention but never encoded the dispatches (they live in the fused CMD2,
gated off for GGUF), so o_proj silently read stale `buf_attn_out` from token 32
onward — per-token vs chunked diverged at cos 0.818 on the 90-token soak.
GGUF per-token/decode now use CPU attention at sl ≥ 32; 90-token cross-path
is bitwise (cos 1.000000).

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

### M4 Pro benchmark (2026-08-19)

Measured on an M4 Pro (Mac16,7, 24 GB unified memory, 926 GB SSD), running the
`finchmoe/` tree directly (prebuilt binary + `quant_clean` weights, same
`bench_suite.sh`). All runs warm — the 14 GB expert working set plus the
engine's ~3.3 GB footprint fit the 24 GB machine, so every expert read hits
the page cache (`pread_wait` 0.024 ms/layer; no cold-cache regime to
benchmark):

| Metric | M4 Pro (24 GB) | M4 baseline (README) | Ratio |
|--------|----------------|----------------------|-------|
| Decode (K=8, 50 tokens, n=3) | **38.4-42.4 tok/s** (fp32) | 16-22 tok/s (n=6, warm) | ~2× |
| Decode 200-tok (attn-heavy) | **38.6-39.8 tok/s** | 17.4-19.1 tok/s | ~2× |
| Prefill 90 tok, chunked (chunk 8) | **1.21-1.23 s** | 1.9-2.4 s (warm) | 1.6-2× |
| Prefill 90 tok, per-token (chunk 0) | **2.2-2.3 s** | 5.0-5.5 s (warm) | ~2.3× |
| Expert `pread_wait` (chunked) | 0.024 ms/layer | 0.019 ms/layer (warm) | same |
| KV modes (`--kv-fp16` / `--kv-turbo`) | 39.9-40.9 tok/s decode; 1.23-1.23 s prefill — within noise of fp32 | within noise | same |

Consistent with M4 Pro vs M4: ~2× decode/prefill, I/O-bound signature
unchanged. Full log: `/tmp/finchmoe_bench_m4pro_2026-08-19.log`.

### KV cache quantization (`--kv-fp16` / `--kv-turbo`) — 2026-08-14

Three KV storage modes (CPU-side cache; the GPU attention kernels and mirrors
are unchanged): FP32 (default), FP16 (**2× smaller KV**, logits cos 0.999999
vs FP32), and TURBO — K int8 + V 4-bit (**~5× smaller KV**, logits cos
0.999793). Run `./bench_suite.sh` for the full comparison.

| Mode | Decode 50 tok (n=3) | Prefill 90-tok flag-0 | Prefill 90-tok chunk-8 | 200-tok decode |
|------|---------------------|-----------------------|------------------------|----------------|
| fp32 | 16.5-22.0 tok/s | 5.2-5.3 s | 1.9 s | — |
| `--kv-fp16` | **21.9-22.6 tok/s** | 5.3-5.5 s | 1.93-1.95 s | 17.4 tok/s |
| `--kv-turbo` | 17.0-21.9 tok/s | 5.0-5.1 s | 2.0-2.4 s | 19.1 tok/s |
| M1 mini fp32 | 4.05-4.20 tok/s | 23.0 s | 24.8 s | 4.57 tok/s |
| M1 mini `--kv-fp16` | 3.93-4.17 tok/s | 22.5 s | 24.4 s | 4.50 tok/s |
| M1 mini `--kv-turbo` | 3.84-4.15 tok/s | 22.2 s | 24.7 s | 4.45 tok/s |

On the M4 (warm cache) the quant modes are within run-to-run noise — the
expert I/O dominates and the quantized-attention cost (~10-15% on the
CPU-attention layers) is invisible. fp16's short decodes came out slightly
faster than fp32 (less cache traffic). Same finding on the M1 mini (8 GB):
all three modes are within noise (pread_wait ~33.0 ms/layer in every mode —
expert I/O dominates completely there). The M1 win is memory, not speed:
`--kv-turbo` shrinks the KV cache ~5×, which matters when the 3 GB bootstrap
memory gate is the binding constraint on an 8 GB machine.

### Native quant vs GGUF (`--gguf`) — 2026-08-18

Head-to-head of the two model-loading paths on the **same** M1 mini (8 GB):
the native 3-bit expert pack (`quant_clean/`, the `bench.sh` path) versus the
cross-validated **GGUF Q4_K_M** tier (`--gguf models/Qwen3.6-35B-A3B-Q4_K_M.gguf`).
Same binary, same K=8 / 40 layers, same prompts. (The `finchmoe-m1/` deploy
copy is git-ignored; this run used the 2026-08-18 build with the GGUF path.)

| Metric | Native quant | GGUF Q4_K_M | Ratio |
|--------|--------------|-------------|-------|
| Decode 50 tok (n=2) | **4.29-4.33 tok/s** | 2.35-2.43 tok/s | native **1.8×** |
| Decode 200 tok (attn-heavy) | **4.31 tok/s** | 2.55 tok/s | native **1.7×** |
| Prefill 53 tok, per-token (chunk 0) | **12.7 s** | 21.2 s | native **1.7×** |
| Prefill 53 tok, chunked (chunk 8) | 15.2 s | **11.3 s** | GGUF **1.3×** |
| Quick 30 tok, e2e | **6.8 s** (4.28 tok/s) | 9.9 s (2.51 tok/s) | native **1.7×** |

**Decode is the headline: native quant wins ~1.7-1.8×** and holds it across
short and long generation. The per-token gap is mostly `lm_head`: native does
the 3-bit→logit matvec in ~15 ms, the GGUF path ~266 ms (~17× — Q4_K_M must be
dequantized before the final projection). That fixed per-token cost is what a
token-streaming workload pays every step.

**Prefill is mixed.** GGUF's *chunked* path was faster (11.3 s vs 15.2 s), but
its *per-token* path was much slower (21.2 s vs 12.7 s). Both modes emitted the
same non-fatal `fused_gate_up_swiglu_qk_pool_*` shader warnings and fell back to
the same path, so the comparison is fair — the difference is real, not a
fallback artifact.

Caveats: the 8 GB machine ran under constant memory pressure (4.4 GB available,
"may trigger SIGKILL" banner); a few longer runs were jetsam-killed when the
whole suite ran back-to-back and succeeded when re-run individually. This is a
throughput/latency comparison on one machine — it does **not** measure
perplexity or quality, where the GGUF tier is the cross-validated one
(cos 0.9982 vs llama.cpp).

### SSD wear: streaming weights is read-only — no meaningful impact

The engine streams ~420 MB of expert weights from disk per generated token
(40 layers × 8 experts × 1.31 MB, 3-bit). Should you worry about SSD
lifespan? No — the numbers:

- **SSD wear counts WRITES** (NAND program/erase endurance, rated in TBW).
  The engine writes almost nothing to the disk — the weights are read-only
  (zero-copy mmap + `pread`), and the only writes are optional debug dumps.
- **Reads are effectively free for endurance.** The read-disturb effect is
  real but managed by the controller's background refresh, which writes a
  tiny fraction of the read volume. Consumer SSDs (150-600 TBW rated) treat
  this workload as noise.
- **Worst-case scenario**: an 8 GB machine (always-cold cache) at 4.1 tok/s
  sustains ~1.7 GB/s of SSD reads — ~53 PB/year if run 24/7. Even then the
  write-endurance budget is essentially untouched; the *real* cost of
  SSD-streaming is the **speed** (the I/O bottleneck, see the table above),
  not the drive's life.

### Why the M4 decode moved from 9-10 to 16-22 tok/s (page cache)

The engine streams ~420 MB of expert weights from disk **per generated token**
(40 layers × 8 active experts × 1.31 MB each, 3-bit). Whether those reads hit
RAM or the SSD is entirely the OS page cache:

| Machine state | Expert reads | Decode | Prefill `pread_wait` |
|---|---|---|---|
| Cold restart (page cache empty) | SSD | 8.8-10.3 tok/s | 6.2 ms/layer |
| Warm (hot expert pages cached in ~6 GB of reclaimable RAM) | RAM | **16-22 tok/s** | 0.019 ms/layer |

The earlier 9-10 tok/s figure came from a post-restart machine with a cold
cache; after repeated runs the frequently-routed expert pages become resident
and the I/O wait disappears. The M1 mini (8 GB) can never warm up: the 14 GB
expert working set does not fit alongside the engine's ~3.3 GB footprint, so
every read stays an SSD read (33 ms/layer) — that single gap, not compute,
explains its 4.1 tok/s.

**Where the warm-state RAM goes (measured on the M4, 2026-08-14):**

| Consumer | Resident size |
|---|---|
| Engine process (peak RSS, 50-token run) | **~0.7 GB** — KV 320 MB + delta state 63 MB + expert buffers 64 MB + prefill pool 268 MB + touched weight pages |
| Expert page cache (OS "inactive" RAM) | **~3.7 GB** — the hot 3-bit expert pages the OS keeps resident |
| **Total warm-state cost** | **~4.4 GB** |

(The engine's ~2.3 GB GPU allocation is mostly *virtual* — the 1.82 GB weight
file is a zero-copy mmap and only the touched pages become resident; RSS is
the physically-used figure.) The warm setup needs ~4.4 GB of RAM — beyond the
8 GB mini's practical headroom (its available pool fluctuates 2.5-3.8 GB),
which is why it stays permanently cold.

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

# Server (OpenAI-compatible, SSE streaming) + native chat app
./finchmoe-infer -R 9000 -k 8                    # terminal 1
cd finchmoe-chat && ./package_app.sh             # terminal 2 (builds FinchmoeChat.app)
open FinchmoeChat.app                            # SwiftUI client: sessions, streaming,
                                                # collapsible think blocks, tok/s inspector
./chat --show-think                              # or the terminal TUI

# Cross-validation (expect CosSim 1.000000 for gated/o_proj/h_mid/h_post)
FINCHMOE_DUMP_STAGES=1 ./finchmoe-infer -t 1 -k 8 -e 0 -P "Explain what a MoE transformer is in one sentence."
FINCHMOE_REF_MANIFEST=quant_clean/model_weights_quant.json FINCHMOE_REF_WEIGHTS=quant_clean/model_weights_quant.bin python3 finchTool/tools/debug_gdn_reference.py /tmp/stage_dump.bin

# GGUF mode (llama.cpp files, Q4_K/Q6_K GPU dequant, chunked prefill default)
./finchmoe-infer --gguf ../models/Qwen3.6-35B-A3B-Q4_K_M.gguf -L --prompt "The capital of France is Paris." -t 40
./bench_gguf.sh 13 prompt_tokens_gguf.bin        # llama.cpp + CPU + GPU cross-validation
# 90-token soak regression (per-token vs chunked must be bitwise):
#   --prefill-chunk 0 vs 8 --dump-logits on the capitals prompt, compare with
#   python3 finchTool/tools/compare_gguf_logits.py
```

### Key Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `-P TEXT` | — | Input prompt (chat template applied) |
| `-p FILE` | — | Prompt from token-id file |
| `-t N` | 20 | Max tokens to generate |
| `-e F` | 0.7 | Temperature (0 = greedy) |
| `--rep-penalty F` | 1.15 | Repetition penalty |
| `-k N` | 8 | Active experts per layer (model trained with 8) |
| `-3 / -4 / -2 / -8` | **-3** | Expert bit-width (3-bit is the default) |
| `-B N` | 200 | Think budget: force `</think>` after N reasoning tokens |
| `-R PORT` | off | HTTP server (OpenAI-compatible: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`) |
| `-J` | off | MTP speculative decoding (experimental — currently slower, see issues) |
| `-N N` | 262144 | CPU KV context (256k does NOT fit 16 GB — use 16384-32768) |
| `-Q N` | 8192 | GPU KV pre-allocation |
| `--prefill-chunk N` | **8** | Chunked batched prefill (0 = per-token path; 8 = pooled+batched-attn sweet spot) |
| `--low-memory` | off | Skip Metal weight wrap |
| `--kv-fp16` | off | KV cache in FP16 — 2× smaller KV (logits cos 0.999999 vs FP32) |
| `--kv-turbo` | off | KV cache K int8 + V 4-bit — ~5× smaller KV (logits cos 0.999793; ~10% slower CPU attention) |
| `--min-p F` | 0.05 | min_p tail filter (llama.cpp-style): zero tokens below min_p × top-prob — narrows long-form drift |
| `--gguf FILE` | off | Load a GGUF file directly (Q4_K/Q6_K GPU dequant, chunked prefill default ON, ~1.06 tok/s decode) |
| `-I FILE` | — | Dump first-token logits (cross-validation with finchTool/tools/compare_gguf_logits.py) |

## Model Sizes

| Configuration | Weights | Expert disk | Total | Speed (K=8, M4) | Quality loss vs BF16 | Notes |
|---------------|---------|-------------|-------|-------------|----------------------|-------|
| **Default (quant_clean)** | **1.95 GB** (4-bit GDN, 8-bit embed/lm_head) | 14 GB 3-bit | **~16 GB** | **16-22 tok/s** warm cache (10.3 cold) | **Small**: 3-bit experts weight CosSim 0.966-0.979, 4-bit non-experts ≥0.995 (requant 2026-08-13), typo-prompt PASS | current production config; ~4.8× smaller than BF16 |
| Protected tier | 2.45 GB (8-bit GDN, `FINCHMOE_GDN8=1`) | 13 GB 3-bit | ~16 GB | ~9.1 tok/s | **Minimal**: 8-bit non-experts near-lossless; experts as default tier | quality-safe fallback |
| BF16 (source) | 67 GB | — | **71.9 GB** | — | None (reference) | reference; the intended requant base |
| 2bit-dense-v2 (legacy) | 4.96 GB BF16 | 9.4 GB 2-bit | ~14 GB | ~5 tok/s | **Large**: edge prompts degrade into repetition loops (llama.cpp Q4_K_M of the clean base handles the same prompts) | marginal quality — replaced by the clean rebuild |
| MTP (optional) | 4.96 GB | 0.45 GB (layer_40) | 5.4 GB | n/a | n/a (head not shippable) | loaded only with `--mtp`; deletable by default |

**Known limitation — long-form generation**: on requests that invite long
structured output (~500+ words, essays), the model drifts into a
meta-planning loop ("I'll finalize… I'll ready now…") and then a
synonym-spiral ("firm steadfast determined resolute…") regardless of
temperature (0 / 0.3 / 0.7), rep penalty (1.15-1.35), min_p (0.05), or the
n-gram blocker — all tested 2026-08-15. The engine-side state is proven
clean (fresh-prefill differential cos 0.99942) — this is the quant's
long-generation stability limit:

| Tier | Drift onset (essay prompt) | Notes |
|------|---------------------------|-------|
| 4-bit GDN (default) | ~100-200 tokens | loops in the think + answer phases |
| **8-bit GDN protected** (`quant_clean_8gdn`, ~9.1 tok/s) | **~150-250 tokens** | coherent longer; the recommended serve tier for essay-style use |

llama.cpp Q4_K_M of the same model stays clean at 400 tokens (Aug-11
baseline) — a GGUF importer or a higher-precision expert pack is the real
long-form fix. Short and interactive queries are unaffected.

Quality figures are weight-level CosSim from the requant validation plus spot
checks, plus one end-to-end number: **HumanEval pass@1 = 49/164 (29.9%)** on the
M4 native 3-bit tier (2026-08-18, T=0 greedy, raw completions without the chat
template, `--no-think`, 512-token cap; harness: `humaneval_m1/`, results in
`humaneval_m1/he_results.jsonl`). Full protocol matrix measured on the same
engine (all T=0 greedy, `rep-penalty 1.0`): raw+no-think **29.9%** (49/164);
chat-template+think **11.0%** (18/164, 1536-token cap — the model restates the
function then EOS's mid-body); chat-template+no-think **0/10** on a probe. The
chat-template protocols are *worse* on this quant — instruct-template adherence
degrades under 3-bit quantization, while raw continuation rides the base-model
weights — so the raw number is the tier's published protocol, not a pessimism
artifact. The same harness runs on the M1 mini deploy for a second data point.

Four-way comparison on the same M4, same raw harness (2026-08-19):
| engine + model | tier | decode | pass@1 |
|---|---|---|---|
| finchMoE + Qwen 3.6 35B | 4-bit native experts (`--4bit`, `4bit-dense` pack) | ~5.5 tok/s | **53/164 = 32.3%** |
| finchMoE + Qwen 3.6 35B | 3-bit native experts (default) | 10-16 tok/s | 49/164 = 29.9% |
| finchMoE + Qwen 3.6 35B | GGUF Q4_K_M | 3.4-5.5 tok/s | 36/164 = 22.0% |
| turbo-fieldfare + Gemma 4 26B-A4B | their 4-bit repack | ~3.5 tok/s (their M4-mini number) | **142/164 = 86.6%** (chat protocol) |

Read the turbo-fieldfare row carefully: it is a *different model* (Gemma 4 26B,
chat protocol — its only API) and a different quantization pipeline. The
comparison splits into two findings: (1) our custom quant pipeline beats
llama.cpp's Q4_K_M on the same model (+10 points at 3-bit, +13 at 4-bit), but
loses ~30 points vs the FP16 reference (61.6%) — the quantization gap is the
dominant quality cost, not the engine; (2) Gemma 4 26B's chat-mode score sits
near its FP16 expectation, i.e. turbo-fieldfare's 4-bit repack loses far less
than our 3-bit — its quantization is the reference to study.

**Quant-plan closure (2026-08-20)** — both hypotheses tested and closed
(details in [QUANT_QUALITY_PLAN.md](finchmoe/QUANT_QUALITY_PLAN.md)): (a)
near-lossless weights (8-bit experts cos 0.99996 + 8-bit GDN) tie the default
tier — quantization is not the lever; (b) protocol variants tie or lose to
raw+no-think (think-allowed is a no-op — raw prompts never emit `<think>`;
chat template 16.7% vs raw 46.7% on the hard slice). Verdict: **32.3% is the
capability ceiling of Qwen 3.6 35B A3B through our harness**; the fraQtl
number is a different model, not reachable by weight or protocol changes.

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
| [finchmoe/QUANT_QUALITY_PLAN.md](finchmoe/QUANT_QUALITY_PLAN.md) | Quantization quality plan — Phase 0 audit results, E1/E2/E3' experiments, runbook |
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
| `build_hot_sets.py` | Per-layer hot expert sets from `--collect-routing` logs → hot_sets.bin (prefill prefetch) |

### Benchmarks (finchmoe/)

| Tool | Purpose |
|------|---------|
| `bench_suite.sh` | Decode/prefill comparison matrix (KV modes, chunk sizes) |
| `bench_prefill.sh` | Prefill benchmark + bitwise parity matrix (chunk sizes × timings) |
| `bench_gguf.sh` | GGUF gate runner: llama.cpp reference + finchmoe CPU + GPU, cos comparison |
| `run_logit_dump_safe.sh` | llama.cpp logit_dump with the memory watchdog |

### Debugging & verification (finchTool/tools/)

| Tool | Purpose |
|------|---------|
| `compare_gguf_logits.py` | cos/argmax/max-diff of `-I` logit dumps (used by bench_gguf.sh) |
| `compare_pb_traces.py` | First-diverging (token, layer) on Phase-B traces |
| `analyze_s4_dumps.py` | Phase C S4 kernel-parity dumps |
| `analyze_longgen.py` | Long-generation health: n-gram repetition, EOS, think tags, drift |
| `debug_gdn_reference.py` | Numpy GDN reference (stage cross-validation, CosSim 1.000000) |
| `debug_compare.py` / `debug_e2e_logits.py` | Weight/logits consistency checks |
| `debug_full_forward.py` / `debug_layer_compare.py` / `debug_layer_diff.py` | Full-forward and per-layer reference diffs |
| `debug_full_attn_moe.py` / `debug_2token_gdn.py` | Attention/MoE and GDN targeted references |
| `debug_bf16_vs_4bit.py` / `debug_mlx_inference.py` | Format and MLX-loader verification |
| `verify_clean_rebuild.py` | CosSim verification of the clean-rebuild weights (3-bit experts, non-experts) |
| `quant_audit.py` | Per-tensor quantization audit vs pristine BF16 (259 non-experts + 3/4/8-bit expert packs); heavy-job lock + memory guards (see Machine safety) |
| `extract_mtp_experts.py` / `mtp_reference.py` | MTP head extraction + numpy reference |
| `wobble_hunt.sh` / `wobble_trace_hunt.sh` | Run-to-run wobble hunting |
| `finchTool/` | Standalone Metal kernel verification suite (matvec/attention kernels) |
| `test_engine_path.m` | Engine-path standalone test (finchmoe/) |

### Machine safety (2026-08-20)

Two kernel panics were caused by memory-heavy quant jobs exhausting the 16 GB
mini: **2026-08-15** `logit_dump` at 10.5 GB RSS, and **2026-08-20** two
concurrent audit runs at 21.1 + 15.3 GB RSS — compressor 100% of segments,
~14 MB free, watchdogd starved 92 s. The operating rule — **one heavy job at
a time, never into swap** — is now enforced by the tools:

- **Shared job lock** (`/tmp/finchmoe_heavy_job.lock`) in `quant_audit.py`
  and `repack_experts.py` — a second heavy job exits with the holder's pid.
- **Memory guards** — abort at <6 GB available at startup, <4 GB per expert
  layer, <3 GB per big tensor (free+inactive+speculative+purgeable).
- **Streamed expert refs** in `quant_audit.py` (per-expert instead of all
  256 materialized) — peak 4.5 GB → **1.7 GB**, verified: 3-pack sweep at
  1.69 GB max RSS, 0 swaps.

### Runtime diagnostics (engine flags & env vars)

| Flag / Env | Purpose |
|------------|---------|
| `--timing` / `FINCHMOE_PF_TIMING=1` | Per-phase timing (per-token / chunked prefill; now includes per-commit idle gaps: cmdA/delta/cmdB/cmd3_gap — the GPU wake-tax attribution) |
| `--debug-layers` | Per-layer hidden-state statistics |
| `--compare-experts N` | GPU vs CPU expert outputs for layer N |
| `--dump-logits FILE` / `-I FILE` | First-token logits for cross-validation |
| `--collect-routing FILE` | Routing logs (layer, hidden, top-K, top-24) for predictor training |
| `FINCHMOE_DUMP_HIDDEN` / `FINCHMOE_PF_DUMP` / `FINCHMOE_DUMP_STAGES` | Per-layer hidden / chunked-stage dumps for parity forensics |
| `FINCHMOE_PF_PREFETCH=1` | Force-enable hot-set prefetch (bypasses the memory gate) |
| `FINCHMOE_GGUF_GDN_GPU` | GGUF linear chain: 0 = CPU chain (default), 1 = fused GPU chain (S5), 2 = cmdA matvecs + fused chain in ONE CB (S6a) |
| `FINCHMOE_PF_TIMING=1` + `FINCHMOE_PF_CMD3WAIT=1` | Isolate the deferred CMD3 tail from cmdA_wait |
| `FINCHMOE_PF_KLOOP/GKLOOP/C3LOOP=N` | Repeat cmdA/fused-GDN/CMD3 encodes N× in one CB (kernel-only timing) |
| `FINCHMOE_PF_C3SKIP=N` | Skip CMD3 expert dispatches (timing-only — outputs go stale) |
| `FINCHMOE_PF_NODEDUP` | Disable expert-pread dedup (escape hatch) |
| `FINCHMOE_PF_PINGPONG=1` | Split-pool CMD3 groups (opt-in; measured neutral on M4) |
| `FINCHMOE_PF_NOPREAD=1` | Direct-mmap CMD3 (probe — known slow/wrong, see GGUF status) |
| `FINCHMOE_GGUF_STAGE2=1/2` | 2MB-aligned copies of GPU-read QK tensors (probe — wins kernels, costs expert-slab cache) |
| `FINCHMOE_CBLAT=N/K/KA` | Command-buffer latency probes (empty / kernel / keep-alive ticker) |

## Project Layout

```
finchMoE/
├── README.md, design.md, BUGS.md
├── finchmoe/                     # engine + tools (see tables above)
│   ├── infer.m                   # main engine (C/Metal, ~15k lines)
│   ├── shaders.metal             # Metal kernels (4/3/2/1/8-bit dequant, fused GDN, routing, attention)
│   ├── chat.m                    # chat TUI client
│   └── finchTool/                # Metal kernel verification suite + tools/ (python/shell diagnostics)
├── models/                       # Qwen3.6-35B-A3B variants + GGUF references
│   └── Qwen3.6-35B-A3B-bf16/     # pristine BF16 base (requant source, expert packs)
├── quant_clean/                  # current production weights (model_weights_quant.bin/.json)
├── quant_*/                      # quant experiments (self, 4gdn, 8gdn, visual, …)
└── finchmoe-m1/                  # M1 mini (8GB) deploy copy: binary + weight symlinks + bench.sh (untracked)
```
