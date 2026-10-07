<p align="center">
  <img src="docs/assets/finchmoe-app-icon.png" alt="FinchMoE logo" width="280">
</p>

<h1 align="center">FinchMoE</h1>

<p align="center">
  <strong>Out-of-core MoE inference on Apple Silicon — running Qwen 3.6 35B-A3B and Qwen 3.8 Flash-Next 125B</strong><br>
  A custom Swift + Metal runtime that streams MoE experts from SSD, so large models run on Macs with 8 GB of RAM.
</p>

<p align="center">
  <img alt="Swift 6.2" src="https://img.shields.io/badge/Swift-6.2-F05138?logo=swift&logoColor=white">
  <img alt="Metal 4" src="https://img.shields.io/badge/Metal-4-5E5CE6">
  <img alt="macOS 26 or later" src="https://img.shields.io/badge/macOS-26%2B-000000?logo=apple&logoColor=white">
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache%202.0-2ea44f"></a>
</p>

<p align="center">
  <a href="#try-it">Quick start</a> ·
  <a href="#model-performance-comparison">Model performance</a> ·
  <a href="#qwen-36-35b-a3b-port">Qwen 3.6 port</a> ·
  <a href="#qwen-38-flash-next-125b-port">Qwen 3.8 port</a> ·
  <a href="docs/OPENAI_SERVER.md">Local server</a> ·
  <a href="docs/BENCHMARKS.md">Benchmarks</a> ·
  <a href="docs/SYSTEM_DESIGN.md">How it works</a> ·
  <a href="docs/IMPLEMENTATION_REFERENCES.md">References</a>
</p>



## What this is

FinchMoE is a Swift + Metal runtime that runs a MoE LLM **without loading the
whole model into RAM**. It keeps the shared core and KV/linear state in
memory, then streams only the experts a token needs from SSD through a bounded
LFU cache. That is what lets a tens-of-billions-parameter model run on an 8 GB
Mac. It is model-specific, not a wrapper around MLX or llama.cpp.

The **FinchMoE** name comes from this project's first generation — a C/Metal
engine that stalled at ~13% HumanEval and was abandoned in August 2026. What
you see here is its second generation: the Swift + Metal runtime (formerly
**FlashQwen**) that built on the out-of-core design of
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) and reached the
quality the first generation could not (90.9% HumanEval — llama.cpp parity).
The first-generation codebase and payload live on, archived, under
`archive/`.

The upstream project ran **Gemma 4 26B-A4B**. This engine's port goal was
**Qwen 3.6 35B-A3B** (`model_type: qwen3_5_moe`), and that is the working
reference model today; the Gemma path is intact, and family dispatch runs
both from the same binary.

## Current state

- The engine ships as **FinchMoE** (renamed from FlashQwen when the two
  generations consolidated under this name, 2026-09-07): the module, target,
  product, and binary names all use `FinchMoE`, and the on-disk model format is
  **`.finch`** (binary magic `FINCH`) in place of the upstream `.gturbo`.
- The Gated-DeltaNet (linear-attention) decode unit — the piece with no Gemma
  analogue — is implemented and reference-checked against the
  `qwen3_5_moe` Transformers source (**Phase 1 of the port, done**).
- The **complete Qwen 3.6 decode-layer path is wired into the forward pass**,
  behind `ArchConfig.modelFamily == "qwen3_6"`: GDN layers (silu causal conv,
  fused gate, recurrent step, gated RMSNorm), the 10 full-attention layers
  (partial RoPE, `attn_output_gate`), the Qwen MoE tail (silu experts, plain
  softmax top-8 router, sigmoid-gated shared expert), per-layer recurrent
  state, and the untied `lm_head`. The Gemma path is unchanged and the whole
  suite stays green. Every new kernel is reference-checked before wiring.
- The port is **complete and closed (2026-09-07)**: Qwen chunked prefill
  (GDN conv + recurrent + full-attention chunk kernels), the bf16 →
  `.finch` quantizing repack, the interface surface (tokenizer family,
  ChatML, dual stop set, softcap-0 sampling), end-to-end quality (degenerate
  output resolved), benchmarks, a long-form quality pass, and a 4096-token
  context soak have all landed. The Qwen 3.6 install built and validated
  locally is the working reference model today; the Gemma 4 26B-A4B path
  stays intact and family dispatch runs both from the same binary.
- The **Qwen 3.8 Flash-Next 125B path is wired and validated** behind
  `ArchConfig.modelFamily == "qwen3_8"`: hyper-connections replace RMSNorm,
  QSA sparse-block attention runs on the full-attention layers, and the PLE
  n-gram head is served from shard files. The shipping install is
  `Qwen3.8-Flash-Next-125B-ple4bit.finch` — **97 GiB**, with the PLE table
  quantized to int4/group-32 and the expert and attention tensors as before.
  It loads and decodes inside the 16 GB operating budget; the earlier 162 GiB
  install (PLE table at raw bf16) stays directly behind it as the fallback the
  app's default-install resolution drops to. EvalPlus HumanEval on the
  shipping install scores **95.1% base / 92.1% HumanEval+**.
- **Abliterated (uncensored) variants of both Qwen models are built, measured
  and published** (2026-09-27). They are repacked by the same pipeline at the
  same quantization settings, so only the weights differ, and EvalPlus shows
  **no measurable regression** against the base weights — 3.6 abliterated
  scores 93.3% base / 90.2% HumanEval+ against the base install's
  90.9% / 87.8%, and 3.8 abliterated 94.5% / 92.7% against 94.5% / 92.1%. The
  app's Preset picker lists them and prefers them when a checkout holds one.
  See [Abliterated (uncensored) installs](#abliterated-uncensored-installs).
- **The prefill projections are batched.** The linear-attention (GDN) weights
  are int8 on every shipped install, and the prefill used to feed them one GEMV
  per token — 426 re-reads of each weight matrix per chunk. A tiled kernel that
  dequantizes the weight tile once and reuses it across the token dimension cut
  the projection stages by 3.5x and the prefill end to end by **~1.4x on both
  models** (3.6: 13.4 s to 9.4 s at 426 tokens; 3.8: 29.2 s to 20.8 s). It is on
  by default; `FQ_INT8_GEMM=0` restores the per-token path.
- The pristine upstream TurboFieldfare source is archived in `reference/`
  (gitignored, alongside `models/`).

The Qwen 3.6 port is documented end-to-end — target model, locked GDN math,
and the phase plan — in [docs/QWEN36_PORT.md](docs/QWEN36_PORT.md).
The Qwen 3.8 Flash-Next port is documented in
[docs/QWEN38_PORT.md](docs/QWEN38_PORT.md).

## Try it

```bash
git clone https://github.com/haihengh/finchMoE.git
cd finchMoE
swift build -c release
.build/release/FinchMoEMac
```

On the first run, Swift Package Manager downloads and builds the packages
required by the tokenizer. The complete release build produces the Mac app and
its sibling decode-service executable.

When the app opens, choose **Download** and let FinchMoE fetch and repack the
pinned model. Once it is ready, choose **Load Model**, type your prompt, and
press **Generate**.

> **Note:** the local install this repo builds and validates is **Qwen 3.6
> 35B-A3B** (built from `models/Qwen3.6-35B-A3B-bf16` by `FinchMoERepack`:
> int8 GDN linear-attention projections, affine 4-bit MoE group 64, ~20 GB on
> disk, streamed out-of-core). The upstream Gemma 4 26B-A4B path is intact;
> family dispatch keeps both runnable from the same binary.

## At a glance

Measured 2026-09-30 on a 24 GiB Apple M4 Pro (`Mac16,7`, macOS 26.7), 2026-09-18
on a 16 GB Apple Silicon Mac mini, and 2026-10-01 on an 8 GB Apple M1 Mac mini
(`Macmini9,1`, macOS 27.0.1), all with the engine's release CLI against the
shipping `Qwen3.6-35B-A3B-4bit.finch` (page cache warm). The M4 Pro is the
faster host on both axes: it decodes **~20 tok/s** against the mini's ~8.4 and
the 8 GB M1's ~5.2, and the same 2,940-token prompt it prefills in **41.2 s**
takes the 16 GB mini 70.1 s and the 8 GB M1 130.3 s. The M4 Pro figures are at
the app sampling defaults (temperature 0.2, Top-K 64, Top-P 0.95); see the
[model comparison](#model-performance-comparison) for all four hosts.

| Metric   | Qwen 3.6 35B-A3B (`qwen3_5_moe`) install                                                                                                |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Model    | 35B total parameters, ~3B active per token; 30 Gated-DeltaNet linear-attention layers + 10 full-attention; MoE 256 experts top-8 + shared |
| Weights  | GDN projections int8; router int8; shared/routed experts affine 4-bit group 64; fp16 activations, fp32 Metal accumulators                 |
| Storage  | ~19 GB installed text-only `.finch` (streamed from disk during decode)                                                           |
| Memory   | ~1.1-1.2 GiB peak resident while decoding (out-of-core expert streaming; OS page cache additional)                                            |
| Decode   | **~20 tok/s** (24 GiB M4 Pro) / ~8.4 tok/s (16 GB Mac mini) / ~5.2 tok/s (8 GB M1), 2,940-token prompt, 128-token decode                                                                                             |
| Prefill  | **~71 tok/s** on a 2,940-token prompt (24 GiB M4 Pro, 41.2 s) / ~42 tok/s (16 GB Mac mini, 70.1 s) / ~22.6 tok/s (8 GB M1, 130.3 s); short prompts skip the SHA-256 pass when a verified-install receipt is present (`--verify auto`, the default)            |
| Hardware | Apple Silicon Mac (validated on 8, 16 and 24 GiB RAM)                                                                                                |
| Platform | macOS 26, Metal 4, Swift 6.4                                                                                                              |

### M4 Mac mini performance

Measured 2026-09-18 on a 16 GB M4 Mac mini, release CLI, greedy decode, both
shipping installs, warm page cache, one run each on the same 2,940-token prompt
(`long-synthesis`, 128 generated tokens). Both models now have a published
throughput row; the 3.8 one is new, and the 3.6 prefill figure is more than
twice what this table said before the int8 projections were batched.

| Host | Model | Workload | Prefill tok/s | Decode tok/s | Peak resident |
| --- | --- | --- | ---: | ---: | ---: |
| 16 GB M4 Mac mini | Qwen 3.6 35B-A3B | 2,940-token prompt; 128-token greedy decode | **~42** (69.3 s) | ~8.4 | ~1.1 GiB |
| 16 GB M4 Mac mini | Qwen 3.8 Flash-Next 125B | 2,940-token prompt; 128-token greedy decode | **~19** (157.7 s) | ~2.8 | inside 16 GB budget |

The same two runs at the app's sampling defaults (temperature 0.2, Top-K 64,
Top-P 0.95) land within about 10% of these: 42.0 and 18.9 prefill tok/s, 7.8 and
2.6 decode.

Prompt length, generated length, page-cache state, and hardware all affect
throughput. See [benchmarks](docs/BENCHMARKS.md) for the upstream Gemma
measurements the fork started from.

## Model performance comparison

Four hosts have prompt-suite rows: a 24 GB Apple M4 Pro (`Mac16,7`, macOS
26.7, Swift 6.4, 2026-09-30), a 16 GB M4 Mac mini (2026-09-18), a 16 GB
M6 Mac mini (`Mac18,5`, macOS 27.0, Swift 6.4, 2026-09-30), and an 8 GB
M1 Mac mini (`Macmini9,1`, macOS 27.0.1, 2026-10-01). The throughput
comparison below puts them side by side; the tables after it are the rows behind
it. All four runs are on the current engine. The 8 GB host carries Qwen 3.6 rows
only — the 125B install falls into swap on 8 GB, see below.

All rows use the release `FinchMoECLI` against verified local `.finch` installs,
with the app sampling defaults (`temperature 0.2`, Top-K 64, Top-P 0.95) and a
128-token generation cap. Decode rates exclude model load and prompt prefill.
Prefill rates are reported separately because long prompts exercise a different
path than token-by-token decode.

### Throughput across hosts

Tokens per second on the three frozen cases. Each cell is ordered
`short-explanation / medium-review / long-synthesis`:

| Host | Model | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: |
| 24 GB M4 Pro | Qwen 3.6 35B-A3B | 30.4 / 78.9 / 71.4 | 24.10 / 22.37 / 20.41 |
| 24 GB M4 Pro | Qwen 3.8 Flash-Next 125B | 11.5 / 33.8 / 30.9 | 6.63 / 5.88 / 5.53 |
| 16 GB M4 Mac mini | Qwen 3.6 35B-A3B | 14.6 / 46.8 / 42.0 | 10.03 / 9.73 / 7.82 |
| 16 GB M4 Mac mini | Qwen 3.8 Flash-Next 125B | 6.6 / 20.7 / 18.9 | 3.17 / 2.95 / 2.59 |
| 16 GB M6 Mac mini | Qwen 3.6 35B-A3B | 17.8 / 62.7 / 59.3 | 8.46 / 8.57 / 7.74 |
| 16 GB M6 Mac mini | Qwen 3.8 Flash-Next 125B | 7.6 / 24.9 / 23.2 | 3.21 / 2.67 / 2.43 |
| 8 GB M1 Mac mini | Qwen 3.6 35B-A3B | 11.9 / 24.2 / 22.6 | 5.19 / 4.81 / 4.17 |

### Quality

| Model | Install | HumanEval base pass@1 | HumanEval+ pass@1 |
| --- | ---: | ---: | ---: |
| Qwen 3.6 35B-A3B | ~19 GB `.finch` | 90.9% (149/164) | 87.8% (144/164) |
| Qwen 3.8 Flash-Next 125B | ~167 GB `.finch` | 94.5% (155/164) | 92.1% (151/164) |

Both models also exist as abliterated (uncensored) installs, measured separately
so the rows above stay a single-machine comparison — see
[Abliterated (uncensored) installs](#abliterated-uncensored-installs).

### Per-case detail

The rows behind the comparison table, host by host.

Current engine, 24 GB M4 Pro, 2026-09-30 (one measured run per case after a
discarded warmup, fresh processes, no other FinchMoE process active):

| Prompt-suite case | Model | Prompt tokens | Prefill | Prefill tok/s | Decode | Decode tok/s | Expert reads/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-explanation | Qwen 3.6 35B-A3B | 62 | 2.04s | 30.4 | 5.31s | 24.10 | 222.1 MB |
| short-explanation | Qwen 3.8 Flash-Next 125B | 62 | 5.41s | 11.5 | 19.31s | 6.63 | 613.8 MB |
| medium-review | Qwen 3.6 35B-A3B | 426 | 5.40s | 78.9 | 5.72s | 22.37 | 236.1 MB |
| medium-review | Qwen 3.8 Flash-Next 125B | 426 | 12.61s | 33.8 | 21.75s | 5.88 | 677.3 MB |
| long-synthesis | Qwen 3.6 35B-A3B | 2,940 | 41.18s | 71.4 | 6.27s | 20.41 | 243.6 MB |
| long-synthesis | Qwen 3.8 Flash-Next 125B | 2,940 | 95.10s | 30.9 | 23.17s | 5.53 | 724.3 MB |

Current engine, 16 GB M4 Mac mini, 2026-09-18, same protocol and columns:

| Prompt-suite case | Model | Prompt tokens | Prefill | Prefill tok/s | Decode | Decode tok/s | Expert reads/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-explanation | Qwen 3.6 35B-A3B | 62 | 4.24s | 14.6 | 12.77s | 10.03 | 217.3 MB |
| short-explanation | Qwen 3.8 Flash-Next 125B | 62 | 9.35s | 6.6 | 40.42s | 3.17 | 603.0 MB |
| medium-review | Qwen 3.6 35B-A3B | 426 | 9.10s | 46.8 | 13.15s | 9.73 | 237.2 MB |
| medium-review | Qwen 3.8 Flash-Next 125B | 426 | 20.59s | 20.7 | 43.37s | 2.95 | 682.4 MB |
| long-synthesis | Qwen 3.6 35B-A3B | 2,940 | 70.06s | 42.0 | 16.38s | 7.82 | 247.8 MB |
| long-synthesis | Qwen 3.8 Flash-Next 125B | 2,940 | 155.80s | 18.9 | 49.47s | 2.59 | 726.8 MB |

Current engine, 16 GB M6 Mac mini, 2026-09-30, same protocol and columns:

| Prompt-suite case | Model | Prompt tokens | Prefill | Prefill tok/s | Decode | Decode tok/s | Expert reads/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-explanation | Qwen 3.6 35B-A3B | 62 | 3.49s | 17.8 | 15.13s | 8.46 | 222.0 MB |
| short-explanation | Qwen 3.8 Flash-Next 125B | 62 | 8.20s | 7.6 | 39.93s | 3.21 | 602.5 MB |
| medium-review | Qwen 3.6 35B-A3B | 426 | 6.79s | 62.7 | 14.94s | 8.57 | 232.7 MB |
| medium-review | Qwen 3.8 Flash-Next 125B | 426 | 17.08s | 24.9 | 47.90s | 2.67 | 671.7 MB |
| long-synthesis | Qwen 3.6 35B-A3B | 2,940 | 49.57s | 59.3 | 16.53s | 7.74 | 242.9 MB |
| long-synthesis | Qwen 3.8 Flash-Next 125B | 2,940 | 126.99s | 23.2 | 52.59s | 2.43 | 727.3 MB |

Current engine, 8 GB M1 Mac mini, 2026-10-01, same protocol and columns, Qwen
3.6 only (the 3.8 run fell into swap and was aborted; see below):

| Prompt-suite case | Model | Prompt tokens | Prefill | Prefill tok/s | Decode | Decode tok/s | Expert reads/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-explanation | Qwen 3.6 35B-A3B | 62 | 5.23s | 11.9 | 24.69s | 5.19 | 221.2 MB |
| medium-review | Qwen 3.6 35B-A3B | 426 | 17.62s | 24.2 | 26.59s | 4.81 | 234.3 MB |
| long-synthesis | Qwen 3.6 35B-A3B | 2,940 | 130.29s | 22.6 | 30.72s | 4.17 | 250.8 MB |

On the current engine the M4 Pro now leads the other three hosts on the 3.8 install's
prefill (33.8 / 30.9 tok/s on medium / long against 20.7 / 18.9 on the M4 mini
and 24.9 / 23.2 on the M6 mini). The earlier M4 Pro 3.8 rows (16.5 and 14.8)
predated the quantized-PLE install and the batched int8 projections, so the
comparison above — all three hosts on the same engine — is the first that lines
them up fairly.

Qwen 3.6 decodes about **3.0-3.3x faster** than Qwen 3.8 on the minis and
**3.6-3.8x** on the M4 Pro, while Qwen 3.8 scores **+4.2 points** on HumanEval
base and **+4.3 points** on HumanEval+. The dominant runtime difference is
routed-expert I/O: the 125B install reads roughly 602-727 MB of expert data per
generated token across the three hosts, versus 215-248 MB/token for the 35B
install.

The M4 Pro decodes about **2.6-2.9x faster** than the M6 mini and **2.3-2.6x
faster** than the M4 mini on the 3.6 install, and that is not routing or cache
policy: its per-token expert hit and miss counts match the M6 mini's within
about a point (short-explanation, 58.8% hits against 58.9%; the 3.8 install
agrees likewise, 51.5% against 52.4%). What differs is how fast a miss is
served, which the host's memory and storage path decide.

The M6 mini's edge over the M4 mini is prefill, which is compute-bound: 59.3
tok/s against 42.0 on the 2,940-token 3.6 case, and 23.2 against 18.9 for 3.8.
Decode is level between them — 7.74 against 7.82, and 2.43 against 2.59. The M6
rows are one measured run per case after a discarded warmup, so read differences
of a few percent as noise; that is most visible on short-explanation, the
shortest case.

The 8 GB M1 mini decodes the 3.6 install at about **half the M6 mini's rate**
(5.19 / 4.81 / 4.17 tok/s against 8.46 / 8.57 / 7.74) with the same expert
reads per token (221-251 MB) and hit/miss counts within noise of the other
hosts' — the gap is the older compute and the smaller page cache, not routing.
Medium and long prefill land at about a third of the M6 mini's (24.2 / 22.6
against 62.7 / 59.3 tok/s). The 125B install does not fit an 8 GB budget: with
the host's ambient apps open, its resident core plus the open apps pushes free
memory down to 3-7% and the OS swaps the working set between tokens. The one
measured short-explanation pass decoded at 0.27 tok/s (473 s for 128 tokens,
against 39.9 s on the 16 GB M6 mini) before the run was aborted as meaningless,
so the 8 GB host carries no 3.8 rows. On 8 GB the out-of-core claim covers the
35B install; the 125B needs a 16 GB host.

### M4 Pro rerun, 2026-10-03: what a failed receipt costs

The 24 GB M4 Pro was re-run on 2026-10-03 on the same protocol, and those rows
are **not** comparable to the 2026-09-30 ones above. All six cases carry
`warning: verified-install.json is present but unusable (model directory
mismatch); verified with full SHA-256 instead` on stderr, so `--verify auto`
resolved to `.fullSha256`. Verification is lazy — a layer's `packed_experts`
file and each PLE part are hashed the first time they are read — so the pass
runs inside the prefill window and its cost lands in the prefill column:

| Prompt-suite case | Model | Prompt tokens | Prefill | Prefill tok/s | Decode | Decode tok/s | Expert reads/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-explanation | Qwen 3.6 35B-A3B | 62 | 8.26s | 7.5 | 5.88s | 21.78 | 219.2 MB |
| short-explanation | Qwen 3.8 Flash-Next 125B | 62 | 47.39s | 1.3 | 27.98s | 4.57 | 600.2 MB |
| medium-review | Qwen 3.6 35B-A3B | 426 | 8.98s | 47.5 | 6.02s | 21.27 | 240.5 MB |
| medium-review | Qwen 3.8 Flash-Next 125B | 426 | 50.07s | 8.5 | 23.60s | 5.42 | 676.0 MB |
| long-synthesis | Qwen 3.6 35B-A3B | 2,940 | 45.81s | 64.2 | 7.34s | 17.43 | 247.1 MB |
| long-synthesis | Qwen 3.8 Flash-Next 125B | 2,940 | 132.68s | 22.2 | 25.96s | 4.93 | 735.6 MB |

The pass accounts for the whole prefill gap against the 2026-09-30 rows: +3.6
to +6.2 s on the ~19 GB 3.6 install, +37.5 to +42.0 s on the 97 GB 3.8 one. The
3.6 rate is the higher of the two because that install still fits the 24 GB
page cache, so its hash reads back from RAM rather than the volume.

Decode is not explained by any of this: the rerun lands below the 2026-09-30
rows on all six cases, by 5% (21.27 against 22.37, 3.6 medium) to 31% (4.57
against 6.63, 3.8 short). That run wrote no `system.txt`, so it carries no host
state to attribute the decode gap to, and the 2026-09-30 rows stay the host's
published ones.

The receipt binds to the model directory's *physical* path, so a run that
reaches an install through a different spelling than the receipt was recorded
through falls back to hashing the whole install. Both installs here have
receipts, and they record different spellings of the same symlinked location —
`.../flash-qwen/models/...` for 3.6, `.../finchMoE/models/...` for 3.8, with
`models/` a symlink from the latter into the former. Re-record with
[`--verify-install`](#command-line-interface) from the path the harness will
use, or pass `--verify full-sha256` deliberately and price the hash in.

### Prompt cache: cold vs hot

The app's chat turns resume from the previous turn's KV cache (see
[docs/SPEEDUP_EXECUTION_PLAN.md](docs/SPEEDUP_EXECUTION_PLAN.md) §2). Measured
2026-10-03 on a 16 GB M4 Mac mini through the shipping app client with the
same three frozen cases and sampling as the tables above. Per case: a
discarded warmup cold run, a measured cold run, then a hot turn that resumes
from that run's KV and appends the 426-token medium prompt — enough new
tokens to price the resumed prefill, enough decode for a TG sample. Spotlight
indexing was off on the model volume (with it on, this box had produced
stalled rows up to 20× slow). These rows come from the debug test build of
the same client the app uses; they land close to this host's published rows
above — 3.6 medium/long within 2% (47.1 / 41.2 tok/s against 46.8 / 42.0) and
3.8 within 7% (19.5 / 17.6 against 20.7 / 18.9) — while short-explanation,
where fixed costs dominate, reads as noise (17.3 / 6.6 against 14.6 / 6.6).

| Case | Model | Cold prefill | Hot: cached / computed | Hot prefill | Cold TG | Hot TG |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| short | Qwen 3.6 35B-A3B | 3.59s (17.3 tok/s) | 189 / 429 | 9.13s (47.0 tok/s) | 8.5 tok/s | 7.6 tok/s |
| short | Qwen 3.8 125B | 9.34s (6.6 tok/s) | 189 / 429 | 22.4s (19.2 tok/s) | 2.9 tok/s | 2.2 tok/s |
| medium | Qwen 3.6 35B-A3B | 9.05s (47.1 tok/s) | 553 / 429 | 9.68s (44.3 tok/s) | 8.0 tok/s | 6.5 tok/s |
| medium | Qwen 3.8 125B | 21.9s (19.5 tok/s) | 553 / 429 | 23.6s (18.2 tok/s) | 2.9 tok/s | 2.2 tok/s |
| long | Qwen 3.6 35B-A3B | 71.4s (41.2 tok/s) | 3,067 / 429 | 12.2s (35.2 tok/s) | 6.7 tok/s | 6.5 tok/s |
| long | Qwen 3.8 125B | 166.6s (17.6 tok/s) | 3,067 / 429 | 24.3s (17.7 tok/s) | 2.7 tok/s | 2.6 tok/s |

The per-token prefill rate is the same cold and hot: reuse removes tokens
from prefill, it does not make the path faster. The wall-clock win therefore
scales with the cached context and shrinks with the size of the new message —
the long case's 426-token follow-up prefills in 12.2 s instead of re-reading
3,496 tokens (~7× on 3.6, ~8× on 3.8), and a small follow-up on the same
conversation resumes in seconds against ~85 s cold (a 3,840-token soak
measured 2.8 s against 100.1 s). Decode is untouched by reuse: the cold/hot
TG columns sit within sampling noise of each other on both installs.

Reproduce with the install-gated benchmark (skipped on CI; installs are read
from `<repo>/models` unless `FQ_BENCH_MODEL_DIR` says otherwise; rows append
to `benchmark-results/prefix-cold-hot.log`, each block tagged with the host's
`hw.model` so runs from different machines stay comparable):

```bash
swift test --no-parallel --filter PrefixReuseBenchmarkTests
# or one case on one model:
FQ_BENCH_CASES=long-synthesis swift test --no-parallel --filter qwen38ColdAndHot
```

### Routed-expert width: 4-bit vs 3-bit vs 2-bit

Routed experts ship at 4-bit by default; `FinchMoERepack
--routed-expert-bits 3|2` converts the same bf16 snapshot to narrower
experts in the same affine group-64 family (int3 packs eight values per
24-bit little-endian triplet, int2 four values per byte; scales/biases stay
BF16 per group). All rows: Qwen 3.6 35B-A3B, 16 GB M4 mini, the frozen
`real-generation-v1` cases at the published sampling, one measured run per
cell after a discarded warmup — prompt processing (PP) over the
50/414/2,928-token prompts, decode (TG) over 128 generated tokens; scores
from the EvalPlus harness (`quality/humaneval/`). 3-bit and 2-bit are
experimental; 4-bit remains the default.

| | 4-bit | 3-bit | 2-bit |
| --- | ---: | ---: | ---: |
| Install | 18.68 GiB | 14.93 GiB | 11.18 GiB |
| HumanEval pass@1 | 90.9% (149/164) | 90.9% (149/164) | 76.8% (126/164) |
| HumanEval+ pass@1 | 87.8% (144/164) | 86.6% (142/164) | 70.7% (116/164) |
| Prompt processing (short/medium/long) | 17.0 / 49.9 / 42.2 tok/s | 21.4 / 54.6 / 46.9 tok/s | 27.9 / 58.2 / 50.3 tok/s |
| Decode (short/medium/long) | 8.5 / 8.3 / 7.2 tok/s | 10.0 / 10.8 / 12.1 tok/s | 17.8 / 17.2 / 12.0 tok/s |

3-bit is the sweet spot: quality parity (base identical, HumanEval+ two
problems behind) for 20% less disk and 1.4-1.7x the decode rate. 2-bit buys
another 25% of disk and up to 2.4x short-context decode for ~14-17 points —
a usable tier, not a near-parity one. Prompt processing gains less (+11% /
+19% on the long case) because prefill is compute-bound; decode gains track
the per-token expert bytes.

An earlier 3-bit experiment (2026-09-23) scored 28/164 and parked the
format. The re-run above attributes that to the archived runtime's int3
kernels reading the blobs incorrectly at production row widths — a failure
small-case checks cannot see — not to the 3-bit format itself;
`docs/SESSION_HANDOFF_3BIT_EXPERTS.md` carries the corrected verdict and
the kernel/writer verification that guards it now.

## The Qwen 3.8 Flash-Next 125B port

Qwen 3.8 Flash-Next 125B (`qwen4_exp_text`, GGUF `qwen4exp`, Finch family
`qwen3_8`) extends the Qwen runtime beyond the 3.6 Gated-DeltaNet baseline.
The text path has 48 layers: 36 GDN layers and 12 full-attention layers at
layers 3, 7, ..., 47. It keeps the Qwen tokenizer family and GDN foundation,
but adds the model-specific pieces that make Flash-Next different:

| Component | Qwen 3.8 Flash-Next detail |
| --- | --- |
| Hyper-connections | 4 parallel 2,560-wide streams replace every RMSNorm path, including final norm |
| QSA indexer | 4 query heads plus 1 shared key head select sparse attention blocks on full-attention layers |
| PLE n-gram head | Layer 1 uses 128 bf16 shard files, 3-gram hashing, 8 heads per n-gram, and 160-wide rows |
| MoE | 512 experts, top-10 routing, 640-wide routed/shared experts, sigmoid shared gate, per-expert router scale |
| Install | The shipping install is **97 GiB** with the PLE table at int4/group-32; the earlier install is 174,403,168,940 bytes, of which 95 GiB is the PLE table at raw bf16 |

The shipping install is `models/Qwen3.8-Flash-Next-125B-ple4bit.finch` (the
app's default); `models/Qwen3.8-Flash-Next-125B.finch` is the pre-quantization
install and stays as the fallback. Both have completed the M1-M4 port path:
local repack, schema/load support, hyper-connection + QSA + PLE forward wiring,
prefill/decode smoke, llama.cpp oracle checks, and — on the quantized one —
EvalPlus HumanEval at 95.1% base / 92.1% plus. The
available oracle evidence is argmax-level rather than whole-vocab cosine-level:
tokenization is byte-exact, argmax and generation match on the checked prompts,
and top-10/top-100 logit cosine reached 0.995/0.985, while whole-vocab cosine
is 0.88 under cross-quantization. See [docs/QWEN38_PORT.md](docs/QWEN38_PORT.md)
for the full validation notes and caveats.

## The Qwen 3.6 35B-A3B port

Qwen 3.6 35B-A3B (`qwen3_5_moe`) is a 40-layer MoE where 30 layers use a
**Gated-DeltaNet linear-attention** (a fixed-size recurrent state instead of a
KV cache) and 10 use full attention, with 256 routed experts (top-8) plus a
shared expert. The MoE shape differs from Gemma 4 in scale only, so the
expert-streaming runtime transfers directly; the new compute is the GDN
linear-attention layer.

| #  | Work                                                                                          | Status |
| -- | --------------------------------------------------------------------------------------------- | ------ |
| 1  | GDN unit: fp32 CPU reference, `gdn_conv_update` + `gdn_gate` + `gdn_recurrent` + `gdn_rmsnorm_gated` + `gdn_gate_gemv` Metal kernels, wrapper, tests | done   |
| 2  | Full-attention path for the 10 `F` layers (partial RoPE, output gate, chunked prefill) | done |
| 3  | MoE: 256-expert routing and streamed execution (top-8, silu experts, shared expert 512, sigmoid gate) — decode and prefill | done |
| 4  | Embedding + untied `lm_head` (vocab 248320), sampling, stop on 248044 | done |
| 5  | Repack writer: bf16 shards → `.finch` (int8 linear-attention + int4 affine experts + int8 router), Qwen manifest, SHA-256s | done |
| 6  | `ArchConfig` preset for Qwen3.6-35B-A3B; wire `fullAttentionLayerMask` and the GDN dims        | done   |
| 7  | End-to-end: load → prefill → decode → sample; coherent generation vs the bf16 reference      | done   |

Phase 1 was the gate: nothing in the engine exercised a linear-attention state
before it, and the recurrence order (decay → read → update → read-out) is the
part most likely to be subtly wrong. Kernels are validated against the Swift
fp32 reference before any layer is wired in. Full details, the locked GDN
math, and the target-model spec live in
[docs/QWEN36_PORT.md](docs/QWEN36_PORT.md).

## Abliterated (uncensored) installs

Both Qwen models also exist here in an **abliterated** variant: the refusal
direction removed from the weights *upstream*, not by prompting and not by
anything the runtime does. They are repacked by the same `FinchMoERepack`
pipeline with the same quantization settings as the base installs, so every
engine path is identical and the weights are the only variable.

| | Qwen 3.6 35B-A3B | Qwen 3.8 Flash-Next 125B |
| --- | --- | --- |
| Hugging Face | [finchmoe-4bit-abliterated](https://huggingface.co/haihengh/Qwen3.6-35B-A3B-finchmoe-4bit-abliterated) | [finchmoe-4bit-ple4bit-abliterated](https://huggingface.co/haihengh/Qwen3.8-Flash-Next-125B-finchmoe-4bit-ple4bit-abliterated) |
| Install size | 18.7 GiB | 96.9 GiB |
| Pinned revision | `e1998dcb` | `d19beab2` |
| In-app download | yes | yes |
| Upstream abliteration | [`huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated) | [`windowsxp811203/Qwen3.8-Flash-Next-Abliterated`](https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-Abliterated) |
| Licence | Apache-2.0 | Qwen Community License 1.0 |

Both are published as finished `.finch` installs, so the app downloads them
directly — pick the preset and choose **Download**. The Qwen 3.8 *base* install
is published the same way at
[`finchmoe-4bit-ple4bit`](https://huggingface.co/haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple4bit);
the "Upstream abliteration" row names the checkpoint each was repacked *from*,
which is provenance rather than a download source.

The app's Preset picker lists all four Qwen entries, with each abliterated entry
directly after the base entry it is a variant of. When a checkout holds an
abliterated install, that install is the one the app starts on.

### Measured quality

EvalPlus HumanEval, greedy, 164 problems, the frozen server protocol — same
harness, same 768-token cap, context 4096 — measured 2026-09-27 on the 16 GB
mini. Each pair differs **only** in the weights: same engine, same quantization,
same context, so the delta is attributable to the abliteration.

| Install | HumanEval base pass@1 | HumanEval+ pass@1 |
| --- | ---: | ---: |
| Qwen 3.6 35B-A3B (base weights) | 90.9% (149/164) | 87.8% (144/164) |
| **Qwen 3.6 35B-A3B abliterated** | **93.3% (153/164)** | **90.2% (148/164)** |
| Qwen 3.8 Flash-Next 125B (base weights) | 94.5% (155/164) | 92.1% (151/164) |
| **Qwen 3.8 Flash-Next 125B abliterated** | **94.5% (155/164)** | **92.7% (152/164)** |

**Do not read the 3.6 gain as abliteration improving coding.** The binomial
standard error at p≈0.91, n=164 is 2.2 points, so +2.4 pt is about one standard
error — it is noise, and problems moved in both directions (gained 95, 99, 113,
124, 147, 160; lost 54, 116). The defensible claim, and the one worth having, is
**no measurable regression** — which is what an abliterated derivative needs to
demonstrate and is not obvious in advance, since abliteration rewrites
residual-writing tensors.

Read the residual failures with the cap in mind. Of 3.6 abliterated's 11
failures, 4 are `length`-capped by the harness budget rather than genuinely
wrong (HumanEval/116, 129, 130, 132); of 3.8 abliterated's 9, **7 are capped**
(32, 93, 113, 116, 129, 130, 132) and only 2 are genuine (145, 163). A stable
core — HumanEval/32, 93, 129, 130, 132, 145, 163 — fails on the base weights too,
so it is model difficulty, not abliteration.

**Refusal behaviour itself is not measured here.** EvalPlus says nothing about
what a model will or will not refuse, and no cell in this project probes it. What
the numbers above establish is that general capability survived; they are not a
claim about what these models decline to do.

### Measured throughput

The same pairs on the prompt suite — 16 GB M6 Mac mini, 2026-09-30, plus the
3.6 pair on an 8 GB M1 Mac mini, 2026-10-01 (the 3.8 install does not fit 8 GB),
the protocol in [Model performance comparison](#model-performance-comparison),
each cell ordered `short-explanation / medium-review / long-synthesis`:

| Install | Prefill tok/s | Decode tok/s | Expert reads/token |
| --- | ---: | ---: | ---: |
| Qwen 3.6 35B-A3B (base weights) | 17.8 / 62.7 / 59.3 | 8.46 / 8.57 / 7.74 | 222 / 233 / 243 MB |
| **Qwen 3.6 35B-A3B abliterated** | 17.4 / 61.3 / 59.6 | 11.13 / 8.48 / 8.00 | 216 / 229 / 251 MB |
| Qwen 3.8 Flash-Next 125B (base weights) | 7.6 / 24.9 / 23.2 | 3.21 / 2.67 / 2.43 | 603 / 672 / 727 MB |
| **Qwen 3.8 Flash-Next 125B abliterated** | 7.6 / 25.7 / 23.3 | 3.17 / 2.64 / 2.49 | 605 / 703 / 729 MB |
| Qwen 3.6 35B-A3B (base weights), 8 GB M1 | 11.9 / 24.2 / 22.6 | 5.19 / 4.81 / 4.17 | 221 / 234 / 251 MB |
| **Qwen 3.6 35B-A3B abliterated, 8 GB M1** | 11.9 / 24.0 / 22.3 | 4.70 / 4.36 / 3.89 | 231 / 238 / 253 MB |

Abliteration rewrites a handful of tensors and nothing else — same shapes, same
quantization, same routing — so each pair lands within run-to-run noise of the
other, and the gap between the two models stays where the base installs put it.
The exception is 3.6 on short-explanation, 11.13 against 8.46 tok/s: that is the
shortest and noisiest case, at a single sample per install.

### How the app tells them apart

An abliterated install and its base twin carry the **same
`sourceSnapshotHash`** — that hash covers the tensor index (names, shapes,
layout), which abliteration does not change, so both 3.6 installs hash to
`41b93561…` and both 3.8 installs to `99e81524…`. Identification therefore keys
on `model_weights.bin`'s SHA-256, which does differ
(`9644b61a…` vs `f6862341…` for 3.6, `c522877f…` vs `6af82b55…` for 3.8), with
the snapshot hash retained as the fallback for unrecognised installs. See
`AppModelInstallDescriptor.weightsSHA256`.

## Using FinchMoE

FinchMoE provides a native Mac app, a command-line interface, and an
experimental loopback OpenAI-compatible server. They share the same
`.finch` model directory, but only one model-owning product should run at a
time.

The Swift package exposes six products:

| Product | Purpose |
| --- | --- |
| `FinchMoE` | Swift library containing the runtime and Metal kernels |
| `FinchMoEMac` | Native Mac app for installation and generation |
| `FinchMoEDecodeService` | One-shot local model and Metal owner used by the Mac app |
| `FinchMoECLI` | Command-line instruction chat and raw completion |
| `FinchMoEServer` | Loopback OpenAI-compatible Chat Completions server |
| `FinchMoERepack` | Streaming model installer and install verifier |

### Requirements

- An Apple Silicon Mac; validated on a 16 GB Mac mini (the ~20 GB Qwen
  install streams out of core; the 8 GB M2 MacBook Air target applied to the
  upstream Gemma 4-bit install). An 8 GB M1 Mac mini runs the Qwen 3.6
  install at about half the M6 mini's decode (measured 2026-10-01); the
  97 GB Qwen 3.8 install thrashes in swap on 8 GB and needs a 16 GB host
- macOS 26 with Metal 4
- Xcode 26 and Swift 6.2 or newer
- Enough free storage for the model installation
- An internet connection for the first model install (or a local checkpoint)

The package is arm64-only. Older macOS and Metal versions are not supported.

### Mac app

Clone the repository, then run the app from its root:

```bash
swift build -c release
.build/release/FinchMoEMac
```

Build the complete package so the app and its sibling decode service are both
available. When launched from this checkout the app resolves its model from
`models/`, preferring the Qwen 3.6 family over 3.8 (a 125B needs far more memory
to run) and, within a family, the abliterated install over its base; with no
`.finch` install present it falls back to `scratch/gemma4.finch`.

The **Preset** picker in the inspector lists every entry — Gemma, both Qwen
families in base and abliterated form, and **Local directory** for an install
elsewhere on disk. Selecting a preset while a model is loaded is ignored; unload
first.

#### Install the model

On first launch with no install present, the app checks available storage and
shows the download and installed sizes for the selected checkpoint. Choose
**Download** to begin.

There are two remote install routes, and the preset decides which one runs.

- **Repack from an upstream checkpoint** — Gemma. The installer never
  materializes the full source checkpoint: it streams the required byte ranges
  from the pinned Hugging Face revision and repacks them directly into the
  `.finch` layout as they arrive, which avoids a second full checkpoint on disk
  and keeps scratch memory bounded.
- **Fetch a published `.finch`** — Qwen 3.6 and 3.8, base and abliterated. Here
  the layout already exists on the remote, so there is nothing to repack: the
  download is exactly the files the remote `manifest.json` lists, each verified
  against the digest that named it, and the receipt is written locally for the
  path it was installed to. An interrupted download resumes when the checkpoint
  still describes the same repository, commit and manifest, and is refused
  rather than half-mixed when it does not.

Qwen 3.6 base has no published `.finch` distribution, so it stays local-only:
build it with `FinchMoERepack` and the app will load it, but its Download button
remains disabled.

The completed installation is accepted only after its manifest and every file
hash validate.

#### Load and generate

1. Choose **Load Model**.
2. Enter a message in the composer.
3. Choose send, or press <kbd>Return</kbd>; <kbd>Shift</kbd>+<kbd>Return</kbd>
   adds a line. Use **Settings > Send Message With** to send with
   <kbd>Command</kbd>+<kbd>Return</kbd> instead.
4. Use the stop button or <kbd>Escape</kbd> to end generation early.

The status bar shows generation progress, decode speed, and memory use. Use the
right pane to configure sampling, context length, expert-cache slots, and
runtime options. See [Runtime controls](docs/RUNTIME_CONTROLS.md) for details
and defaults.

#### Chats

The window is a chat client: the left column lists every conversation, the
middle column is the transcript, and either column can be collapsed.

- Each session keeps its own history. Earlier turns of the open chat are
  replayed with the next message, so follow-up questions can refer to what was
  already said. Finished answers are rendered as markdown — headings, lists,
  quotes, code panels and tables, plus the charts and box-drawn tables a model
  draws out of characters, which are kept line for line in a monospace panel.
  A reply that is still streaming stays plain text, and a stopped or failed
  reply keeps the partial text it reached.
- **New Chat** (<kbd>Command</kbd>+<kbd>N</kbd>) starts a fresh conversation,
  **Regenerate Reply** (<kbd>Command</kbd>+<kbd>R</kbd>) asks the last question
  again, and **Clear Chat** empties the open conversation without deleting it.
  Rename, copy or delete a chat from its context menu in the sidebar.
- History is stored in
  `~/Library/Application Support/FinchMoE/chat-sessions.json` and reloaded at
  launch. A conversation longer than the context window is trimmed from the
  oldest exchange forward, so the newest question always fits.
- The composer's **Prompt tips** popover covers what this model answers well.

### Command-line interface

The CLI uses an existing `.finch` installation. If you installed through the
Mac app it is already at `scratch/gemma4.finch`; otherwise install it from
the command line:

```bash
swift run -c release FinchMoERepack \
  --output scratch/gemma4.finch \
  --overwrite
```

Continue a cancelled or interrupted download, or remove saved download state:

```bash
swift run -c release FinchMoERepack \
  --output scratch/gemma4.finch \
  --overwrite \
  --resume

swift run -c release FinchMoERepack \
  --discard-partial \
  --output scratch/gemma4.finch
```

Download a published `.finch` install — the base and abliterated Qwen models —
without repacking anything:

```bash
swift run -c release FinchMoERepack \
  --download-finch haihengh/Qwen3.6-35B-A3B-finchmoe-4bit-abliterated \
  --output models/Qwen3.6-35B-A3B-abliterated-4bit.finch
```

`--revision <commit>` pins a revision other than `main` and `--concurrency <n>`
sets how many files are fetched at once (default 6). The download set is exactly
what the remote `manifest.json` declares, each file verified against its own
digest, and the receipt is rewritten for the path installed here — the published
receipt names the uploader's directory, so it is never reused verbatim. An
interrupted download resumes automatically when the checkpoint still describes
the same repository, commit and manifest; `--discard-partial` clears the state
instead.

Verify an existing installation without loading the model:

```bash
swift run -c release FinchMoERepack \
  --verify-install \
  --input-finch scratch/gemma4.finch
```

The runtime accepts only a completed `.finch` directory with a final
`manifest.json`.

#### Instruction chat

Put chat messages in a JSON array and pass it with `--messages-file`:

```json
[
  {"role": "user", "content": "Explain why chunked prefill reduces time to first token while keeping memory bounded."}
]
```

```bash
swift run -c release FinchMoECLI \
  --model scratch/gemma4.finch \
  --messages-file messages.json
```

This formats messages the same way as the Mac app. The CLI response limit is
set with `--max-new` (default 1,024 tokens); the Mac app can generate until the
selected context window is full. Common generation options include
`--max-context`, `--temperature`, `--top-k`, `--top-p`,
`--repetition-penalty`, `--seed`, and repeatable `--stop` strings. The public
CLI uses production runtime defaults — run `FinchMoECLI --help` for the full
list. Generated text goes to standard output; timing statistics go to standard
error, with `--quiet` to suppress the footer.

Model files are verified with `--verify auto` by default. When a repack
receipt (`verified-install.json`) is present and valid it is used: the CLI
still hashes `manifest.json`, `model_weights.bin` and `packed_experts/layout.json`
at load, but size-checks the layer and PLE files against the receipt instead of
hashing them on first use. Without a usable receipt it falls back to hashing
everything, so the default never verifies less than `--verify full-sha256`
would. On the 125B Qwen install that is prefill **63.5 s → 4.6 s** for a
19-token prompt, with identical output.

`--verify full-sha256` forces the full hash and `--verify trusted-install`
requires the receipt instead of falling back (it fails when there is none).
The Mac app exposes the same three modes as a picker, and its diagnostics pane
reports which one the load actually took.

### Local OpenAI-compatible server

Build the server and point it at an installed model:

```bash
swift build -c release --product FinchMoEServer
.build/release/FinchMoEServer \
  --model scratch/gemma4.finch
```

It listens on `http://127.0.0.1:8080/v1` and supports Chat Completions,
streaming, function tools, and single-prefix prompt reuse. The client must
authorize and run every tool call. Keep the server on loopback; it has no
remote authentication or TLS. See
[Local server](docs/OPENAI_SERVER.md) for a test request, setup, and the
supported API subset.

## How the inference engine works

At each transformer layer, Metal computes attention and the router from
resident weights. The CPU uses the router's top-8 expert IDs to plan against
the layer's 16-slot LFU cache, then fills misses with bounded parallel `pread`
calls into Metal-visible buffers. Metal computes the resident shared-expert
branch while those reads run, then combines the shared and routed outputs.

Prompt prefill uses chunks of up to 128 tokens so one fetched expert can serve
multiple rows. Generation repeats the routed layer loop one token at a time.
The installer applies the same bounded-memory rule: it repacks remote ranges
directly into `.finch` without staging a full shard or tensor.

The Qwen 3.6 GDN layers replace the KV cache on 30 of the 40 layers with a
per-value-head recurrent state (a 2 MiB device buffer per layer, updated each
decode step). The decode path for both Qwen layer types —
`Sources/FinchMoE/Metal/LinearAttn/gdn.metal` plus
`Sources/FinchMoE/Metal/Qwen/qwen.metal`, dispatched from
`RealForwardRunner` by `modelFamily` — is wired in and reference-checked
against the `qwen3_5_moe` Transformers source.

[System design](docs/SYSTEM_DESIGN.md) explains the `.finch` layout, memory
ownership, prefill, router handoff, the `cb1`/`io`/`cb2` phases, the Metal
kernels, and the correctness invariants.

## Status and scope

FinchMoE currently includes:

- Remote streaming repack into the `.finch` model format
- The Qwen 3.6 35B-A3B `.finch` install as the working reference model
  (the upstream Gemma 4 26B-A4B path stays intact and runnable)
- 4-bit MLX affine embedding, attention, shared-expert, and routed-expert
  weights, with an 8-bit router
- Custom Metal kernels for quantized GEMV, attention, MoE, normalization,
  RoPE, sampling, and production fusions
- The Qwen 3.6 decode-layer path — Gated-DeltaNet layers, full-attention
  layers with the output gate, the Qwen MoE tail, and the untied `lm_head` —
  wired into the forward pass and reference-checked against `qwen3_5_moe`
- The Qwen 3.8 Flash-Next layer path — hyper-connections in place of RMSNorm,
  QSA sparse-block attention on the full-attention layers, and the PLE n-gram
  head — behind `modelFamily == "qwen3_8"`
- GPU timers over prefill and decode (`gpu_*` counters per stage), per-token
  I/O and expert-read counters, and opt-in diagnostic instruments: a
  fourteen-stage row-hash map, a GDN sub-stage split (`FQ_GDN_SPLIT=1`), and a
  prefill attention determinism fuzz. [Runtime controls](docs/RUNTIME_CONTROLS.md)
  lists them all
- SSD-backed routed-expert streaming with a bounded expert cache
- A Swift library, streaming installer, command-line interface, loopback
  OpenAI-compatible server, and native SwiftUI/AppKit Mac app with a one-shot
  local decode service

The target is text-only Qwen inference on Apple Silicon Macs with at least 8 GB
of RAM — in practice the two shipped tiers: Qwen 3.6 35B-A3B (the reference
install, ~19 GB on disk) and Qwen 3.8 Flash-Next 125B (~97 GB, streamed). The
Qwen 3.6 vision path is out of scope — this port targets the `text_config`
only, consistent with the engine being text-only.

### Future work

- Close the gaps the 2026-09-07 readiness review left open: an aggregate
  numeric-fidelity measurement (perplexity-style) and a throughput comparison
  against reference engines on the same Qwen installs; and broaden the
  validated envelope beyond the 16 GB loopback single-model setup. Quality
  itself is now measured on both installs (EvalPlus, above) rather than
  inferred.
- Long prefills used to be non-reproducible run to run: the indexer's prefill
  store wrote every key at twice its position, and past half the context
  length past the end of its own buffer. That store is fixed and pinned by a
  test, and the configuration that diverged every time now reproduces
  bit-for-bit. A second, **intermittent** divergence remains in the dense
  attention — unrelated to that store, seen once in five pairs at the same
  configuration, as one row of layer 3's attention output — and is
  instrumented but not yet explained. Recorded with
  the streaming path's remaining read deficit in
  [the optimization plan](docs/OPTIMIZATION_PLAN.md).
- Build iPhone and iPad apps, then measure inference speed and memory on
  mobile hardware.

## Experiments and technical documentation

The [experiments that shaped the upstream engine](docs/OPTIMIZATION_JOURNEY.md)
and the detailed
[experiment record](docs/experiments/EXPERIMENT_INVENTORY.md) document the
kernel, caching, I/O, prefill, and decode measurements the fork started from.

Useful entry points:

- [Qwen 3.6 35B-A3B port](docs/QWEN36_PORT.md)
- [Local OpenAI-compatible server](docs/OPENAI_SERVER.md)
- [System design](docs/SYSTEM_DESIGN.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [The experiments that shaped the engine](docs/OPTIMIZATION_JOURNEY.md)
- [Experiment inventory and summaries](docs/experiments/EXPERIMENT_INVENTORY.md)
- [Implementation references](docs/IMPLEMENTATION_REFERENCES.md)

## License and model terms

FinchMoE's source and documentation are licensed under the
[Apache License 2.0](LICENSE).

Model weights are not included. The installer downloads them separately from
the pinned checkpoint, and the weights remain governed by their source terms.

Repack-made `.finch` installs are published on Hugging Face and carry the terms
of the checkpoint they were repacked from, which differ between the two families:

| Install | Licence |
| --- | --- |
| Qwen 3.6 35B-A3B, base and abliterated | Apache-2.0 |
| Qwen 3.8 Flash-Next 125B, base and abliterated | Qwen Community License 1.0 |

Abliteration does not change a model's licence — it is a derivative work and
inherits the terms of the checkpoint it was derived from, along with the
copyright and permission notices those terms require.

## Credits

The Swift + Metal runtime is a fork of
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare) by
**Andrey Mikhaylov** (an iOS and Metal engineer), whose out-of-core MoE
streaming, Metal kernel conventions, and test harness this project builds on.
The dedication from the original project, reproduced with thanks:

> I dedicate this project to my wife, Sasha, the most supportive person I
> know. She stands by me even through the hardest times. She loves wildlife,
> goes birdwatching, and volunteers with our local birding community. Because
> of her, I have also grown closer to birds and nature.
>
> TurboFieldfare is named after the fieldfare, a member of the thrush family
> and my favourite bird. It is not the most noticeable or brightly coloured
> bird, but it definitely has a character and unique features of its own. I
> think the same is true of this project: it may not be the most practical,
> but I built it with my favourite tools, especially Metal, in my favourite
> field, on-device ML inference.

FinchMoE is not affiliated with, sponsored by, or endorsed by Google or
Alibaba.
