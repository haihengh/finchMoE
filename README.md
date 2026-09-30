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

Measured 2026-09-18 on a 16 GB Apple Silicon Mac mini (macOS 26, Metal 4) with
the engine's release CLI, greedy decode, on the local Qwen install (page cache
warm), against the shipping `Qwen3.6-35B-A3B-4bit.finch`. Earlier rows in this
table were measured 2026-09-05 and reproduced 2026-09-08 on a 24 GiB Apple M4
Pro (macOS 26.6.2) against a freshly repacked install from the public
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) bf16
checkpoint; **the prefill figure in particular has moved since**, because the
int8 projections are now batched — the same 2,940-token prompt that took 81.4 s
on the M4 Pro in September takes **69.3 s on the 16 GB mini** today.

| Metric   | Qwen 3.6 35B-A3B (`qwen3_5_moe`) install                                                                                                |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Model    | 35B total parameters, ~3B active per token; 30 Gated-DeltaNet linear-attention layers + 10 full-attention; MoE 256 experts top-8 + shared |
| Weights  | GDN projections int8; router int8; shared/routed experts affine 4-bit group 64; fp16 activations, fp32 Metal accumulators                 |
| Storage  | ~19 GB installed text-only `.finch` (streamed from disk during decode)                                                           |
| Memory   | ~1.1-1.2 GiB peak resident while decoding (out-of-core expert streaming; OS page cache additional)                                            |
| Decode   | ~8.4 tok/s (16 GB Mac mini, 2,940-token prompt) / ~17-19 tok/s (24 GiB M4 Pro), greedy, flat over 100-300 tokens                                                                                             |
| Prefill  | **~42 tok/s** on a 2,940-token prompt (16 GB Mac mini, 69.3 s) / ~44 tok/s (1,020 tok, M4 Pro); short prompts skip the SHA-256 pass when a verified-install receipt is present (`--verify auto`, the default)            |
| Hardware | Apple Silicon Mac (validated on 16 GB and 24 GiB RAM)                                                                                                |
| Platform | macOS 26, Metal 4, Swift 6.3                                                                                                              |

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

Three hosts have prompt-suite rows: a 24 GB Apple M4 Pro (`Mac16,7`, macOS
26.6.2, Swift 6.2.4, 2026-09-14), a 16 GB M4 Mac mini (2026-09-18), and a 16 GB
M6 Mac mini (`Mac18,5`, macOS 27.0, Swift 6.4, 2026-09-30). The throughput
comparison below puts them side by side; the tables after it are the rows behind
it.

The M4 Pro run is kept because it measures both models in one sitting on one
machine, but **its 3.8 rows predate two changes that both move prefill**: the
shipping install is now the quantized-PLE one rather than the 167 GB install it
names, and the int8 projections are batched. Both mini runs are the current
engine.

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
| 24 GB M4 Pro † | Qwen 3.6 35B-A3B | 27.1 / 40.7 / 36.1 | 23.61 / 23.46 / 19.28 |
| 24 GB M4 Pro † | Qwen 3.8 Flash-Next 125B | 10.7 / 16.5 / 14.8 | 6.08 / 5.52 / 4.92 |
| 16 GB M4 Mac mini | Qwen 3.6 35B-A3B | 14.6 / 46.8 / 42.0 | 10.03 / 9.73 / 7.82 |
| 16 GB M4 Mac mini | Qwen 3.8 Flash-Next 125B | 6.6 / 20.7 / 18.9 | 3.17 / 2.95 / 2.59 |
| 16 GB M6 Mac mini | Qwen 3.6 35B-A3B | 17.8 / 62.7 / 59.3 | 8.46 / 8.57 / 7.74 |
| 16 GB M6 Mac mini | Qwen 3.8 Flash-Next 125B | 7.6 / 24.9 / 23.2 | 3.21 / 2.67 / 2.43 |

† Pre-quantized-PLE install and per-token int8 projections, so both 3.8 numbers
are not comparable to the mini rows — see the note above.

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

24 GB M4 Pro, 2026-09-14 — the run whose 3.8 rows predate the quantized-PLE
install and the batched projections:

| Prompt-suite case | Model | Prompt tokens | Prefill | Prefill tok/s | Decode | Decode tok/s | Expert reads/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-explanation | Qwen 3.6 35B-A3B | 62 | 2.29s | 27.1 | 5.42s | 23.61 | 214.7 MB |
| short-explanation | Qwen 3.8 Flash-Next 125B | 62 | 5.81s | 10.7 | 21.04s | 6.08 | 607.1 MB |
| medium-review | Qwen 3.6 35B-A3B | 426 | 10.48s | 40.7 | 5.46s | 23.46 | 236.0 MB |
| medium-review | Qwen 3.8 Flash-Next 125B | 426 | 25.81s | 16.5 | 23.21s | 5.52 | 695.1 MB |
| long-synthesis | Qwen 3.6 35B-A3B | 2,940 | 81.44s | 36.1 | 6.64s | 19.28 | 243.5 MB |
| long-synthesis | Qwen 3.8 Flash-Next 125B | 2,940 | 198.33s | 14.8 | 26.01s | 4.92 | 719.3 MB |

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

Both M4 mini 3.8 prefill figures are roughly 25% better than the September M4 Pro
table recorded for the same cases (16.5 and 14.8 tok/s), on a smaller machine —
the batched projections and the quantized-PLE install, against a much slower
host.

Qwen 3.6 decodes about **3.0-3.3x faster** than Qwen 3.8 on the minis and
**3.9-4.3x** on the M4 Pro, while Qwen 3.8 scores **+4.2 points** on HumanEval
base and **+4.3 points** on HumanEval+. The dominant runtime difference is
routed-expert I/O: the 125B install reads roughly 602-727 MB of expert data per
generated token across the three hosts, versus 215-248 MB/token for the 35B
install.

The M4 Pro decodes about **2.4-2.8x faster** than either mini on the 3.6
install, and that is not routing or cache policy: its per-token expert hit and
miss counts match the M6 mini's within about a point (short-explanation, 60.2%
hits against 58.9%; the 3.8 install agrees likewise, 52.1% against 52.5%). What
differs is how fast a miss is served, which the host's memory and storage path
decide.

The M6 mini's edge over the M4 mini is prefill, which is compute-bound: 59.3
tok/s against 42.0 on the 2,940-token 3.6 case, and 23.2 against 18.9 for 3.8.
Decode is level between them — 7.74 against 7.82, and 2.43 against 2.59. The M6
rows are one measured run per case after a discarded warmup, so read differences
of a few percent as noise; that is most visible on short-explanation, the
shortest case.

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
| Upstream abliteration | [`huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated) | [`windowsxp811203/Qwen3.8-Flash-Next-Abliterated`](https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-Abliterated) |
| Licence | Apache-2.0 | Qwen Community License 1.0 |

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

The same pairs on the prompt suite — 16 GB M6 Mac mini, 2026-09-30, the protocol
in [Model performance comparison](#model-performance-comparison), each cell
ordered `short-explanation / medium-review / long-synthesis`:

| Install | Prefill tok/s | Decode tok/s | Expert reads/token |
| --- | ---: | ---: | ---: |
| Qwen 3.6 35B-A3B (base weights) | 17.8 / 62.7 / 59.3 | 8.46 / 8.57 / 7.74 | 222 / 233 / 243 MB |
| **Qwen 3.6 35B-A3B abliterated** | 17.4 / 61.3 / 59.6 | 11.13 / 8.48 / 8.00 | 216 / 229 / 251 MB |
| Qwen 3.8 Flash-Next 125B (base weights) | 7.6 / 24.9 / 23.2 | 3.21 / 2.67 / 2.43 | 603 / 672 / 727 MB |
| **Qwen 3.8 Flash-Next 125B abliterated** | 7.6 / 25.7 / 23.3 | 3.17 / 2.64 / 2.49 | 605 / 703 / 729 MB |

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
  upstream Gemma 4-bit install)
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
shows the download and installed sizes for the Gemma checkpoint. Choose
**Download** to begin. (Qwen installs are made by `FinchMoERepack` and are
only ever *loaded* by the app — in-app download is Gemma-only.)

The installer never materializes the full source checkpoint. It streams the
required byte ranges from the pinned Hugging Face revision and repacks them
directly into the `.finch` layout as they arrive, which avoids a second full
checkpoint on disk and keeps scratch memory bounded. The completed
installation is accepted only after its manifest and file hashes validate.

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
