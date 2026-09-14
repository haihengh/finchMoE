<p align="center">
  <img src="docs/assets/finchmoe-logo.jpg" alt="FinchMoE logo" width="280">
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
- The pristine upstream TurboFieldfare source is archived in `reference/`
  (gitignored, alongside `models/`).

The Qwen 3.6 port is documented end-to-end — target model, locked GDN math,
and the phase plan — in [docs/QWEN36_PORT.md](docs/QWEN36_PORT.md).

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

Measured 2026-09-05 on a 16 GB Apple Silicon Mac mini (macOS 26, Metal 4)
with the engine's release CLI, greedy decode, on the local Qwen install
(page cache warm). Reproduced 2026-09-08 on a 24 GiB Apple M4 Pro
(macOS 26.6.2) against a freshly repacked install from the public
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) bf16
checkpoint.

| Metric   | Qwen 3.6 35B-A3B (`qwen3_5_moe`) install                                                                                                |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Model    | 35B total parameters, ~3B active per token; 30 Gated-DeltaNet linear-attention layers + 10 full-attention; MoE 256 experts top-8 + shared |
| Weights  | GDN projections int8; router int8; shared/routed experts affine 4-bit group 64; fp16 activations, fp32 Metal accumulators                 |
| Storage  | ~20.0 GB installed text-only `.finch` (streamed from disk during decode)                                                           |
| Memory   | ~1.1-1.2 GiB peak resident while decoding (out-of-core expert streaming; OS page cache additional)                                            |
| Decode   | ~10.5 tok/s (16 GB Mac mini) / ~17-19 tok/s (24 GiB M4 Pro), greedy, flat over 100-300 tokens                                                                                             |
| Prefill  | ~20 tok/s on long prompts (705 tok, Mac mini) / ~44 tok/s (1,020 tok, M4 Pro); short prompts skip the SHA-256 pass when a verified-install receipt is present (`--verify auto`, the default)            |
| Hardware | Apple Silicon Mac (validated on 16 GB and 24 GiB RAM)                                                                                                |
| Platform | macOS 26, Metal 4, Swift 6.3                                                                                                              |

Prompt length, generated length, page-cache state, and hardware all affect
throughput. See [benchmarks](docs/BENCHMARKS.md) for the upstream Gemma
measurements the fork started from.

## Model performance comparison

Measured 2026-09-14 on a 24 GB Apple M4 Pro MacBook Pro (`Mac16,7`, macOS
26.6.2, Swift 6.2.4), using the release `FinchMoECLI`, verified local
`.finch` installs, app sampling defaults for the prompt-suite rows
(`temperature 0.2`, Top-K 64, Top-P 0.95), and a 128-token generation cap.
Decode rates exclude model load and prompt prefill. Prefill rates are reported
separately because long prompts exercise a different path than token-by-token
decode.

| Model | Install | HumanEval base pass@1 | HumanEval+ pass@1 |
| --- | ---: | ---: | ---: |
| Qwen 3.6 35B-A3B | ~19 GB `.finch` | 90.9% (149/164) | 87.8% (144/164) |
| Qwen 3.8 Flash-Next 125B | ~167 GB `.finch` | 94.5% (155/164) | 92.1% (151/164) |

| Prompt-suite case | Model | Prompt tokens | Prefill | Prefill tok/s | Decode | Decode tok/s | Expert reads/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-explanation | Qwen 3.6 35B-A3B | 62 | 2.29s | 27.1 | 5.42s | 23.61 | 214.7 MB |
| short-explanation | Qwen 3.8 Flash-Next 125B | 62 | 5.81s | 10.7 | 21.04s | 6.08 | 607.1 MB |
| medium-review | Qwen 3.6 35B-A3B | 426 | 10.48s | 40.7 | 5.46s | 23.46 | 236.0 MB |
| medium-review | Qwen 3.8 Flash-Next 125B | 426 | 25.81s | 16.5 | 23.21s | 5.52 | 695.1 MB |
| long-synthesis | Qwen 3.6 35B-A3B | 2,940 | 81.44s | 36.1 | 6.64s | 19.28 | 243.5 MB |
| long-synthesis | Qwen 3.8 Flash-Next 125B | 2,940 | 198.33s | 14.8 | 26.01s | 4.92 | 719.3 MB |

In this apples-to-apples local run, Qwen 3.6 decodes about **3.9-4.3x faster**
than Qwen 3.8 across the 128-token prompt suite, while Qwen 3.8 scores **+3.7
points** on HumanEval base and **+4.3 points** on HumanEval+. The dominant
runtime difference is routed-expert I/O: the 125B install reads roughly
607-719 MB of expert data per generated token here, versus 215-244 MB/token for
the 35B install.

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
available. When launched from this checkout, the app prefers the repack-made
Qwen 3.6 install at `models/Qwen3.6-35B-A3B-4bit.finch` when it is present;
otherwise it targets `scratch/gemma4.finch`.

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
2. Enter a prompt in the composer.
3. Choose **Generate**, or press <kbd>Command</kbd>+<kbd>Return</kbd>. Use
   **Settings > Send Message With** to choose Return or Command-Return.
4. Use the stop button or <kbd>Escape</kbd> to end generation early.

The status bar shows generation progress, decode speed, and memory use. Use the
right pane to configure sampling, context length, expert-cache slots, and
runtime options. See [Runtime controls](docs/RUNTIME_CONTROLS.md) for details
and defaults.

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
- SSD-backed routed-expert streaming with a bounded expert cache
- A Swift library, streaming installer, command-line interface, loopback
  OpenAI-compatible server, and native SwiftUI/AppKit Mac app with a one-shot
  local decode service

The target is text-only Qwen 3.6 35B-A3B inference on Apple Silicon Macs with
at least 8 GB of RAM. The Qwen 3.6 vision path is out of scope — this port
targets the `text_config` only, consistent with the engine being text-only.

### Future work

- Close the gaps the 2026-09-07 readiness review left open: an aggregate
  numeric-fidelity measurement (perplexity-style) and a speed comparison
  against reference engines; and broaden the validated envelope beyond the
  16 GB loopback single-model setup.
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
