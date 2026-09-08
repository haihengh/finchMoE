<p align="center">
  <img src="icon.jpg" alt="robot Fhinc" width="280">
</p>

<h1 align="center">FinchMoE</h1>

<p align="center">
  <strong>Out-of-core MoE inference on Apple Silicon — running Qwen 3.6 35B-A3B</strong><br>
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
  <a href="#qwen-36-35b-a3b-port">Qwen 3.6 port</a> ·
  <a href="docs/OPENAI_SERVER.md">Local server</a> ·
  <a href="#at-a-glance">Benchmarks</a> ·
  <a href="docs/SYSTEM_DESIGN.md">How it works</a> ·
  <a href="docs/IMPLEMENTATION_REFERENCES.md">References</a>
</p>

## What this is

FinchMoE's first attempt streamed weights for **Qwen3.5-397B-A17B** off SSD
using a hand-written Objective-C/Metal engine (`archive/finchmoe/`). It hit
every memory and throughput goal, but engine quality never got past ~12%
pass@1 on HumanEval and stopped improving — the wrong direction to invest
further in.

The current engine is a full rebuild: a from-scratch Swift + Metal out-of-core
MoE runtime that streams routed experts from SSD through a bounded LFU cache,
keeping only the shared core and KV/recurrent state resident. That design
produced ~90% HumanEval on a 26B-A4B Gemma-family model, so it was retargeted
to run **Qwen 3.6 35B-A3B** (`model_type: qwen3_5_moe`) — the model this repo
is built and validated against today. The legacy engine is kept for reference
under [archive/](archive/); it is no longer developed.

## Goal

- **Phase 1 (current, done):** Qwen 3.6 35B-A3B — GDN linear-attention port,
  end-to-end quality, and benchmarks. See [Status and scope](#status-and-scope).
- **Phase 2:** Qwen 3.8 Flash Next implementation.
- **Phase 3:** DeepSeek V4 Flash implementation.
- **Phase 4:** GLM 5.3 Flash implementation.

Each later phase targets a different model family on the same out-of-core
runtime: add the family's `ArchConfig` preset, port whatever attention/MoE
variant it uses (GDN was Qwen 3.6's), and repack a `.finch` install —
the streaming, caching, and Metal execution core stays shared. None of this
is scheduled yet; Phase 1 is the only phase with a doc, plan, and working
install.

## Current state

The Qwen 3.6 35B-A3B port (Phase 1) is complete and closed: the GDN
linear-attention layers, the 10 full-attention layers, the Qwen MoE tail, the
bf16 → `.finch` quantizing repack, and the interface surface (CLI, Mac
app, OpenAI-compatible server) all load and run against the local Qwen
install (see [At a glance](#at-a-glance) for the measured numbers). The
Gemma 4 26B-A4B path this engine was built on stays intact and runnable from
the same binary. Phases 2-4 have no work started yet.

- The Qwen 3.6 port is documented end-to-end — target model, locked GDN math,
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
> disk, streamed out-of-core). The Gemma 4 26B-A4B path is intact; family
> dispatch keeps both runnable from the same binary.

## At a glance

Measured 2026-09-05 on a 16 GB Apple Silicon Mac mini (macOS 26, Metal 4)
with the engine's release CLI, greedy decode, on the local Qwen install
(page cache warm).

| Metric   | Qwen 3.6 35B-A3B (`qwen3_5_moe`) install                                                                                                |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Model    | 35B total parameters, ~3B active per token; 30 Gated-DeltaNet linear-attention layers + 10 full-attention; MoE 256 experts top-8 + shared |
| Weights  | GDN projections int8; router int8; shared/routed experts affine 4-bit group 64; fp16 activations, fp32 Metal accumulators                 |
| Storage  | ~20.0 GB installed text-only `.finch` (streamed from disk during decode)                                                           |
| Memory   | ~1.1 GiB peak resident while decoding (out-of-core expert streaming; OS page cache additional)                                            |
| Decode   | ~10.5 tok/s greedy, flat over 200–300 tokens                                                                                             |
| Prefill  | ~20 tok/s on long prompts (705 tok); short prompts pay SHA-256 verification unless `--verify trusted-install` (~1 s vs ~8 s)            |
| Hardware | Apple Silicon Mac (validated on 16 GB RAM)                                                                                                |
| Platform | macOS 26, Metal 4, Swift 6.3                                                                                                              |

Prompt length, generated length, page-cache state, and hardware all affect
throughput.

## The Qwen 3.6 35B-A3B port

Qwen 3.6 35B-A3B (`qwen3_5_moe`) is a 40-layer MoE where 30 layers use a
**Gated-DeltaNet linear-attention** (a fixed-size recurrent state instead of a
KV cache) and 10 use full attention, with 256 routed experts (top-8) plus a
shared expert. The MoE shape differs from Gemma 4 in scale only, so the
expert-streaming runtime transfers directly; the new compute is the GDN
linear-attention layer.

| # | Work                                                                                                                                                          | Status |
| - | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 1 | GDN unit: fp32 CPU reference,`gdn_conv_update` + `gdn_gate` + `gdn_recurrent` + `gdn_rmsnorm_gated` + `gdn_gate_gemv` Metal kernels, wrapper, tests | done   |
| 2 | Full-attention path for the 10 `F` layers (partial RoPE, output gate, chunked prefill)                                                                      | done   |
| 3 | MoE: 256-expert routing and streamed execution (top-8, silu experts, shared expert 512, sigmoid gate) — decode and prefill                                   | done   |
| 4 | Embedding + untied `lm_head` (vocab 248320), sampling, stop on 248044                                                                                       | done   |
| 5 | Repack writer: bf16 shards →`.finch` (int8 linear-attention + int4 affine experts + int8 router), Qwen manifest, SHA-256s                             | done   |
| 6 | `ArchConfig` preset for Qwen3.6-35B-A3B; wire `fullAttentionLayerMask` and the GDN dims                                                                   | done   |
| 7 | End-to-end: load → prefill → decode → sample; coherent generation vs the bf16 reference                                                                    | done   |

Phase 1 was the gate: nothing in the engine exercised a linear-attention state
before it, and the recurrence order (decay → read → update → read-out) is the
part most likely to be subtly wrong. Kernels are validated against the Swift
fp32 reference before any layer is wired in. Full details, the locked GDN
math, and the target-model spec live in
[docs/QWEN36_PORT.md](docs/QWEN36_PORT.md).

## Using FinchMoE

### Requirements

- An Apple Silicon Mac; validated on a 16 GB Mac mini (the ~20 GB Qwen
  install streams out of core; the 8 GB M2 MacBook Air target applied to the
  Gemma 4-bit install)
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

Model files are verified with `--verify full-sha256` by default: the CLI
hashes `model_weights.bin` at load and every `packed_experts` layer file on
first use (~8 s one-time cost for a large install). Pass
`--verify trusted-install` to trust the repack receipt
(`verified-install.json`) and size-check instead — on a ~20 GB Qwen install
this cuts the fixed per-run cost before the first token from ~8 s to under a
second. Same choice the Mac app exposes as its verification setting.

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
  (the Gemma 4 26B-A4B path stays intact and runnable)
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

- [System design](docs/SYSTEM_DESIGN.md) — `.finch` layout, memory ownership, prefill/decode phases, Metal kernels
- [Qwen 3.6 port](docs/QWEN36_PORT.md) — the GDN math, port plan, and status
- [Runtime controls](docs/RUNTIME_CONTROLS.md) — sampling, context length, expert-cache, and I/O knobs
- [Local OpenAI-compatible server](docs/OPENAI_SERVER.md)
- [Optimization journey](docs/OPTIMIZATION_JOURNEY.md) — experiments that shaped the runtime
- [Experiment inventory](docs/experiments/EXPERIMENT_INVENTORY.md) — the full measured record
- [Implementation references](docs/IMPLEMENTATION_REFERENCES.md)

## License and model terms

FinchMoE's source and documentation are licensed under the
[Apache License 2.0](LICENSE).

Model weights are not included. The installer downloads them separately from
the pinned checkpoint, and the weights remain governed by their source terms.
