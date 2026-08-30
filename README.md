# QwenFieldfare

A custom **Swift + Metal** inference engine for **Qwen3-30B-A3B** (30.5B total /
3.3B active MoE), built for low-spec Apple Silicon Macs. It streams routed
expert weights from SSD through the OS page cache instead of holding them in
RAM, so it can run large models with **32K+ context** on machines with as
little as 8 GB of memory.

- 🚀 Pure Swift + Metal — **no Python, no llama.cpp, no MLX runtime** at inference time
- 💾 SSD-streamed experts via thread-safe `pread` + LFU cache + SSD prefetch hints
- 🧠 Full 48-layer causal attention with a linear KV cache (no sliding-window degradation)
- 🔌 CLI **and** OpenAI-compatible local server (`/v1/chat/completions`, `/v1/models`)
- 🍎 Apple Silicon only (arm64), macOS 15+, Metal 3+

---

## Architecture

QwenFieldfare converts the MLX 4-bit checkpoint into a custom `.qturbo`
container that separates **resident** weights (always in RAM) from **routed
experts** (streamed on demand):

```
model.qturbo/
├── manifest.json            # model config + tensor index + expert layout
├── model_weights.bin        # resident tensors (magic "QTURBO1\0" header)
└── packed_experts/
    ├── layer_00.bin         # 128 experts × (gate/up/down w+scales+biases)
    ├── layer_01.bin         #   each expert page-aligned to 16 KiB
    └── … layer_47.bin
```

**Resident** (`model_weights.bin`): embed_tokens, lm_head, all layer norms, all
attention projections (q/k/v/o), all shared experts, all routers, final norm.

**Streamed** (`packed_experts/`): the routed MoE experts
`model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.{weight,scales,biases}`.
Only the 8 experts selected by the router per token per layer are read from
disk; a bounded LFU cache (default 16 slots) keeps hot experts warm, and the
next layer's experts are prefetched (`F_RDADVISE`) while the GPU computes the
current layer.

### Package layout

| Target | Kind | Purpose |
|---|---|---|
| `QwenFieldfareFormat` | lib | `.qturbo` format constants, manifest schema, tensor-layout planner |
| `QwenFieldfareRepack` | lib | Safetensors reader, HF downloader, expert packer, repack command |
| `QwenFieldfareRuntime` | lib | KV cache, expert streamer, Metal kernels, forward pass, sampler, tokenizer |
| `QwenFieldfareServer` | lib | OpenAI-compatible HTTP server on `Network.framework` |
| `qwen-fieldfare` | exe | CLI: `repack`, `run`, `serve` |
| `qwen-fieldfare-server` | exe | Standalone OpenAI server |

### Qwen3-30B-A3B configuration

| Field | Value |
|---|---|
| hidden_size | 2048 |
| head_dim | 128 |
| num_attention_heads | 32 |
| num_key_value_heads | 4 (GQA group = 8) |
| num_hidden_layers | 48 |
| num_experts / per-tok | 128 / 8 |
| moe_intermediate_size | 768 |
| vocab_size | 151936 |
| rope_theta | 1,000,000 (NeoX) |
| quantization | MLX affine 4-bit, group_size 64 |

---

## Installation

```bash
git clone <this-repo> QwenFieldfare
cd QwenFieldfare
swift build -c release
```

Binaries land in `.build/release/qwen-fieldfare` and
`.build/release/qwen-fieldfare-server`.

> Requires Xcode 16 / Swift 6 toolchain, macOS 15+, and an Apple Silicon Mac.

---

## Usage

### 1. Repack the model

Download the MLX 4-bit checkpoint from HuggingFace and convert it to `.qturbo`:

```bash
# Download + repack in one step (uses HF_TOKEN if set for gated repos)
export HF_TOKEN=hf_xxx        # optional
./.build/release/qwen-fieldfare repack \
    --source ~/qwen-src \
    --output ~/models/qwen3-30b-a3b.qturbo \
    --download

# Or repack from an already-downloaded directory of safetensors shards
./.build/release/qwen-fieldfare repack \
    --source ~/.cache/huggingface/.../snapshots/<hash> \
    --output ~/models/qwen3-30b-a3b.qturbo
```

Source repo: [`mlx-community/Qwen3-30B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3-30B-A3B-4bit)
(4 safetensors shards, ~17 GB).

### 2. Run a single prompt

```bash
./.build/release/qwen-fieldfare run \
    --model ~/models/qwen3-30b-a3b.qturbo \
    --prompt "Explain the Metal shading language in two sentences." \
    --max-tokens 256 \
    --temperature 0.7 \
    --top-p 0.9
```

Use `--raw` to skip the Qwen3 chat template, `--system "..."` to set a system
prompt, `--max-seq` to change the context window, and `--cache-slots` to size
the expert cache.

### 3. Start the OpenAI-compatible server

```bash
./.build/release/qwen-fieldfare serve \
    --model ~/models/qwen3-30b-a3b.qturbo \
    --host 127.0.0.1 --port 11434
# (identical to the standalone ./.build/release/qwen-fieldfare-server)
```

Then call it like any OpenAI endpoint:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "qwen3-30b-a3b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true,
        "temperature": 0.7,
        "max_tokens": 256
      }'
```

Supported endpoints: `POST /v1/chat/completions` (streaming SSE and
non-streaming) and `GET /v1/models`.

---

## Metal kernels

All GPU compute lives in `Sources/QwenFieldfareRuntime/Metal/Kernels.metal`:

| Kernel | Role |
|---|---|
| `gemv_int4_q64` | Affine 4-bit GEMV (group_size 64), `val = (nibble-8)*scale + bias` |
| `rms_norm` | RMSNorm with learned weight |
| `rope_neox` | NeoX RoPE (θ = 1e6), in-place on Q/K |
| `gqa_attention_causal` | Full causal GQA attention with online softmax |
| `silu_mul` | SwiGLU: `silu(gate) * up` |
| `moe_combine` | Weighted expert accumulation |
| `sample_argmax` | Greedy decode |
| `sample_top_p` | Temperature + nucleus sampling |

The forward pass (`ForwardRunner.swift`) runs attention/FFN/expert matmuls on
the GPU as int4 GEMV; the tiny router matmul and (if unquantized) the lm_head
run on the CPU.

---

## Memory budget

At 32K context the linear KV cache uses:

```
48 layers × 32768 tokens × (4 KV heads × 128 dim × 2 B) × 2 (K+V) ≈ 3.22 GB
```

Resident weights + a 16-slot expert cache keep the working set within a
~2–3 GB inference budget on 8 GB machines; routed experts are paged in from
SSD as needed.

---

## Notes & limitations

- Sampling defaults to a CPU path (vocab ≈ 152K is trivial for the CPU and
  avoids a GPU round-trip); the equivalent Metal kernels are provided for a
  fully-on-GPU path.
- The forward pass processes one token per step for both prefill and decode,
  which keeps the code uniform and correct for arbitrary context lengths.
- Expert streaming uses `pread` (not `read`) for thread-safe concurrent access,
  and `F_RDADVISE` (macOS) / `posix_fadvise` (fallback) for prefetch hints.
- Apple Silicon only. `#if arch(arm64)` guards protect the Metal device path.
