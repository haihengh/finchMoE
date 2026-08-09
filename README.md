# FinchMoE

A C/Metal inference engine for **Qwen 3.6 35B A3B** on Apple Silicon, targeting 12 tok/s on M4 and 3-5 tok/s on iPhone.

## Current Status

| Metric | Value |
|---|---|
| Output quality | ✅ Coherent — "Hello! How can I help you today?" |
| Speed (M4, 2-bit, K=2) | **8.3 tok/s** |
| Speed (M4, 2-bit, K=4) | **7.5 tok/s** |
| RAM (runtime) | ~2 GB + page cache |
| Disk (2-bit-dense, active) | **21 GB** |
| Disk (4-bit-dense) | **36 GB** |

## Model Size & Quality Tradeoff

| Model | Safe-tensors | Experts | Total | Speed (K=4) | Quality Loss |
|-------|-------------|---------|-------|-------------|-------------|
| BF16 (source) | 67 GB | — | 67 GB | — | 0% (reference) |
| 4-bit-dense (active) | 19 GB | 17 GB | **36 GB** | 7.2 tok/s | ~1-2% |
| 2-bit-dense ✅ | 11 GB | 9.4 GB | **21 GB** | 7.5 tok/s | ~5% |

### Quantization Strategy

| Component | Format | Why |
|-----------|--------|-----|
| Embeddings + lm_head | 8-bit | Vocabulary projections — near-lossless |
| Attention/GDN projections | 4-bit | Large tensors, 4-bit is sufficient |
| Shared expert | 4-bit | Always active, small dim |
| Routed experts | 4-bit or 2-bit | MoE is quantization-tolerant: 8/256 fire, errors cancel |
| Norms + routing gate | BF16 | Tiny, not worth quantizing |

### Why MoE Tolerates Aggressive Quantization

Only 8/256 experts fire per token. If quantization degrades one expert, the router weights it less. The weighted sum of 8 expert outputs smooths individual errors. This is why 2-bit MoE models often match 4-bit quality on benchmarks.

## Comparison Models

We track these reference models to benchmark our speed and quality:

| Model | Arch | Active Params | Size | tok/s (M4) | Notes |
|-------|------|--------------|------|------------|-------|
| **FinchMoE 2-bit (ours)** | MoE 35B A3B | 3B | 21 GB | 8.3 (K=2), 7.5 (K=4) | Custom C/Metal engine |
| [turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) | MoE 26B A4B | 4B | 13 GB | **10.7 tok/s** | Swift/Metal, measured on our M4 |
| [Ternary-Bonsai-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) | Dense 27B | 27B | ~7 GB | TBD | GGUF PQ2_0, downloading |
| [Bonsai-27B-1bit](https://huggingface.co/prism-ml/Bonsai-27B-mlx-1bit) | Dense 27B | 27B | 1.7 GB | TBD | MLX 1-bit, downloading |

Key insight: Dense 27B models (Bonsai) have 9× more active params per token than our MoE 3B — they trade speed for quality. MoE with SSD streaming is the right architecture for low-RAM devices.

### Measured on M4 Mac mini 16GB (Aug 2026)

| Model | Size | "Hello" (20 tok) | "Poem" (50 tok) | Quality |
|-------|------|-----------------|-----------------|---------|
| **FinchMoE 2-bit (ours)** | 21 GB | ~2 GB | TBD | 8.3 tok/s | TBD | TBD | TBD | TBD | 8MB/400MB/2GB | "You are a helpful assistant." |
| **FinchMoE 4-bit** | 36 GB | ~2.5 GB | TBD | 7.5 tok/s | TBD | TBD | TBD | TBD | 8MB/400MB/2GB | "You are a helpful assistant." |
| **turbo-fieldfare** | 13 GB | ~3 GB | TBD | **10.7 tok/s** | TBD | TBD | TBD | TBD | TBD | "The salt-crust clings..." |
| Ternary-Bonsai-27B | ~7 GB | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD (downloading) |
| Bonsai-27B-1bit | 1.7 GB | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD (downloading) |

**PP** = prompt processing (prefill tok/s), **TG** = token generation (decode tok/s).  
**KV** = KV cache at 1K/50K/256K tokens (10 full-attention layers × 2×K+V × 2 heads × 256d × 4B).  
GatedDeltaNet layers (30/40) use fixed 2.1MB state — no KV growth with context.  
All models on Samsung 990 Plus NVMe via TB4 enclosure, M4 Mac mini 16GB.

## Origin

The project started from searching for an LLM that is both capable enough for coding and fast enough on a $599 M4 Mac mini. Gemma 4 E2B was too small (low quality), Bonsai 27B was too slow (dense, all 27B params active per token). turbo-fieldfare proved the MoE + SSD streaming approach works with Gemma 4 26B A4B at 3.5 tok/s. Qwen 3.6 35B A3B was a natural fit — fewer active params (3B vs 4B) means less memory bandwidth, more speed.

[flash-moe](flash-moe/) is the starting codebase — a production C/Metal engine running Qwen3.5-MoE at 4.36 tok/s on M3 Max. It provides:

- **SSD Expert Streaming** — 4-bit expert weights streamed from NVMe on demand
- **FMA-Optimized Dequant** — fused multiply-add in Metal shaders (+12% throughput)
- **GatedDeltaNet via Accelerate BLAS** — 64% faster than scalar
- **GPU Fused Attention** — batched Q@K^T, softmax, scores@V on Metal
- **Trust the OS Page Cache** — no custom expert cache (every attempt was slower)

## Architecture

### Model: Qwen 3.6 35B A3B

```
40 layers: 30× GatedDeltaNet + 10× full attention (3:1 pattern)
Full attention at layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
```

| Parameter | Value |
|---|---|
| Hidden dim | 2048 |
| Attention heads | 16 (GQA: 16Q, 2KV) |
| Head dim | 256 |
| Vocab | 248,320 |
| Experts | 256 (top-8) + 1 shared |
| MoE intermediate | 512 |
| Max position | 262,144 |
| RoPE theta | 10,000,000 |
| Partial rotary | 0.25 |
| MRoPE | interleaved [11, 11, 10] |

### GatedDeltaNet Layer

Pure delta-rule recurrence (not Mamba/SSM). Projects input → Q/K/V/Z/A/B, runs depthwise conv1d(kernel=4), then recurrent state update.

| Component | Dimensions |
|---|---|
| in_proj_qkv | [8192, 2048] |
| in_proj_z | [4096, 2048] |
| in_proj_a / in_proj_b | [32, 2048] |
| conv1d | [8192, 4, 1] |
| A_log, dt_bias | [32] |
| norm | [128] |
| out_proj | [2048, 4096] |
| Recurrent state | [32, 128, 128] ≈ 2.1 MB |

### Full Attention Layer

Standard GQA with Q/output-gate fusion (`attn_output_gate: true`).

| Component | Dimensions |
|---|---|
| q_proj (doubled) | [4096, 2048] |
| k_proj / v_proj | [512, 2048] |
| o_proj | [2048, 2048] |

### MoE Expert (4-bit, per expert)

| Component | Shape | Packed Size |
|---|---|---|
| gate_proj | [512, 2048] INT4 | 590 KB |
| up_proj | [512, 2048] INT4 | 590 KB |
| down_proj | [2048, 512] INT4 | 590 KB |
| **Total per expert** | | **~1.69 MB** |
| **Total experts** | 256 × 40 layers | **~16.9 GB** |

Quantization: MLX affine INT4, group-64, BF16 scale+bias.

## Project Structure

```
finchMoE/
├── README.md              # This file
├── BUGS.md                # Bug documentation and lessons learned
├── design.md              # Detailed design document
├── finchmoe/              # FinchMoE inference engine (adapted from flash-moe)
│   ├── infer.m            #   Main engine (~7800 lines C/Metal, includes HTTP server)
│   ├── shaders.metal      #   Metal compute kernels (~1300 lines)
│   ├── Makefile           #   Build system
│   ├── chat.m             #   Interactive chat TUI (streaming markdown, sessions)
│   ├── tokenizer.h        #   C BPE tokenizer (248K vocab)
│   ├── linenoise.c/h      #   Line editing + history
│   ├── extract_weights.py #   Non-expert weight extraction
│   ├── repack_experts.py  #   Expert weight repacking (dtype-aware)
│   ├── generate_expert_index.py # Expert index generator
│   └── quantize_model.py  #   BF16 → MLX 4-bit quantization
├── flash-moe/             # Starting codebase (Qwen3.5-397B engine, unmodified)
├── turbo-fieldfare/       # Performance benchmark (Swift, Gemma 4)
├── omlx/                  # Qwen-specific Metal kernel reference
├── models/
│   ├── Qwen3.6-35B-A3B-bf16/              # Source model (67 GB)
│   ├── Qwen3.6-35B-A3B-2bit-dense/        # Active model (21 GB) ✅
│   ├── Qwen3.6-35B-A3B-4bit-dense/        # Balanced quality (36 GB) ✅
│   ├── Ternary-Bonsai-27B-PQ2_0.gguf      # Reference: dense 27B @ 2-bit (~7 GB)
│   └── Bonsai-27B-mlx-1bit/               # Reference: dense 27B @ 1-bit (1.7 GB)
└── archive/               # Original finchMoE code (pre-reboot)
```

## Reference Projects

| Project | What We Use It For |
|---|---|
| **flash-moe** | Starting codebase — already runs qwen3_5_moe architecture |
| **turbo-fieldfare** | Performance benchmark — 3.5 tok/s, ~2 GB RAM, M4 mini |
| **omlx** | Qwen-specific Metal kernel optimizations (GDN, FA256, MoE) |

## Status

- [x] Coherent output — "Hello! How can I help you today?"
- [x] 8.3 tok/s on M4 (2-bit-dense, K=2), 7.5 tok/s (K=4)
- [x] Model size: 21 GB (2-bit-dense), 36 GB (4-bit-dense)
- [x] GDN verified bit-identical to llama.cpp reference
- [x] Int8/4-bit/2-bit expert support with GPU kernels
- [x] Dense weight quantization (embeddings 8-bit, attention/shared 4-bit)
- [x] ChatML template with thinking mode support
- [x] HTTP server with OpenAI-compatible API
- [x] Layer-by-layer differential debug scripts
- [ ] 12-15 tok/s target (needs expert prefetch or further I/O reduction)
- [ ] iPhone port (A-series chips, ≤3 GB RAM target)

## Running the Engine

### Quick Start

If you already have a prepared model (pre-quantized or self-quantized):

```bash
cd finchmoe
make && make chat             # Build engine + chat client
./finchmoe-infer --serve 9000 # Start OpenAI-compatible API server
```

Then test it (in another terminal):

```bash
# Health check
curl http://localhost:9000/health

# Generate
curl -N -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":100,"stream":true}'
```

Or use the built-in TUI (in another terminal, while the server is running):

```bash
cd finchmoe
./chat --port 9000
```

### Build

```bash
cd finchmoe
make          # Build finchmoe-infer
make chat     # Build interactive chat TUI
```

Requires: Xcode Command Line Tools (`xcode-select --install`), macOS 14+.

### Model Preparation

**Option A: Pre-quantized model (recommended)**

Download a pre-quantized model (e.g. from mlx-community), then run the one-time preparation:

```bash
cd finchmoe
make extract MODEL_DIR=../models/Qwen3.6-35B-A3B-4bit-custom
make index MODEL_DIR=../models/Qwen3.6-35B-A3B-4bit-custom
make repack
```

**Option B: Self-quantize from BF16**

If you have the original BF16 model (~67 GB):

```bash
cd finchmoe
python3 quantize_model.py \
  --model ../models/Qwen3.6-35B-A3B-bf16 \
  --output ../models/Qwen3.6-35B-A3B-4bit-custom \
  --bits 4
make extract MODEL_DIR=../models/Qwen3.6-35B-A3B-4bit-custom
make index MODEL_DIR=../models/Qwen3.6-35B-A3B-4bit-custom
make repack
```

This produces:
- `model_weights.bin` / `model_weights.json` — non-expert weights (mmap'd at startup)
- `expert_index.json` — expert tensor layout, offsets, and shapes
- `packed_experts/layer_00.bin` … `layer_39.bin` — 4-bit expert weights per layer (~16.9 GB total)

### Basic Usage (CLI)

```bash
cd finchmoe
./finchmoe-infer --prompt "Hello world" --tokens 50
./finchmoe-infer --prompt "Write a haiku about coding." --tokens 200 --timing
```

The engine auto-detects `model_weights.bin`, `vocab.bin`, and `../models/Qwen3.6-35B-A3B-4bit-custom/packed_experts/` relative to the current directory. Override with `--model`, `--weights`, `--manifest`, or `--vocab`.

**Note:** CLI mode sends prompts as-is (base model completions). For Q&A behavior, use Server Mode which automatically wraps prompts in the Qwen chat template.

### Server Mode (OpenAI-Compatible API)

Start FinchMoE as a standalone HTTP server — works with Open WebUI, Continue.dev, Jan, LM Studio, and any OpenAI-compatible client.

```bash
cd finchmoe

# Basic server (default context: 256K max, 8K GPU-accelerated)
./finchmoe-infer --serve 9000

# Agentic workloads: 100K GPU context, full 256K window
./finchmoe-infer --serve 9000 --gpu-kv-seq 100000 --max-seq-len 262144
```

Then connect any tool to `http://localhost:9000/v1`.

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Streaming chat completions (SSE) |
| `GET` | `/v1/models` | Model list |
| `GET` | `/health` | Health check + model name |

#### Chat Completions

```bash
# Streaming (SSE)
curl -N -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}],
    "max_tokens": 200,
    "stream": true
  }'

# Multi-turn with session persistence
curl -N -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Follow-up question..."}],
    "max_tokens": 200,
    "stream": true,
    "session_id": "my-session-42"
  }'
```

Each SSE response includes a `usage` block with timing stats in the final chunk:

```json
{
  "id": "chatcmpl-1",
  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 350,
    "total_tokens": 375,
    "prefill_ms": 62,
    "generation_ms": 23000,
    "tokens_per_second": 15.2
  }
}
```

#### Request Queue

The server uses a **worker thread + FIFO queue** for concurrent clients:

- **Idle**: Request is picked up immediately, response streams in real-time
- **Busy**: Request is enqueued (up to 16 deep), connection held open, processed in order
- **Overflow**: Returns `HTTP 503` with `Retry-After: 3` and `{"error": "server busy", "queue_depth": 16}`
- **Health checks** (`GET /health`) and **model list** (`GET /v1/models`) always respond instantly — they bypass the queue

This means tools like Open WebUI can fire off multiple requests without connection errors. If the queue overflows, the client gets a proper retry hint rather than a dropped connection.

#### Built-in Chat Client

```bash
cd finchmoe
./chat                 # Connect to localhost:9000 (default)
./chat --port 9000     # Explicit port
./chat --show-think    # Show <think> blocks (dimmed)
./chat --resume ID     # Resume a previous session
./chat --sessions      # List saved sessions
```

Features: streaming markdown rendering (bold, italic, code blocks, headers), session persistence to `~/.flash-moe/sessions/`, command history via linenoise.

#### System Prompt

Customize the system prompt by creating `~/.flash-moe/system.md`. The server hot-loads it at startup and pre-caches it so every request starts from the cached system prompt state (saves ~6 seconds of prefill).

Default: `"You are a helpful assistant."`

#### Tool Integration Examples

**Open WebUI:**
```bash
OPENAI_API_BASE=http://localhost:9000/v1 OPENAI_API_KEY=not-needed open-webui
```

**Continue.dev (VS Code):**
```json
{
  "models": [{
    "title": "FinchMoE Qwen3.6",
    "provider": "openai",
    "apiBase": "http://localhost:9000/v1",
    "apiKey": "not-needed"
  }]
}
```

**ChatGPT-style web UI** — any frontend that speaks OpenAI-compatible `/v1/chat/completions` will work. Point it at `http://localhost:9000/v1`.

### Key Flags

| Flag | Purpose |
|---|---|
| `--prompt TEXT` | Input prompt text |
| `--tokens N` | Max tokens to generate (default: 20, max: 32768) |
| `--timing` | Per-layer timing breakdown |
| `--k N` | Active experts per layer (default: 4) |
| `--cache-entries N` | Expert LRU Metal cache size (default: 0 = trust OS page cache) |
| `--cpu-linear` | CPU delta-net path (disable fused GPU path) |
| `--cpu-experts` | CPU expert path (~2 tok/s, for debugging correctness) |
| `--debug-layers` | Print hidden state statistics per layer |
| `--compare-experts N` | Verify GPU vs CPU expert outputs for layer N |
| `--freq` | Expert frequency tracking + analysis |
| `--serve PORT` | Run as HTTP server (OpenAI-compatible API, default port: 9000) |
| `--max-seq-len N` | Max context length for KV cache (default: 262144 = 256K, model's RoPE limit) |
| `--gpu-kv-seq N` | GPU KV buffer pre-allocation in tokens (default: 8192, falls back to CPU past this) |
| `--think-budget N` | Max thinking tokens before force `<`/`think>` (default: 2048, 0=unlimited) |
| `--model PATH` | Model directory containing `packed_experts/` |

#### Context Window Configuration

The engine has two context-length knobs:

| Flag | Default | What it controls |
|---|---|---|
| `--max-seq-len` | 262,144 (256K) | KV cache allocation cap. Matches the model's `max_position_embeddings` — the RoPE embeddings are trained for 256K and cannot generalize beyond without YaRN scaling. |
| `--gpu-kv-seq` | 8,192 | GPU Metal buffer pre-allocation for accelerated attention. Past this limit, attention **automatically falls back to CPU** — the engine doesn't crash, it just slows down. |

The 256K limit comes directly from the model config (`max_position_embeddings: 262144`). Both Qwen 3.6 35B and Qwen 3.5 397B share this limit. Setting `--max-seq-len` higher won't help — the RoPE embeddings have no signal beyond 256K.

For agentic workloads with long context:

```bash
# 100K GPU-accelerated context, full 256K max
./finchmoe-infer --serve 9000 --gpu-kv-seq 100000 --max-seq-len 262144
```

GPU KV buffers at 100K tokens: ~4.1 GB (10 full-attention layers × 2 (K+V) × 512 dims × 4 bytes × 100K tokens).

The 30 GatedDeltaNet layers have **no length limit** — they use a fixed-size recurrent matrix, not a growing KV cache.

### Running Benchmarks

```bash
# Generation speed: 100 tokens with per-layer timing
./finchmoe-infer --prompt "Once upon a time" --tokens 100 --timing

# Prompt processing speed: long prompt, minimal generation
./finchmoe-infer --prompt "Long text here..." --tokens 5 --timing

# Monitor memory during a run (separate terminal)
memory_pressure
vm_stat
```

## Benchmarks

All benchmarks run from a **Samsung 990 Plus NVMe in a Thunderbolt 4 enclosure** (faster than internal SSD on both M1 and M4 Mac minis).

### M4 Mac mini (16 GB) — Development Machine

| Metric | Value |
|---|---|
| Generation speed | **10–15 tok/s** |
| Memory usage | ~1.6 GB engine + page cache |
| Storage | Samsung 990 Plus NVMe via TB4 |

### Mac mini M1 (8 GB) — Tested 2026-08-07

Same Samsung 990 Plus NVMe via Thunderbolt 4 enclosure.

| Metric | Value |
|---|---|
| **Generation speed (avg 100 tok)** | **5.4 tok/s** |
| Cold start (first tokens) | 3.3–3.8 tok/s |
| Warm (page cache filling) | 5–7 tok/s |
| Hot (fully cached, peak) | 7–8.2 tok/s |
| **Prompt processing** | **~3–4 tok/s** (270–350 ms/token) |
| TTFT (11-token prompt) | 5,849 ms |
| TTFT (103-token prompt) | 30,465 ms |
| **Memory usage** | **~3.8 GB** engine footprint |
| System free after runs | ~1.6 GB (52%) |
| Swap used | **0** (none) |
| Expert cache in RAM | Not viable (malloc-cache crashes at 500 entries) |

**Per-layer timing (warm, 4.1 ms total):**

| Phase | Time | % |
|---|---|---|
| cmd1_wait (GPU attention projections) | 1.6 ms | 40% |
| expert_io (SSD read + dequant) | 1.5 ms | 37% |
| cmd2_wait (GPU o_proj + norm + routing + shared) | 0.8 ms | 20% |
| cmd3_encode (GPU expert compute) | 0.06 ms | 1.5% |

**Key findings:**
- Both machines use a **Samsung 990 Plus NVMe in a Thunderbolt 4 enclosure**, which is faster than the internal SSDs on M1 and M4 Mac minis. This is significant: the engine's SSD-streaming architecture benefits directly from fast external storage.
- Generation speed **ramps up** as OS page cache warms (3.3 → 8.2 tok/s over 100 tokens)
- **Expert I/O from SSD is the bottleneck** (37% of per-layer time), not GPU compute — even on a fast external NVMe
- Memory compression keeps the 8 GB system from swapping (~2.4 GB compressed pages)
- On M1 8GB the engine delivers **~40-50% of M4 16GB throughput**, limited primarily by slower GPU compute and lower memory bandwidth, not storage (both share the same external NVMe)
- The 3.5 tok/s project minimum target is comfortably met even on this entry-level Apple Silicon machine

## Known Limitations

**Base model behavior**: Qwen 3.6 35B A3B is a base (pre-trained) model, not instruction-tuned. The server automatically wraps prompts in the Qwen chat template (`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`). For direct CLI use (`--prompt`), prompts are sent as-is (base model completions) — use `--serve` for Q&A behavior.

**Single-worker generation**: The worker thread processes one request at a time. Concurrent requests are queued (up to 16), then get HTTP 503. This is fine for personal use and agentic workloads (tools typically send one request at a time and wait for the response). For high-throughput multi-user serving, continuous batching would be needed.

**GPU context limit**: GPU-accelerated attention caps at `--gpu-kv-seq` tokens (default: 8,192). Past this, the engine falls back to CPU attention automatically — slower but functionally correct. Increase with `--gpu-kv-seq` if you have GPU memory headroom.

**No sampling controls yet**: Temperature, top-p, top-k are not exposed. The engine always uses greedy decoding (argmax). This is fine for factual/agentic use cases but limits creative diversity.

## Bugs & Debugging

All bugs discovered and fixed during development are documented in **[BUGS.md](BUGS.md)** — 6 bugs covering data pipeline errors, quantization format mismatches, and performance issues, plus 6 lessons learned about safetensors offsets, dtype semantics, namespace collisions, and more.
