# FinchMoE

A C/Metal inference engine for **Qwen 3.6 35B A3B** on Apple Silicon, targeting ≥3.5 tok/s at ≤3 GB RAM — and eventually iPhone.

## Goals

| Target | Value |
|---|---|
| Model | Qwen 3.6 35B A3B (4-bit MLX quantized) |
| Hardware | M4 Mac mini 16GB (dev), A-series iPhone (target) |
| Speed | ≥3.5 tok/s |
| Memory | ≤3 GB RAM for engine |
| Disk | ~19 GB model + ~20 GB repacked experts |
| Features | Text generation + built-in internet search |

## Why flash-moe?

We evaluated the original `finchmoe_engine` (C++/ObjC++) and found it was built for a pre-release Qwen MoE spec that never shipped. The published Qwen 3.6 has a completely different architecture.

[flash-moe](flash-moe/) is a production C/Metal engine that already runs the Qwen3.5-MoE family (same `qwen3_5_moe` model type as Qwen 3.6) at 4.36 tok/s on M3 Max. It implements:

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
│   ├── infer.m            #   Main engine (~7100 lines C/Metal)
│   ├── shaders.metal      #   Metal compute kernels (~1300 lines)
│   ├── Makefile           #   Build system
│   ├── extract_weights.py #   Non-expert weight extraction
│   ├── repack_experts.py  #   Expert weight repacking
│   ├── generate_expert_index.py # Expert index generator
│   ├── chat.m             #   Interactive chat TUI
│   ├── tokenizer.h        #   C BPE tokenizer
│   └── export_tokenizer.py#   Tokenizer export utility
├── flash-moe/             # Starting codebase (Qwen3.5-397B engine, unmodified)
├── turbo-fieldfare/       # Performance benchmark (Swift, Gemma 4)
├── omlx/                  # Qwen-specific Metal kernel reference
├── models/
│   ├── Qwen3.6-35B-A3B-4bit-custom/  # Target model (~19 GB) ✅
│   └── Qwen3.5-397B-A17B-4bit/       # Baseline model (~209 GB)
└── archive/               # Original finchMoE code (pre-reboot)
```

## Reference Projects

| Project | What We Use It For |
|---|---|
| **flash-moe** | Starting codebase — already runs qwen3_5_moe architecture |
| **turbo-fieldfare** | Performance benchmark — 3.5 tok/s, ~2 GB RAM, M4 mini |
| **omlx** | Qwen-specific Metal kernel optimizations (GDN, FA256, MoE) |

## Status

- [x] Qwen3.6-35B-A3B-bf16 downloaded (67 GB — original BF16)
- [x] Qwen3.6-35B-A3B-4bit-custom self-quantized (18.5 GB, clean)
- [x] flash-moe adapted for Qwen3.6 dimensions
- [x] Self-quantization pipeline working (BF16 → 4bit → extract → repack)
- [x] FP16/BF16 format mismatch fixed
- [x] Qwen3_5RMSNorm weights fixed (1+weight_param)
- [x] GPU expert path verified bit-identical to CPU (`--compare-experts`)
- [x] Engine runs at **10–15 tok/s** on M4 (GPU experts + GPU delta-net, default)
- [x] Coherent output — produces grammatical English text completions
- [ ] 397B baseline (downloaded, not yet tested)
- [ ] KV cache quant, MTP, TurboQuant, flash attention optimizations
- [ ] iOS port (A-series chips)

## Running the Engine

### Build

```bash
cd finchmoe
make          # Build finchmoe-infer
make chat     # Build interactive chat TUI (optional)
```

Requires: Xcode Command Line Tools (`xcode-select --install`), macOS 14+.

### Model Preparation

If you've downloaded a pre-quantized model from mlx-community or self-quantized with `quantize_model.py`, run these one-time preparation steps:

```bash
cd finchmoe
make extract MODEL_DIR=../models/Qwen3.6-35B-A3B-4bit-custom
make index MODEL_DIR=../models/Qwen3.6-35B-A3B-4bit-custom
make repack
```

This produces:
- `model_weights.bin` / `model_weights.json` — non-expert weights (mmap'd at startup)
- `expert_index.json` — expert tensor layout, offsets, and shapes
- `packed_experts/layer_00.bin` … `layer_39.bin` — 4-bit expert weights per layer

### Basic Usage

```bash
cd finchmoe
./finchmoe-infer --prompt "Hello world" --tokens 50
./finchmoe-infer --prompt "Write a haiku about coding." --tokens 200 --timing
```

The engine auto-detects `model_weights.bin`, `vocab.bin`, and `../models/Qwen3.6-35B-A3B-4bit-custom/packed_experts/` relative to the current directory. Override with `--model`, `--weights`, `--manifest`, or `--vocab`.

### Key Flags

| Flag | Purpose |
|---|---|
| `--prompt TEXT` | Input prompt text |
| `--tokens N` | Max tokens to generate (default: 20) |
| `--timing` | Per-layer timing breakdown |
| `--k N` | Active experts per layer (default: 4) |
| `--cache-entries N` | Expert LRU Metal cache size (default: 2500, 0=disabled) |
| `--cpu-linear` | CPU delta-net path (disable fused GPU path) |
| `--cpu-experts` | CPU expert path (~2 tok/s, for debugging correctness) |
| `--debug-layers` | Print hidden state statistics per layer |
| `--compare-experts N` | Verify GPU vs CPU expert outputs for layer N |
| `--freq` | Expert frequency tracking + analysis |
| `--serve PORT` | Run as HTTP server (OpenAI-compatible `/v1/chat/completions`) |
| `--think-budget N` | Max thinking tokens before force `</think>` (default: 2048) |
| `--model PATH` | Model directory containing `packed_experts/` |

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

**Base model behavior**: Qwen 3.6 35B A3B is a base (pre-trained) model, not instruction-tuned. Without the Qwen chat template (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n`), it produces next-token completions rather than direct answers. Output quality varies with temperature sampling (default 0.8). For Q&A use, pipe prompts through the chat template first.

## Bugs & Debugging

All bugs discovered and fixed during development are documented in **[BUGS.md](BUGS.md)** — 6 bugs covering data pipeline errors, quantization format mismatches, and performance issues, plus 6 lessons learned about safetensors offsets, dtype semantics, namespace collisions, and more.
