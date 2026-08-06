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
| gate_proj | [512, 2048] INT4 | 196 KB |
| up_proj | [512, 2048] INT4 | 196 KB |
| down_proj | [2048, 512] INT4 | 590 KB |
| **Total per expert** | | **~0.94 MB** |
| **Total experts** | 256 × 40 layers | **~9.6 GB** |

Quantization: MLX affine INT4, group-64, BF16 scale+bias.

## Project Structure

```
finchMoE/
├── README.md              # This file
├── design.md              # Detailed design document
├── flash-moe/             # Starting codebase (Qwen3.5-397B engine)
├── turbo-fieldfare/       # Performance benchmark (Swift, Gemma 4)
├── omlx/                  # Qwen-specific Metal kernel reference
├── models/
│   ├── Qwen3.6-35B-A3B-4bit/   # Target model (~19 GB)
│   └── Qwen3.5-397B-A17B-4bit/ # Baseline model (~209 GB, downloading)
└── archive/               # Original finchMoE code (pre-reboot)
```

## Reference Projects

| Project | What We Use It For |
|---|---|
| **flash-moe** | Starting codebase — already runs qwen3_5_moe architecture |
| **turbo-fieldfare** | Performance benchmark — 3.5 tok/s, ~2 GB RAM, M4 mini |
| **omlx** | Qwen-specific Metal kernel optimizations (GDN, FA256, MoE) |

## Development Plan

1. **Baseline**: Download flash-moe's Qwen3.5-397B model, run on M4 mini 16GB, measure tok/s
2. **Adapt dimensions**: Port flash-moe constants to Qwen3.6 (hidden=2048, layers=40, experts=256)
3. **Repack experts**: Update `repack_experts.py` for Qwen3.6 expert sizes
4. **Extract weights**: Update `extract_weights.py` for Qwen3.6 non-expert tensors
5. **Test**: Run Qwen3.6 inference on M4 mini, measure tok/s
6. **Optimize**: Apply omlx kernel improvements, target turbo-fieldfare performance
7. **Search**: Integrate internet search (inspired by llm-search pattern)
8. **iOS**: Port to A-series chips

## Status

- [x] Qwen3.6-35B-A3B-4bit downloaded (19 GB)
- [ ] Qwen3.5-397B-A17B-4bit baseline downloading (120/209 GB)
- [ ] flash-moe adapted for Qwen3.6 dimensions
- [ ] Baseline benchmark on M4 mini
- [ ] FinchMoE benchmark on M4 mini
