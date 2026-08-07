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
│   ├── Qwen3.6-35B-A3B-4bit/   # Target model (~19 GB) ✅
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

1. ~~**Adapt dimensions**: Port flash-moe constants to Qwen3.6~~ ✅
2. ~~**Repack experts**: Update `repack_experts.py`~~ ✅
3. ~~**Extract weights**: Update `extract_weights.py`~~ ✅
4. ~~**FP16→BF16 fix**: MLX stores scales/biases as FP16, not BF16~~ ✅
5. ~~**Self-quantization pipeline**: BF16 → clean MLX 4-bit~~ ✅
6. ~~**Norm weight fix**: Qwen3_5RMSNorm (1+weight) formulation~~ ✅
7. ~~**Coherent output**: Layer-by-layer comparison, 5 bugs fixed~~ ✅
8. ~~**GPU expert path**: Verified bit-identical to CPU, 5-8× speedup~~ ✅
9. **Optimize**: KV cache quant, MTP, TurboQuant, flash attention
10. **iOS**: Port to A-series chips

## Status

- [x] Qwen3.6-35B-A3B-bf16 downloaded (67 GB — original BF16)
- [x] Qwen3.6-35B-A3B-4bit-custom self-quantized (18.5 GB, clean)
- [x] flash-moe adapted for Qwen3.6 dimensions
- [x] Self-quantization pipeline working (BF16→4bit→extract→repack)
- [x] FP16/BF16 format mismatch fixed
- [x] Qwen3_5RMSNorm weights fixed (1+weight_param)
- [x] Engine runs at **10-15 tok/s** (GPU experts + GPU delta-net, default)
- [x] Coherent output — produces grammatical English text completions
- [x] GPU expert path **verified bit-identical** to CPU path (`--compare-experts`)
- [ ] 397B baseline (downloaded, not yet tested)

## Known Limitations

**Base model behavior**: Qwen 3.6 35B A3B is a base (pre-trained) model, not instruction-tuned. Without the Qwen chat template (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n`), it produces next-token completions rather than direct answers. Output quality varies with temperature sampling (default 0.8). For Q&A use, pipe prompts through the chat template first.

## Bugs Found & Fixed

This section documents the debugging session that brought the engine from producing incoherent CJK/English word fragments to coherent text at 10-15 tok/s.

### Bug 1: MTP Layer Confusion in `generate_expert_index.py`

**Symptom**: Expert weights in `packed_experts/` produced astronomically large outputs (RMS 528 billion), triggering the NaN guard and zeroing out expert contributions. The hidden state after layer 0 exploded to ~74 million (should be ~0.05).

**Root cause**: The script scanned all safetensors keys matching `*.switch_mlp.*` and parsed the layer index from the first `layers.N` segment found. Both `model.layers.X.mlp.switch_mlp.*` and `mtp.layers.X.mlp.switch_mlp.*` (Multi-Token Prediction) matched. Since Python dict insertion order depends on the index file, MTP entries would overwrite model entries for the same layer index, causing the repacker to pack MTP weights instead of the actual model expert weights.

**Fix**: Skip tensor names starting with `mtp.` or containing `.mtp.`. Only match layers preceded by `model` (not `mtp`):
```python
if tensor_name.startswith('mtp.') or '.mtp.' in tensor_name:
    continue
# Only match 'layers' preceded by 'model'
if p == 'layers' and i > 0 and parts[i-1] == 'model':
    layer_idx = int(parts[i + 1])
```

### Bug 2: Safetensors Offset Miscalculation in `generate_expert_index.py`

**Symptom**: Even after fixing the MTP confusion, expert outputs were still wrong. Packed file data didn't match source safetensors.

**Root cause**: The safetensors format stores `data_offsets` relative to the **data section** (after the 8-byte header length + JSON header), not relative to the start of the file. `generate_expert_index.py` stored `data_offsets[0]` as `abs_offset`, but `repack_experts.py` uses `os.pread(fd, size, abs_offset)` which requires an **absolute file position**. The offset was off by `8 + header_len` bytes (~53 KB per file), causing reads from wrong positions that happened to contain other tensors' data.

**Fix**: Add the data section start to each offset:
```python
ds = file_data_starts[filename]  # 8 + header_len
byte_start = ds + data_offsets[0]
byte_end   = ds + data_offsets[1]
```

**Note**: `extract_weights.py` did NOT have this bug — it correctly computes `data_start = 8 + header_len` and uses `sf.seek(data_start + tensor_offsets[0])`.

### Bug 3: FP16/BF16 Double-Conversion in `repack_experts.py`

**Symptom**: Expert scales read from packed files were negative (e.g., -1.29) instead of positive MLX quantization scales (~0.004). This caused expert weight values to be ~300× too large.

**Root cause**: Two different quantization data formats exist:
- **mlx-community models**: Store quantization scales/biases as **FP16** but label them with `dtype='BF16'` in safetensors metadata (known MLX quirk). These need FP16→BF16 conversion.
- **Self-quantized models** (`quantize_model.py`): Store scales/biases as **BF16** with `dtype='U16'`. These are already in the correct format.

`repack_experts.py` unconditionally applied FP16→BF16 conversion to ALL scales/biases:
```python
arr = np.frombuffer(data, dtype=np.uint16)
f16 = arr.view(np.float16).astype(np.float32)     # WRONG for BF16 data!
bf16 = (f16.view(np.uint32) >> 16).astype(np.uint16)
```

For BF16-encoded data (0x3B84 = 0.00403), interpreting as FP16 gives 0.939, then re-encoding as BF16 gives a completely different value.

**Fix**: Only convert when `dtype == 'BF16'` (mlx-community convention):
```python
needs_fp16_convert = is_scale_or_bias and info.get('dtype') == 'BF16'
```

**Note**: `extract_weights.py` correctly had this guard: `if is_scale_or_bias and dtype == 'BF16'`. Only `repack_experts.py` was missing it.

### Bug 4: `force_cpu_experts = 1` — Performance Bottleneck

**Symptom**: Engine ran at 1.94 tok/s instead of the expected 10+ tok/s. Per-layer timing showed `cmd3_encode: 10.4ms` (82% of total time).

**Root cause**: `infer.m:5204` had `int force_cpu_experts = 1` — a debug flag set during development when corrupted expert weights caused NaN in the GPU path. This forced ALL expert matmul operations (12 dequant-matvecs per layer: 4 experts × 3 matrices) to run on the CPU instead of the GPU Metal shaders.

**Fix**: Changed to `int force_cpu_experts = g_cpu_experts ? 1 : 0`, defaulting to GPU path. Added `--cpu-experts` flag for debugging, `--gpu-experts` kept as explicit opt-in.

**Verification**: Added `--compare-experts N` diagnostic that runs both GPU and CPU expert computation on the same input and diffs every intermediate tensor (gate_proj, up_proj, swiglu, down_proj). Confirmed GPU outputs are **bit-identical** to CPU outputs across all stages — the `dequant_matvec_4bit_v3` Metal shader is numerically correct.

### Bug 5: Qwen3_5RMSNorm Weight (Previously Fixed)

**Symptom**: Norm weights stored as ~0.03 instead of ~1.03. The Qwen model stores `weight_param ≈ 0` where effective weight = `1 + weight_param`. Quantizing `weight_param` directly (without adding 1.0) produced near-zero norm weights, killing the signal after the first RMSNorm.

**Fix**: In `quantize_model.py`, add 1.0 before quantizing norm weights:
```python
if 'norm.weight' in nn or 'layernorm.weight' in nn:
    arr = arr + 1.0
```

### Debugging Tooling Added

- **`--debug-layers`**: Prints per-layer hidden state statistics (mean, rms, std, min, max) at input and output of each transformer layer. Used to discover the 74M hidden state explosion.
- **`--compare-experts N`**: For layer N, runs both GPU and CPU expert computation on the same input and prints per-stage comparison (gate_proj, up_proj, swiglu, down_proj) with max_diff, avg_diff, and first mismatching element. Used to verify GPU bit-identical correctness.
- **`--timing`**: Per-phase timing breakdown (cmd1_submit, cmd1_wait, cpu_attn, cmd2_encode, cmd2_wait, routing_cpu, expert_io, cmd3_encode) averaged across all layers and tokens.
- **`finchmoe/debug_compare.py`**: Python script that reads `model_weights.bin` and verifies embedding lookup, norm weights, and key naming conventions.
- **`finchmoe/debug_mlx_inference.py`**: Loads model via MLX for inference comparison (slow to load, not the primary debugging tool).

### Lessons Learned

1. **Trust but verify offsets**: Safetensors `data_offsets` are relative to the data section, not absolute file positions. Always add `8 + header_len` when using `os.pread`. Python `file.seek()` with `data_start + offset` is the safer pattern.

2. **Don't assume dtype semantics**: MLX community models label FP16 data as `dtype='BF16'` in safetensors metadata. Self-quantized models label actual BF16 data as `dtype='U16'`. Always check the actual byte format, not just the dtype label.

3. **Name collisions are subtle**: Both `model.layers.X` and `mtp.layers.X` exist in the same safetensors file with the same internal structure. Pattern matching on `layers.N` without namespace awareness silently picks up wrong tensors.

4. **One debug flag can mask performance**: `force_cpu_experts = 1` was left over from NaN debugging. A single hardcoded `1` cost 5-8× throughput. Always make debug flags explicit command-line options.

5. **CPU reference is the gold standard**: The `--compare-experts` approach — running both paths on identical data and diffing — found that the GPU shader was correct all along. The bugs were all in the data pipeline (wrong offsets, wrong data, wrong format).
