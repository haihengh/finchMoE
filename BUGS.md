# Bugs Found & Fixed — FinchMoE Debugging Session (2026-08-07)

This document chronicles the debugging session that brought FinchMoE from producing
incoherent CJK/English word fragments to coherent text at 10–15 tok/s.

## Timeline

```
Incoherent output (CJK + English fragments)
  → --debug-layers: found hidden state = 74 million after layer 0
  → --debug-layers: found post-norm correct (rms=1.0), projections correct
  → --debug-layers: found h_mid correct (rms=0.05), but moe_out = 74M
  → Python check: expert scales in packed file are NEGATIVE (-1.29)
  → Python check: safetensors source has correct scales (0.004)
  → Bug 1: MTP confusion → packed wrong tensors
  → Bug 2: safetensors offset → read from wrong positions
  → Bug 3: FP16/BF16 double-conversion → corrupted scale values
  → Fix all 3, repack → model produces coherent output at 1.94 tok/s
  → timing: cmd3_encode = 10.4ms (82% of time) — why?
  → Bug 4: force_cpu_experts = 1 → all expert matmul on CPU
  → --compare-experts: GPU == CPU bit-identical → enable GPU by default
  → 10-15 tok/s ✅
```

---

## Bug 1: MTP Layer Confusion in `generate_expert_index.py`

**Symptom**: Expert weights in `packed_experts/` produced astronomically large outputs.
Hidden state after layer 0 exploded to RMS ~74 million (should be ~0.05).
Expert gate_proj output had RMS ~13,500 (should be ~3).

**Root cause**: `generate_expert_index.py` scanned all safetensors keys matching
`*.switch_mlp.*` and parsed the layer index from the first `layers.N` segment found.
Both `model.layers.X.mlp.switch_mlp.*` and `mtp.layers.X.mlp.switch_mlp.*`
(Multi-Token Prediction) matched. MTP entries overwrote model entries for the same
layer index, causing the repacker to pack MTP weights instead of actual model expert
weights.

The MTP layer is a separate auxiliary prediction head with its own expert weights.
It has the same structure as the main model layers (same shapes, same component names),
so the packed file format was valid — just the wrong data.

**Discovery method**: After `--debug-layers` showed the hidden state exploding after
layer 0, we ran a Python script (`debug_compare.py`) that read the packed expert file
and printed scale/bias values. The scales were negative (-1.29) instead of positive
(~0.004). Reading the same tensor from the source safetensors (using the correct key
name) gave correct positive values, confirming the packed data was wrong.

**Fix in `generate_expert_index.py`**:
```python
# Skip MTP layers — only process main model tensors
if tensor_name.startswith('mtp.') or '.mtp.' in tensor_name:
    continue

# Only match 'layers' preceded by 'model' (not 'mtp')
for i, p in enumerate(parts):
    if p == 'layers' and i > 0 and parts[i-1] == 'model':
        layer_idx = int(parts[i + 1])
        break
```

---

## Bug 2: Safetensors Offset Miscalculation in `generate_expert_index.py`

**Symptom**: Even after fixing the MTP confusion, packed file data didn't match source
safetensors. The first expert scale read as 0x81D5 (BF16 ≈ -0.0) instead of the
expected 0x3B84 (BF16 ≈ 0.00403).

**Root cause**: The [safetensors format](https://github.com/huggingface/safetensors)
stores tensor `data_offsets` relative to the **data section** (after the 8-byte header
length + JSON header), not relative to the start of the file. `generate_expert_index.py`
stored `data_offsets[0]` directly as `abs_offset`, but `repack_experts.py` uses
`os.pread(fd, size, abs_offset)` which requires an **absolute file position**.

The offset was off by `8 + header_len` bytes (~53 KB per shard file). This caused
reads from positions in the file that happened to contain other tensors' data — data
that looked structurally valid (correct sizes) but had completely wrong values.

**Discovery method**: We manually read the safetensors file using two methods:
1. `file.seek(data_offsets[0])` → wrong data
2. `file.seek(8 + header_len + data_offsets[0])` → correct data

This confirmed the offset needed the data section start added.

**Note**: `extract_weights.py` did NOT have this bug — it correctly computes
`data_start = 8 + header_len` and uses `sf.seek(data_start + tensor_offsets[0])`.

**Fix in `generate_expert_index.py`**:
```python
# Track data section start for each file
file_data_starts[filename] = 8 + header_len

# Use absolute offsets for os.pread
ds = file_data_starts[filename]
byte_start = ds + data_offsets[0]
byte_end   = ds + data_offsets[1]
```

---

## Bug 3: FP16/BF16 Double-Conversion in `repack_experts.py`

**Symptom**: Expert scales in packed files were negative (e.g., -1.29 as BF16)
instead of positive MLX quantization scales (~0.004). Effective weight values were
~300× too large, causing expert outputs to reach RMS ~528 billion, triggering the
NaN guard for most layers. Layer 0 barely passed the guard (output RMS 74M < 1e20
threshold), corrupting the hidden state.

**Root cause**: Two different quantization data formats exist for MLX models:

| Source | Scale/bias storage | Safetensors dtype | Needs conversion? |
|---|---|---|---|
| mlx-community models | FP16 | `BF16` | Yes: FP16→BF16 |
| Self-quantized (`quantize_model.py`) | BF16 | `U16` | No: already BF16 |

`repack_experts.py` unconditionally applied FP16→BF16 conversion to ALL scales/biases:
```python
arr = np.frombuffer(data, dtype=np.uint16)
f16 = arr.view(np.float16).astype(np.float32)     # Treats as FP16!
bf16 = (f16.view(np.uint32) >> 16).astype(np.uint16)
```

For BF16-encoded data (0x3B84 = scale 0.00403):
1. Interpret as FP16 → 0.939
2. Re-encode as BF16 → completely different value

The original value (0.00403) became -1.29 after this double-conversion, flipping the
sign and changing the magnitude by ~300×.

**Discovery method**: We read the same tensor from both the safetensors source and the
packed file and compared the raw uint16 values:
- Safetensors: 0x3B84 → BF16 = 0.00403 ✅
- Packed file: 0xBFA5 → BF16 = -1.29 ❌

Tracing through the conversion code revealed the FP16 misinterpretation.

**Note**: `extract_weights.py` correctly had this guard:
```python
if is_scale_or_bias and dtype == 'BF16':  # only convert when dtype says BF16
```
Only `repack_experts.py` was missing the dtype check.

**Fix in `repack_experts.py`**:
```python
# Only convert when source dtype is 'BF16' (mlx-community convention)
needs_fp16_convert = is_scale_or_bias and info.get('dtype') == 'BF16'
```

---

## Bug 4: `force_cpu_experts = 1` — Performance Bottleneck

**Symptom**: Engine ran at 1.94 tok/s instead of 10+ tok/s. `--timing` showed
`cmd3_encode: 10.44 ms` consuming 82% of per-layer time.

**Root cause**: `infer.m:5204` had `int force_cpu_experts = 1` — a debug flag set
during development when corrupted expert weights (bugs 1–3 above) caused NaN in the
GPU path. This forced ALL expert matmul operations (12 dequant-matvecs per layer:
4 experts × 3 matrices each) to run on the CPU.

Per-layer timing breakdown with CPU experts:
```
cmd1_wait:     1.25 ms  (10%)
cmd2_wait:     0.88 ms  ( 7%)
cmd3_encode:  10.44 ms  (82%)  ← CPU dequant-matvec
total_layer:  12.65 ms
```

40 layers × 12.65 ms = 506 ms/token → 1.94 tok/s.

With GPU experts:
```
cmd3_encode:   0.02 ms  ( 1%)  ← GPU encoding only
total_layer:   2.38 ms
```

40 layers × 2.38 ms = 95 ms/token → 10.5 tok/s (5–8× faster).

**Discovery method**: `grep -n "force_cpu_experts"` found the hardcoded `= 1` at
line 5204. The `--timing` flag revealed `cmd3_encode` as the bottleneck.

**Verification**: Added `--compare-experts N` diagnostic that runs both GPU and CPU
expert computation on the same input and diffs every intermediate tensor (gate_proj,
up_proj, swiglu, down_proj). Confirmed GPU outputs are **bit-identical** to CPU
outputs — max_diff < 1e-7 across all stages. The `dequant_matvec_4bit_v3` Metal
shader is numerically correct. The "GPU quality regression" observed earlier was
sampling variability from the base model, not a GPU bug.

**Fix in `infer.m`**:
```c
// Old: int force_cpu_experts = 1;  // always CPU, 1.94 tok/s
// New:
int force_cpu_experts = g_cpu_experts ? 1 : 0;  // GPU by default, 10-15 tok/s
```

Added `--cpu-experts` flag for debugging, `--gpu-experts` kept as alias.

---

## Bug 5: Qwen3_5RMSNorm Weight (Fixed Earlier)

**Symptom**: Norm weights stored as ~0.03 instead of ~1.03. After input RMSNorm,
hidden state RMS dropped to near zero, killing the signal before any attention
computation.

**Root cause**: Qwen uses `Qwen3_5RMSNorm` where the stored weight parameter is
`weight_param ≈ 0` and the effective weight is `1 + weight_param ≈ 1.0`.
`quantize_model.py` quantized `weight_param` directly (near zero) without adding
the constant 1.0 offset. The C engine's `cpu_rms_norm` multiplies by the stored
weight directly, producing near-zero outputs.

**Fix in `quantize_model.py`**:
```python
if 'norm.weight' in nn or 'layernorm.weight' in nn:
    arr = arr + 1.0  # Qwen3_5RMSNorm: effective = 1 + weight_param
```

---

## Bug 6: 8-Bit Gate Dequant (Fixed Earlier)

**Symptom**: Routing gate (mlp.gate) and shared_expert_gate produced wrong outputs.

**Root cause**: These two weight matrices use 8-bit quantization (not 4-bit like
everything else). The C engine was calling `cpu_dequant_matvec(..., bits=4)` instead
of `bits=8`.

**Fix**: Pass `bits=8` for gate and shared_expert_gate tensors. The `quantize_model.py`
script already handles this:
```python
EIGHT_BIT = ['.mlp.gate.weight', '.mlp.shared_expert_gate.weight']
bits = 8 if any(p in nn for p in EIGHT_BIT) else 4
```

---

## Debugging Tools Added

### In the C engine (`infer.m`)

**`--debug-layers`**
Prints per-layer hidden state statistics (mean, rms, std, min, max) at the input
and output of each transformer layer. Used to discover the 74M hidden state explosion.

**`--compare-experts N`**
For layer N, runs both GPU and CPU expert computation on the same input (`h_post`)
and prints per-stage comparison:
- gate_proj, up_proj, swiglu, down_proj
- max_diff, avg_diff, first mismatching element index
- cpu_rms vs gpu_rms for each stage
- First 5 values of each output vector
- Scale/bias raw values for sanity checking

Used to verify GPU bit-identical correctness.

**`--timing`**
Per-phase timing breakdown (ms) averaged across all layers and tokens:
- cmd1_submit, cmd1_wait, cpu_attn, cmd2_encode, cmd2_wait
- routing_cpu, expert_io, cmd3_encode, total_layer
- Counts of command buffers, sync waits, and GPU encoders per layer

**`--cpu-experts` / `--gpu-experts`**
Explicit control over expert computation path. GPU is default (10–15 tok/s).

### Python scripts (`finchmoe/`)

**`debug_compare.py`**
Reads `model_weights.bin` and verifies:
- Tensor naming conventions
- Embedding lookup values
- Norm weight statistics (checks Qwen3_5RMSNorm fix)
- Key name consistency

**`debug_mlx_inference.py`**
Loads a quantized model via `mlx-lm` and runs inference. Useful for verifying
the quantized model format is compatible with MLX's loader. Note: loading a
20GB model takes several minutes.

---

## Key Architecture: How the Engine Processes a Token

Understanding the pipeline is essential for debugging:

```
Per-layer pipeline (fused_layer_forward):
┌─────────────────────────────────────────────────────────┐
│ CMD1: attention projections (GPU batch matvec)           │
│   Full attn: q_proj, k_proj, v_proj                     │
│   Linear:    in_proj_qkv, in_proj_z, in_proj_a, in_proj_b │
│   + GPU delta-net: conv1d→rms_norm→decay→recur→gate     │
│   Submit + wait for GPU                                  │
├─────────────────────────────────────────────────────────┤
│ CPU: attention compute                                   │
│   Full attn: RoPE → KV cache → GQA softmax → sigmoid    │
│   Linear:    conv1d → split q/k/v → delta net recurrence │
│              → gated_rms_norm                            │
├─────────────────────────────────────────────────────────┤
│ CMD2: o_proj + residual + norm + routing (GPU fused)     │
│   8 encoders, 1 commit+wait                              │
│   Reads h_mid, h_post from GPU buffers                   │
├─────────────────────────────────────────────────────────┤
│ CPU: softmax + top-K routing                             │
│   K=4 experts selected from 256                          │
├─────────────────────────────────────────────────────────┤
│ Expert I/O: parallel pread (or page cache hit)           │
│   K=4 experts × 1.69 MB = 6.76 MB per layer             │
├─────────────────────────────────────────────────────────┤
│ CMD3: expert forward + shared expert + combine (GPU)     │
│   gate_proj → up_proj → SwiGLU → down_proj (per expert) │
│   + shared expert SwiGLU + down_proj                     │
│   + weighted combine + residual + next-layer input norm  │
│   Deferred commit (async, don't wait)                    │
└─────────────────────────────────────────────────────────┘
```

The expert weights are stored as 4-bit packed uint32 with BF16 per-group scales/biases:
```
Each expert (1,769,472 bytes):
  [0]       gate_proj.weight  U32[512,256]  524,288 bytes
  [524288]  gate_proj.scales  BF16[512,32]   32,768 bytes
  [557056]  gate_proj.biases  BF16[512,32]   32,768 bytes
  [589824]  up_proj.weight    U32[512,256]  524,288 bytes
  [1114112] up_proj.scales    BF16[512,32]   32,768 bytes
  [1146880] up_proj.biases    BF16[512,32]   32,768 bytes
  [1179648] down_proj.weight  U32[2048,64]  524,288 bytes
  [1703936] down_proj.scales  BF16[2048,8]   32,768 bytes
  [1736704] down_proj.biases  BF16[2048,8]   32,768 bytes
```

---

## Lessons Learned

1. **Trust but verify offsets**: Safetensors `data_offsets` are relative to the data
   section, not absolute file positions. Always add `8 + header_len` when using
   `os.pread`. Python `file.seek()` with `data_start + offset` is the safer pattern.

2. **Don't assume dtype semantics**: MLX community models label FP16 data as
   `dtype='BF16'` in safetensors metadata. Self-quantized models label actual BF16 data
   as `dtype='U16'`. Always check the actual byte format, not just the dtype label.

3. **Name collisions are subtle**: Both `model.layers.X` and `mtp.layers.X` exist in
   the same safetensors file with the same internal structure. Pattern matching on
   `layers.N` without namespace awareness silently picks up wrong tensors.

4. **One debug flag can mask performance**: `force_cpu_experts = 1` was left over from
   NaN debugging. A single hardcoded `1` cost 5–8× throughput. Always make debug flags
   explicit command-line options.

5. **CPU reference is the gold standard**: The `--compare-experts` approach — running
   both paths on identical data and diffing — found that the GPU shader was correct all
   along. The bugs were all in the data pipeline (wrong offsets, wrong data, wrong format).

6. **Per-layer statistics first**: Before diving into full Python reference
   reimplementation, adding `--debug-layers` (printing hidden state RMS per layer)
   immediately revealed the explosion at layer 0, which directed all further
   investigation toward the expert weights.

---

## Bug 7: switch_mlp Weights Excluded from Extraction (2026-08-07)

### Status: **FOUND, NOT YET FIXED**

### Symptom
Model produces degenerate output regardless of sampling settings or chat template fixes:
- Raw prompts: repeating patterns (`". The a. The a..."`, `"time: time: time:"`)
- Chat template: single-character repeats (`".........."`, `"-.-.-.-."`) or random byte sequences (`"â\"6â\"6..."`)
- Output is never coherent, never grammatical, at any temperature (0.8–1.2) or top-k (40–80)

### Root Cause
**`extract_weights.py` filters out `switch_mlp` tensors**, treating them as routed expert weights. But `switch_mlp` is a **dense per-layer FFN** (like a standard transformer MLP block) that must run on every token at every layer. It is NOT a routed expert — it has no expert dimension, no routing, and the same weights apply to all tokens.

The filter at `extract_weights.py:69`:
```python
expert_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
```

This matches and skips all `model.layers.X.mlp.switch_mlp.*` tensors. As a result:
- `model_weights.json` contains **0 switch_mlp tensors** (out of 240 expected: 40 layers × 6 tensors)
- `model_weights.bin` has **no switch_mlp weights**
- The C engine has **no code to compute switch_mlp** forward pass
- Every layer's output is missing the switch_mlp contribution

### Evidence
- MLX loading confirms `switch_mlp` exists on every layer: `model.layers.X.mlp.switch_mlp.{down_proj,gate_proj,up_proj}.{weight,scales,biases}`
- MLX reports 1024 unexpected parameters when loading the model without switch_mlp support
- `model_weights.json` has 0 switch_mlp entries confirmed via Python check
- Config.json shows the model type is `qwen3_5_moe` which includes switch_mlp per layer

### Impact
**Catastrophic** — every token through every layer is missing the switch_mlp computation. This explains all degenerate output patterns observed. The model is effectively producing random noise because a major component of each transformer layer is absent.

### Fix Required (3 parts)
1. **`extract_weights.py`**: Remove `switch_mlp` from the expert exclusion pattern. Add it to the non-expert weight extraction. It should be categorized under a new `switch_mlp` weight category.
2. **`infer.m`**: Add `switch_mlp_forward()` function that computes: `gate_proj(hidden) → SiLU → * up_proj(hidden) → down_proj`. This is identical to the shared_expert computation but uses switch_mlp weights. Must be called for EVERY layer (both GatedDeltaNet and full attention).
3. **Model weights manifest**: Add switch_mlp weight pointers to the layer config struct so `fused_layer_forward` can access them.

### Similarity to Known Architecture
The switch_mlp is architecturally identical to the `shared_expert` — both are dense FFNs with gate/up/down projections. The shared_expert already works correctly. The switch_mlp forward pass can be modeled directly on the existing shared_expert code.

---

## Bug 8: Shared Expert Metal Command Buffer Synchronization (2026-08-09)

**Symptom**: Model output degenerated into repeated tokens ("Con Con Con") or Chinese gibberish. All individual components verified correct but full pipeline was broken. llama.cpp ground truth (GGUF conversion) confirmed the model weights were fine.

**Root Cause**: In `fused_layer_forward` CMD3, the shared expert SwiGLU activation was dispatched into a Metal command buffer, but the CPU immediately read `buf_shared_act` for the BF16 down-projection BEFORE `[cmd_experts commit]`. The buffer contained stale/zero data, corrupting the shared expert output across all 40 layers.

**Fix** (infer.m, 1 line):
```c
// Before: CPU reads stale buffer
cpu_dequant_matvec(sdw, NULL, NULL, act, out, ...);

// After: commit and wait first
[cmd_experts commit];
[cmd_experts waitUntilCompleted];
float *act = (float *)[g_metal->buf_shared_act contents];
cpu_dequant_matvec(sdw, NULL, NULL, act, out, ...);
cmd_experts = [g_metal->queue commandBuffer];  // fresh buffer for remaining dispatches
```

**Discovery Method**: Systematic bisection — GPU-all vs CPU-linear vs CPU-experts vs both-CPU. Only CPU-experts produced coherent output, isolating the bug to the GPU expert path. GPU-vs-CPU comparison (--compare-experts) showed routed experts matched but didn't test the shared expert path.

**Lessons**:
1. Test the FULL pipeline, not just individual components. All sub-modules passed unit tests.
2. Metal command buffers are asynchronous — data written by GPU kernels is NOT visible to CPU until the command buffer completes.
3. Bisection debugging (binary search over GPU/CPU combinations) is effective for isolating async bugs.
4. A ground truth reference (llama.cpp GGUF) is invaluable for confirming whether the model or the engine is at fault.
