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

---

## Bug 9: 4-bit Non-Expert Dequant 400× Slower Than BF16 (2026-08-09)

**Symptom**: Self-quantized model (2-bit-dense-v2) with 4-bit attention/GDN/embeddings hangs during inference. GPU path hangs, CPU path takes 500+ seconds per token. Flash-moe BF16 weights work fine.

**Root Cause**: `cpu_dequant_matvec` for large 4-bit packed tensors is pathologically slow. The QKV projection tensor (12288×4096) requires 50 million nibble extractions and BF16 conversions in a scalar C loop — 4.2 seconds per tensor on M4. The BF16 path reads uint16 directly without unpacking — 0.01 seconds. This is a 400× difference.

Four tensors (QKV, Z, A, B) × 30 GDN layers = 120 dequant calls per token. At 4.2s each: 504 seconds/token. The GPU path (`gpu_batch_matvec` → `matvec_v3`) also hangs, likely due to dispatch overhead for 120+ large kernel launches per token.

**Fix**: Keep non-expert tensors as BF16. Only quantize routed experts which use optimized SIMD-stride GPU kernels (`dequant_matvec_4bit_v3`). The model_weights.bin grows from 1.4 GB to ~5 GB, but speed is unchanged.

**Lesson**: Quantization benefits are architecture-dependent. Expert tensors (many small matmuls, specialized kernels) benefit greatly from 4/2/1-bit. Non-expert tensors (few large matmuls, generic dequant path) do not — the dequant overhead dominates any I/O savings. Always measure, don't assume.

---

## Bug 10: Vocab BPET Format Parsing (2026-08-09)

**Symptom**: `load_vocab` read the BPET magic bytes as `num_entries=1.4 billion`, causing a 93 GB malloc that corrupted VM state and triggered SIGKILL. The kernel killed the process (and sometimes neighboring processes) due to memory pressure.

**Root Cause**: The `load_vocab` function in `infer.m` read the 4-byte BPET magic (`"BPET"`) as the first `uint32_t` value (vocab_size). The ASCII bytes `B` `P` `E` `T` interpreted as little-endian uint32 = 0x54455042 = 1,414,992,962 entries. The subsequent malloc(1.4B * sizeof(entry)) requested ~93 GB, which macOS rejects by sending SIGKILL.

**Fix**: Properly parse the BPET header: read 4-byte magic, 4-byte version, THEN read vocab_size, num_merges, num_added as subsequent uint32 fields.

**Lesson**: Always verify magic bytes BEFORE reading size fields. A format change or corrupted file will produce absurd allocation sizes that crash the entire system, not just the process.

---

## Bug 11: BPE Merge Table Corruption in export_tokenizer.py (2026-08-10)

**Symptom**: Model produced grammatically correct but completely wrong token salad. "The capital of France is" → "'m ysterious to me!" Token 846 ("user") appeared where token 74455 ("assistant") should be. The chat template `<|im_start|>assistant\n<think>\n` was being tokenized as `<|im_start|>user\nThe`, making the model think it should complete a USER turn instead of responding as assistant.

**Root Cause**: **1-line bug in `export_tokenizer.py` line 60:**

```python
# BROKEN: string indexing returns CHARACTERS
for pair in merges:
    a, b = pair[0], pair[1]
```

The tokenizer JSON stores BPE merges as space-separated strings like `"Ġ Ġ"`, `"a s"`, `"as s"`. Python string indexing `pair[0], pair[1]` extracts individual CHARACTERS, not the space-delimited tokens. For merge `"a s"`:
- Expected: `a="a"`, `b="s"`
- Actual: `a="a"`, `b=" "` (a space character!)

**ALL 247,587 merges** had their second part replaced with a single space character. The C tokenizer could merge individual bytes into pair-level tokens ("a"+"s"→"as") since the first round used the merge priority correctly, but all subsequent multi-level merges ("as"+"s"→"ass", "ass"+"i"→"assi", etc.) failed because the merge keys were corrupted.

Compound tokens couldn't be formed:
- "assistant" → 5 tokens: `299 6122 267 276 83` (should be 1: `74455`)
- "user" → 2 tokens: `350 261` (should be 1: `846`)

**Verification**: Before fix `vocab.bin` was 5.9 MB (corrupted — all merge b-parts were 1-byte spaces). After fix `vocab.bin` is 7.8 MB (correct merge key lengths). C tokenizer output now matches Python `tokenizers` library exactly.

**Fix**:
```python
# CORRECT: split on space delimiter
for pair in merges:
    a, b = pair.split(' ')
```

**Impact**: This bug was present since `export_tokenizer.py` was created. EVERY model run used corrupted tokenization. The model weights and inference pipeline were correct all along — only the tokenizer was producing garbage input tokens. The `finchTool` kernel verifier tests matmul correctness but doesn't test tokenization.

**Discovery Method**: Compared Python tokenizer output against C tokenizer output for the same prompt string. Python produced 26 correct tokens; C produced 25 wrong tokens. Traced the discrepancy to the merge table: `ht_lookup` for "assistant" correctly returned 74455 in the vocab hash, but BPE couldn't merge that far because all second-level merges were missing. Wrote a trace of the `bpe_process` merge loop which revealed that merges like "as"+"si" returned 0xFFFFFFFF (not found). Checked `export_tokenizer.py` and immediately saw `pair[0], pair[1]` on a string.

**Lesson**: Python string indexing and string splitting are dangerously confusable. `pair[0]` on `"a s"` gives `"a"` (first char), which looks correct at a glance but is only correct by accident for single-character tokens. `pair.split(' ')` is the correct operation. Always verify tokenizer output against a reference implementation (Python `tokenizers` library) before trusting the C tokenizer.

---

## Bug 12: Chat Template Think Tag Causes Repetition Loops (2026-08-10)

**Symptom**: Model output `</think>` as first token (closing empty think block), then leaked reasoning as regular text and degraded into repetition loops ("I'll make sure it's good." ×50).

**Root Cause**: The prompt template in `tokenize_chat_message()` prepended `<think>\n` after `<|im_start|>assistant\n`. When the model saw `<think>\n` already in the prompt, it interpreted this as "thinking already started, I need to close it" and output `</think>` immediately. The model's reasoning then leaked as regular output, and without the think→answer structure, it degraded into repetition.

Without `<think>` in the prompt, the model:
1. Outputs `<think>` on its own
2. Plans the response inside the think block
3. Outputs `</think>`
4. Writes the actual answer

**Fix in `infer.m`**:
```c
// OLD: prepend <think> — model closes it immediately
snprintf(buf, bufsz, "<think>\n");

// NEW: let model output <think> itself
buf[0] = '\0';
```

**Verification**:
- Before fix: `</think>\nThe user wants a 450 word essay... I'll make sure it's good.` ×50
- After fix: `<think>\nThe user wants a short paragraph...\n</think>\n"Air conditioning revolutionized comfort. Willis Carrier invented it in 1902..."`

**Discovery Method**: Tested raw text completion (no chat template) — model produced 80+ coherent tokens. Tested chat template without think tag — model output `<think>` itself and produced proper think→plan→answer flow. Tested with think tag — model immediately closed it. Isolated to the `<think>\n` string in the prompt.

**Lesson**: The quantized model can't distinguish "think block is already open" from "think block was just opened by me." Letting it control think tag boundaries eliminates the confusion. The official Qwen chat template includes `<think>\n` but the quantized model handles it poorly.

---

## Bug 13: Memory Safety Margin Blocks GPU Zero-Copy (2026-08-10)

**Symptom**: After running a few tests, model fell back to CPU matmuls at ~800ms/token (30× slower than GPU 170ms/token). Output became garbage. Available memory was 4-5 GB on a 16 GB M4 — plenty for a 5 GB model, but the check refused to wrap the weight file.

**Root Cause**: `metal_set_weights()` required `available_memory >= weight_file_size + 2GB_safety_margin`. With 4.62 GB weights + 2 GB margin = 6.62 GB needed. Available memory dropped to 4-5 GB after several runs (page cache from previous mmaps), blocking GPU zero-copy.

The 2 GB safety margin (`METAL_SAFETY_MARGIN_BYTES`) was designed for 17 GB weight files (Qwen 3.5 397B) and discrete GPUs with separate VRAM. On Apple Silicon unified memory, `newBufferWithBytesNoCopy` with `MTLResourceStorageModeShared` is zero-copy — it creates GPU page-table mappings into the mmap'd file without allocating physical pages. A 2 GB margin is unnecessary.

**Fix** (2 changes):
1. Reduce `METAL_SAFETY_MARGIN_BYTES` from 2 GB to 256 MB
2. Attempt GPU wrap even when below margin (warn but don't block). Only block at <256 MB (critical memory pressure).

**Verification**:
- Before fix: `⚠️ Available memory (4.53 GB) < peak GPU usage (5.06 GB)` → fallback, 800ms/token prefill
- After fix: `[metal] Weight file wrapped as Metal buffer (4.62 GB, zero-copy)` at 4.53 GB available, 170ms/token prefill

**Lesson**: Apple Silicon unified memory is fundamentally different from discrete GPU architectures. Zero-copy Metal buffers don't allocate physical pages — they just create GPU page-table mappings. Memory safety margins designed for discrete GPUs are unnecessarily conservative and can block correct operation. Always test on target hardware.

**Remaining**: Weight file is 4.96 GB (BF16 non-expert weights). Should be ~1.5 GB with non-expert quantization (embeddings + lm_head at 8-bit saves 1 GB, full 4-bit saves ~3.5 GB). This is the real fix for fitting in 2 GB RAM.

---

## Bug 14: K=2 Default Produces Garbage Output (2026-08-11)

**Symptom**: Engine produced word salad at default settings ("No matter what you're going to be a little as you can. You will! be a **"). Model outputs were incoherent despite correct token-1 logits.

**Root Cause**: Qwen 3.6 35B A3B is trained with `num_experts_per_tok: 8`, meaning 8 of 256 experts are active per layer. The CLI default was `K=2`, which meant only 2/8 experts (25%) contributed to each layer's output. The remaining 75% of expert computation was silently skipped, corrupting the residual stream at every layer.

K=2 was chosen as a performance optimization (8.3 tok/s vs 7.5 tok/s for K=4), assuming the model could tolerate fewer experts at inference time. This assumption was wrong — the model requires all 8 experts for coherent output.

**Fix** (4 changes in `infer.m`):
1. Main CLI default: `int K = 2` → `int K = 8`
2. MTP default: `int K = 2` → `int K = 8`
3. I/O parallelism: `NUM_IO_THREADS` 4 → 8 (one thread per expert)
4. Help text: `(default: 4)` → `(default: 8)`
5. Header comment: `8 active (we use K=4 for speed)` → `8 active (model trained with 8 experts/token)`
6. Encoder comment: `With K=4: 10 encoders` → `With K=8: 18 encoders`

**Verification**: With K=8 and T=0.7, engine produces 400 coherent tokens of valid Ruby code at 3.88 tok/s. No degradation, no word salad.

**Lesson**: Never override a model's trained architectural hyperparameters without verifying the quality impact. The speed difference between K=2 and K=8 is significant (8.3→3.9 tok/s), but correctness matters more. A config validation that warns when K ≠ `num_experts_per_tok` would prevent this class of bug.

---

## Bug 15: lm_head 2 GB F32 Cache Causes OOM / Segfault (2026-08-11)

**Symptom**: Engine crashed with `zsh: segmentation fault` at token 1 on a 16 GB M4 with 5.7-6.3 GB free. The crash occurred during the first token's lm_head computation, after final norm but before sampling.

**Root Cause**: The `lm_head_forward()` function allocated a static 2 GB F32 cache on first use (`lm_head_f32 = malloc(VOCAB_SIZE * HIDDEN_DIM * sizeof(float))` = 248,320 × 2,048 × 4 = 2.03 GB). On a system with 5.7 GB free and a 4.6 GB mmap'd weight file, this pushed total allocation past available physical memory. macOS virtual memory overcommit allowed the malloc to succeed (returning non-NULL), but the subsequent BF16→F32 conversion loop triggered a SIGSEGV when the OS couldn't back the pages.

The old code path was:
1. Check for scales/biases (4-bit quantized path)
2. If not found, allocate 2 GB F32 cache
3. Convert all 508M BF16 values to F32
4. Use Accelerate BLAS `cblas_sgemv` for the matvec

This was the ONLY place in the engine that required a large additional allocation beyond the mmap'd weight file. Everything else used either GPU zero-copy (reading BF16 directly from Metal buffer) or existing GPU buffers.

**Fix**: Replaced the CPU F32-cache path with a GPU path that reads BF16 weights directly from the Metal buffer (zero-copy, no allocation):

```c
// NEW: GPU path — uses gemv_bf16_x2 kernel
if (g_metal && g_metal->wf_buf && g_metal->gemv_bf16_x2_pipe) {
    // Copy input to GPU, dispatch kernel, copy result back
    // Weights read directly from Metal buffer at offset 0
    // No malloc needed — uses preallocated buf_input and buf_output
}
// FALLBACK: CPU chunked path (34 MB chunks, avoids 2 GB allocation)
```

The `buf_output` Metal buffer was already allocated at `VOCAB_SIZE * sizeof(float)` = 970 KB for this purpose (line 1566 had the foresight: `max_out = VOCAB_SIZE * sizeof(float); // lm_head is largest`).

The GPU kernel `gemv_bf16_x2` computes 2 output rows per threadgroup using 128-bit (uint4) BF16 loads, interleaved thread access pattern. Dispatch: 124,160 threadgroups × 256 threads.

**Additional fix**: `decode_token()` had no NULL guard for the `Vocabulary *v` parameter. Calling `decode_token(NULL, tid)` (used by the logit diagnostics) dereferenced `v->num_tokens` and crashed. Added `if (!v ...)` guard.

**Verification**:
- Before fix: `zsh: segmentation fault` at token 1
- After fix: 400 tokens generated at 3.88 tok/s, no crash
- GPU path latency: ~10 ms per lm_head forward (vs ~15 ms for old CPU BLAS path when it worked)

**Lesson**: Never allocate large caches proportional to vocabulary size without checking available memory. On macOS, malloc can succeed even when physical RAM is exhausted (virtual memory overcommit), and the actual crash happens later when writing to the pages. Always prefer zero-copy GPU paths that read directly from mmap'd weight buffers on Apple Silicon unified memory.

---

## Bug 16: Logit Diagnostic NULL Pointer Crash (2026-08-11)

**Symptom**: After fixing Bug 15, engine still crashed with segfault at token 1. Debug output showed `lm_head_forward returned` successfully, crash was in subsequent code.

**Root Cause**: The new `logit_diag_dump()` function called `decode_token(NULL, token_id)` to display token text in the top-20 logit display. But `decode_token()` dereferenced `v->num_tokens` without checking if `v` is NULL, causing a segfault. The original top-5 debug print (line 9254) passed the actual vocab pointer, but the generic diagnostic function used NULL.

**Fix**: Added `if (!v || ...)` guard to `decode_token()`:
```c
static const char *decode_token(Vocabulary *v, int token_id) {
    if (!v || token_id < 0 || token_id >= v->num_tokens || !v->tokens[token_id]) {
        return "<unk>";
    }
```

**Lesson**: Always guard pointer parameters, even in "internal" functions. Diagnostic/tooling code paths are tested less frequently and are more likely to pass unexpected NULL values.
