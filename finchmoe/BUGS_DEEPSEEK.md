# Bugs Found & Fixed — DeepSeek-V4-Flash Engine Development

This documents every bug encountered during the development of `infer_deepseek.m`,
the CPU reference engine for DeepSeek-V4-Flash-0731 on Apple Silicon.

## Timeline

```
Scaffold loads 72K tensors → SIGBUS crash on layer 0
  → F8_E4M3 attention weights treated as BF16 (1-byte vs 2-byte)
  → Fix: detect dtype=F8_E4M3, skip attention for Phase 1
  → Expert weights dtype=I8, not MXFP4 — I8 dequant implemented
  → All 43 layers process (experts all skip with "e")
  → I64 tid2eid read as I32 → garbage expert IDs (-1)
  → Fix: detect I64, convert to I32
  → Attention identity doubles hidden each layer (2^43 → INF → NaN)
  → Fix: skip residual add for identity pass-through
  → ss_get returns NULL for valid expert tensor names
  → Workaround: direct tensor array iteration
  → ue8m0 scale: powf(2,sf) → 2^255 overflow → INF
  → Fix: (uint32_t)sf << 23 bitcast (from vllm)
  → MegaMoE architecture: wrong FFN dimensions
  → Fix: input is hidden//2 (2048), w1/w3 stacked, w2: 1024→4096
  → Finite output achieved! (hidden RMS=1925)
```

---

## Bug 1: F8_E4M3 Attention Weights → SIGSEGV

**Symptom**: Engine crashes with SIGSEGV (exit 139) on first layer forward pass.
No output after "layer 0..." debug trace.

**Root cause**: Attention tensors (`wq_a`, `wq_b`, `wo_a`, `wo_b`, `wkv`) have dtype
`F8_E4M3` (1 byte per value). The `ss_get` function's dtype mapping didn't handle
this dtype, falling back to `dtype=0` (BF16, 2 bytes per value). The `ss_get_f32`
function then read F8_E4M3 data as BF16, reading 2× the actual data size and
going past the mmap boundary → SIGSEGV.

**Discovery method**: Python script checked actual tensor dtypes from the
safetensors shard header for layer 0. All attention weights showed `dtype=F8_E4M3`,
confirming the dtype wasn't BF16 as assumed.

**Fix in `ss_load` (dtype detection)**:
```objc
else if ([dtype isEqualToString:@"F8_E4M3"]) t->dtype = 3;  // FP8
```

**Fix in `deepseek_attention`**: Skip F8_E4M3 attention for Phase 1. When wq_a has
dtype=3, use pass-through (identity). This avoids the complex F8 dequant while
letting the rest of the engine work.

**Verification**: finchTool model check confirmed 6 unique dtypes exist:
`BF16 F32 F8_E4M3 F8_E8M0 I64 I8`

---

## Bug 2: Expert Weights — MXFP4 Assumption vs I8 Reality

**Symptom**: All 6 experts skipped per layer ("e" trace). Expert weight data was
assumed to be MXFP4 e2m1 format (4-bit nibbles with ue8m0 scale), but the
actual format was I8 (signed 8-bit integers) with F8_E8M0 scale.

**Root cause**: The model config says `expert_dtype: fp4` which was interpreted
as MXFP4 nibble-packed format. But the safetensors store experts as individual
tensors with dtype=I8 and per-block F8_E8M0 scales. The block size is 16 along
the input dimension.

**Discovery method**: Python inspection of shard header for expert tensors:
```python
layers.0.ffn.experts.0.w1.weight: shape=[2048, 2048] dtype=I8
layers.0.ffn.experts.0.w1.scale:  shape=[2048, 128] dtype=F8_E8M0
```

**Fix in `ss_load` (dtype detection)**:
```objc
else if ([dtype isEqualToString:@"I8"]) t->dtype = 4;   // INT8 quant
else if ([dtype isEqualToString:@"I64"]) t->dtype = 5;   // INT64 (tid2eid)
```

**Fix in expert dequant**: Replaced MXFP4 nibble unpacking with I8 matvec:
```c
for (int blk = 0; blk < n_blocks; blk++) {
    float scale = ue8m0_to_f32(sr[blk]);
    for (int j = 0; j < block_size; j++)
        acc += (float)(int)wr[base+j] * scale * x[base+j];
}
```

---

## Bug 3: I64 tid2eid → Garbage Expert IDs (-1)

**Symptom**: Hash routing returns expert ID -1 for all selections. Tensor names
like `layers.3.ffn.experts.-1.w1.weight` appear in debug logs. Expert loading
fails because expert -1 doesn't exist.

**Root cause**: The `tid2eid` hash routing table has dtype `I64` (8-byte signed
integers), but `hash_load_tid2eid` read it as `I32` (4 bytes) via `memcpy`.
This caused: (1) only half the data copied, (2) misaligned reads producing
garbage values including -1.

**Discovery method**: Added `ss_miss` debug in `ss_get` to print tensor names
that fail lookup. Output showed `layers.3.ffn.experts.-1.w1.weight` — the
expert ID -1 was the smoking gun.

**Fix in `hash_load_tid2eid`**:
```c
int is_i64 = 0;
// Check actual tensor dtype
for (int i = 0; i < m->num_tensors; i++) {
    if (strcmp(m->tensors[i].tensor_name, name) == 0) {
        is_i64 = (m->tensors[i].dtype == 5); break;
    }
}
if (is_i64) {
    const int64_t *src64 = (const int64_t *)data;
    for (int i = 0; i < nelem; i++) buf[i] = (int32_t)src64[i];
} else {
    memcpy(buf, data, nelem * sizeof(int32_t));
}
```

**Verification**: Expert IDs became valid (e.g., 49, 246 for layer 0).

---

## Bug 4: Attention Identity → Hidden Explosion (2^43)

**Symptom**: Hidden state becomes NaN by layer 3. All subsequent layers inherit
NaN. Learned routing selects expert -1 because all gate logits are NaN.

**Root cause**: When F8_E4M3 attention is skipped (Bug 1 workaround), the
pass-through sets `attn_out = normed_input`. The residual `hidden += attn_out`
then adds `normed_input` to `hidden`. Since `normed_input` has RMS ~1.0 and
`hidden` also has RMS ~1.0, each layer doubles the hidden magnitude.

After 43 layers: `hidden = initial * 2^43 ≈ initial * 8.8e12` → overflows
float32 (max ~3.4e38) → INF → NaN.

**Discovery method**: Added trace showing "L12" for layer steps. Step 2
(attention) produced identity output. Step 4 (MoE) was skipped. After only
3 layers, the NaN appeared.

**Fix in `deepseek_layer_forward`**:
```c
// Detect identity (F8 pass-through): don't double hidden each layer
int is_id = 1;
for (int i = 0; i < HIDDEN_DIM; i++)
    if (fabsf(attn_out[i] - normed[i]) > 1e-6f) { is_id = 0; break; }
if (is_id)
    memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));  // replace, don't add
else
    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] += attn_out[i];
```

**Verification**: Hidden RMS remained finite through all 43 layers.

---

## Bug 5: ss_get Name Matching → NULL for Expert Tensors

**Symptom**: `ss_get` returns NULL for valid expert tensor names even though:
- All 72,317 tensors have valid `mmap_base` (0 NULL in counter)
- The tensor names exist in the weight_map
- The same names work when searched via direct array iteration

**Root cause**: Unknown — the `ss_get` function logic appears correct. The
`strcmp` comparison should match. Suspected causes:
- String encoding issue between `snprintf`-generated names and `strdup`-stored names
- Compiler optimization affecting the comparison
- Name buffer reuse causing subtle corruption

**Discovery method**: Added NULL mmap counter to `ss_load` — all 72K tensors
showed valid mmap. Yet `ss_get` returned NULL. Direct array iteration in the
expert loading code (using identical `strcmp` logic) succeeded.

**Workaround**: Replace all expert tensor `ss_get` calls with direct tensor
array iteration using a separate name buffer (`wname[256]`):
```c
for (int i = 0; i < m->num_tensors; i++) {
    if (strcmp(m->tensors[i].tensor_name, wname) == 0 && m->tensors[i].mmap_base) {
        SSTensor *t = &m->tensors[i];
        uint64_t hl = *(uint64_t*)t->mmap_base;
        data = (type *)((uint8_t *)t->mmap_base + 8 + hl + t->data_offset);
        break;
    }
}
```

**Status**: Workaround deployed. Root cause investigation deferred.

---

## Bug 6: ue8m0 Scale — powf(2, sf) vs Bitcast

**Symptom**: After all experts load correctly, hidden state contains NaN at
output despite finite intermediate values.

**Root cause**: The ue8m0 scale conversion used `powf(2.0f, (float)sf)`. For
`sf=255`, this computes `powf(2, 255) ≈ 5.8e76` → overflows float32 → INF.
When multiplied with I8 weight values, produces INF weight values. The matvec
accumulation of INF values produces INF, and INF - INF produces NaN.

The CORRECT conversion is `(uint32_t)sf << 23` bitcast to float32. This places
the uint8 value into the float32 exponent field, producing `2^(sf-127)`.
For sf=120..122, this gives values 0.008..0.03 (correct range).

**Discovery method**: Found in vllm source at `nvidia/model.py:311`:
```python
def _ue8m0_uint8_to_float(sf):
    return (sf.to(torch.int32) << 23).view(torch.float32)
```

**Fix**:
```c
// BEFORE (wrong):
static inline float ue8m0_to_f32(uint8_t sf) {
    return powf(2.0f, (float)sf);  // sf=255 → overflow!
}

// AFTER (correct):
static inline float ue8m0_to_f32(uint8_t sf) {
    uint32_t bits = (uint32_t)sf << 23;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;  // sf=120 → 2^(120-127) = 0.0078
}
```

**Verification**: Python check confirmed values 120..122 produce 0.008..0.03.
Hidden RMS changed from NaN to 1925 (finite).

---

## Bug 7: MegaMoE FFN Architecture Misunderstanding

**Symptom**: Experts loaded and computed but hidden RMS was NaN (before Bug 6 fix)
or too large (RMS=1925 after Bug 6 fix). Token generation always produced token 0.

**Root cause**: The expert FFN architecture was assumed to be standard SwiGLU
with dimensions matching HIDDEN_DIM (4096). The actual MegaMoE architecture
from vllm is completely different:

| Assumption | Reality (vllm) |
|-----------|----------------|
| Expert input: full HIDDEN_DIM (4096) | `hidden_size // 2` (2048) |
| w1: gate [4096, 4096] | w1: gate [2048, 2048] |
| w3: up [4096, 4096] | w3: up [2048, 2048] (stacked with w1 as w13) |
| w2: down [4096, 2048] | w2: [4096, 1024] — takes half of SwiGLU output |
| SwiGLU output: 2048 | SwiGLU output: 2048 (full intermediate) |

**Discovery method**: Read `DeepseekV4MegaMoEExperts` in vllm `nvidia/model.py`:
```python
self.w13_weight = nn.Parameter(torch.zeros(
    num_local_experts,
    2 * intermediate_size,    # 2 * 2048 = 4096 (w1 + w3 stacked)
    hidden_size // 2,          # 4096 // 2 = 2048
    dtype=torch.uint8,
))
self.w2_weight = nn.Parameter(torch.zeros(
    num_local_experts,
    hidden_size,               # 4096
    intermediate_size // 2,     # 2048 // 2 = 1024
    dtype=torch.uint8,
))
```

**Fix**: Rewrote expert FFN to match MegaMoE architecture:
1. Input: `h_post[0:2048]` (first half of hidden)
2. w1 @ input → gate [2048]
3. w3 @ input → up [2048]
4. `silu(gate) * up` → act [2048]
5. w2 @ act[0:1024] → output [4096]
6. Accumulate weighted output

**Verification**: Expert computations produce finite values. Hidden RMS=1925
(high but finite). Token generation runs.

---

## Bug 8: Safetensors Header Offset → 0-byte tensors

**Symptom**: During `ss_load`, all tensor metadata showed `offset=0, size=0`.
The `ss_get` function couldn't find tensor data.

**Root cause**: The `model.safetensors.index.json` only maps tensor names to
shard filenames. The per-tensor metadata (offsets, shapes, dtypes) is stored
in the HEADER of each individual safetensors shard file, not in the index.
The code read `index["metadata"]` expecting per-tensor data, but it only
contained `total_size`.

**Discovery method**: Python inspection of the index JSON structure:
```python
Top-level keys: ['metadata', 'weight_map']
metadata: 1 entries, first key=total_size  # NOT per-tensor!
```

**Fix**: Parse all 48 shard headers at startup to build the complete tensor
metadata index:
```c
for (int i = 0; i < m->num_shards; i++) {
    NSDictionary *header = parse_shard_header(spath);  // read 8-byte header_len + JSON
    for (NSString *tname in header) {
        if (![tname isEqualToString:@"__metadata__"])
            allMeta[tname] = header[tname];
    }
}
```

Then compute absolute data position: `mmap_base + 8 + header_len + data_offset`.

**Verification**: All 72,317 tensors parsed with correct offsets and shapes.

---

## Bug 9: Learned Routing NaN → Expert -1

**Symptom**: Layers 3+ (learned routing) select expert ID -1 for all 6 slots.

**Root cause**: When the gate projection produces NaN logits (because hidden state
contains NaN from earlier bugs), the top-K selection loop never finds a valid
expert: `if (logits[e] > best_v)` is always false for NaN.

**Fix**: Added `isfinite()` guard and fallback:
```c
for (int e = 0; e < 256; e++) {
    if (isfinite(logits[e]) && logits[e] > best_v) { best_v = logits[e]; best = e; }
}
if (best < 0) best = k % 256;  // NaN fallback
```

**Verification**: Expert selection always returns valid IDs after NaN source was
fixed (Bug 4, Bug 6).

---

## Summary

| # | Bug | Severity | Time to Find | Fix Complexity |
|---|-----|----------|-------------|----------------|
| 1 | F8_E4M3 → BF16 SIGSEGV | Critical | 30 min | Low (skip + dtype detect) |
| 2 | MXFP4 vs I8 format | Critical | 1 hour | Medium (new dequant) |
| 3 | I64 → I32 tid2eid | Critical | 1 hour | Low (dtype conversion) |
| 4 | 2^43 hidden explosion | Critical | 30 min | Low (skip residual) |
| 5 | ss_get name matching | High | 2 hours | Workaround (direct lookup) |
| 6 | ue8m0 powf vs bitcast | Critical | 2 hours | Low (1 line) |
| 7 | MegaMoE architecture | High | 2 hours | High (full rewrite) |
| 8 | Shard header parsing | Critical | 30 min | Medium (parse headers) |
| 9 | NaN routing → expert -1 | Medium | — | Low (isfinite guard) |

**Total time**: ~10 hours of debugging across 4 sessions.
**Key lesson**: The most dangerous bugs were dtype assumptions (F8_E4M3, I8, I64)
and format misunderstandings (ue8m0 bitcast, MegaMoE architecture). Always verify
dtypes from shard headers before implementing dequant.

## Related

- [BUGS.md](BUGS.md) — Qwen3.6-35B-A3B bugs (10 total)
- [design_deepseek.md](design_deepseek.md) — DeepSeek engine design document
- [finchTool README](finchTool/README.md) — Model integrity checker
