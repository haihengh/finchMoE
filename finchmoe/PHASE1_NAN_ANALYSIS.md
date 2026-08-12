# CMD2 NaN Analysis — Phase 1 Non-Expert Quantization

**Status**: Unresolved. Blocking Phase 1 (non-expert weight quantization).

---

## 1. What Changed

Switched `model_weights.bin` from BF16 non-expert weights (4.96 GB, extracted from
`Qwen3.6-35B-A3B-2bit-dense-v2`) to 4-bit quantized non-expert weights (1.39 GB,
extracted from `Qwen3.6-35B-A3B-4bit`).

| Component | Before (BF16) | After (4-bit) |
|-----------|--------------|---------------|
| Attention Q/K/V/O | BF16, `gemv_bf16_x2` kernel | U32-packed, `dequant_matvec_4bit_v3` |
| GDN projections | BF16 | 4-bit dequant |
| Shared expert | BF16, CPU fallback in CMD3 | 4-bit, GPU dequant in CMD3 |
| Routing gate | BF16 | 4-bit dequant |
| Embeddings | BF16 lookup | 4-bit dequant row lookup |
| lm_head | BF16, GPU gemv_bf16_x2 | 4-bit, GPU dequant matvec |
| Expert weights | **UNCHANGED** (same packed_experts files) | **UNCHANGED** |

**Everything that changed is in the non-expert weight path. Expert weights,
expert kernels, and the CMD3 routed-expert dispatch are byte-identical.**

## 2. Symptom Pattern

```
[prefill] 3/4 tokens: 367 ms    ← tokens 0-2 process CLEAN (no NaN)
[CMD2-NAN] layer=1: h_mid or h_post contains NaN!   ← token 3, layer 1
[LOOP] layer 1: hidden rms=nan!
[CMD2-NAN] layer=2 ... layer=39 (all NaN)
```

Key observations:
1. **Tokens 0-2 are clean** — no NaN during the first 3 prefill tokens.
2. **Token 3, layer 0 is clean** — no CMD2-NAN at layer 0.
3. **Token 3, layer 1+ all produce NaN** in CMD2 output.
4. `--cpu-experts` **fixes everything** (correct output, ~10× slower).
5. Prefill speed: 367 ms vs 3893 ms (BF16) — **5× faster**, so I/O is not the issue.

## 3. What Was Verified Correct

| Component | Method | Result |
|-----------|--------|--------|
| GPU dequant kernel (v3) | Standalone test with REAL weight data | CosSim=1.0 for [8192,2048], [4096,2048], [2048,4096] |
| GPU dequant via engine path | Single Metal buffer + offsets (exact engine code) | CosSim=1.0, MaxDiff < 1e-4 |
| Embeddings (4-bit lookup) | Engine debug | rms=7.7, finite ✓ |
| CMD1 attention projections | Engine debug (NaN check) | No NaN ✓ |
| Tensor offsets in manifest | Python verification | All offsets within file bounds ✓ |
| Tensor shapes vs kernel expectations | Python verification | All match ✓ |
| Weight/scale/bias triplets | Python verification | All present ✓ (0 missing in attention/MoE) |

**Conclusion: the GPU kernels, the data, and the CMD1 path are all correct.
The NaN originates somewhere between CMD1 and CMD2 output readback.**

## 4. The Failure Chain (Hypothesis A)

The NaN in CMD2's `h_mid` requires either:
1. `o_proj` output is NaN, OR
2. `residual` (input to residual_add) is NaN

CMD2's residual comes from `buf_residual`, populated by:
```c
memcpy([g_metal->buf_residual contents], residual, HIDDEN_DIM * sizeof(float));
```
where `residual` is the CPU-side hidden state.

For layer 1+ (FAST PATH), the hidden state comes from:
```c
finalize_deferred_experts():
    memcpy(g_deferred.hidden, [g_metal->buf_moe_hidden contents], ...)
```

`buf_moe_hidden` is written by CMD3(N-1)'s `moe_combine_residual` kernel, which reads:
1. `buf_multi_expert_out[k]` — routed expert outputs (**unchanged weights**)
2. `buf_shared_out` — shared expert output (**NOW 4-bit dequant down_proj**)
3. `buf_h_mid` — residual

**If `buf_shared_out` is NaN → combine output NaN → buf_moe_hidden NaN → hidden NaN → buf_residual NaN → CMD2 h_mid NaN.**

This explains every symptom:
- Layer 0 clean: slow path, hidden from embedding (not buf_moe_hidden)
- Layer 1+ NaN: fast path, hidden from buf_moe_hidden
- `--cpu-experts` fixes it: CPU fallback computes combine on CPU (correct)
- Tokens 0-2 clean: NaN propagates but is DISCARDED by `discard_deferred_experts()`
  (only the GDN state and KV cache persist, not the hidden state)

Wait — tokens 0-2 should also show CMD2-NAN at layers 1+! Unless the NaN only
appears at token 3 due to GDN state accumulation.

## 5. The Failure Chain (Hypothesis B) — GDN State Corruption

Tokens 0-2 update the 30 GDN recurrent states. If the GDN state is corrupted
(even slightly) by 4-bit quantized GDN projection weights, the corruption
accumulates over tokens. By token 3, the state produces NaN in the attention
output, which flows into o_proj → CMD2 h_mid NaN.

This explains why tokens 0-2 are clean but token 3 fails.

However, CMD1-NAN checks showed the GDN projections are finite at token 3.
The GDN state update happens on GPU (fused GPU delta-net) with the SAME
buffers as before — only the projection INPUTS changed.

## 6. What `--cpu-experts` Actually Changes

`--cpu-experts` (infer.m:6422) skips the ENTIRE GPU CMD3 encoding and uses:
1. CPU dequant matvec per expert
2. CPU combine (`hidden = h_mid + moe_out + shared_out`)

It also means `gpu_combine` is never set, so `finalize_deferred_experts()`
reads from CPU memory (`g_deferred.hidden`) instead of `buf_moe_hidden`.

This narrows the bug to ONE of these CMD3 GPU components:
1. `moe_combine_residual` kernel inputs (shared_out with 4-bit weights)
2. Shared expert SwiGLU + down_proj 4-bit dequant in CMD3
3. The GPU-side combine + norm chain (buf_moe_hidden → buf_input)

## 7. Most Likely Root Causes (Ranked)

### #1: Shared expert down_proj 4-bit dequant produces wrong values (60% confidence)

The CMD3 shared expert path changed from CPU BF16 to GPU 4-bit dequant.
The dimensions are [2048, 512] (out=2048, in=512), which is DIFFERENT from
the attention tensors tested standalone ([8192,2048], [2048,4096]).

The v3 kernel with in_dim=512: num_groups=8, packed_per_group=8.
Not verified standalone with these exact dimensions.

### #2: GPU combine reads stale buf_h_mid (25% confidence)

The combine kernel reads buf_h_mid from CMD2. CMD2 is committed and waited,
so buf_h_mid should be current. But the kernel ordering within CMD3 could
have a subtle issue: the shared expert down_proj reads buf_shared_act, which
is written by the SwiGLU dispatch earlier in CMD3. If the encoders aren't
properly ordered within CMD3...

### #3: Metal resource limit with more encoders (15% confidence)

CMD3 now has MORE encoders than before (the shared expert went from CPU to
GPU). More encoders = more resources. Could hit a Metal limit silently.

## 8. Next Debugging Steps

1. **Standalone test for shared expert dimensions**: Test v3 with
   [2048, 512] (down_proj) and [512, 2048] (gate/up) using real weight data.
2. **Check buf_shared_out directly**: Add debug print after CMD3 completes to
   see if buf_shared_out is NaN.
3. **Check buf_moe_hidden directly**: Verify whether the combine output is NaN.
4. **Disable GPU combine**: Force the slow path (CPU combine) and see if NaN
   disappears — would confirm the combine chain as the source.
5. **Isolate CMD3 shared expert**: Use BF16 for shared expert only (keep 4-bit
   for attention) by modifying the manifest, then test.

## 9. Files and Reproduction

```bash
cd finchmoe
# Extract 4-bit non-expert weights
python3 extract_weights.py --model ../models/Qwen3.6-35B-A3B-4bit --output ./quant_test

# Switch engine to quantized weights
ln -sf quant_test/model_weights.bin model_weights.bin
ln -sf quant_test/model_weights.json model_weights.json

# Reproduce NaN
./infer -t 3 -k 8              # NaN at layers 1+

# Verify correct with CPU experts
./infer -t 3 -k 8 --cpu-experts   # works correctly

# Verify GPU kernels standalone
clang -O2 -fobjc-arc -framework Metal -framework Foundation test_engine_path.m -o test_engine_path
./test_engine_path            # all PASS (attention dims only)
```
