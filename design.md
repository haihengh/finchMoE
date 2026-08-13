# FinchMoE Design Document

*Last updated 2026-08-13 — reflects commits through 753dde3.*

## 1. Overview

FinchMoE is a C/Metal inference engine for Qwen 3.6 35B A3B on Apple
Silicon. The engine is **bit-exact** against an independent numpy
reference of the GDN chain (CosSim 1.000000 per stage) and runs at
**11.4-12.4 tok/s** on M4 with **1.95 GB weights** and ~2.8 GB total RAM
at 8k context. See [finchmoe/OPTIMIZATION_PLAN.md](finchmoe/OPTIMIZATION_PLAN.md)
for the commit-by-commit progress log and open issues.

## 2. Model Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 40 (30 linear-attention GatedDeltaNet + 10 full attention, every 4th) |
| Hidden dim | 2048 |
| Attention | 16 Q heads, 2 KV heads, head_dim 256 (GQA) |
| GDN | 16 K heads / 32 V heads, key/value dim 128, conv kernel 4 |
| MoE | 256 routed experts, K=8 active + 1 shared expert, intermediate 512 |
| Vocab | 248,320 tokens |

## 3. Weight Inventory (Current Production — quant_self/)

| Component | Format | Size |
|-----------|--------|------|
| GDN qkv/z/out_proj | 4-bit affine, group 64 (the default tier; `FINCHMOE_GDN8=1` → 8-bit) | |
| Attention Q/K/V/O | 4-bit affine, group 64 | |
| beta/alpha gates, norms, conv, routing gate | BF16 (kept) | |
| Embeddings + lm_head | 8-bit, group 64 | |
| **Total non-expert** | | **1.95 GB** |
| Routed experts | **3-bit** affine, group 64 (8 values/24 bits, 1.31 MB/expert) | 13 GB disk, streamed |

Bit-width is unambiguous from manifest shapes (row_u32 == groups×16 → 8-bit,
== groups×8 → 4-bit); the engine detects it at cache-build time
(`tensor_bits()`).

## 4. Inference Pipeline (Per Token, per Layer)

```
CMD1+CMD2 — ONE command buffer, ONE commit+wait:
  [1] attention projections: qkv/z/alpha/beta (4 matvecs, v3/8-bit/BF16 per tensor)
  [2] fused_gdn_full: conv1d + qk-norm + decay/beta + delta recurrence +
      gated norm — ONE kernel, zero intermediate global round trips
  [3] o_proj (v3 tiled / 8-bit / BF16 by tensor bits)
  [4] residual_norm_fused: h_mid = residual + o_proj AND h_post = rms_norm(h_mid)
      — ONE kernel; fast path reads buf_moe_hidden directly as the residual
      (GPU-ordered via the serial queue, no CPU memcpy)
  [5] routing_batch_fused: gate(8-bit) + sg/su(4-bit) + seg(8-bit) — ONE
      mixed-bits kernel
CPU: softmax + top-K + parallel pread of K experts (3-bit, 1.31 MB each)
CMD3 — ASYNC (deferred, overlaps the next layer):
  K expert forwards + shared SwiGLU/down + moe_combine_residual
  → buf_moe_hidden (next layer's input norm + residual source)
```

**Key mechanisms**
- **One round trip per layer** (CMD1+CMD2 fused; CMD3 deferred) — the
  structural win worth ~0.4 ms/layer.
- **Fused GDN kernel** (`fused_gdn_full`): one threadgroup per v-head
  computes its 384 conv elements into threadgroup memory, normalizes q/k
  in place, and runs the recurrence + gated norm without touching global
  memory for intermediates.
- **Weight memory bandwidth is the fused-wait wall**: ~42 MB/layer at
  8-bit GDN vs ~21 MB at 4-bit GDN — the 4-bit tier is the +25% speed win
  (measured). Dispatch count is NOT the wall (proven by neutral
  micro-fusions).
- **3-bit experts**: 8 values per 24-bit triplet; byte-addressed tiled
  kernel. 9.09 tok/s with 4-bit-quality output; beats 2-bit on speed too
  (page-cache friendliness).

## 5. Memory Layout (8k context)

| Component | Size |
|-----------|------|
| Weights (mmap + zero-copy Metal wrap) | 1.95 GB |
| CPU KV (10 full-attn layers, K+V fp32, `-N`) | 0.34 GB @ 8k / 10.7 GB @ 256k ⚠️ |
| GPU KV (`-Q`, default 8192) | 0.34 GB |
| Delta-net state (30 × 32×128×128 fp32) | 63 MB |
| Expert buffers (16 × 2 MB-aligned) | 64 MB |
| **Peak GPU (budget report)** | **2.25 GB** |

256k context does NOT fit 16 GB (10.7 GB CPU KV alone) — disk-backed KV
is a future item. The startup budget report only counts the GPU side;
CPU KV is not budgeted (known gap).

## 6. GPU Kernels (shaders.metal)

| Kernel | Purpose | Notes |
|--------|---------|-------|
| `dequant_matvec_4bit_v3` | 4-bit dequant matvec | 8 rows/TG × 256 threads, x_shared cache |
| `dequant_matvec_8bit` | 8-bit dequant matvec | same tiled structure |
| `dequant_matvec_3bit` | 3-bit dequant matvec | byte-addressed 24-bit triplets |
| `dequant_matvec_2bit` / `1bit` | low-bit experts | |
| `dequant_matvec_4bit_fast` | 1 row/TG × 64 threads | in_dim > 4096 only |
| `fused_gdn_full` | conv1d+qk-norm+decay+delta+gated | ONE kernel, threadgroup-local |
| `fused_gdn_core` | decay+delta+gated (fallback chain) | |
| `residual_norm_fused` | residual_add + rms_norm | writes buf_h_mid + buf_input |
| `routing_batch_fused` | gate+sg+su+seg in one mixed-bits dispatch | section descriptors |
| `moe_combine_residual` | expert combine + residual + norm | reads buf_moe_hidden residual |
| `attn_scores/softmax/values_batched` | GPU full attention (kv ≥ 32) | |
| `gated_delta_net_step`, `conv1d_step`, `rms_norm_qk` | fallback chain pieces | |

All kernels verified against CPU references (CosSim 1.0); the layer-0
chain verified end-to-end by `debug_gdn_reference.py`.

## 7. Per-Layer Timing (3-bit experts + 4-bit GDN, K=8, M4)

| Phase | ms/layer | Bound by |
|-------|----------|----------|
| Fused GPU wait (CMD1+CMD2) | 1.33 | weight memory bandwidth (~21 MB/layer) |
| Expert I/O | 0.94 | SSD pread (3-bit, page-cached) |
| CPU routing + readbacks | ~0.15 | trivial |
| **Total** | **~2.5** | → ~11.4 tok/s |

Prefill = decode speed (~12 tok/s; every prompt token runs the serial
40-layer pipeline) — batched GPU prefill is the agentic-workload priority.

## 8. Key Design Decisions

1. **Quantized everything** (4-bit GDN tier + 3-bit experts default):
   the community-model failures were calibration issues, not bit-width;
   our affine quantizer is coherent at these widths. 8-bit GDN tier
   available via `FINCHMOE_GDN8=1` (quality-safe fallback).

2. **3-bit experts beat 2-bit on speed AND quality**: 1.31 MB/expert
   page-caches better than 0.98 MB suggests, with near-4-bit quality.

3. **GPU-side residual from buf_moe_hidden**: the fast path's
   residual_add reads the previous layer's combine output directly on
   GPU — eliminates the CPU round trip and enables the single
   commit+wait per layer.

4. **K=8 always**: the model is trained for 8 experts/token; K=2
   produces garbage (historical trap documented in the optimization log).

5. **No custom expert cache**: OS page cache wins; temporal prediction
   (--predict) measured net-negative (~50% hits, overhead > savings).

6. **Think budget (-B)**: forces `</think>` after N reasoning tokens;
   needed because story/edge prompts can loop in the thinking phase.

7. **Weights must come from the pristine BF16 base**: the current build
   quantized the marginal `2bit-dense-v2` variant, whose edge-prompt
   behavior degrades (repetition loops) while llama.cpp Q4_K_M of the
   clean base handles the same prompts correctly. Re-quantizing from
   `Qwen3.6-35B-A3B-bf16` is Priority 0.

## 9. Optimization History

| Step | tok/s | What |
|------|-------|------|
| Roadmap baseline (4-bit exp, K=8) | 3.85 | reference point |
| Self-quantized weights (8-bit GDN tier) | 6.90 | Phase 1 quality fix |
| CMD1+CMD2 single round trip | 7.32 | structural sync win |
| 3-bit experts | 9.09 | +32%; quality held |
| **4-bit GDN tier (default)** | **11.4-12.4** | bandwidth halved on the dominant traffic |
| micro-fusions (residual+norm, routing batch) | neutral | proved dispatch count ≠ the wall |
| full GDN fusion | neutral | proved intermediates ≠ the wall |

## 10. Future

See [finchmoe/OPTIMIZATION_PLAN.md](finchmoe/OPTIMIZATION_PLAN.md). In order:
1. Requant from pristine `Qwen3.6-35B-A3B-bf16` (edge-prompt quality).
2. Batched GPU prefill (agentic workloads; 5-10× PP target).
3. Server multi-turn session fix.
4. MTP speculative decoding (α-gated; forward-math verification first).

## 11. Hardware Targets

| Device | RAM | Current | Target |
|--------|-----|---------|--------|
| M4 Mac mini 16 GB | 16 GB | **11.4-12.4 tok/s** | 12-15 tok/s (MTP) |
| iPhone | 8 GB | — | 3-5 tok/s (all-4-bit, SSD streaming) |
