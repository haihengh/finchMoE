# FinchMoE Optimization Roadmap: 12 tok/s @ ~2 GB RAM

## Where We Are vs Where We Need to Be

| Metric | Current (Aug 2026) | Target | Gap |
|--------|-------------------|--------|-----|
| Decode speed (M4 16GB) | 3.88 tok/s | 12 tok/s | **3.1×** |
| RAM footprint | ~6.0 GB | ~2.0 GB | **3.0×** |
| Common weight file | 4.96 GB (all BF16) | ~1.5 GB (quantized) | **3.3×** |
| Expert bit-width | 4-bit (1.77 MB/expert) | 2-bit (0.98 MB/expert) | **1.8× I/O** |

**Reference target**: turbo-fieldfare achieves 5.1-6.3 tok/s at ~2 GB on **M2 8GB** with Gemma 4 26B (30 layers, 128 experts, 4-bit everywhere). M4 is ~1.4× faster than M2, so turbo-fieldfare-level optimization on M4 would give **7-9 tok/s**. To reach 12 tok/s, we need to go beyond.

## Benchmark: Where Every Millisecond Goes

Per-token breakdown from 400-token run at K=8, 4-bit experts, warm cache:

```
                    ms/token   %     Bound by
┌────────────────────────────────────────────┐
│ CMD1 (attn projections)          90   35%  │ GPU dispatch overhead (5-9 encoders)
│ CPU attention compute            10    4%  │ Memory bandwidth (mostly memcpy)
│ CMD2 (o_proj+norm+routing)       60   23%  │ GPU dispatch (8 encoders)
│ CPU routing (softmax+topK)        5    2%  │ CPU (trivial)
│ Expert I/O (parallel pread)       40   15%  │ SSD random read IOPS
│ CMD3 (expert forwards, ASYNC)    30   12%  │ GPU compute (K×2 + shared)
│ CMD3 deferred completion         15    6%  │ GPU wait + CPU readback
│ Overhead (sync, misc)            10    4%  │ State management
├────────────────────────────────────────────┤
│ TOTAL per layer                 260  100%  │
│ TOTAL per token (40 layers)     260        │ → 3.85 tok/s
└────────────────────────────────────────────┘
```

## The Plan: 4 Phases to 12 tok/s

### Phase 1: Non-Expert Quantization → ~2.5 GB RAM, ~4 tok/s

**Status**: BLOCKED — see risks below. Must solve first.

**What**: Quantize all non-expert weights (embeddings, attention, GDN, shared experts) from BF16 to 4-bit/8-bit using the same MLX affine scheme as experts.

| Component | Current | Target | Savings |
|-----------|---------|--------|---------|
| Embeddings | 0.97 GB BF16 | 0.49 GB 8-bit | 0.48 GB |
| lm_head | 0.97 GB BF16 | 0.49 GB 8-bit | 0.48 GB |
| Attention Q/K/V/O | 1.84 GB BF16 | 0.46 GB 4-bit | 1.38 GB |
| GDN projections | 0.78 GB BF16 | 0.20 GB 4-bit | 0.58 GB |
| Shared experts | 0.26 GB BF16 | 0.07 GB 4-bit | 0.19 GB |
| Norms + gates | 0.05 GB BF16 | 0.05 GB BF16 | — |
| **Total** | **4.96 GB** | **~1.7 GB** | **3.3 GB** |

**Implementation**:
1. Modify `quantize_model.py`: enable `FOUR_BIT_DENSE` and `EIGHT_BIT_DENSE` lists
2. Fix GPU hang for large non-expert tensors — investigate `dequant_matvec_4bit_v3` with [2048, 2048] and larger shapes
3. Add 8-bit GPU dequant kernel (needed for embeddings/lm_head)
4. Run `extract_weights.py` against quantized safetensors
5. Update C engine: `gpu_batch_matvec` already handles scales/biases → 4-bit path
6. Quality validation: compare logits before/after quantization, ensure CosSim > 0.99

**Risk**: The `quantize_model.py` comments say "GPU path also hangs" for large non-expert tensors. This must be root-caused and fixed. The hang is NOT a fundamental limitation (turbo-fieldfare proves it works). Likely causes:
- Incorrect packed size calculation (in_dim must be divisible by 64 for 4-bit, 32 for 8-bit)
- Buffer offset alignment issue
- Threadgroup count exceeding Metal limits (though 2048/8=256 TG is well within limits)
- Scale/bias shape mismatch

**Speed impact**: GPU 4-bit dequant matvec should be similar speed to GPU BF16 matvec. The kernel does 8 values per uint32 load, same compute pattern. Slight overhead from scale/bias application.

### Phase 2: 2-Bit Experts → ~6 tok/s

**Status**: Already working (`--2bit` flag). Quality tested.

**What**: Switch from 4-bit to 2-bit expert quantization. Each expert drops from 1.77 MB to 0.98 MB.

| Metric | 4-bit | 2-bit | Change |
|--------|-------|-------|--------|
| Expert size | 1,769,472 B | 983,040 B | **1.8× less I/O** |
| Per-token I/O (8 experts × 40 layers) | 566 MB | 315 MB | **251 MB saved** |
| Expert I/O time (est.) | 40 ms | 22 ms | **18 ms saved per layer** |
| Total disk | 17 GB | 9.4 GB | Fits smaller SSDs |

**Speed projection**: I/O drops from 40ms to 22ms per layer. New per-layer: 260 - 18 = 242 ms → **4.13 tok/s**. With page cache warming: **~5-6 tok/s** (measured at tokens 300+ from the 400-token run, where speed reached 5-6 tok/s as pages stayed resident).

**Quality**: 2-bit-dense-v2 produces coherent output at K=8, T=0.7, rep_penalty=1.15. Slight quality difference vs 4-bit is acceptable for the speed gain. The model's 8-expert redundancy helps mask individual expert quantization errors.

**Implementation**: Already done. Just make `--2bit` the default and verify quality on long generations.

### Phase 3: GPU Pipeline Optimization → ~9 tok/s

**What**: Structural improvements to the Metal command buffer pipeline.

#### 3a. Fuse CMD1+CMD2 → ~4.5 tok/s

Currently CMD1 and CMD2 are separate command buffers with a commit+wait between them. CMD1 does attention projections (3-4 matvecs), then waits. CMD2 does o_proj+norm+routing (8 encoders), then waits.

**Proposal**: Merge into a single command buffer with 11-12 encoders, one commit+wait.

- Saves: ~10 ms per layer (driver commit+wait overhead)
- New per-layer: 242 - 10 = 232 ms → **4.31 tok/s**

**Implementation**: Combine `cmd1` and `cmd_fused` into a single command buffer. The only dependency is that CMD2 reads `buf_output` which CMD1's attention projections don't write. CMD1 writes `batch_out[0-3]`. CMD2 reads `batch_out[6]` (set by CPU after attention compute) and `buf_h_mid` (set by CMD2's own residual_add). So they're independent — they can share a command buffer.

#### 3b. ICB for CMD3 → ~5.5 tok/s

Metal Indirect Command Buffers (ICB) eliminate per-encoder encoding overhead. Instead of encoding K×2 individual expert dispatches, we pre-record the dispatch pattern once and replay it per layer.

- Saves: ~20 ms per layer (CMD3 encoding overhead for 16 expert encoders)
- New per-layer: 232 - 20 = 212 ms → **4.72 tok/s**

Combined with page cache warming: **~5.5 tok/s**.

#### 3c. Persistent GPU Workgroups (turbo-fieldfare's key insight)

The current expert kernels dispatch one threadgroup per expert output row (e.g., 512 TGs for gate_proj [512, 512]). turbo-fieldfare's breakthrough was using persistent workgroups that claim rows until the dispatch completes, giving the GPU more scheduling flexibility.

- Saves: ~15 ms per layer (GPU occupancy improvement)
- New per-layer: 212 - 15 = 197 ms → **5.08 tok/s**

Combined: **~6 tok/s**.

#### 3d. MTP Speculative Decoding → ~9-12 tok/s

Multi-Token Prediction infrastructure is already complete in the codebase. The MTP head predicts 1 future token; the main model verifies. At 50% acceptance rate (measured previously), effective throughput doubles.

- Effective multiplier: 1.5-2.0×
- Final speed: 6 × 1.5 = **~9 tok/s** (conservative)
- With 2-bit experts: 6 × 1.5 × (additional I/O savings) = **~10-12 tok/s**

**Implementation**: Fix latent MTP bugs, enable by default. The MTP code path exists but was disabled after quality issues. Need to:
1. Verify MTP forward math against reference
2. Tune acceptance threshold
3. Handle MTP+sampling interaction (temperature, rep penalty)

### Phase 4: Advanced → Beyond 12 tok/s

These are higher-risk, longer-term items that could push past 12 tok/s:

| Optimization | Est. Gain | Complexity | Notes |
|-------------|-----------|------------|-------|
| Single-kernel multi-expert | +20% | Very High | All K experts in one GPU dispatch. Occupancy goes up, encoding overhead eliminated. |
| Batched GPU prefill | 5-10× PP | High | Current prefill is CPU-only. GPU batching needed for production. |
| KV cache FP16 | -448 MB | Medium | Already in turbo-fieldfare. Our KV is FP32. |
| 3-bit experts | +15% speed, better quality | Medium | Sweet spot between 2-bit (fast, some quality loss) and 4-bit (slow, reference quality). Need 3-bit dequant kernel + repacker support. |
| Expert prediction v2 | 10-20% I/O reduction | Medium | Current predictor 41% accurate. Better predictor (attention-based, learned) could reach 60-70%. |

## RAM Budget Projection

| Phase | Common Weights | GPU Buffers | Expert Slots | KV Cache (4K) | Total |
|-------|---------------|-------------|-------------|---------------|-------|
| Current | 4.96 GB | 0.45 GB | 0.06 GB | 0.02 GB | **~5.5 GB** |
| Phase 1 | 1.70 GB | 0.45 GB | 0.06 GB | 0.02 GB | **~2.2 GB** |
| Phase 2 | 1.70 GB | 0.45 GB | 0.06 GB | 0.02 GB | same (2-bit doesn't change RAM) |
| Phase 3 | 1.70 GB | 0.45 GB | 0.06 GB | 0.02 GB | same |
| +FP16 KV (Phase 4) | 1.70 GB | 0.23 GB | 0.06 GB | 0.01 GB | **~2.0 GB** ✅ |

turbo-fieldfare's 1.35 GB common weights + ~0.3 GB KV + ~0.2 GB scratch ≈ 1.9-2.1 GB. We can match this.

## Implementation Order & Dependencies

```
Phase 1 (Non-expert quant) ──────┐
  ├─ Fix GPU 4-bit hang          │
  ├─ Add 8-bit GPU dequant       │
  └─ Verify quality              │
                                  │
Phase 2 (2-bit experts) ─────────┤  ← Independent, can start immediately
  └─ Make default, verify        │
                                  │
Phase 3a (Fuse CMD1+CMD2) ───────┤  ← Independent
Phase 3b (ICB for CMD3) ─────────┤  ← Independent
Phase 3c (Persistent workgroups)─┤  ← Independent
Phase 3d (MTP enable) ───────────┘  ← Needs quality verification

Phase 4 (Advanced) ← Can run in parallel with Phase 3
```

**Critical path**: Phase 1 is the bottleneck. Without non-expert quantization, RAM stays at ~5.5 GB and we can't reach the 2 GB target. All other phases build on this.

## What turbo-fieldfare Does That We Don't (Yet)

| Feature | turbo-fieldfare | FinchMoE | Gap |
|---------|----------------|----------|-----|
| Non-expert quantization | 4-bit everywhere | BF16 everywhere | **Must fix** |
| Expert streaming | LFU cache, 16 slots/layer | LRU (optional), 8 slots/layer | Small gap |
| Prefill | Chunked GPU (MPP) | CPU-only (slow) | Large gap |
| KV cache | FP16, ring buffers | FP32, linear | Medium gap |
| Output head | Fused 4-bit | GPU gemv_bf16 (good) | Equivalent |
| MoE kernel | Persistent workgroups | Independent dispatches | Medium gap |
| Pipeline | cb1→io→cb2 (3-phase) | CMD1→CPU→CMD2→CPU→CMD3 (5-phase) | We have MORE phases |
| Weight file | 1.35 GB | 4.96 GB | 3.7× |
| Install | Remote streaming repack | Local copy | Different use case |

## First Actions (This Week)

1. **Reproduce turbo-fieldfare's non-expert quantization success**
   - Take a single attention tensor (e.g., Q-projection [2048, 2048])
   - Quantize to 4-bit with group_size=64
   - Test GPU dequant matvec with our existing kernel
   - If it hangs: debug with smaller tensor, check buffer sizes
   - If it works: quantize all non-experts, run full model

2. **Benchmark 2-bit experts at K=8**
   - Run 400-token generation with `--2bit`
   - Measure speed and quality
   - If acceptable, make default

3. **Fuse CMD1+CMD2**
   - Merge two command buffers into one
   - Measure per-layer time reduction
   - Verify correctness with `--compare-experts`

4. **Profile current GPU dispatch overhead**
   - Use Metal profiler to measure per-encoder cost
   - Quantify ICB potential savings
   - Determine if CMD3 encoding is the bottleneck

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| GPU hang with non-expert quantization | Critical | Incremental testing, start with smallest tensor, check buffer sizes |
| 2-bit quality degradation at K=8 | Medium | Already tested and produces coherent output; monitor long generations |
| MTP quality issues | Medium | Latent bugs from previous attempt; needs thorough testing |
| ICB complexity | Low | Metal supported since M1; well-documented API |
| Fuse CMD1+CMD2 correctness | Low | Independent dispatches, no data dependency between attention proj and o_proj |

## Success Criteria

- ✅ 12 tok/s decode on M4 16GB with 4K context
- ✅ <2.5 GB process RSS
- ✅ Coherent 400+ token generation (no degradation)
- ✅ Same token IDs as reference for first 50 tokens (quality checkpoint)

If we reach 9-10 tok/s with phases 1-3, that's a 2.5× improvement over current and validates the approach. The jump to 12 tok/s depends on MTP acceptance rates and expert cache performance.
