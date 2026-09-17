# Prefill experiments

[Previous: Attention and KV cache](05-attention-and-kv-cache.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Fusions, head, and orchestration](07-fusions-head-and-orchestration.md)

Production prefill uses 128-token chunks, projection-specific GEMV/QMM
selection, bounded staged affine MPP for INT4 projections, and batched routed
MoE. Several candidates improved one shape, allocation count, or host but
failed the M2 long-row gate.

| Current result | Disposition |
| --- | --- |
| Chunk 128, staged affine MPP, and batched routed MoE | Production |
| Apple10 TensorOps full attention | Production on Apple10; tiled fallback elsewhere |
| Shared/fetch overlap v3 | Rejected and removed |
| Shared INT8 QMM, deeper lookahead, and argument-buffer rings | Rejected on M2 |

## Shape and scheduling

<a id="pf-01"></a>
### PF-01: Mixed QMM and GEMV dispatch

- **Hypothesis:** Matrix kernels help only where token count and projection shape
  amortize setup.
- **Variants tested:** QMM and repeated GEMV by projection at `T = 32`.
- **Evidence:** QMM was slower for full Q and neutral for SWA Q; KV/O and
  other projection families benefited.
- **What changed the conclusion:** The
  shape matrix replaced a single global rule.
- **Final disposition:** Production
  policy.
- **Lesson:** Select prefill kernels by projection and shape.

<a id="pf-02"></a>
### PF-02: Chunk size 128

- **Hypothesis:** Larger chunks would amortize per-chunk work without breaking
  the 8 GB budget.
- **Variants tested:** 32 and 128 tokens per chunk on M2.
- **Evidence:** Prefill fell from 15.80 to 9.34 s at 121 tokens, 49.80 to 26.61 s
  at 527, and 92.89 to 52.35 s at 1017.
- **What changed the conclusion:** Nothing;
  memory and correctness gates passed.
- **Final disposition:** Production.
- **Lesson:** A configuration sweep can unlock more than a new kernel.

<a id="pf-03"></a>
### PF-03: Deeper tile lookahead

- **Hypothesis:** Two pending tiles and four experts per tile would hide more
  expert reads.
- **Variants tested:** Production depth one/eight experts and depth
  two/four experts.
- **Evidence:** The candidate regressed 527-token prefill from
  26.61 to 33.38 s and 1017 from 52.35 to 73.88 s because tile dispatches
  doubled.
- **What changed the conclusion:** Scheduling cost exceeded hidden I/O.
- **Final disposition:** Rejected.
- **Lesson:** More lookahead is useful only when
  it does not multiply dispatch work.

<a id="pf-04"></a>
### PF-04: SIMD-cooperative routed MoE

- **Hypothesis:** SIMD cooperation would accelerate grouped routed prefill.
- **Variants tested:** Row/expert-parallel and SIMD-cooperative kernels.
- **Evidence:** Kernel time regressed from 1.410 to 4.619 ms.
- **What changed the conclusion:** Cooperation removed useful row and expert
  parallelism.
- **Final disposition:** Rejected and removed.
- **Lesson:** The decode
  parallelism lesson also applies to prefill's grouped expert work.

<a id="pf-05"></a>
### PF-05: Shared-expert INT8 QMM v2

- **Hypothesis:** The dense shared expert should use QMM once token count exceeds
  the `M = 1` regime.
- **Variants tested:** Legacy GEMV, full QMM v2, and a phase-1
  hybrid.
- **Evidence:** QMM was faster for `M >= 16` and slower at `M = 1`. Medium rows
  improved modestly, but the final 1017-token M2 row regressed from 45.59 to
  52.01 s. The hybrid beat legacy but not full v2.
- **What changed the conclusion:**
  Long-row M2 behavior broke the shape-local promotion case.
- **Final disposition:** Rejected as production.
- **Lesson:** A mixed-length policy needs
  representative short, medium, and long gates.

## Allocation and overlap

<a id="pf-06"></a>
### PF-06: Routed metadata reduction

- **Hypothesis:** One sorted-pair buffer could replace three routed metadata
  buffers.
- **Variants tested:** Three-buffer and one-buffer grouping.
- **Evidence:** Allocations fell from 90 to 30 in the small case and 360 to 120
  in the larger case.
- **What changed the conclusion:** No throughput claim was
  needed; the change improved bounded allocation and clarity.
- **Final disposition:** Production; bounded-allocation cleanup.
- **Lesson:** Record allocation improvements
  without inventing a speed result.

<a id="pf-07"></a>
### PF-07: Argument-buffer reuse v1

- **Hypothesis:** Reusing bindings across streamed tiles would reduce encoder
  work.
- **Variants tested:** Per-tile allocation and a first reuse design.
- **Evidence:** The candidate failed its runtime gate and raised lifetime
  concerns.
- **What changed the conclusion:** Allocation reduction did not improve
  the production schedule.
- **Final disposition:** Rejected and removed.
- **Lesson:** GPU resource reuse requires both lifetime proof and wall-time value.

<a id="pf-08"></a>
### PF-08: Argument-buffer ring v2 and v3

- **Hypothesis:** A small ring could eliminate most argument-buffer allocations.
- **Variants tested:** Per-dispatch allocation and two ring designs on M2 and M5.
- **Evidence:** Allocations fell from 21,217 to two, but M2 1,017-token prefill
  and TTFT regressed about 9%. On M5, v3 improved prefill and TTFT by only
  0.2-0.3%, below the 2% action gate, and one 527-token repeat reversed
  direction slightly.
- **What changed the conclusion:** Allocation count moved dramatically while wall time did not.
- **Final disposition:** Rejected and removed.
- **Lesson:** Mechanism counters are
  diagnostics, not acceptance metrics.

<a id="pf-09"></a>
### PF-09: Shared/fetch overlap on M2

- **Hypothesis:** Decode's shared-MLP/I/O overlap would transfer to chunked
  prefill.
- **Variants tested:** Serial and overlapping shared expert/fetch paths.
- **Evidence:** The candidate improved 121 and 527 tokens by about 1% and 5.3%,
  then materially regressed 1017 tokens.
- **What changed the conclusion:** The
  long M2 row reversed the short and medium gains.
- **Final disposition:**
  Rejected on production M2.
- **Lesson:** A successful decode schedule may not
  transfer to long prefill.

<a id="pf-10"></a>
### PF-10: Shared/fetch overlap v3 on M5

- **Hypothesis:** M5 scheduling and tensor paths might change the overlap
  trade-off.
- **Variants tested:** Balanced, thermally controlled control/candidate rows
  at 527 and 1017 tokens.
- **Evidence:** The candidate improved about 2.2-2.3% at
  both lengths.
- **What changed the conclusion:** The balanced M2 revalidation was
  order-sensitive and materially regressed long prefill. The user judged the
  2.2-2.3% M5 gain too small to justify a separate scheduler branch.
- **Final disposition:** Rejected and removed.
- **Lesson:** A small host-specific win may cost more complexity than it
  returns.

## Installation and startup policy

<a id="pf-11"></a>
### PF-11: Trusted installation receipt

- **Hypothesis:** A trusted receipt could skip model SHA verification and reduce
  time to first token.
- **Variants tested:** Full hash and receipt-trusted load.
- **Evidence:** The receipt avoided 6.741 s of hashing in one 527-token
  attribution row. Clean medians improved 37.95% at 121 tokens and 11.52% at
  527, but regressed 38.65% at 1,017 because full hashing had warmed expert-file
  pages.
- **What changed the conclusion:** The switch changed
  integrity semantics and cache state.
- **Final disposition:** Conditional integrity policy, not a production
  kernel-speed claim.
- **Lesson:** Separate trust policy from
  compute performance.

## Production compute paths and gates

<a id="pf-12"></a>
### PF-12: Staged affine MPP INT4

- **Hypothesis:** Expand only a 32x64 affine INT4 tile into 4 KiB threadgroup
  memory, then use MPP matmul without a full expanded buffer.
- **Variants tested:**
  Packed QMM and bounded staged MPP across projection shapes and quality
  endpoints.
- **Evidence:** Weighted M128 work improved about 73.8%; stable
  512-token prefill improved 11.421% and TTFT 10.947%. A 36-endpoint quality gate
  passed.
- **What changed the conclusion:** An initial 0.09375 logit delta had
  caused a false rejection; distributional quality showed acceptable or improved
  behavior.
- **Final disposition:** Production; reversed rejection.
- **Lesson:**
  Bounded dequantization can unlock matrix hardware without violating the model
  memory rule. See [METH-01](09-validation-and-measurement-lessons.md#meth-01).

<a id="pf-13"></a>
### PF-13: Direct shader-local UInt4

- **Hypothesis:** A direct packed UInt4 path could avoid staged half tiles.
- **Variants tested:** Retired packed QMM, direct UInt4, and current staged MPP.
- **Evidence:** At M128, direct UInt4 beat the retired path by 68.46% but lost to
  staged MPP by 20.40%; at M32 it lost to staged MPP by 38.61%.
- **What changed the conclusion:** The current winner, not the retired baseline, set the gate.
- **Final disposition:** Rejected.
- **Lesson:** Compare new work with today's best
  path.

<a id="pf-14"></a>
### PF-14: QMM threadgroup reuse

- **Hypothesis:** Reusing activation tiles across output rows would accelerate
  the remaining packed QMM families.
- **Variants tested:** Current and reuse
  kernels by projection.
- **Evidence:** Individual families improved 3.2-9.7%
  with exact output, but fresh attribution left only about 0.41% whole-prefill
  opportunity.
- **What changed the conclusion:** Earlier promotions had shrunk the
  target.
- **Final disposition:** Rejected after re-attribution.
- **Lesson:** Local
  speedup times current share determines value.

<a id="pf-15"></a>
### PF-15: Batched routed MoE

- **Hypothesis:** Grouping same-expert token rows would amortize affine setup
  within the existing bounded scratch budget.
- **Variants tested:** Pair and
  batched routes with 495,616 bytes of scratch, isolated kernels, balanced M2
  rows, and quality endpoints.
- **Evidence:** Isolated time fell from 4943.704 to
  3415.312 microseconds, a 30.91% gain. Balanced 121/527 prefill improved about
  2.0-2.2%; quality matched the reference gate.
- **What changed the conclusion:**
  The first removal used invalid exact cross-process output comparison. A later
  same-process/distributional gate reversed it.
- **Final disposition:** Production;
  reversed rejection.
- **Lesson:** Batching can improve streamed MoE without
  expanding model-scale scratch. See
  [METH-01](09-validation-and-measurement-lessons.md#meth-01).

<a id="pf-16"></a>
### PF-16: Long prefill endpoint gate

- **Hypothesis:** Chunked production prefill would retain quality and bounded RSS
  beyond the short fixtures.
- **Variants tested:** Sixteen endpoints through 3707
  tokens against the scalar reference.
- **Evidence:** Aggregate delta-NLL was
  +0.002588; top-1 matched 16/16. Chunked RSS was 888.3 MiB versus 1517.9 MiB for
  scalar, with no model-scale heap staging.
- **What changed the conclusion:** A single sensitive 1,024-token endpoint
  outlier did not persist at neighboring lengths.
- **Final disposition:** Production validation.
- **Lesson:** Long endpoint sweeps
  separate a local tie from a systematic quality change.

<a id="pf-17"></a>
### PF-17: Apple10 TensorOps full-prefill attention

- **Hypothesis:** One threadgroup could process all eight Q heads in a full
  attention GQA group, reuse K/V reads, and map QK/PV onto cooperative matrix
  operations.
- **Variants tested:** Production tiled attention, four-head grouped online
  softmax, and an eight-head TensorOps path with FP32 QK/PV accumulation.
- **Evidence:** TensorOps was 11.24x faster at isolated 16K and 11.63x at 64K.
  Same-input 32K prefill fell from 491.09 to 204.29 seconds, a 2.404x
  end-to-end speedup, with identical post-prefill RSS. Direct attention
  reference checks passed through 64K, and frozen MLX-relative endpoints
  matched top-1 at 8K/16K/32K/64K.
- **What changed the conclusion:** Exact production-logit identity falsely
  rejected a valid floating-point reduction order. Direct attention error and
  independent MLX quality gates isolated top-k routing amplification instead
  of a shader semantic defect.
- **Final disposition:** Production on Apple10; automatic causal-tiled fallback
  on earlier GPU families and a named rollback remain.
- **Lesson:** Reordered floating-point kernels need a direct numerical oracle
  plus model-quality gates, not identity with one reduction order.

### PF-18: What the prefill's I/O actually costs, and the tile pipeline that does not change it

- **Hypothesis:** the prefill streams expert bytes at 1.14 GB/s where decode is
  drive-bound at 3.4, so the drive must be idle most of the prefill, and the
  cause is the routed-expert **tile** loop: a tile is read and then computed on,
  so overlap depends on how many tiles ahead the reads are issued. That depth
  was hardcoded at 1 (`PrefillRoutedTileSchedulerConfig`), which caps the duty
  cycle at roughly read/compute however fast the drive is.
- **Change:** the pair became a runtime setting —
  `RuntimeConfiguration.prefillTileDepth` / `.prefillTileExperts` plus
  `--prefill-tile-depth` / `--prefill-tile-experts` — with the existing slot
  budget still enforced (`(depth + 1) * tileExperts <= slots`, refused with the
  count it needs). Verified live: depth 7 x 8 experts is refused at 16 slots and
  accepted at 32.
- **Evidence, and the answer is no.** Five arms x 2 rounds (reverse-ordered),
  426-token prompt at chunk 512: `{16,1,8}` 29.30/28.87 s, `{32,3,8}`
  29.10/28.84, `{32,7,4}` 29.05/29.01, `{32,1,8}` 28.95/28.87, `{16,3,4}`
  28.99/28.93 — a 1.6% spread over a 4x change in tiles in flight, identical
  bytes (33.11 GB), and **bit-identical logits on every arm**. Back-to-back
  repeats of one arm are flat too (28.98/28.98/29.01), so the page cache is not
  hiding anything either.
- **What the same run did establish, for the first time.** The prefill's own
  breakdown is now printable (`--counters` emits a `scope=prefill` line from the
  snapshot already taken at the prefill/decode boundary; without it a
  prefill-dominated run reported only the decode delta, which is all zeros). For
  426 tokens, one chunk: the reads occupy **10.59 s of the 29.03 s wall (36%)**,
  moving 33.11 GB at **3.13 GB/s — essentially the drive's ceiling** — with 6.6
  reads in flight on average against decode's 5.59. So the drive is idle 64% of
  the prefill, but that is a **consequence**, not the lever: when the reads do
  run they are already at the drive's limit, and more lookahead does not extend
  the windows. Roughly 18.4 s is the rest of the prefill — GPU work and host
  per-tile cost — and the `gpu_*` counters read zero in the prefill scope
  because only the decode path accumulates them. **That number is currently
  uninstrumented, and it is the thing to measure next**: 36% I/O with 64%
  unaccounted is not yet an optimization target.
- **How this reads against the Edge0 comparison** (`/Volumes/samsung 2t/code/Edge0`,
  Apache-2.0, same Qwen3.6-35B-A3B base on Apple Silicon): their prefill win
  comes from loading the **whole layer** per layer — `load_full_layer` reads 9
  stacked tensors and notes "9 direct whole-tensor loads, NOT 256x9 per-expert
  builds" — which needs no routing decision, and `mx.async_eval` per layer
  overlaps the host load under the previous layer's GPU work. Their target is
  therefore the *host* cost of assembling a layer, not the drive's throughput,
  which is the same conclusion this entry reaches from the other side.
- **The split, measured (the follow-up this entry called for).** The prefill
  now records GPU time. Its four CBs per layer are all committed *and waited*
  already — the layer's own forward, the shared expert, the routed tiles and the
  tail — so `gpuStartTime`/`gpuEndTime` are available on each, into the same
  accumulators the decode path uses. Reusing them is sound because every counter
  here is read as a **delta**: the `scope=prefill` snapshot holds prefill-only
  values and a decode-scoped line holds decode-only ones, from one set of
  fields. `totalPrefillCommandBuffers` counts the prefill's commits separately
  (both profiles commit through the same queue) and is printed as `cbs` on the
  prefill-scoped line, where the decode count is zero by construction — that is
  what makes the sample coverage checkable. Only the 3.8 prefill body is
  instrumented; a 3.6 or Gemma run reports zeros here, which is why the
  `scope=prefill` unit label drops the `/step` suffix rather than dividing a
  total by a decode step count.
- **The answer, on 426 tokens.** At chunk 512 the prefill is 29.65 s: **GDN
  layers 13.07 s (44%)**, expert I/O 10.68 s (36%), routed MoE 6.19 s (21%),
  full attention 0.92 s (3%). Coverage is 1720 samples against 1768 commits
  (97%), and GPU + I/O = 30.87 s against a 29.65 s wall — so the prefill is
  **essentially serialized**, with about a second of overlap, which is the
  quantitative form of what the flat tile-depth sweep showed. Per layer, a GDN
  layer costs ~363 ms against a full-attention layer's ~77 ms: **4.7x**, and the
  GDN stack is 36 of the 48 layers.
- **Two independent controls.** (1) The *decode* line must not move with a
  prefill knob, and does not: at chunk 128 against 512 the decode-scoped
  per-step values are 265.5 / 283.9 ms of I/O, 60.8 / 63.8 ms of `cb1` and 186 /
  182 commits per step. (2) The I/O term must move with the chunk size while the
  GPU terms do not, and that is what happens: chunk 128 re-reads each chunk's
  expert union, so over the same 426 tokens its I/O is 27.07 s of a 46.77 s
  prefill (58%) against 10.68 s of 29.65 s, **2.5x**, while `gpu_cb1` moves
  14.45 -> 14.00 s and the GDN term 13.32 -> 13.07 s. The GPU is doing the same
  work; only the reads changed.
- **The GDN stack split four ways, and the surprise.** `FQ_GDN_SPLIT=1` commits
  the GDN layer's four sub-stages as separate command buffers, committing each
  without waiting — only submission order matters, so the layer's one existing
  wait completes them all and the sync pattern (and so the overlap) is
  unchanged. What it changes is `gpu_cb1_gdn`, which then measures only the
  layer's post-GDN remainder; the stages are read against the *unsplit* total.
  The knob's own overhead is measured: prefill 29.95 / 29.53 s off against
  29.51 / 29.42 s on.
- **The answer: it is the projections, by a lot.** Over the same 426-token
  prefill, the four stages sum to **12.64 s against the unsplit GDN total of
  13.15 s** (95%; the remainder is the seq mix and the plane combines). Of that
  12.64 s: **input projections 8.66 s, output projection 3.24 s, chunked
  recurrent scan 0.72 s, conv1d with its gated activation 0.009 s.** So the GDN
  stack is **95% GEMM and 6% scan** — the reverse of what the sequential chunked
  scan's shape suggests, and it means a GDN fusion pass (item 2.3) would be
  folding a 6% term while 11.9 s of a 29.5 s prefill sits in the two projection
  stages. Per GDN layer that is 241 ms of input projection and 90 ms of output
  projection against 20 ms of scan.
- **Why the projections cost what they do, read out of the kernel.** Both
  projection stages run at ~150 GFLOP/s, which is ~3.5% of this GPU's fp16
  peak, and neither is read-bound: the int4 weights are 21.1 MB per layer, 759 MB
  across the 36 layers, which is 88 MB/s of effective weight traffic against a
  drive that serves 3.4 GB/s. `prefill_dequant_int4_qmm_f16_block` says why:
  **one thread computes one (token, output-row) element and walks the whole
  weight row itself**, so each weight row is dequantized once *per token* — at
  T=426 that is 426x redundant dequantization per layer — and consecutive
  threads in a threadgroup differ in `n`, so their weight reads are a whole row
  apart and X, W and Y are all uncoalesced. The 8x8 threadgroup tiles (token,
  row) but nothing is reused across the tile.
  With K = 2560, qkvDim = 10240, valueDim = 6144 and D = 2560, the input
  projections are 18.0 GMAC per layer and the output projection 6.7 GMAC, both
  landing on the same ~150 GFLOP/s — one kernel, one reason.
- **The prediction this makes, and it is checkable.** A kernel with no reuse
  across the tile should scale linearly in T, so a prefill token should cost
  about what a decode token costs. Measured, a prefill token is ~2.5x cheaper
  (0.57 ms against ~1.4 ms per GDN layer per token) — so there is *some*
  amortization from the extra threadgroups hiding latency, and it is nowhere
  near the order of magnitude a real GEMM would give at T=426. That is the
  shape of the gap, not just its size.
- **The batched int8 projection, and what it is worth.** The GEMV path exists
  because the prefill had no batched int8 kernel at all: the linear-attention
  weights are int8 on every shipped 3.8 install, and `encodeRepeatedInt8` issued
  **one GEMV dispatch per token** — at T=426, 426 re-reads of the weight matrix
  per projection, ~46,000 dispatches per chunk. `prefill_dequant_int8_gemm_f16_block`
  tiles 64 output rows by 32 tokens, dequantizes the weight tile once per K-step
  into threadgroup memory and reuses it across the token dimension, with W, X and
  Y coalesced. Behind `FQ_INT8_GEMM=1`, at qkv, z and out_proj (the small a/b
  pair interleaves in one buffer with a doubled stride, which this store does not
  express). Measured, two rounds each, with the scan as an unchanged control:

  | stage | per-token GEMV | batched kernel | |
  | --- | --- | --- | --- |
  | input projections | 8377.8 / 8362.9 ms | **2436.9 / 2420.3 ms** | 3.45x |
  | output projection | 3142.3 / 3148.8 ms | **796.9 / 796.3 ms** | 3.95x |
  | recurrent scan (control) | 728.2 / 712.0 ms | 720.6 / 714.6 ms | unchanged |

  End to end on the same prompt: **prefill 29.17 s -> 20.78 s (1.40x), 14.6 ->
  20.5 prefill tok/s**, with decode identical (2.46 s, 3.253 tok/s) and the
  generated token IDs identical.

  The declaration order matters and was checked rather than assumed: the tile is
  stored **fp32**, not half. A dequantized weight reaches ~120, where fp16 carries
  0.03 of absolute error — 0.8% of a small output — and the GEMV this replaces
  keeps its weights in fp32 registers. The first version of the kernel stored the
  tile as half and the oracle caught it as a 0.8% disagreement, which reassociation
  cannot explain; storing fp32 brought it inside the reassociation band. That is
  the difference between "reordered" and "less accurate", and it is the one that
  would have quietly cost model quality.
- **The battery, and why the default did NOT flip.** Three cases at the
  documented protocol's sampling. Below the 2051-token boundary, where the
  engine repeats exactly and tokens are therefore a real signal:
  `short-explanation` identical for all 32 generated tokens, and
  `medium-review` identical for **27 then divergent** (`668 5073 55606 13514`
  against `10814 795 19963 383`). On the 2940-token prompt the engine's own
  run-to-run variation is the same order as the cross-arm difference (same-arm
  max |d| 0.34 / median 0.031 against cross-arm 0.61 / 0.042), so that prompt
  cannot resolve this at all; its top-10 rank band agrees 10/10 in every pair,
  same-arm and cross-arm alike, which is a coarse pass and no more.
  **So the stated gate — zero numerical edge-case regressions — is not met as
  written**, and the default stayed off.
- **But identity is the wrong bar here, and the repo already said so.**
  [PF-17](#pf-17) settled the same question for a reordered attention kernel:
  "reordered floating-point kernels need a direct numerical oracle plus
  model-quality gates, not identity with one reduction order." This kernel has
  the oracle — it agrees with the GEMV's arithmetic read from the same bytes,
  to within the reassociation slack of a 2560-term fp32 sum — and the divergence
  above is exactly the thin-margin case that lesson anticipates: 27 tokens of
  agreement then a branch, on a greedy trajectory whose top-1 margin at some
  step was smaller than the reassociation. What it does not yet have is the
  second half: a quality gate. EvalPlus HumanEval is the one this project already
  ran for the PLE change, so it is the flip's precondition, not more identity
  chasing on prompts that stop after two tokens.
- **A measured no-op, kept as a note.** Hand-vectorizing the inner loop's shared
  loads (float4 weights, half4 inputs) moved the projection stages by nothing:
  2423 ms against 2420-2437 before it, with the scan and the output projection
  unmoved too. The compiler was already coalescing them. The source records that
  the loop's apparent 2:1 FMA-to-load ratio is therefore not what limits the
  kernel — which is what the tuning below has to start from rather than from the
  assumed load bound.
- **What the kernel is worth on the long prompt.** The 2940-token case at chunk
  512, two runs each: input projections 60.1 / 58.9 s -> **16.7 / 17.1 s**, output
  projection 22.5 / 22.1 s -> **5.6 / 5.5 s**. That is 82.6 s of projection work
  down to 22.3 s on a prefill whose total at that size is ~215 s.
- **Where the target was not met.** The plan's bar was the projections reaching
  under 1.5 s; they reached 3.23 s (11.5 -> 3.23). Still ~537 GFLOP/s, so the
  kernel is now compute-bound on something other than weight traffic — larger
  tiles, more accumulators per thread, or a simdgroup-tiled inner loop are the
  next increments, and `gpu_gdn_proj_ms/split` prices each one.
- **Final disposition:** the knobs ship (default unchanged at depth 1, tile 8,
  which the sweep says is not leaving anything on the table at these sizes);
  the tile-depth hypothesis is **refuted**; the prefill's cost is attributed;
  and the GDN split answers the question it was built for: **projection kernel
  tuning, not GDN fusion.** The scan and conv a fusion pass would fold are
  0.73 s of a 29.5 s prefill; the two projection stages are 11.9 s. A rewritten
  int4 projection that dequantizes each weight row once and reuses it across the
  tile is the next piece of work, and `gpu_gdn_proj_ms/split` is the metric to
  A/B it against. (Estimate, not measurement: at a plausible 2 TFLOP/s for a
  dequant-then-GEMM kernel the input projections would fall from 8.66 s to
  under 1 s.)

[Previous: Attention and KV cache](05-attention-and-kv-cache.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Fusions, head, and orchestration](07-fusions-head-and-orchestration.md)
