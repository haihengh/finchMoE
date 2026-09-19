# Attention and KV-cache experiments

[Previous: RDADVISE](04-rdadvise.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Prefill](06-prefill.md)

The current runtime uses exact split-K/V attention and an FP16 KV cache. Packed
K4/V4 saved only about 82 MiB at 4K, grew across all 30 attention layers, and
failed the quality gate. It was rejected and removed.

| Current result | Disposition |
| --- | --- |
| Split-KV plus GQA-aware SWA | Production |
| Exact split full attention | Production |
| FP16 KV ring | Production |
| Packed K4/V4 and alternate codecs | Rejected and removed |

## Decode attention geometry

<a id="kv-01"></a>
### KV-01: Split-KV attention

- **Hypothesis:** Partitioning the sequence and combining partial online-softmax
  states would expose decode parallelism.
- **Variants tested:** Single-pass and
  split-KV attention across SWA, full attention, and long context.
- **Evidence:**
  SWA improved from about 995 to 289-302 microseconds; full attention from 1473
  to 360-382; full 4K from 6103 to 1471, about 4.1x. Long-context cb1 GPU time
  fell about 28%, while short-context end-to-end speed stayed nearly flat.
- **What changed the conclusion:** Whole-step share explained the short-row
  neutrality.
- **Final disposition:** Production.
- **Lesson:** A large kernel win
  matters only where that kernel occupies enough of the step.

<a id="kv-02"></a>
### KV-02: GQA-aware sliding-window attention

- **Hypothesis:** Query heads that share KV heads should share decode work.
- **Variants tested:** Generic and GQA-aware SWA kernels.
- **Evidence:** The generic split-partial control took 295.04 microseconds; the
  GQA-aware candidate took 273.69 microseconds and passed the runtime gate.
  Full attention required a different geometry.
- **What changed the conclusion:**
  Nothing.
- **Final disposition:** Production for SWA.
- **Lesson:** Architectural
  reuse helps only when the kernel maps the sharing relationship directly.

<a id="kv-03"></a>
### KV-03: Full-attention GQA staging

- **Hypothesis:** The SWA GQA strategy would transfer to full attention.
- **Variants tested:** Default full attention and the A3 full-GQA candidate at
  1024 and 4096.
- **Evidence:** In isolated kernel rows, the candidate was 21.2% faster at 1024 and 54.6%
  slower at 4096.
- **What changed the conclusion:** Long-context scaling reversed
  the short-row win.
- **Final disposition:** Rejected.
- **Lesson:** Attention
  geometry needs at least one long-sequence gate.

<a id="kv-04"></a>
### KV-04: Local full-attention variants

- **Hypothesis:** Broadcast softmax, `exp2`, K16 combine, or another threadgroup
  width would trim the full-attention floor.
- **Variants tested:** Each change at
  1024 and 4096, including 64, 128, and 256 threads.
- **Evidence:** In isolated
  rows, broadcast softmax, `exp2`, and K16 improved 1024 by 0.9%, 1.5%, and 3.1%,
  respectively, but were neutral or slower at 4096. TG64, TG128, and TG256
  regressed at both lengths.
- **What changed the conclusion:** The sweep showed a
  geometry problem rather than one expensive instruction.
- **Final disposition:**
  Rejected.
- **Lesson:** Stop local tweaking when all nearby variants preserve the
  same bound.

<a id="kv-05"></a>
### KV-05: MLX geometry v1

- **Hypothesis:** MLX's vectorized full-attention layout would solve the geometry
  problem.
- **Variants tested:** The current split kernel and the first MLX-style
  mapping.
- **Evidence:** Isolated time improved 65-68%; early end-to-end rows
  improved 2.8-3.7%. A forced-prefix exact-output gate then failed and the
  candidate was removed.
- **What changed the conclusion:** A corrected v2 reopened the geometry family,
  but the instruction checkpoint's full quality gate later rejected it too.
- **Final disposition:** Rejected; exact split remains production.
- **Lesson:** Select the quality oracle from the candidate's mathematical
  contract without retroactively declaring an old candidate safe. See
  [METH-01](09-validation-and-measurement-lessons.md#meth-01).

<a id="kv-06"></a>
### KV-06: MLX geometry v2

- **Hypothesis:** A corrected MLX-style mapping could retain the speed signal and
  pass an appropriate quality gate.
- **Variants tested:** Exact reference repair,
  isolated geometry, a 9,216-row quality corpus, and balanced M2 end-to-end
  rows.
- **Evidence:** Isolated speed improved 71-74%, and an earlier decode path
  improved 1.30-1.59%. On the instruction checkpoint's 9,216-row evaluation,
  exact split-K/V matched the FP16 reference while MLX-v2 exceeded the accepted
  mean delta-NLL threshold.
- **What changed the conclusion:** Revalidation against the shipping instruction
  checkpoint replaced the earlier promotion result.
- **Final disposition:** Rejected for the instruction checkpoint; exact split is
  production.
- **Lesson:** Revalidate a strong speed signal when the rejection instrument
  was wrong. See [METH-01](09-validation-and-measurement-lessons.md#meth-01).

## Packed KV and quality

<a id="kv-07"></a>
### KV-07: Packed K4/V4 attention

- **Hypothesis:** Keeping K and V in packed 4-bit form would save memory and
  reduce bandwidth enough to pay for decode.
- **Variants tested:** Single-pass and
  split packed attention, Q pretransform, function constants, K staging, and
  full-GQA forms.
- **Evidence:** Split packed attention beat packed single-pass by
  about 2.4x but remained slower than FP16. Q pretransform and function constants
  helped; K staging and full-GQA regressed.
- **What changed the conclusion:**
  Optimization could not erase packed access and dequantization cost.
- **Final disposition:** Rejected and removed.
- **Lesson:** Improve a
  candidate against the production control, not only against its first version.

<a id="kv-08"></a>
### KV-08: Alternative K/V codecs

- **Hypothesis:** `vllm4nc` or `affine4g32` could improve the packed-cache
  balance.
- **Variants tested:** Current K4/V4 stores one scale per head role;
  `vllm4nc` uses norm metadata for K and affine scale/zero metadata for V;
  `affine4g32` stores scale/zero metadata per group of 32. Each ran in isolated
  writer and packed-attention rows, with FP16 as a separate production control.
- **Evidence:** The
  alternate codecs improved some writer rows but made packed attention much slower.
- **What changed the conclusion:** Writer speed was not the controlling end-to-end cost.
- **Final disposition:** Rejected.
- **Lesson:** Evaluate a storage format across
  write, read, attention, quality, and memory together.

<a id="kv-09"></a>
### KV-09: K4/V4 quality characterization

- **Hypothesis:** The packed cache's memory saving would fit the quality budget.
- **Variants tested:** Packed K4/V4 against FP16 over the full quality corpus.
- **Evidence:** Mean delta-NLL was +0.015197, p95 +0.287202, top-1 agreement
  dropped 5.0781 percentage points, and top-8 dropped 5.5990 points. Every split
  failed. Against the final FP16 ring, packed storage saves only about 82 MiB.
- **What changed the conclusion:** The full corpus and ring-relative memory
  comparison replaced earlier short and obsolete-linear comparisons.
- **Final disposition:** Rejected and removed.
- **Lesson:** Approximation must clear quality against the best exact memory
  layout.

<a id="kv-10"></a>
### KV-10: FP16 full-attention island

- **Hypothesis:** Keeping five full-attention layers exact while packing SWA
  layers could recover most quality.
- **Variants tested:** All-packed and an FP16
  full-attention island.
- **Evidence:** A 16-row sample looked better. At 256 rows,
  mean delta-NLL improved relative to all-packed, but top-1 still fell 1.5625
  points and top-8 fell 0.5371. Advancement required mean delta-NLL to improve
  by at least 0.002 nat/token and both agreement metrics to improve by at least
  1.0 percentage point.
- **What changed the conclusion:** The larger gate
  reversed the small sample.
- **Final disposition:** Rejected and removed.
- **Lesson:** Small quality samples are useful for screening, not promotion.

<a id="kv-11"></a>
### KV-11: Packed-attention chunk 32

- **Hypothesis:** More final SWA chunks would improve packed-attention occupancy.
- **Variants tested:** 16 and 32 final chunks with isolated, short-decode,
  and independent quality rows.
- **Evidence:** Isolated speed improved 5.42-6.30%;
  an earlier short decode rose from 6.511 to 7.110 tok/s. Holdout delta-NLL was
  +0.005853 and top-1 agreement fell 0.846 points, both outside the gate.
- **What changed the conclusion:** The independent quality holdout rejected the
  speed winner.
- **Final disposition:** Rejected and removed.
- **Lesson:** Keep the
  quality holdout independent from candidate selection.

## Exact FP16 ring

<a id="kv-12"></a>
### KV-12: FP16 KV ring

- **Hypothesis:** Sliding-window layers need only a ring, not the retired linear
  4K allocation.
- **Variants tested:** Linear and ring FP16 storage with split-KV
  attention.
- **Evidence:** Against the retired linear FP16 allocation, the ring saved about
  575-591 MiB. Near-4K decode was neutral to slightly faster, 4.357 to 4.417
  tok/s; 1K was slightly slower.
  Token parity held after a PSO-selection asymmetry was fixed.
- **What changed the conclusion:** Correctness repair established an exact storage optimization.
- **Final disposition:** Production.
- **Lesson:** Reclaim exact lifetime waste
  before spending quality on compression.

<a id="kv-13"></a>
### KV-13: Ring-specific kernel follow-up

- **Hypothesis:** The promoted ring kernel still had meaningful local headroom.
- **Variants tested:** The retired linear control and current ring-1152 kernel,
  used to decide whether a new modulo-hoist or segment-split candidate was
  justified.
- **Evidence:** The initial ring row improved 4.5% over the linear control,
  implying about 0.14% whole-step opportunity. A terminal alternating median
  favored ring by 10.9%, but weighted opportunity remained about 0.34%, below
  the 0.5% action gate.
- **What changed the conclusion:** Earlier work had reduced the target's share.
- **Final disposition:**
  Stopped without promotion.
- **Lesson:** Recompute whole-step value after every
  major stack change.

## Prefill attention correctness

<a id="kv-14"></a>
### KV-14: Prefill tiled-attention race

- **Hypothesis:** Production-shaped validation could expose synchronization
  defects hidden by toy tests.
- **Variants tested:** The original shared scratch
  plus a corrected single-bank path with a third barrier, then a two-bank layout.
- **Evidence:** The unsafe bank lacked a reader-to-next-writer edge. A third
  barrier made it correct but slowed production shapes 5.1-14.2%. Two alternating
  banks retained byte-identical output and recovered 2.23-6.43% versus corrected
  single-bank.
- **What changed the conclusion:** The layout removed the cost of an
  otherwise necessary barrier.
- **Final disposition:** Two-bank correctness repair
  in production.
- **Lesson:** Every reused threadgroup-memory cycle needs an explicit
  reader-to-next-writer edge. The surrounding compute path is covered in the
  [prefill summary](06-prefill.md).

### KV-15: Prefill reproducibility tracks how long the prefill *runs*, not how it is chunked

- **Hypothesis, since refuted (see the last two cells):** the engine stops being bit-reproducible
  past 2051 tokens, and the QSA ranking path (which only dispatches there) is where it happens — the
  reading `FQ_QSA_OFF=1` supported, since it removed the sparse-block selector entirely and made a
  diverging pair at 2511 tokens bit-identical.
- **Instrument.** `FQ_QSA_DUMP=<path>` (docs/RUNTIME_CONTROLS.md) appends one line per full-attention
  layer with the selected cell count, the pooled-block count, and an FNV-1a hash of the selected
  indices — from the end of `produceToken` after its wait (`decode` records), and from the chunk loop
  in `prefillChunked` after a drain (`prefill` records, which describe each chunk's *last* row). The
  aggregate is the wrong instrument: two runs differ in ~99% of their logits, which says a divergence
  happened somewhere earlier and nothing about where.
- **First reading, since withdrawn.** Four runs of one 2940-token prompt at prefill chunk 128 gave
  **four distinct outcomes**; four at **512** gave **one**. The ranking dispatches at both sizes, so
  this looked like chunking — a tiling hazard, KV-14's shape — rather than context length.
- **The ladder.** {128, 256, 512} x 3 runs, same prompt, same binary, back to back. The dose-response
  reproduced, and so did the confound: prefill time is 343.9 / 269.9 / 215.0 s, so chunk size and run
  duration move together by construction and this design cannot separate them.
  `chunk 128 -> 3 distinct; 256 -> 2 distinct; 512 -> 1 distinct`.
- **The duration control, which refutes the first reading.** Chunk **512** on a longer prompt (4606
  tokens) so the prefill takes **347.6 / 341.2 / 351.1 s** — matched to the 128 case. Three runs,
  **three distinct outcomes** (all pairs differ, ~247k of 248,320). Chunk 512 is not the safe setting;
  it was the *short* setting.
- **What the four cells together settle.** Length alone cannot be the variable: the same 2940-token
  prompt diverges at 344 s and does not at 215 s. Chunk size alone cannot be it either: 512 diverges
  at 348 s and does not at 215 s. Chunk *count* is disfavored: at chunk 256 (12 chunks, 270 s) two of
  three runs matched each other, while 9 chunks at 348 s gave three distinct outcomes. What survives
  every comparison is **elapsed prefill time**: 215 s has never diverged in 7 runs, 270 s diverged in
  2 of 3, and 344-348 s diverged in every pair tried (6 runs: the ladder's 128 arm and the control).
- **It is a rising probability, not a threshold.** An earlier, uncontrolled set of four 2940-token
  runs had two match bit-exactly — including across a binary rebuild. That set's chunk size was not
  recorded (the default at the time was 128, so it was most likely 344 s duration, where the ladder
  and control above diverged 4 of 4). Either way it is the reason to write this as a probability that
  rises with duration rather than a switch that flips at ~345 s. It is also independent evidence that
  the computation *can* reproduce, which is what makes this a race and not a second arithmetic path.
- **Named honestly: that is a correlate, not a mechanism.** Nothing about wall time changes
  arithmetic. What accumulates over a long run is exposure to *timing perturbation* — the drive's own
  read-latency regime, memory pressure and paging, thermal/power state, background activity — and any
  of those could be what actually opens a race window. This experiment set separates duration from
  length and tiling; it does not name the state variable, and the next step is to perturb timing
  directly (see disposition) rather than to keep lengthening prompts.
- **The second refutation: the selector is not necessary.** `FQ_QSA_OFF=1` removes the ranking
  dispatches outright — dense attention at every position, `encodeFull` never `encodeFullCells`, and
  the `Int.max` selection width takes the dense branch in prefill, not a degenerate cells call. Two
  such runs on the same 4606-token prompt, at **359.6 and 353.7 s** of prefill, differ in **246,482 of
  248,320 logits**. The earlier 2511-token `FQ_QSA_OFF` pair that came back bit-identical was ~295 s —
  inside the regime where the duration curve above is still often reproducible. Both attributions this
  entry was built on (the 2051 boundary, the ranking path) were proxies for how long the run takes.
- **Where it is *not*.** A diverging pair at chunk 128, fingerprinted on both sides: **0 of 276
  prefill records differ** — the selection at every chunk boundary, every layer, is the same in both
  runs — while 276 of 372 decode records differ, first at the *first* decode step. That kills the
  reading that the ranking's own output diverges: if it did at a chunk boundary, these would show it.
  The logits dumped at the prefill/decode boundary *do* differ (246,623 elements), so the divergence
  is in the prefill's output but not at its sampled selection points — between them, in a row the
  sampling does not cover, which is now the leading reading rather than a fallback.
- **Consistent detail.** In all 276 differing decode records the cell count and pooled count are
  identical (`cells=2051`, one `pooled` per position); only the chosen cells differ. So whatever
  moved did not change *how many* cells the ranking keeps.
- **Three exclusions, measured.** With duration established as the correlate and no mechanism in hand,
  the branches that would not be found by measuring longer runs were closed directly.
  *ThreadSanitizer* (verified against a deliberate race first) reports **no host-side data race**
  across 908 tests, a real-install run at chunk 128, and — the one that matters — **the exact
  configuration that diverges**: 2940 tokens, chunk 128, 356.6 s of prefill, 0 reports, exit 0.
  *Byte stability*: 96.8 GiB of the 97 GiB install (49 expert files, the resident weights, all 128 PLE
  shards) read cold three times with `F_NOCACHE` — **37,266 block-hash comparisons, zero differing
  blocks**, so the weights are not changing under the engine. *Read paths*: both streamers loop on
  short reads and verify the total, so a partial read throws rather than leaving stale bytes in a slot.
  A fourth check, cheaper than any of them: the MoE route pairs sort on
  `routedExpertPhysicalOffsets`, a static property of the install, so the summation order does not
  move with cache state — a cache-dependent order would have produced exactly this symptom legally.
- **What is left, and what TSan cannot see.** The Metal driver is not instrumented, so a host-write
  versus GPU-read hazard is invisible to TSan, and that is the shape the surviving suspects take; GPU
  kernel execution is likewise outside every instrument used here. Note also that nothing in this
  entry localizes the divergence *within* the prefill: the chunk-boundary selections agree while the
  final logits differ, and the code has no per-row fingerprint to bisect further — adding one is the
  work that remains, not more runs of the same length.
- **The row-level instrument, and what it localized.** `FQ_ROW_HASH=<path>` fingerprints the residual plane
  *per row* on the GPU — one FNV-1a hash per (layer, stage, row) at three stages of every layer, written to
  a side buffer and read back once, so no per-chunk commit or wait is added to the run being measured. On
  the diverging configuration (2940 tokens, chunk 128, two runs at 350.6 and 350.5 s of prefill, logits
  differing in 246,897 of 248,320): **104,561 of 423,360 hashes differ (24.7%)**, and the first one is
  **layer 7, stage `attn`, row 2304**. Layer 7 is a full-attention layer (the mask is every fourth layer
  from 3), and 2304 is exactly **chunk 18's first row** — in other words **chunks 0-17 are bit-identical in
  both runs at every stage of every layer, and the divergence starts at chunk 18 and persists through
  chunk 22.** That is a chunk-scoped trigger, not a scattered one, and it is the first evidence in this
  investigation that points at a specific place rather than at a duration.
- **One thing in that map does not have a causal explanation yet, and it is worth stating plainly.** From
  layer 11 onward the affected row set widens *backward*, to rows ≥2053 — but those rows agreed at layers
  7-10, and causal attention cannot let a difference at row 2304 change row 2053's earlier-layer output.
  Either a shared per-layer structure (the ranking's pooled blocks and cells are per-layer state, and rows
  ≥2051 are the ones that use them) is carrying influence backwards, or the row index means something
  other than position in a way this instrument does not capture. Resolving that comes before reading
  kernels on the strength of the layer-7 result.
- **Second row-level pass, four stages deeper, and this one names the mechanism.** The instrument
  was extended from three plane stages to seven — adding, on full-attention layers, the q/k/v output
  after the RoPE and norm epilogue, **the QSA selection**, the attention output, and the block output.
  That split is what the plane alone could not give: it separates "the projections moved" from "the
  selection moved". Same configuration as before (2940 tokens, chunk 128, two runs at 365.0 and
  363.6 s, logits differing in 247,655 of 248,320), and the map is far sharper:

  | layer | in | attn | post | qkv | idxcells | core | oproj |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | **3** | - | **1** | 1 | - | - | **1** | **1** |
  | 4 | 1 | 1916 | 1916 | - | - | - | - |
  | 7 | 1916 | 1916 | 1916 | 1916 | 105 | 1916 | 1916 |
  | 15 | 1916 | 1916 | 1916 | 1916 | 844 | 1916 | 1916 |

  **The first difference is one row of one layer: layer 3's attention output, row 1024** — and at that
  layer the projections and the selection are *bit-identical*. So the divergence is born inside the
  attention computation for a single row, from identical inputs: a race in the kernel, not a
  data-flow divergence. Layer 3 is the first full-attention layer (the mask is every fourth from 3),
  and 1024 is a chunk boundary.
- **The amplification is causal, and that resolves the anomaly the earlier pass recorded.** One row
  becomes **1916** at the next layer — and 2940 - 1024 = **1916** exactly: every row at or after the
  perturbed one, which is what causal attention permits and nothing more. The earlier run's
  "widening backward" (a first difference at row 2304 with a later layer reaching back to 2053) is
  therefore not non-causal: each full-attention layer can land the race at a **different row
  independently**, so a later layer's landing at an earlier row moves that layer's minimum backward
  while its input still agreed. That also explains why the first affected layer differed between runs
  — layer 7 then, layer 3 now — for a race whose per-layer probability is small. The anomaly is
  retired as an anomaly.
- **Where to look now, specifically.** Rows at or below 2051 take the *dense* prefill path
  (`attention.encodeFull`, which builds its causal mask from `seqLen`), so the hazard is in the dense
  prefill attention kernel — not in the cells path, and not in the ranking, which this map exonerates
  for the rows in question. That is the same subsystem and the same shape as [KV-14](#kv-14): a
  prefill attention hazard exposed by how the work is tiled. KV-14 was one instance of it, fixed with
  a two-bank layout; this is a second, and it lands at a chunk boundary.
- **Third pass: the kernel fuzz comes back clean, and clears four suspects.** The kernel audit the
  localization pointed at was done by reading and then by repetition. Read and cleared:
  `block_reduce_sum` has both barriers (the write→read edge *and* the read→next-write edge KV-14 was
  about); the empty-chunk case is handled explicitly — a chunk with no positions writes
  `(-inf, 0, 0)`, which the combine weights to zero via `e^{-inf}`, and `chunkLength = ceil(len/N)`
  means no chunk is skipped anyway; `partialPipeline` only selects the 16-chunk specialization when
  the runtime count *is* 16, so there is no function-constant/geometry mismatch, and this model takes
  the generic PSOs regardless (head_dim 256, 24 query heads, 2 KV heads matches neither prebuilt
  pair); and both the attention scratch and the KV cache are `.storageModeShared` with **tracked**
  hazards, so cross-encoder and cross-kernel ordering on them is the driver's.
- **The repetition test, which is the one that could have found a rare race.**
  `PrefillAttentionDeterminismTests` runs the same dispatch on fixed inputs and compares bit-exactly,
  over both paths and across both boundaries the maps landed on — 1024 (below the selection width,
  dense) and 2051/2064 (above it, cells): **20,000 rounds x 9 lengths x 2 paths = 360,000 dispatches,
  zero mismatches**, and the same again under four CPU burners (360,000 more). A pass is a bound and
  not a proof, but it rules the hazard out as something reachable by repeating the dispatch in
  isolation — which is consistent with everything else this hunt has found: what matters is the
  machine's state during a long prefill, not the dispatch itself.
- **What the two maps disagree on, stated rather than smoothed.** The first landing was layer 3's
  *dense attention output* with q, k, v and the cells all identical. The second was layer 7's
  **cells**, which moved *before* the attention output did, at row 2064 — and there the attention's
  own q/k/v were identical too. Both are boundary rows, but they are different paths, so either there
  are two hazards or one upstream that neither pass has instrumented. **The gap is now specific: the
  indexer has its own q and k** (`index_qk_proj` into `qwen38IdxQ`/`qwen38IdxK`), and no stage hashes
  them. Hashing those is the next instrument, not another pair of prefill runs.
- **Fourth pass: eleven stages, and it lands on the ranking.** Two more stages — the indexer's own
  query (`qwen38IdxQ`) and its **raw key timeline** (`lay.rawKeys`, a persistent per-layer buffer this
  chunk writes in place at the row's position and a later chunk pools in completed blocks) — separate
  "the indexer's inputs moved" from "the selection moved". On a fresh diverging pair (2940 tokens,
  chunk 128, 361.9 and 361.0 s, logits differing in 247,303 of 248,320):

  | layer | in | attn | post | qkv | idxcells | core | oproj | krot | vrot | idxq | idxk |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | **3** | - | **837** | 837 | - | **837** | 837 | 837 | - | - | **-** | **-** |
  | 7 | 875 | 876 | 876 | 875 | 791 | 876 | 876 | 875 | 875 | 875 | - |

  At layer 3 the attention's q/k/v are identical, **and so are the indexer's query and its raw keys**,
  while the cells differ. So the divergence is inside **pool → score → radix-select → cells-write**:
  the QSA ranking's own kernels, on identical inputs.
- **What that does and does not change.** It does not contradict the earlier refutation — a
  selector-less pair still diverged, so the ranking is not *necessary* for the phenomenon. But it is
  now a *sufficient* meeting point, with far better evidence than the `FQ_QSA_OFF` isolation had
  before that. Combined with the first landing (dense attention output, row 1024, cells identical),
  the honest reading is **at least two meeting points**: the ranking's pipeline and the attention's.
  They share a shape — a reduction over a variable-length set, dispatched per row — and the ranking
  version is the subsystem that already produced one UB bug
  ([[metal-divergent-threadgroup-barrier]]: a barrier inside `if (simd_group == 0)` in the radix
  select, which silently returned the wrong rank; found by reading, fixed).
  The next stage to hash is between the two ends of this one: the pooled keys and the block scores,
  which would separate the pool from the selection and put the radix select in or out.
- **Fifth pass: the pool is clean, so it is the scoring and the radix select.** One more checkpoint
  between the two ends of the previous pass — the **pooled keys**, hashed where the pool writes them
  and before any score reads them — splits the ranking's pipeline in two. On a fresh pair (2940
  tokens, chunk 128, 361.9 and 361.7 s, logits differing in 247,731 of 248,320), the first landing is
  layer 15 row 2063, and it reads:

  | attn | qkv | idxcells | core | oproj | krot | vrot | idxq | idxk | **idxpool** |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | **842** | - | **842** | 842 | 842 | - | - | - | - | **-** |

  **The pooled keys are identical and the cells differ.** So the multi-chunk page assembly is not
  where this starts: with the plane, the attention's q/k/v, the indexer's query, its raw key timeline
  *and* the pooled keys all bit-identical, the divergence is in **score → radix-select → cells-write**
  — the reduction and the selection, on identical inputs. That is the isolation this hunt has been
  working toward, and it is a different place from where the `FQ_QSA_OFF` story pointed.
- **An instrument bug found on the way, which had made one stage lie.** The hash kernel indexed its
  source from row 0 and offset only the destination, so a chunk-local buffer (the plane, the attention
  scratch) was handled correctly while a buffer indexed by position or block was not: every chunk
  fingerprinted the timeline's *first* rows again. That is the whole of the `idxk` stage's first
  result — it read as identical because it was measuring the same 128 rows each time. The kernel now
  takes a `srcRowBase`, with a test that pins it (a source base of 0 would hash row 0). The pooled
  checkpoint above is the first report from the corrected instrument.
- **Seventh pass: the second mechanism is the dense attention itself.** The selector-off configuration —
  no indexer in the runner at all, so no store, no pool, no ranking — is untouched by the fix, and it
  still diverges, so it was run with the row fingerprints on (stages 9-11 are the indexer's and stay
  zero there; 0-8 cover the plane and the attention block). The first diverging pair, two runs at
  316.9 / 314.2 s of prefill with logits differing in 247,480 of 248,320, reads:

  | layer | in | attn | post | qkv | core | oproj | krot | vrot |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | **3** | - | **1** (row 1024) | **1** | - | **1** | **1** | - | - |
  | 4 | 1 (row 1024) | 3582 | 3582 | - | - | - | - | - |

  **At layer 3 the queries, the rotated keys and the values have no differing row at all, and the
  attention's output differs at exactly one row — 1024.** Everything from that row onward is then
  carried (3582 of 4606 rows from layer 4 on). So the second meeting point is the dense attention
  block, and in this pair it landed exactly where the very first map's did — layer 3, row 1024,
  attention output, inputs equal — which was the one thing the store bug never explained. That map's
  landing and this one are therefore one mechanism, not two: the hunt's "at least two meeting points"
  is now (a) the indexer store, fixed, and (b) this. (How stable that row is, is next.)

  **The mechanism is intermittent, and row 1024 is a landing and not a property.** Re-running the same
  selector-off configuration gave **0 of 248,320 logits differing** — an agreeing pair — where the
  first had differed in 247,480, and three further pairs did the same: **one divergent pair in five on
  the day, all at chunk 512 and 4606 tokens.** Two consequences follow.

  One detail cuts against the duration story that governs the *store* bug: the diverging selector-off
  pair was the **fastest** of the five (264.5 s of prefill against 312-317 s for the four that
  agreed), so for this mechanism longer is not more exposed — or elapsed time is simply a coarse
  proxy for a machine state that moved underneath both.

  First, **no deterministic arithmetic condition can be the whole of it.** A row that is simply
  computed *wrongly* on fixed inputs would be wrong identically in both runs, which is the trap this
  hunt has fallen into before (a deterministic bug cannot by itself make two runs differ; the
  difference needs something whose *content* varies run to run — a stale slot, an unordered write, a
  read the driver does not order). Row 1024 is nevertheless the row both diverging runs landed on, and
  it is both the first row of a prefill chunk (8 at chunk 128, 2 at chunk 512) and the first row whose
  sequence length exceeds 1024, where the dense split's `chunkLength = ceil(effLen/16)` steps 64 → 65
  so the last chunk stops being exactly full. **The earlier chunk boundaries did not fire** — 128, 256,
  …, 896 in one run, 512 in the other — which is evidence against "any chunk boundary" as the trigger
  and mildly for the geometry step; it is not positive identification, and the CLI's
  `allowedPrefillChunkTokens = [32, 64, 128, 256, 512, 1024]` means a chunk size that does not divide
  1024 cannot be asked for, so the two cannot be separated by chunking alone. The constants that could
  have made the geometry stale are also exonerated: **no Swift code defines `FC_ATTN_NUM_CHUNKS` or
  `FC_ATTN_RING_CAP`**, so `attention_decode_combine` and both partial kernels read their geometry from
  the arguments.

  Second, **the landing row is not stable across maps**, which is what an intermittent mechanism should
  look like: the fifth pass landed at layer 15 row **2063** where these land at layer 3 row 1024.

  **Reproduce it with `tools/qsaoff-hash-until.sh [pairs]`** — the loop that pairs the two runs, keeps
  the prefill logits as the control, and prints this same map on the first pair that diverges. It uses
  the prompt the pairs actually ran on, committed for the purpose as
  `docs/benchmark-prompts/real-generation-v1/long-matched.json` (`long-synthesis` plus an appended
  tail, 4606 tokens), so the arm is a command rather than a memory of one.

  A third reading stays live and is now instrumented: **the attention's inputs are not all hashed.**
  Stages 7/8 fingerprint the K/V *stage* scratch, and the cache the attention actually reads — the
  destination of `copyPrefillKVToCache` — was not fingerprinted at all, so a cache-side difference
  would look exactly like this: identical stage rows, one differing attention output. Stages 12/13
  (`kcache`, `vcache`) hash those rows immediately after the copy; a differing attention output
  beside *equal* cache rows means the kernel, and *unequal* cache rows means the copy.

  Note also what the indexer-side evidence said about this: the fifth pass's landing at layer 15, row
  **2063**, is *not* a chunk boundary (2048 is) — consistent with the store bug being driven by the
  position it wrote to rather than by any tiling, and now measured rather than assumed.
- **Eighth pass: contention catches it, and the K/V cache is clean, so the copy is out.** The
  mechanism is a race, and waiting on a quiet box pays one hit in eleven valid pairs. Contention is
  the cheaper lever, and it worked on the **first** pair: four CPU hammer threads (the harness's
  `QSAOFF_BURNERS=4`), two runs at 312.7 / 313.6 s of prefill — the *same* durations as the quiet
  pairs, so the burners did not slow the work, they only perturbed the schedule — and logits differing
  in **185,754 of 248,320**. Fourteen stages this time, including the two added for exactly this
  question, and the map reads:

  | layer | in | attn | post | qkv | core | oproj | krot | vrot | **kcache** | **vcache** |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | **43** | - | **1** (row 4096) | **1** | - | **1** | **1** | - | - | **-** | **-** |
  | 44 | 1 | 510 | 510 | - | - | - | - | - | - | - |

  **Every K/V cache row is bit-identical** — zero differing rows across the whole dump — while the
  attention's output differs at one row, 4096. That is the branch the two-stage instrumentation was
  built for, and it resolves the way the earlier evidence could not: `copyPrefillKVToCache` and its
  source are both out, and what the kernel read is what the map says it read. Layers 43-47 carry it
  from row 4096 on (510 rows).

  The landing row moved again — 4096 where the quiet pairs landed on 1024 and the indexer-on fifth
  pass on 2063 — and 4096 is again the first row whose sequence length exceeds a multiple of 1024,
  where `chunkLength = ceil(effLen/16)` steps 256 -> 257 so the last chunk stops being exactly full.
  That is now three landings out of four on such a row, which is a pattern and not yet a mechanism: the
  remaining landings need a hypothesis that predicts them.

- **Ninth pass (audit): the split path's structure is clean, and one buffer in it is un-hashed.** With
  the copy excluded, the audit went into `attention.encodeFull`'s split — `encodeSplit`, the two
  partial kernels and `attention_decode_combine`. Reading found no data-dependent control flow: every
  `threadgroup_barrier` sits at the same level for all threads (including the two inside
  `block_reduce_sum`, which the p-loop calls once per cell), both partial kernels get their geometry
  from kernel arguments, and the combine is a plain reduction with no barriers at all. The one
  register-array bound, `kPerThread = ceil(kAttnMaxHeadDim / kAttnThreads) = 2` against
  `headDim = 256` at a dispatch width of 256, holds with one iteration to spare — it is safe only
  while the dispatch width stays `>= 128`, which is why the encoders' `min(threadsPerGroup,
  pso.maxTotalThreadsPerThreadgroup)` matters.

  Two structural facts came out of it. **Qwen 3.8 never uses the specialized split pipelines**:
  `partialPipeline`/`combinePipeline` select on `(headDim, numQHeads, numKVHeads)` with 16 q-heads,
  and the 3.8 full layer has 24, so every row takes the generic `psoPartial`/`psoCombine`. The
  `numChunks == 16` specializations — and with them `FC_ATTN_NUM_CHUNKS` — are therefore dead for this
  model, which is why the earlier "no stale geometry specialization" check passed, though for a
  different reason than it claimed. Harmless today, and worth knowing that the two paths are untested
  against the only model that could exercise them.

  **And the one buffer in this path that no stage hashes is the split's own scratch**:
  `mPartial`/`dPartial`/`oPartial`, one shared allocation per `Attention` instance, rewritten by every
  row of every layer. Everything the kernel *reads* is fingerprinted; the accumulator it *writes and
  re-reads* between its two passes is not. A combine that folded in a partial it should not have would
  show exactly the measured signature — identical q/k/v, identical cache, one differing row, nothing
  upstream to point at — so that is the hypothesis the next experiment tests.
- **Prior art, now a weaker analogy:** [KV-14](#kv-14) was a prefill tiled-attention race whose
  exposure depended on the tiling. That is what the withdrawn reading looked like. The duration
  result moves this away from "a shared-memory reuse hazard in a specific kernel" and toward a race
  whose *window* opens under timing perturbation the run accumulates.
- **Sixth pass: the prefill was storing its indexer keys at twice their position, and the fix removes
  the divergence.** The fifth pass left exactly one un-hashed link between "the pooled keys are
  identical" and "the cells differ": the `scores` array. Reading the scoring and select kernels for a
  guard whose coverage depends on the input found nothing — every barrier there sits at the same
  control-flow level for all threads, and the one early return (`b·r >= tail_start`) is threadgroup-
  uniform. Reading the *call site* instead found the store's index arithmetic: `encodeQKPost` takes a
  `kRawOffset` buffer binding **and** a `pos` argument, and the kernel addresses the store as
  `k_raw + pos·idxDim`; the chunked prefill passed both, so every prefill key landed at `2·pos`. It
  has been there since the M3.4 chunked-prefill commit.

  Two consequences, one arithmetic and one about memory. **Arithmetic:** a block pooled during a
  multi-chunk prefill is the mean of the wrong cells, so the prefill's ranking scored the wrong blocks
  — that is a selection that is wrong but plausible, which is why the engine stayed coherent and why
  the toy-level chunk-vs-decode test (tier-1 equality at a GDN layer, tier-2 logits cosine ≈ 0.98)
  never saw it. **Memory:** `rawKeys` is `maxContext · idxDim` halves, so the store leaves the buffer
  entirely once `pos >= maxContext/2` — at the hunt's `--max-context 4096` that is position **2048**,
  and the 2940-token pairs wrote up to **456 KB past the end of a 1 MB buffer, per full layer**, into
  whatever the allocator had placed next (per layer, in allocation order: `pooled`, `scores`, `cells`,
  `cellCount`, `qkProj`, `qIdx`). Metal cannot associate that write with the dispatch that made it, so
  its ordering against the dispatches that legitimately read and write those buffers is undefined —
  which is the shape this hunt has been looking for, and it is invisible to ThreadSanitizer for the
  same reason the other surviving suspects are.

  Note what that does to the boundary evidence. The original framing put the trigger at
  `idxCapacity = min(maxContext, budget + r − 1) = 2051`; the out-of-bounds threshold for the same
  runs is `maxContext/2 = 2048`. **The two are three positions apart**, and the truncated probes are
  equally consistent with both: 2065 tokens is 17 rows past either threshold and reproduced, 2511 is
  463 rows past and diverged. That evidence never separated them because both stories predict the same
  ordering of the two probes.

  **Pinned by a test.** `QSAIndexerTests/chunkFormPostsUseAbsolutePositions` drives the chunk form with
  the engine's own argument shape — a chunk starting at position 8 that pools blocks 2 and 3. Before the
  fix it reports `timeline slot 8 [0] = 0.0 but the chunk's post for that position stored nothing
  there` (the key went to slot 16) and `chunk-form pool relErr=1.0` (the pool read zeros); after it,
  all eleven indexer tests pass. The `kRawOffset` parameter is **removed** rather than defaulted — a
  chunk has no reason to want a base, and an unused parameter that double-shifts is a trap, not an
  option.

  **The experiment.** `--max-context 4096`, chunk 128, the same 2940-token prompt, the configuration
  that had diverged in every pair tried before: **four runs in two pairs, at 361.64 / 359.14 s and
  360.26 / 358.88 s of prefill, now agree bit-for-bit** — 0 of 248,320 logits differ, all 1,693,440
  row hashes are identical within each pair, and all four dumps carry the same digest
  (`5a49bf89320002a1`). The pre-fix pair at the same durations differed in 247,731 of 248,320. Two
  agreements at a duration where the old code never reproduced is not a proof — the phenomenon was
  probabilistic, and this file's own history has an uncontrolled set of four pre-fix runs of which two
  matched — but it is the first evidence this hunt has produced that the divergence is *removable*
  rather than merely re-describable.

  **What this does not settle.** The `FQ_QSA_OFF` pair at 4606 tokens diverged with **no indexer at
  all**, so the fixed store cannot be its cause: re-run on the day of the fix it still diverges —
  234,042 of 248,320 logits — on a runner that reports `built without the QSA indexer; attention takes
  the dense path for every layer`. That is a second mechanism, in the dense path, and it is now the
  whole of this hunt's open surface.

  **A caveat the fix's day produced, about duration as a proxy.** The same selector-off configuration
  measured **359.6 s of prefill on 2026-09-16 and 264.5 s on 2026-09-18** — 12.8 → 17.4 prefill tok/s
  for a byte-identical invocation. Whatever that 36% is (drive state, compressor, page cache), it means
  elapsed time is a proxy for the machine's state and not a stable name for it: two runs can share a
  duration on different days and not share the state. The comparison above is therefore stated as *the
  same duration band with opposite outcomes* — pre-fix 361.9/361.7 s diverged, post-fix 358.9-361.6 s
  agreed four times — rather than as a clean causal chain, and this sentence is the reason. And this finding says nothing about the ranking kernels
  themselves: the 720,000-dispatch fuzz bound stands, and the two landings at the attention's own
  output (`core` differing with `idxcells` equal) are still unexplained by a mis-stored key.
- **Final disposition:** open, with **both earlier attributions withdrawn** — the chunk-size reading
  and the ranking-path isolation were each refuted by a control of their own. What is left: the
  instrument, the negative on the ranking's *sampled selections*, the two refutations, and duration as
  the surviving correlate. Note what the refutations do *not* do — they do not clear the ranking's
  kernels (the selector changed no *outcome* here, which is not the same as being race-free), and they
  do not identify a replacement suspect. The prefill path as a whole is back in scope, together with
  the possibility that this is environmental rather than a kernel defect at all.
- **The next experiment has to decouple work from wall time.** In every cell above, elapsed time and
  the amount of work are confounded — at a fixed chunk size duration is proportional to token count,
  and the ladder varied them together by construction. Work can be held fixed while wall time moves by
  slowing the *I/O* (the contention load priced in [IO-22](01-model-install-and-expert-io.md#io-22),
  or `FINCHMOE_IO_NOCACHE=1`) or by inserting delay between chunks. If a stretched short run diverges
  while its unstretched twin does not, wall time is causal; if it does not, the remaining variable is
  the number of dispatches, and the search belongs in the prefill kernels after all.

[Previous: RDADVISE](04-rdadvise.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Prefill](06-prefill.md)
