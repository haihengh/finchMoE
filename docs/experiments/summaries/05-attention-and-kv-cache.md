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
- **Prior art, now a weaker analogy:** [KV-14](#kv-14) was a prefill tiled-attention race whose
  exposure depended on the tiling. That is what the withdrawn reading looked like. The duration
  result moves this away from "a shared-memory reuse hazard in a specific kernel" and toward a race
  whose *window* opens under timing perturbation the run accumulates.
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
