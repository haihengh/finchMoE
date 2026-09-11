# Validation and measurement lessons

[Previous: Sampling, tokenization, and output](08-sampling-tokenization-and-output.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md)

Several conclusions changed because the original measurement answered the
wrong question. The project now separates exact lossless gates from numerical
quality gates, profiler attribution from clean throughput, and training traces
from representative holdouts.

| Current rule | Purpose |
| --- | --- |
| Exact identity for claimed-lossless changes | Detect unintended math or storage changes |
| Distributional quality for reordered floating-point work | Avoid false rejection near ties |
| Interleaved production A/B plus isolated attribution | Separate mechanism from outcome |
| Short, medium, long, and holdout rows | Detect scaling and generalization failures |

## Correctness and quality gates

<a id="meth-01"></a>
### METH-01: Numerical reordering needs a quality oracle

- **Hypothesis:** Exact cross-process tokens or raw logits could validate any
  candidate.
- **Variants tested:** That oracle was applied to MLX attention,
  staged MPP INT4 prefill, batched routed MoE, and TensorOps prefill attention.
- **Evidence:** Each candidate
  had a strong speed signal but crossed known near ties under reordered
  floating-point work. Reference-relative delta-NLL, top-k agreement, and
  endpoint gates later passed.
- **What changed the conclusion:** The validation
  contract, not the candidates, was wrong.
- **Final disposition:** Methodology
  corrected; four false rejections reversed.
- **Lesson:** Numerical equivalence
  is a distributional claim, not a bit-identity claim.

<a id="meth-02"></a>
### METH-02: Lossless changes still require exact identity

- **Hypothesis:** Once quality gates replaced universal exact comparison, exact
  checks might be unnecessary.
- **Variants tested:** Cache policy, packed-load
  width, FP16 ring storage, and lossless fusion rollbacks.
- **Evidence:** These
  changes claim identical model math and therefore passed identical bits or
  tokens.
- **What changed the conclusion:** Nothing; the gate depends on the
  claim.
- **Final disposition:** Methodology retained.
- **Lesson:** Use the strictest gate appropriate to the candidate's contract,
  not one universal tolerance.

<a id="meth-03"></a>
### METH-03: Production-shaped offsets

- **Hypothesis:** An offset-zero parity buffer represented the standalone INT4
  path.
- **Variants tested:** Aligned test buffers and live resident tensors.
- **Evidence:** `uint` loads passed at offset zero but failed when BF16 planes
  left packed weights only 2-byte aligned.
- **What changed the conclusion:** The
  real layout invalidated the test's alignment assumption.
- **Final disposition:**
  Real-shape offset tests became required.
- **Lesson:** Include live offset and
  stride contracts in kernel fixtures.

<a id="meth-04"></a>
### METH-04: Threadgroup versus grid synchronization

- **Hypothesis:** A local barrier fix could stand in for a complete ownership and
  scope proof.
- **Variants tested:** The production shader inventory was audited
  across threadgroup scratch, disjoint device partials, and queue-ordered
  dispatches.
- **Evidence:** The only proven bug was a reused FP16 prefill
  reduction bank whose old cycle lacked a reader-to-next-writer edge. Other
  cross-threadgroup data relied on disjoint ownership and queue order, not a
  threadgroup barrier.
- **What changed the conclusion:** Synchronization scope
  became an explicit part of the proof.
- **Final disposition:** Methodology and correctness repair retained; the
  concrete failure is recorded in [KV-14](05-attention-and-kv-cache.md#kv-14).
- **Lesson:** Name the producer, consumer, memory
  region, ownership, and synchronization domain before choosing a barrier.

## Measurement discipline

<a id="meth-05"></a>
### METH-05: Profiler throughput is diagnostic

- **Hypothesis:** A profiled run's tok/s could serve as a production comparison.
- **Variants tested:** A clean production row, an attribution profile, and a
  production-behavior timeline row.
- **Evidence:** The clean row measured 6.144
  tok/s. Profile instrumentation disabled the normal command-buffer pipeline and
  measured 4.228 tok/s; the timeline retained production behavior and measured
  6.337 tok/s.
- **What changed the conclusion:** Clean, profile, and timeline rows answer
  different questions: production throughput, detailed attribution, and
  production-behavior timing. They are not interchangeable throughput samples.
- **Final disposition:** Methodology retained.
- **Lesson:** Use profiler
  spans to choose a target and clean A/B rows to decide promotion.

<a id="meth-06"></a>
### METH-06: First-run and thermal state

- **Hypothesis:** One control/candidate pair could establish a small shader win.
- **Variants tested:** Single-order runs, interleaved warm rounds, and balanced
  M5 thermal sequences.
- **Evidence:** Apparent 15-100% swings, including apparent 2x wins on the first
  run, disappeared under interleaving.
- **What changed the conclusion:**
  Order and thermal state explained the signal.
- **Final disposition:** Balanced
  interleaved measurement adopted.
- **Lesson:** Small GPU deltas require an order
  that cannot assign warm-up or throttling to one variant.

<a id="meth-07"></a>
### METH-07: Mechanism counts are not outcomes

- **Hypothesis:** Fewer allocations, dispatches, or command buffers implied a
  faster path.
- **Variants tested:** Argument-buffer rings and merged command
  buffers.
- **Evidence:** The ring cut 21,217 allocations to two but slowed long
  prefill about 9%. The merge cut submissions but slowed decode from 4.013 to
  3.707 tok/s.
- **What changed the conclusion:** Both mechanism counts improved
  while their end-to-end gates regressed.
- **Final disposition:** End-to-end time
  remains the promotion gate.
- **Lesson:** Mechanism counters explain results; they do not replace results.

<a id="meth-08"></a>
### METH-08: Training traces need holdouts

- **Hypothesis:** A policy optimized on one or several related traces would
  generalize.
- **Variants tested:** Single-trace packed layout and trace-trained
  heterogeneous cache allocation.
- **Evidence:** The layout won natural-text
  replay and failed near 4K; the allocation lost on its held-out trace.
- **What changed the conclusion:** Workload-specific locality did not transfer.
- **Final disposition:** Representative holdouts required; see
  [CACHE-04](03-expert-cache-prediction-and-layout.md#cache-04) and
  [CACHE-08](03-expert-cache-prediction-and-layout.md#cache-08).
- **Lesson:** Treat
  layout and cache policies like learned models.

<a id="meth-09"></a>
### METH-09: Greedy repetition-loop artifact

- **Hypothesis:** The long fixture's hit-rate decline revealed a general cache
  defect.
- **Variants tested:** Offline token-period and route-period analysis of
  the existing trace.
- **Evidence:** Late output locked into a period-44 loop with
  94-100% rolling ID match. One cycle touched a median 62 distinct experts per
  layer versus 16 slots. The apparent hit-rate decline from 66.6% to 54.2% was
  cyclic thrash from repeated text.
- **What changed the conclusion:** The fixture
  was an artificial routing workload.
- **Final disposition:** Long-decode
  methodology corrected.
- **Lesson:** Label repetition onset before treating a
  greedy long run as representative generation.

<a id="meth-10"></a>
### METH-10: A bucket named for a kernel prices the dispatch around it

- **Hypothesis:** A `cb1` sub-bucket named for a subsystem reports that
  subsystem's cost, so it can price the subsystem's kernels.
- **Variants tested:** The Qwen decode split — attention, GDN
  projections/conv-gate/recurrent, router, and an `io` hit/miss count — on both
  installs; then the expert-cache slot sweep those counters made measurable.
- **Evidence:** Every CPU encode bucket together summed to 0.8% of a token
  (2.82 ms of `cb1` against 367 ms/token), because `cb1` is an encode-and-commit
  clock that excludes the pipeline wait. The sweep is the matching case on the
  `io` side: 16 to 32 slots cut reads 21% (766.6 to 603.4 MB/step) and raised the
  hit rate 39.5% to 52.4% for +0.8% throughput, inside the run-to-run spread.
- **What changed the conclusion:** In both cases the instrument moved as
  designed and the end-to-end number did not. The counters closed the *encode*
  question and left the *kernel* question open — they were read as an answer to
  something they cannot see.
- **Final disposition:** Buckets are for attribution, not for pricing kernels;
  kernel claims need GPU spans. Both families reconcile exactly
  (`identity=exact`), the instrumented build reproduces the baseline token
  stream, and slot counts 16/24/32 produce byte-identical output.
- **Lesson:** A bucket is named for the code surrounding it, not for the
  hardware work it triggers. State the clock kind before quoting a share.

<a id="meth-11"></a>
### METH-11: A GPU span does not license the subtraction you build from it

- **Hypothesis:** METH-10 closed with "kernel claims need GPU spans", so adding
  device timestamps to the same readout would settle which of the two decode
  stacks owns the GPU time, and `wait` minus GPU time would be dispatch overhead.
- **Variants tested:** `MTLCommandBuffer` start/end timestamps read after
  completion at the two existing wait sites (no extra buffer, no extra wait, no
  change to commit order), split by layer kind, across three runs on two
  installs and two context lengths.
- **Evidence:** The first half worked: `gpu_cb1_fullattn + gpu_cb1_gdn ==
  gpu_cb1` exactly on every run, GDN costs 1.718 ms/layer against attention's
  1.127 on Qwen 3.8 and 84.3% of `gpu_cb1` on 3.6, and the two-install split
  identified the target stack that CPU encode clocks could not. The second half
  did not. `wait` minus `gpu_cb1` came out at 46.8 ms/step, which reads as 12.7%
  of the token in dispatch tax. Subtracting `gpu_routed` as well leaves 10.40 /
  5.60 / 8.69 ms/step, or 0.055 / 0.035 / 0.055 ms per command buffer -- an order
  of magnitude *below* the ~0.26 ms a kernel-bearing buffer costs under S6.
- **What changed the conclusion:** The commit order, not the timestamps. Layer
  N+1's `cb1` is committed after layer N's routed tail, so N+1's wait necessarily
  drains that tail. Any subtraction over `wait` that omits a term charges the
  omitted block to overhead -- and the omitted block here is the largest kernel
  group in the model, so the error is large enough to look like a finding.
  `gpu_cb1_fullattn + gpu_cb1_gdn == gpu_cb1` held exactly throughout and did not
  catch it; a correct internal identity is not evidence that the quantity built
  on top of it is being read correctly.
- **Final disposition:** The split is retained and both GPU figures are now
  printed beside `wait`, with the subtraction warning in `SYSTEM_DESIGN.md`; item
  2.2 is recorded as refuted rather than deferred. `gpu_samples` is the coverage
  gate and is exact, not approximate -- `cbs - gpu_samples` was 62 on all three
  runs, exactly two buffers per forward.
- **Lesson:** Before subtracting two spans, name what each one covers and in what
  order it is committed. A span that *contains* the thing you meant to exclude
  cannot be corrected by adjusting the coefficients afterwards.

<a id="meth-12"></a>
### METH-12: An awaited I/O window is a latency measurement, not a bandwidth one

- **Hypothesis:** A wall clock wrapped around an `await`ed expert fetch measures
  the bytes read, so bytes/time is the achieved read bandwidth, and comparing it
  to a measured cold-stripe ceiling says whether the drive is saturated.
- **Variants tested:** Six decode runs on Qwen 3.6 holding prompt, context length
  and `--max-new` fixed and varying only `--expert-cache-slots` across 16/24/32,
  in both sweep orders; plus a controlled 16-versus-24 pair on Qwen 3.8.
- **Evidence:** Bytes and time moved in *opposite* directions. On 3.6, going
  32 -> 16 slots cuts io time 12.9% forward and 7.8% reverse for 8.7% and 6.2%
  more throughput, while reading 34% *more* bytes -- monotonically across all
  three slot counts in both orderings. On 3.8, 11.6% fewer bytes bought 1.1%
  less time. Every reverse-sweep run was 2-4 ms/step slower than its forward
  counterpart, so session drift is real and was subtracted first; the reverse
  sweep's *last* run was also its fastest, which is the observation that rules
  drift out as the explanation.
- **What changed the conclusion:** The earlier reading divided bytes by the
  `io` figure and got 2.1-3.2 GB/s against a ~2.1-2.3 GB/s ceiling, which looked
  like saturation and closed the read side for three cycles. The window contains
  issue, queue and first-byte latency that the division attributes to transfer --
  at 16 slots the same arithmetic returns 6.2 GB/s. Worse, the model was
  *falsifiable and falsified*: "misses are issued in parallel, so removing a
  fifth does not shorten the critical path" predicts io independent of miss
  count, and io instead rises as the batch shrinks. Miss count is also the queue
  depth, so more outstanding reads capture more of the drive.
- **Final disposition:** The slot recommendation stands on other grounds (16 is
  the minimum legal count on both installs and the best measured one), but it is
  no longer supported by the reason given for it, and "cache capacity is the only
  read-side axis" is withdrawn. The replacement hypothesis -- re-scope item 2.1
  around in-flight concurrency -- was then tested against the drive and is also
  withdrawn; see below.
- **Refinement, same day:** Before writing any prefetch code, both candidate
  mechanisms were priced against the real expert file. Delivered bandwidth rises
  steeply with outstanding-request depth (2.69 GB/s at depth 1, 4.82 at 2, 5.66
  at 4), saturates by depth 4-8, and then *declines* (5.34 at 16, 4.89 at 32,
  4.24 at 64) -- and decode already runs at depth 3-6, so the queue is at its
  knee and the curve past it points down. Repetition, by contrast, is worth more
  than concurrency at every depth: at fixed depth 4 and volume, a 24-expert
  repeated pool runs at 14.4 GB/s against 7.1 GB/s for diverse offsets with the
  buffer cache bypassed, and 28.2 against 9.8 with it allowed. So the slot
  inversion is two effects pointing the same way -- a smaller cache re-reads a
  more repetitive set that both the OS buffer cache and the SSD's controller
  cache absorb, and it concentrates misses into deeper per-layer batches while a
  large cache starves many layers down to depth 1-2. Neither is a byte effect,
  and both say a wider read set is the wrong direction. **Do not build the
  prefetch.** Item 2.1 is closed with the recommendation unchanged and the
  reasoning replaced.
- **Lesson:** An awaited window bounds the transfer from above; it does not
  measure it. Divide bytes by it only after showing the window is transfer-bound
  -- and show that by varying the bytes, never by comparing the quotient to a
  ceiling. Then price the replacement hypothesis against the device before
  building on it: a mechanism that is real in the aggregate can still be one the
  system is already sitting on.

## Boundaries that were not failed experiments

- ANE/Core ML offload was excluded by the platform and architecture decision;
  no failed ANE performance candidate exists.
- Requantizing weights below INT4 was cancelled by the quality floor; no 3-bit
  or 2-bit weight runtime was built.
- Draft-model speculative decoding and n-gram verification stopped at limited
  scope analysis; they were not complete runtime candidates.
- Cache-conditional routing remains a quality-changing hypothesis. Production
  routing was never cache-conditional.
- APFS preallocation remains unexecuted.
- The iPhone port is deferred; all performance results here are Mac results.

## False rejections and attribution corrections

Three experiment conclusions are genuine reversed quality rejections:

1. [MLX full-attention geometry](05-attention-and-kv-cache.md#kv-06) had a
   genuine speed signal and later shipped as v2 after the quality oracle was
   repaired.
2. [Staged affine MPP INT4 prefill](06-prefill.md#pf-12) shipped after a
   distributional quality gate replaced an exact raw-logit comparison.
3. [Batched routed MoE prefill](06-prefill.md#pf-15) shipped after the same
   class of validation error was corrected.

QKV epilogue fusion is a separate attribution warning. An early row was noisy;
a later joint rollback supported the targeted fusion stack but did not isolate
the epilogue's share.

[Previous: Sampling, tokenization, and output](08-sampling-tokenization-and-output.md) |
[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md)
