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
- **Refinement, 2026-09-11:** The depth curve above does not describe the
  regime the engine operates in, and the correction comes from measuring depth
  on the engine instead of against the device. `FINCHMOE_IO_READ_WAVE` sweeps
  outstanding reads from 5.28 down to 1.81 and the rate stays pinned at
  2.87-2.97 GB/s across the whole range; per-read latency absorbs the
  difference almost exactly linearly
  ([IO-14](01-model-install-and-expert-io.md#io-14)). The probe, by contrast,
  rose steeply from 2.69 GB/s at depth 1 to 5.66 at depth 4 and then declined,
  and its repeated-pool arm reached 14.4 GB/s bypassed and 28.2 allowed. Two
  things follow. The *shape* — a steep rise, a knee at 4-8, a decline past 16 —
  is not reproduced where the engine actually reads, so a recommendation that
  the queue is "at its knee" was drawn from a curve the engine does not sit on.
  And the *absolute* rates on the repeated arm exceed what either install
  delivers cold on this volume by a factor of four to eight. `F_NOCACHE` was
  set, so this is not the unified buffer cache — but it bypasses that cache and
  nothing else, and every cell in the probe re-read the same 433 MiB
  `layer_00.bin`, which the drive's own controller and SLC caches are free to
  serve in a way that a 63 GB working set never is. A paired purge control
  since then shows the OS buffer cache serves none of the engine's decode
  window on either install, which is why the engine's own numbers look nothing
  like the probe's. At minimum the probe measured a state the engine is not in. The repetition result may well be real — the `FINCHMOE_IO_NOCACHE`
  A/B is explained by exactly that mechanism — but its magnitude was measured
  in a state the engine is not in. Treat the probe rows as an upper bound on
  what the device could do, never as what it does; and see
  [METH-15](#meth-15) for the drift
  that makes even paired device rows fragile.
- **Lesson:** An awaited window bounds the transfer from above; it does not
  measure it. Divide bytes by it only after showing the window is transfer-bound
  -- and show that by varying the bytes, never by comparing the quotient to a
  ceiling. Then price the replacement hypothesis against the device before
  building on it: a mechanism that is real in the aggregate can still be one the
  system is already sitting on.

<a id="meth-13"></a>
### METH-13: A near-complete serial sum bounds the overlap, without instrumenting it

- **Hypothesis:** The decode pipeline's stated design — the shared expert on an
  early-committed buffer overlapping the expert pread, cross-layer pipelining —
  means reads and device execution overlap, so the two costs are partly
  alternatives rather than additive, and improving either buys less than its
  share of the token.
- **Variants tested:** None. All seven captured counter runs were re-read rather
  than re-run: `io`, `gpu_cb1`, `gpu_routed`, `head_wall` and `ple_wall` summed
  against the token, across two installs, three context lengths and three slot
  counts.
- **Evidence:** The sum lands at **93.0-97.8%** of the token on every run. The
  device term is complete rather than a sample — `totalGpuRoutedNanos`
  accumulates the routed, shared and phase-1-hit buffers, and `gpu_samples`
  covers 5821 of 5883 buffers, the two per forward it misses being the embed and
  head syncs — so `gpu_cb1 + gpu_routed` is every GPU nanosecond a layer spends.
  The 2.2-7.0% remainder has to hold the CPU encode (1.7-2.6 ms/step), sampling
  and detokenization. Overlap is therefore bounded at a few percent of the
  token. The mechanism is real and documented in the code — `encodeRoutedTail`
  says so in its own comment (`RealForwardRunner.swift:2794-2795`) — but
  undersized: all three tail buffers total 0.87 ms/layer of device time against
  a 4.74 ms/layer read window, so perfect overlap of every one of them fills 18%
  of it. The order is forced, not incidental: the CPU readback of the router's
  top-k indices (`:2813-2819`) is gated by the blocking `waitForCompletion(cb)`
  at `:5236`, and layer N+1 cannot be encoded before layer N's routed output
  exists because that output is its input.
- **What changed the conclusion:** Nothing was re-measured; the numbers were
  already on the counters line. What changed is which arithmetic was done with
  them. Every earlier cycle asked what the read *share* was. Adding the device
  share to it answers a different and more useful question — how much of the
  token is *sequential*.
- **Final disposition:** The serial model replaces the overlap model. Device work
  pays 1:1 against the token, which re-promotes GDN's 61.85 ms/step (17.1% of a
  3.8 token) as a target and demotes int8 KV to a VRAM play on the smaller
  stack, and it exposes the largest unexplained quantity left: 773.7 MiB in
  227.36 ms is 3.57 GB/s against a drive delivering 5.4-5.7 GB/s at this depth,
  so ~84 ms/step (23% of the token) is fixed per-batch cost rather than
  transfer.
- **Lesson:** When two spans are claimed to overlap, sum them. A sum that
  approaches the whole bounds the overlap at the complement, needs no new
  instrumentation, and is not fooled by a span whose *name* says it overlaps.
  Then check that sum against the design comment that promises the overlap: a
  mechanism can be present, documented in the code, and five times too small to
  matter. Corollary, from the same cycle — an argument that prices a change on
  one axis (bytes) while the win would land on another (latency) is not a
  refutation, and reading it as one nearly discarded a live hypothesis.

<a id="meth-14"></a>
### METH-14: A replay is only as faithful as the reuse it cannot see

- **Hypothesis:** METH-13 closed by naming the next measurement — replay the
  engine's own pread offset sequence offline at depth 6 with a cold cache, and
  see whether it reproduces the engine's 4.74 ms/layer read window. An offline
  replay holds the offsets and the depth exactly, so it should price the drive
  and nothing else.
- **Variants tested:** Five conditions on the engine's exact 9,083-pread
  sequence — page cache allowed and bypassed, ten resident destinations against
  48×16 rotating engine-shaped destinations (1.98 GiB), and depth 3 against
  depth 10 — interleaved and round-alternated (METH-06). Then the one condition
  that showed a large win was A/B'd on the engine itself.
- **Evidence:** The replay reproduced *neither* number it was built to choose
  between, landing below both: 130.3-177.9 ms/step against the engine's 225.4,
  with the engine slower than its own worst offline arm. Two differences were
  then measured rather than assumed. Engine-shaped destination pages cost
  19.3 ms/step — real and small. Pool width cost nothing (depth 3: 147.3;
  depth 10: 149.6), the queue-depth knee from the earlier drive probes
  reappearing. The remaining ~75 ms/step stayed unmodelled and was reported as
  such.
- **What changed the conclusion:** The bypassed arms were 42-61 ms/step faster
  offline — a fifth to a quarter of the engine's own window, in both orders —
  and the engine opens its layer files without `F_NOCACHE`, so this looked like
  a free win. Added as an off-by-default knob and A/B'd on the engine in both
  orders: **225.4 → 261.0 ms/step of `io`, −12% tok/s**, token IDs identical
  across all four runs. The sign was
  wrong, not just the size. The trace explains it: 41.7% of the engine's 9,083
  reads repeat a (layer, expert) pair already read during the run, but the
  *median* reuse distance is **3.97 GiB** and not one of the 3,787 repeats lands
  within 512 MiB. Ten recycling 2.64 MiB destination buffers — 26 MB — were
  holding reuse that the engine's own OS cache cannot, so the harness's
  "bypassed" arm was measuring a hit rate the engine has and the engine's
  "allowed" condition could not reach. The harness was not wrong about the
  drive; it was wrong about the *workload*, and it could not tell the two apart
  because it never reported a reuse distance.
- **Final disposition:** The offline replay is retired as a predictor for the
  read path and kept only as a drive pacer. The `F_NOCACHE` knob ships off by
  default, with the engine's number recorded at the call site so the offline
  argument is not re-derived by the next reader. The ~75 ms/step residual is
  recorded as unexplained, with its next discriminator named: the engine's slot
  pages are GPU-shared `MTLBuffer`s under a live Metal heap, which changes the
  vm object's reclamation behaviour in a way a Python replay cannot hold. A
  corollary control was also rebuilt: the first pass's "syscall cost" arm
  re-read one blob 5,000 times and so priced bandwidth, not syscall entry
  (978 µs/call). Redone at one byte per call it is 28.3 µs hot and 107.9 µs on
  a fresh page, and the engine's own batch shape splits **2.0% issue / 98.0%
  completion** — which is what the in-engine split independently said.
- **Lesson:** A replay inherits the offsets, the depth and the block size, and
  silently discards the destination working set and the reuse distance — the
  two quantities that decide whether a cache arm means anything. Before
  trusting a cache-condition result, compute the workload's reuse distances and
  compare them against the cache it claims to model; a harness with a 26 MB
  working set will report a hit rate that a 2 GiB one cannot have. The general
  form: when a microbenchmark predicts a sign, the prediction is a claim about
  the *workload it ran*, and the burden is on showing the engine shares the
  property that made it true. Related, and the same failure in miniature — a
  control arm is only a control for what it actually varies, so check that the
  quantity it names is the quantity it moves before quoting it.

<a id="meth-15"></a>
### METH-15: A rate is only comparable inside the session that measured it

- **Hypothesis:** The read-gap investigation ([IO-11](01-model-install-and-expert-io.md#io-11)
  through [IO-17](01-model-install-and-expert-io.md#io-17)) compares an engine
  run against an offline replay, and the comparison is a subtraction of two
  rates. Both numbers were measured carefully, so the subtraction should hold
  even though the runs happened weeks apart.
- **Variants tested:** Nothing was changed to test this; it was found by
  repeating an arm. The same install's engine was run twice within one session
  (paired, interleaved, identical flags and prompt), one identical synthetic
  cell was repeated minutes apart, and one cell was repeated four times.
- **Evidence:** The drive drifts *within* a session. 3.6's engine read at 4.68
  and then 5.18 GB/s in paired runs eleven minutes apart, while agreeing with
  itself to 0.3% inside any one of those runs. One identical synthetic cell
  gave 1.825 and 3.163 GB/s twenty minutes apart — a 1.7x swing on a variable
  nothing had touched. Four repeats of a single cell gave 2.729 / 3.180 /
  3.113 / 3.152. Across sessions the same engine figure has been recorded at
  3.60, 6.24, 6.55, 6.90 and 4.736 GB/s. Spotlight indexing is live on the
  volume (`mdbulkimport`, `mds_stores`), and while it is not proven to be the
  cause it is not excluded either.
- **What changed the conclusion:** The gap the investigation exists to explain
  is 1.57x. The instrument's own drift, measured on a fixed variable, is of the
  same order or larger. That makes §2.1's published table — an engine row at
  3.60 GB/s against a replay row at 5.42, measured in different sessions —
  uninterpretable as a subtraction. The rows are not wrong measurements; they
  are not a comparison, and reading them as one produced the named
  discriminator that IO-15 then spent a cycle refuting. Two smaller readings
  were also withdrawn on this basis: a file-level claim drawn from one layer
  per install ([IO-13](01-model-install-and-expert-io.md#io-13)), and a
  spacing effect that reversed under a reverse-order run
  ([IO-17](01-model-install-and-expert-io.md#io-17)).
- **Final disposition:** Every arm that is compared against another is now run
  paired and interleaved inside one session, with the rounds ordered one way
  and then the other, so that drift cannot separate the arms and agreement
  cannot be an artifact of order. Cross-session rates are quoted as history,
  never subtracted. The pairing is stated wherever a ratio is claimed.
- **Lesson:** A rate carries its session with it. Before subtracting two
  measurements of the same physical quantity, repeat one of them and see how
  much it moves on its own — if that movement is comparable to the effect you
  are measuring, the effect has not been measured yet, however careful each
  individual number was. The related failure is the single-order sweep: when a
  working set fits in RAM, a parameter sweep over it measures the sweep's
  position rather than its variable, and only a reverse-order run can tell you
  which one you measured. METH-06 covers the same ground for first-run and
  thermal state; this is the magnitude, measured, and the size of the effect it
  invalidates.

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
