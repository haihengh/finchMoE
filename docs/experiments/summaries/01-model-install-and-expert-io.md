# Model installation and expert I/O

[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Decode, MoE, INT4, and router](02-decode-moe-int4-and-router.md)

The model's routed-expert pool is much larger than physical memory, so disk
access is part of every decode step. These experiments established the bounded
remote installer and the demand-read path used by the frozen M2 reference
runtime.
The entries retain their stable ID order, so the installer that creates the
runtime artifact appears last.

The closing section (IO-11 onward) is a later arc on the Swift runtime, and
belongs to a different engine generation than the table below: it asks why
Qwen 3.8 reads packed experts at about 3 GB/s where Qwen 3.6 reads at about
6 GB/s on the same drive. It is recorded here rather than in a new summary
because it is the same subject — the demand-read path — and because IO-11
through IO-15 exist only as commit messages and call-site comments, which is
the wrong place for a refuted hypothesis to live.

| Current result | Disposition |
| --- | --- |
| Remote range repack with 512 KiB peak payload and scratch heap | Production |
| Default bounded parallel miss-read path | Production |
| Whole-pool `mmap`, `mlock`, compression, speculative reads, and MTLIO | Rejected |
| Read request count, read depth, and destination memory | Rejected; knobs retained off by default as falsifiers |

## Runtime expert I/O

<a id="io-01"></a>
### IO-01: `mmap` versus `pread`

- **Hypothesis:** Faulting mapped expert pages might avoid copies and beat
  explicit reads.
- **Variants tested:** Cold and warm `mmap`, sequential and
  parallel `pread`, and a full-token simulator.
- **Evidence:** A cold expert read
  took 9.88 ms with `mmap` and 2.79 ms with `pread`, about a 3.5x difference.
  Warm `mmap` won its local comparison, but the expert working set exceeds usable
  cache on the target machine. The simulator reached 3.97 tok/s with parallel `pread` and
  0.50 tok/s with `mmap`.
- **What changed the conclusion:** Warm-file results
  ceased to matter once the gate modeled continual cold misses.
- **Final disposition:** Production uses bounded parallel `pread`.
- **Lesson:** Benchmark
  the cache state the application can sustain, not the cache state a microbench
  can manufacture.

<a id="io-02"></a>
### IO-02: Parallel reads and command-buffer coalescing

- **Hypothesis:** Concurrent expert reads and fewer GPU submissions would reduce
  the two largest early stalls.
- **Variants tested:** The serial stage-0 path,
  parallel `pread`, and parallel `pread` plus command-buffer coalescing.
- **Evidence:** Decode moved from about 1.13 to 2.08 tok/s with parallel reads and
  to about 2.13 tok/s after coalescing.
- **What changed the conclusion:** Nothing;
  later stages kept the structure.
- **Final disposition:** Production.
- **Lesson:**
  Fix the largest wall-time bucket before tuning arithmetic inside a smaller one.

<a id="io-03"></a>
### IO-03: Darwin file hints

- **Hypothesis:** Darwin-specific caching and read-ahead controls could improve
  miss service without changing the data path.
- **Variants tested:** `F_RDAHEAD=0`, `F_NOCACHE`, and Darwin
  [`F_RDADVISE`](04-rdadvise.md), which advises the kernel about upcoming file
  ranges.
- **Evidence:** The first two were neutral.
  Untimed ahead advice reached 3.61-3.78 GB/s and justified a real-decode
  candidate.
- **What changed the conclusion:** Because advice ran outside the
  timed body, the probe justified only a separate real-decode program.
- **Final disposition:** The simple hints are rejected; RDADVISE is evaluated separately.
- **Lesson:** A
  promising system-call probe earns an end-to-end test, not immediate promotion.

<a id="io-04"></a>
### IO-04: Dedicated expert-I/O executor

- **Hypothesis:** A controlled executor and split-read schedule would outperform
  the existing concurrent reads.
- **Variants tested:** Worker and submission
  counts around the production path.
- **Evidence:** The best executor result was
  8.59 ms; the existing path measured 8.42 ms. Real decode was unchanged at four
  and eight workers.
- **What changed the conclusion:** The candidate failed its
  first production comparison.
- **Final disposition:** Rejected.
- **Lesson:** A new
  scheduler must beat the existing concurrency, not merely expose more controls.

<a id="io-05"></a>
### IO-05: Custom I/O worker pool

- **Hypothesis:** A fixed custom worker pool could beat bounded parallel miss
  reads.
- **Variants tested:** The default path and a four-worker override in
  repeated 256-token decode.
- **Evidence:** The pairs were mixed: 5.964 to 6.132
  tok/s, then 6.093 to 6.022; profile I/O stayed at 71.3 ms/token.
- **What changed the conclusion:** The repeated pair reversed the first result.
- **Final disposition:** The custom worker override was rejected; production retains
  parallel reads over the current miss set.
- **Lesson:** A concurrency wrapper
  must beat the existing parallel work distribution across repeated rows.

<a id="io-06"></a>
### IO-06: `mlock` and resident-model pages

- **Hypothesis:** Expert streaming evicted the 1.58 GB resident model, so pinning
  it should restore GPU time.
- **Variants tested:** Locked resident buffers and a
  diagnostic run that skipped streaming.
- **Evidence:** `mlock` recovered
  essentially 0 ms. Skipping expert streaming recovered about 12-14 ms.
- **What changed the conclusion:** The mechanism was memory-system contention,
  not resident-page eviction.
- **Final disposition:** Rejected.
- **Lesson:** Prove
  the mechanism before choosing a remedy with a large memory-policy cost.

<a id="io-07"></a>
### IO-07: Expert compression

- **Hypothesis:** Compressing expert blobs could reduce bytes read enough to pay
  for decompression.
- **Variants tested:** Zstandard and LZ4 over representative
  packed experts.
- **Evidence:** Zstandard saved about 10%; LZ4 saved 0.06%.
- **What changed the conclusion:** Neither ratio supported the required CPU work
  and format complexity.
- **Final disposition:** Rejected.
- **Lesson:** Already
  quantized weights may contain too little redundancy for transparent runtime
  compression.

<a id="io-08"></a>
### IO-08: Speculative expert reads

- **Hypothesis:** Reading or advising likely future experts could convert misses
  into warm page-cache hits.
- **Variants tested:** Entry probes followed by a
  paired real decode with speculative reads enabled.
- **Evidence:** Probes showed post-advice residency and up to 10.63 GB/s. In
  the first end-to-end pair, decode fell from 4.937 to 4.742 tok/s, total
  prefill wall time rose from 82.50 to 123.64 s, and emitted IDs diverged. The
  pair therefore failed both its speed and token-parity gates.
- **What changed the conclusion:** The probe measured page residency,
  not lead time, contention, or pipeline behavior.
- **Final disposition:**
  Rejected.
- **Lesson:** A mechanism probe cannot substitute for the first clean
  end-to-end pair. Cache prediction and replacement are covered in the
  [cache summary](03-expert-cache-prediction-and-layout.md).

<a id="io-09"></a>
### IO-09: MTLIO

- **Hypothesis:** A GPU-oriented I/O queue could remove CPU staging for routed
  experts.
- **Variants tested:** Warm MTLIO reads and production miss-state
  classification.
- **Evidence:** Warm reads reached 13.1-13.3 GB/s, but only
  5.4-7.5% of observed misses were fully warm.
- **What changed the conclusion:**
  The fast state was too rare to control token time.
- **Final disposition:**
  Rejected for the runtime.
- **Lesson:** Optimize the state distribution the
  application occupies, not the fastest state an API exposes.

## Model installation

<a id="io-10"></a>
### IO-10: Remote streaming repack

- **Hypothesis:** The installer could convert the source model directly from
  remote byte ranges without storing or loading the complete source checkpoint.
- **Variants tested:** A pinned remote revision, range planning, bounded payload
  copying, durable per-range checkpoints, interruption and process-death
  recovery, explicit discard, and atomic final promotion.
- **Evidence:** A
  validated run downloaded 14,952,958,284 bytes in 229 ranges. Its largest
  transfer was 64 MiB; peak payload and scratch heap were each 524,288 bytes.
  The freshly installed output loaded in 0.97 s. A greedy eight-token smoke
  for `The capital of France is` generated at 5.015 tok/s; this validated the
  install/load path, not answer quality.
- **What changed the conclusion:** Nothing; later installs kept the same
  core path. A later interruption gate reused every checkpointed range,
  redownloaded only the unfinished work, and produced output identical to an
  uninterrupted install.
- **Final disposition:** Production.
- **Lesson:** Model installation must obey the same bounded-memory architecture
  as inference, and resumability must trust only durable, digest-verified
  destination bytes.

## The 3.6-versus-3.8 read gap

<a id="io-11"></a>
### IO-11: Decomposing the read window

- **Hypothesis:** `io_read_wall` was the last unmeasured span inside the `io`
  window, and its own comment admitted why: `executeExpertCachePlan` timed
  around `concurrentPerform` inclusive, so the price of waking the pool threads
  — the nested dispatch — sat inside the number that was supposed to be the
  reads. If a meaningful share of the window were submission cost rather than
  transfer, the read path had headroom.
- **Variants tested:** A timestamp pair per miss, one cache line each so the
  fan-out's stores do not contend for the line they are measured through,
  yielding three telescoping parts — fan-out (batch start to first thread
  entry), span (first entry to last exit), drain (last exit to the return) —
  plus summed per-thread time, whose ratio to the span is the achieved width,
  reported as `io_conc`. A printed identity checks the tiling; two runs per
  install, interleaved, greedy, 48 tokens.
- **Evidence:** Qwen 3.8 read 232.4 ms/step with fan-out 0.42, span 231.7 and
  drain 0.27, at `io_conc` 5.60 against 5.59 misses/layer — 100% of the width
  asked for. Qwen 3.6 read 46.0 ms/step with fan-out 0.21, span 45.7 and drain
  0.17, at `io_conc` 2.96 against 4.17 misses/layer — 71% of the width asked.
  Fan-out plus drain is 0.29% of the window on 3.8 and 0.80% on 3.6.
- **What changed the conclusion:** The split refuted the hypothesis it was
  built to test and, on the way past, refuted the obvious next one. 3.8 was
  running *wider* than 3.6 — 5.60 concurrent against 2.96 — and still getting
  half the throughput, so "not enough parallelism" is dead. Bypassing the
  buffer cache does not close it either: under `FINCHMOE_IO_NOCACHE=1` the two
  sit at 241.8 and 49.4 ms/step, a 1.95x gap against the cached 2.01x. What
  the numbers do say is per-request: one 2.64 MiB read on 3.8 costs about
  4.83 ms where one 1.69 MiB read on 3.6 costs about 1.14 ms, a 2.7x per-read
  difference at equal depth.
- **Final disposition:** The submission path is closed — there is no cost
  there to recover. The counters stay unconditional so the measured build is
  the shipped build, and the split is the instrument the next four entries
  were read through.
- **Lesson:** Decompose a span into parts that telescope, then print the
  identity and check it. Two things follow. The part you suspect may be
  two orders of magnitude below the whole — here 0.3% — and the split's real
  value is that it refutes the hypothesis it was built for rather than
  confirming it. And a width counter is worth more than a depth setting: it
  turns "we asked for parallelism" into "we achieved this much of it", which
  is what made the 3.8-is-running-wider finding visible at all.

<a id="io-12"></a>
### IO-12: Read request size

- **Hypothesis:** The drive may serve a smaller request faster per byte, so
  3.8's 2.64 MiB expert reads could cost more per byte than 3.6's 1.69 MiB
  ones. Every other explanation measured so far had been eliminated, and this
  was the only structural difference left that was not the file or the engine.
- **Variants tested:** `FINCHMOE_IO_READ_SPLIT=K`, which issues one expert
  read as K sequential preads — the only way to vary request size while
  holding the install, the offsets and the bytes fixed. Split 1, 2 and 4, both
  installs, two rounds, interleaved. The files were then priced directly at a
  fixed block size, count, offset distribution and thread count, under
  `F_NOCACHE`, plus a fresh copy of one layer file written to the same volume.
- **Evidence:** Flat. Qwen 3.8: split 1 at 231.2/229.0 ms/step, split 2 at
  231.5/232.0, split 4 at 230.4/232.1 — a fifth of a percent across a
  fourfold change in request count. `io_thread_wall`, the summed read time,
  moved under 1%, so the drive serves 2.64 MiB in 4.8 ms whether that arrives
  as one request or four. Qwen 3.6 was equally flat, and its round-2 drift of
  about 20% appeared at every split including the ones that cannot have caused
  it. The file pricing left something behind, though: 3.8's `layer_00` at 2.35
  / 2.86 / 2.62 GB/s (one thread random, four threads random, one thread
  sequential) against 3.6's `layer_00` at 3.29 / 4.63 / 3.61.
- **What changed the conclusion:** Nothing in the engine; the request-size
  hypothesis is refuted. The pricing arm produced the next claim — that the
  gap lives in the installed files — and a fresh copy of 3.8's `layer_00`
  recovered only about a fifth of it (2.80 / 3.45 / 3.15, a consistent
  1.19-1.21x), reported as an upper bound because the SSD's write cache could
  be part of it. What survives from this entry is the refutation and the
  method: sequential reads are slower too, which rules out the offset pattern
  entirely, because a pattern the engine never issues costs the same way.
- **Final disposition:** Rejected, knob kept off by default as the falsifier
  for the refutation — the same disposition `FINCHMOE_IO_NOCACHE` got, and for
  the same reason: it answers a question worth not re-asking. Tested by a
  fixture whose every byte depends on its absolute file offset, since the
  existing uniform-tag layer cannot see a tiling bug in a chunked read (every
  byte of an expert is the same byte, so a skipped chunk reads as valid data);
  splits of 3 and 7 exercise the remainder path against a stride they do not
  divide.
- **Lesson:** Two of them. A shape knob and a depth knob are different
  experiments, and conflating them wastes a cycle — this one varies how a read
  is cut, IO-14 varies how many are outstanding, and only the second is about
  queueing. And design the fixture for the failure you are actually exposed
  to: a uniform-tag fixture will pass a chunked read that skips chunks.

<a id="io-13"></a>
### IO-13: Retracting the file claim

- **Hypothesis:** The previous entry's pricing arm reported that Qwen 3.8's
  layer file is served 1.37x to 1.60x slower than Qwen 3.6's for an identical
  pattern, and called that the shape of the finding.
- **Variants tested:** Five layers per install instead of one, same 2 MiB
  random blocks, four threads, `F_NOCACHE`, on the same volume.
- **Evidence:** 3.8 layers 00, 12, 24, 36 and 47 read at 2.81, 3.27, 3.40,
  3.23 and 3.12 GB/s (median 3.23); 3.6 layers 00, 10, 20, 30 and 39 read at
  3.86, 3.12, 3.18, 3.14 and 3.78 (median 3.18), against a within-install
  range of 2.81-3.86. `layer_00` was picked for both installs because it sorts
  first, and it is the slowest of the five 3.8 files and one of the two fastest
  3.6 files — so the comparison drew the widest available gap out of two
  distributions that overlap completely.
- **What changed the conclusion:** Sample size, and the sampling rule. The
  medians differ by 1.6%. What survives is narrower and still worth having:
  rewriting 3.8's `layer_00` lifts it from 2.81 to 3.45, but 3.45 is what its
  own siblings already reach, so that measures a within-install straggler and
  not an install-level penalty — a 63 GB rewrite of `packed_experts` would buy
  the install nothing. The request-size refutation from IO-12 stands as
  measured.
- **Final disposition:** Claim withdrawn and recorded as withdrawn rather than
  amended away. The honest state at this point is narrower than the previous
  commit claimed: request size refuted, file refuted, and the engine's 1.94x
  gap still unexplained.
- **Lesson:** Pick the sample by the distribution, not by the alphabet. A
  sort-order pick is a convenience that silently selects an extreme when the
  distributions overlap, and with n=1 per arm there is nothing in the result
  that says so. The corollary is about the record: a refuted claim left
  standing in a comment or a commit is how a wrong hypothesis gets re-tried
  six months later, so retract it where it was made.

<a id="io-14"></a>
### IO-14: Capping the read depth

- **Hypothesis:** Qwen 3.8 keeps 5.59 reads and 14.76 MiB in flight where
  Qwen 3.6 keeps 2.95 and 4.99, and 3.8 is the one that is both deeper and
  slower. That is backwards for a queue, so capping how many reads are
  outstanding at once should help.
- **Variants tested:** `FINCHMOE_IO_READ_WAVE=K`, which issues the misses in
  waves of K. Because `concurrentPerform` returns only when every iteration
  has, a wave is a barrier and at most K preads are in the drive at a time;
  zero, the default, is the single unbroken fan-out every earlier number was
  measured through. Waves 0, 2, 3, 4 and 6, two rounds, round 1 ascending and
  round 2 descending so the rounds cannot agree by drift.
- **Evidence:** Refuted, and the shape of the refutation is the useful part.
  Wave 0 gave `io_conc` 5.28 at 250.43/239.61 ms/step and 4.92 ms/read; wave 2
  gave 1.81 at 253.93/252.93 and 1.71 ms; wave 3 gave 2.55 at 246.96/246.84 and
  2.34 ms; wave 4 gave 3.08 at 242.14/242.38 and 2.78 ms; wave 6 gave 3.93 at
  238.80/240.03 and 3.50 ms. Across a 2.9x range of outstanding reads the read
  window moves 6%, toward the *wider* end. Per-read latency scales with depth
  almost exactly linearly while the aggregate rate stays pinned at 2.87-2.97
  GB/s: a saturated drive converting queue depth into latency and returning
  the same bandwidth. Token IDs were identical across the whole sweep, which
  is what makes this about depth rather than content, and inertness was
  checked twice because the two failures differ — the refactor being inert at
  its default (current build against the binary saved before the knob existed)
  and the width changing nothing (same binary, only the environment differs).
- **What changed the conclusion:** The mechanism turned out to be the thing
  the knob was meant to expose, seen from the other side. The flat throughput
  and the linear latency curve are one measurement read two ways, and only the
  split counters make that visible; it is also why the offline replay found
  pool width flat (depth 3 at 147.3 against depth 10 at 149.6). The prior for
  this experiment was weak and the record says so: the read-wave doc comment
  had first cited the replay as independent support for deeper being slower,
  and the number it cited — 177.9 — is the page-cache-allowed replay row, not
  a depth. That citation was corrected in place.
- **Final disposition:** Rejected, knob off by default. The same-depth control
  this produced is the number the rest of the arc is read against: 3.6 at
  `io_conc` 2.95 costs 0.809 ms/read (6.40 GB/s); 3.8 at 2.55 costs 2.340
  (2.87 GB/s) and at 3.08 costs 2.778 (2.92 GB/s) — same engine, same drive,
  same concurrency band, 2.2x apart, with 3.8 running 2.9-3.4x the per-read
  latency even after scaling for its larger stride.
- **Lesson:** When two counters move in opposite directions, the invariant
  between them is the measurement — a rate that holds while its numerator and
  denominator both move is telling you the resource is saturated, and no
  setting of the knob will change it. And correct a wrong citation where it
  stands. A comment that supports a hypothesis with a number that measures
  something else is a loaded gun for the next reader.

<a id="io-15"></a>
### IO-15: Moving the destination out of the Metal heap

- **Hypothesis:** The plan had named this in as many words as the last
  difference standing between the engine and its own offline replay: the
  engine's slot pages are GPU-shared `MTLBuffer`s under a live Metal heap,
  which changes the vm object's reclamation behaviour in a way a Python replay
  cannot hold. If that were it, reading into an ordinary aligned allocation
  and copying into the slot should recover most of the way to the replay's
  5.42 GB/s, with the cost reappearing in the copy.
- **Variants tested:** `FINCHMOE_IO_STAGE` reads each expert into a plain
  `posix_memalign`'d staging buffer and memcpys it into the slot, instead of
  preading straight into it. The staging buffer uses the same `scratchAlignment`
  and the same padded length as a slot, so alignment and page size match
  exactly and the only difference left is that the slot goes to
  `makeBuffer(bytesNoCopy:options:.storageModeShared)` and the staging buffer
  does not. `readFull` was split into `readChunks` plus a thin wrapper so the
  two spans can be timed apart. Off and on, both installs, 16 slots, two
  rounds, round 1 ascending and round 2 descending.
- **Evidence:** 3.8 read 226.71/226.70 ms/step with staging off at 3.28/3.28
  GB/s and 232.32/233.33 with it on at 3.28/3.27 — rounds agreeing to 0.3%,
  the window growing 2.5%, and the memcpy it adds costing 30.5 ms of thread
  time, 2.4% of the window. 3.6 read 47.32/45.10 at 6.24/6.55 with it off and
  47.40/47.60 at 6.90/6.87 with it on. The staging buffer is one hot 2.7 MiB
  page per thread, reused for every read — maximally TLB- and cache-resident,
  and entirely independent of the 2028 MiB slot pool. So pread into the best
  available destination is exactly as slow as pread into the worst.
- **What changed the conclusion:** The plan's named discriminator is flat,
  and cleanly, which closes the page-mapping, TLB, page-table, fragmentation
  and fault-behaviour explanations together rather than only the first. The
  control is what makes it a finding instead of an artifact of the knob: 3.6
  under the same change is not slower, so the staging path is not itself
  expensive and the null result on 3.8 belongs to the read rather than to the
  copy added to it. The copy then separates the installs almost perfectly —
  3.8 at 1.6815 ms/MiB read and 0.0430 copied, 3.6 at 0.5129 and 0.0503, within
  15% on the copy and 3.3x apart on the read. Same memory system, same page
  size, same bytes, same memcpy: symmetric on the operation with no disk
  behind it, asymmetric on the one with a disk behind it. One earlier number
  did not survive: the slot sweep's claim that 3.6 loses a quarter to a third
  of its per-byte read rate when its pool doubles is withdrawn, because its
  absolute read thread time is flat across the change (146.5/138.6 ms/step at
  16 slots against 136.9/144.1 at 32) — the rate fell because 32 slots hit
  more and so read 30% fewer bytes in about the same window, while per-read
  latency rose 38% across 30% fewer reads. The corrected reading is "a bigger
  pool issues fewer reads and each costs more", not "a bigger pool reads
  slower".
- **Final disposition:** Rejected as a setting and kept as a measurement tool:
  it costs a memcpy of every expert read, so it is off when the variable is
  absent or zero, and with it off `readFull` is the byte-for-byte path it
  always was. Gate: the token stream of the parent commit's own binary, the
  instrumented binary with staging off, and the instrumented binary with it on
  are identical, all 48 IDs, against a baseline built in a worktree so the
  uncommitted work was never in the comparison; 893 tests in 151 suites passed
  (was 891), and both accounting identities hold under staging.
- **Lesson:** When a candidate is refuted, check what it refutes on the way
  past. This one killed five explanations at once because its control arm
  made the null result about the read rather than about the copy — a knob
  whose control is not run is a knob whose null result means nothing. And a
  rate is not comparable across two arms that read different byte counts:
  "bytes per second" moves when the numerator changes for a reason that has
  nothing to do with speed, which is exactly the error the slot sweep made.

<a id="io-16"></a>
### IO-16: Reproducing the engine's batch structure offline

- **Hypothesis:** IO-11 established that a step is neither one fan-out nor a
  queue: `executeExpertCachePlan` runs once per layer behind a
  `concurrentPerform` barrier, so a step is 40-48 strictly serialized batches
  of 3.3 or 5.1 reads, and `io_conc` is just misses divided by layers. If the
  cost is in that structure — the barrier, the per-batch ramp, the file walk
  across 48 distinct inodes — then a harness with the same structure should
  reproduce the per-read penalty, and at equal structure the difference
  between the installs should shrink.
- **Variants tested:** `tools/read-sweep/read_batch.c`, which walks a list of
  files in order, issues m concurrent preads from each behind a barrier, and
  moves to the next file. The barrier is `dispatch_apply` on a private
  concurrent queue, the engine's own primitive; a hand-rolled spin/atomic
  handshake was measured alongside it on identical offsets. Controls: 48
  batches against one file (isolating the barrier), and a file-count sweep at
  fixed reads per round and bytes per round — 48 files x5, 24 x10, 16 x15 —
  isolating the distinct-inode walk.
- **Evidence:** The structure reproduces closely — 133 and 249 reads per round
  against the engine's 134.8 and 245.7, batch widths 3.33 and 5.19 against
  `io_conc` 3.26 and 5.30. Cold round 1, against a same-hour single-file
  continuous reader at 3.007 GB/s (3.6) and 2.966 (3.8): the batch structure
  gave 1.917/2.123 on 3.6 and 2.038/2.482 on 3.8, where the engine itself
  reads 4.736 and 3.167. **The harness inverts the engine's ordering** — it
  puts 3.8 ahead, the engine puts 3.6 ahead by 1.5x. Both controls came back
  clean: the barrier is nearly free (48 batches against one file gives 2.759
  against the continuous reader's 2.966, about 7%), and the file walk is not
  the cost either (fewer files is slower, not faster: 3.163 for 48 files,
  3.003 for 24, 2.883 for 16).
- **What changed the conclusion:** The object the harness was built to test is
  not the mechanism, so the structure is falsified as the cause rather than
  left as a candidate. Two by-products are worth keeping. The barrier
  implementation matters more than the barrier: the hand-rolled handshake and
  `dispatch_apply` differ by about 27% on identical offsets (3.6 at 1.572
  against 2.054 GB/s, 3.8 at 1.825 against 2.406), which is the size of
  several effects this harness exists to measure, so `dispatch_apply` was kept
  because it is the engine's primitive and the alternative would have reported
  a harness artifact as a drive property. And the same-file control is
  confounded by its own residency: rounds 1, 2 and 3 gave 2.759, 4.733 and
  7.454 GB/s as a 1352 MiB file stays resident, which means even round 1 is
  partly cache-served by the batches that came before it.
- **Final disposition:** Negative result recorded so the structure is not
  re-proposed; the harness stays in `tools/read-sweep/` and is not wired into
  the engine.
- **Lesson:** A harness that reproduces a shape without reproducing the
  ordering has falsified the shape as the cause, and that is a result — not a
  failed measurement. The control that fails is still informative: here, both
  controls exonerated what they were built to price, which is what let the
  next entry move the question somewhere else. And hold the denominator when
  the metric is a rate: the file-count sweep was first read as showing a
  per-file cost, and the reversal — fewer files is slower — is what showed the
  reading came from bytes per round rather than from the walk.

<a id="io-17"></a>
### IO-17: Replaying the engine's own traces

- **Hypothesis:** IO-16 exonerated the batch structure, and every other
  read-side candidate had been measured away across IO-12 through IO-15. What
  remained untested was the access pattern itself. A synthetic harness cannot
  answer that, because offsets drawn uniformly at random are a different
  workload wearing the same bytes — which is precisely the error METH-14
  records. So the engine's own read sequence was captured and replayed.
- **Variants tested:** `FQ_EXPERT_TRACE` captured for both installs in one
  session (4,179 reads on 3.6, 134.8/step, 227.5 MiB/step; 7,617 on 3.8,
  245.7/step, 648.8 MiB/step). Both traces then replayed through one harness
  in the same process, interleaved and round-alternated, each at its own
  engine concurrency (3 and 5), with the engine's own `F_NOCACHE` setting
  (off) and warm reused destinations, scored on the engine's own metric — the
  **sum of per-batch read spans**, which is what `io_read_wall` measures and
  which excludes anything happening between batches. Then a gap sweep at 0,
  0.5, 1.0 and 2.0 ms between batches, on the last structural difference
  remaining (the engine spaces its batches with GPU work; the replay does
  not), with a reverse-order control; and working-set and reuse-distance
  accounting computed from the traces.
- **Evidence:** The engine and its own replay disagree in opposite directions.
  3.6 runs 47.18-52.19 ms/step against its trace replayed at 62.68 — the
  engine is **1.20-1.33x faster** than a faithful offline replay of its own
  reads. 3.8 runs 213.73-233.15 against a replay of 162.95 — the engine is
  **1.31-1.43x slower**. Replayed identically, the traces put 3.8 ahead of 3.6
  by 1.34-1.49x; the engine reads them the other way by 1.57-1.59x. Both arms
  are generous to the engine (warm reused destinations, where engine-shaped
  destinations were priced at 19.3 ms/step, about 15%), so the 3.8 deficit is
  a lower bound. The working sets give the shape of the asymmetry: 3.6 touches
  2,531 distinct experts, 4,271 MiB, median reuse distance 1,333 MiB, 60.6%
  first-touch; 3.8 touches 4,195, 11,077 MiB, median 3,694 MiB, 55.1%
  first-touch. 3.6's working set fits in 16 GB and warms completely — a fifth
  replay pass reaches 21.4 GB/s — while 3.8's does not, so 3.8 never gets that
  relief. The gap sweep on 3.8, whose working set cannot be flattered by the
  page cache, is flat: 162.95 / 152.29 / 158.99 / 154.63 ms/step at 0 / 0.5 /
  1.0 / 2.0 ms. The 3.6 column of the same sweep looks like a large spacing
  effect (62.68 down to 23.27) and is not one: run in reverse the trend
  reappears in the same monotone order (89.92 ms/step at gap 4.0 first, then
  57.34, 40.91, 28.90 and 11.12 as the run proceeds), so it tracks position in
  the sweep and not gap width.
- **What changed the conclusion:** The access pattern is exonerated and the
  engine's 3.8 runtime is not. This also retracts the previous cycle's interim
  reading, which held that 3.8's engine sat at the device's cold rate for its
  shape and that the section's premise therefore did not survive: that was
  measured against synthetic uniform-random offsets, which are *slower* than
  the engine's real pattern, so the comparison flattered the engine. Against
  its own trace the premise survives — 3.8's engine reads at about 0.7x what
  its own pattern delivers cold.
- **Final disposition:** The read side is closed as "not the pattern". The
  deficit is in the engine's 3.8 runtime path, which is now the only place
  left for it to be. The next discriminator is named and bounded: the slot
  destination working set is 48x16x2.64 MiB = 2.03 GiB of rotating pages on
  3.8 against 40x16x1.69 MiB = 1.08 GiB on 3.6, on a 16 GB box whose peak
  compressed footprint memguard reports at 3.1 GB — price it on the new traces
  with the span-sum metric. If it does not account for 1.3-1.4x, the next step
  is a GPU-overlap experiment and not a read-side one.
- **Lesson:** Replay the workload, not a stand-in for it — and notice that the
  same harness produced both the false acquittal and the finding, because
  synthetic offsets are a different workload wearing the same bytes. A
  correction that flips the sign of a conclusion is worth more than one that
  changes a magnitude, and the way to earn it is to make the two arms
  comparable: interleaved in one process, on the engine's own offsets, scored
  on the engine's own definition of the metric. The gap sweep carries a second
  lesson: a single-order sweep on a working set that fits in RAM measures the
  sweep's position, not its variable, and only a reverse-order run can tell
  you which one you measured.

[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Decode, MoE, INT4, and router](02-decode-moe-int4-and-router.md)
