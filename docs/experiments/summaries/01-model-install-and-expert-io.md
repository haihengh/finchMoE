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

### IO-18: Pricing the slot destination working set, and finding the replay must be run cold

**The measurement IO-17 named, run, and answered no.** IO-17 closed with a bounded
instruction: the engine holds one `PreadExpertStreamer` per layer
(`Model.swift:689` is inside the layer loop), each with its own `slotCount` slots
allocated once at init, so at 16 slots it rotates **48 x 16 x 2,768,896 B = 1.98 GiB**
of anonymous pages on 3.8 against **1.05 GiB** on 3.6 -- price that on the new traces
with the span-sum metric, and if it does not account for 1.3-1.4x, the next step is a
GPU-overlap experiment and not a read-side one.

**Two things were wrong with the price the plan already had.** §2.1's table carried an
engine-shaped-destination row at 149.6 against 130.3 warm, "+19.3 ms/step, real and
small" -- but that row was measured with the page cache **bypassed**, and the engine
does not bypass the cache. The cell that matches what the engine actually does,
`allowed + engine-shaped`, had never been run. It is also the only cell where the
interaction can appear: 2 GiB of *resident* destinations competes with the page cache,
and 3.8's 11 GiB working set does not fit in the first place.

**The harness is committed this time.** IO-17's replay was never checked in, so the
plan's read-side numbers could not be re-derived from the repo -- `tools/read-sweep/README.md`
told the reader to replay `FQ_EXPERT_TRACE` and no committed tool could. Now
`tools/read-sweep/replay_dest.py` takes a trace and a layout, builds the ring the way
the engine builds it (one page-aligned buffer per `(layer, slot)`, pre-faulted outside
every timed span, freed between installs so only one ring is ever live),
`capture-traces.sh` re-captures the engine half, and `paired-engine-replay.sh`
alternates the two. Conditions, at each install's own read concurrency:

| condition | what it isolates |
| --- | --- |
| `allowed, warm dest` | IO-17's condition, reproduced |
| `bypassed, warm dest` | IO-17's condition with the page cache removed |
| **`allowed, slot dest`** | **the engine's own condition AND its own destination shape** |
| `bypassed, slot dest` | both ablations together |
| `bypassed, slot dest, depth 3` | pool width |

**Result: +1.5 ms/step, not +19.3.** Engine and replay were interleaved round by round
(`paired-engine-replay.sh 3.8 ... 5 6`), the engine's `io_read_wall_ms/step` read against
a replay that ran within a minute of it, with condition order alternated by round parity
so the position bias separates from the effect. Solving the paired deltas for the two
unknowns (`D` = true slot cost, `A` = the cost of running second):

| round | engine `io_read_wall` | `io_thread_wall` | `io_conc` | replay, in run order |
| --- | ---: | ---: | ---: | --- |
| 2 | 215.11 | 1142.21 | 5.32 | warm 128.64, slot 129.68 |
| 3 | 215.48 | 1154.09 | 5.36 | slot 129.62, warm 130.73 |
| 4 | 216.09 | 1157.11 | 5.36 | warm 129.48, slot 131.90 |
| 5 | 216.62 | 1169.36 | 5.41 | slot 129.73, warm 130.25 |
| 6 | 216.01 | 1170.38 | 5.43 | warm 129.32, slot 132.28 |

`D` = **+1.5 ms/step** -- 1.2% of the replay window and 1.7% of the 86 ms/step gap it
would have to explain. The destination working set is exonerated a second time, and
this time in the condition the engine actually runs in.

**The engine / replay ratio is tighter than the record had it, and the same on every
round: 1.66x** (215.11/128.64, 215.48/129.62, 216.09/129.48, 216.62/129.73,
216.01/129.32), against IO-17's 1.31-1.43x.

**Three more read-side candidates, all negative, all cold.** The engine's 48 per-layer
batches take 4.48 ms each where the replay's take 3.06 -- so the next question was what
the engine does *between* its batches:

| variable | tested | result |
| --- | --- | --- |
| inter-batch idle (the engine's ~1.5 ms of GPU work) | 0 / 1.5 / 3.0 ms, cold | 152.2 / 155.1 / 165.8 -- a 3 ms idle costs the next batch **0.28 ms**, not the 1.3 ms needed. The drive does not re-ramp. |
| memory pressure | 0 / 2 / 4 / 6 GiB resident ballast | flat within noise (145-164), and the slot ring's ~80,000 re-faults per pass cost nothing in wall time -- they hide inside the drive wait at 5 workers |
| the 3.8-only PLE stream | `ple_wall_ms/step` 3.84, 496 opens / 31 steps | open-dominated, ~16 opens/step; far too few bytes to cost the expert window 63 ms |

**The find of the session is a method error, and it changes how the older numbers read.**
`F_NOCACHE` is not honoured on this volume and `sudo purge` is unavailable, so the only
way to get a genuinely cold pass is to push the trace's pages out by reading something
else. `--evict-gib` streams 17 GB of the *other* install's experts before every pass.
On 3.6, the same condition, in one session:

| 3.6 replay, `allowed, warm dest` | ms/step | GB/s |
| --- | ---: | ---: |
| cold (first run) | 56.71 | 4.21 |
| **evict = on** | **56.96** | **4.19** |
| warm (second run, warmed by the first) | **8.99** | **26.53** |
| evict = on | 56.73 | 4.20 |

**6.3x, from the cache alone, on this box's own 4.3 GiB working set.** The plan's "3.6's
working set fits in 16 GB and warms completely -- a fifth replay pass reaches 21.4 GB/s"
is a property of the *replay*, not of the engine, and the engine -- **51-57 ms/step cold,
the same session** -- is sitting on the cold rate. On 3.8 the same eviction moves the
replay only 144.7 -> 149.1 (3.1%), so 3.8's replay was never cache-inflated and its 4.6 GB/s
is real NAND throughput.

**Which corrects IO-17's headline for 3.6.** IO-17 recorded the 3.6 engine as 1.20-1.33x
*faster* than a faithful replay of its own reads. Cold, the two are at parity
(51-57 against 56.7-57.0). The engine does not beat its replay on 3.6 and it does not
need to; the replay only appeared faster or slower depending on how warm it was run.
**The 3.8 deficit survives the correction and is the only one left: 1.42-1.46x against a
cold replay** (215 against 147-153).

**And it is now a per-request number rather than a window.** `io_thread_wall / misses`
is the average pread, and both arms were at the same achieved concurrency:

| | per-read service time | concurrency |
| --- | ---: | ---: |
| 3.6 engine | 1.22 ms | 3.23 |
| 3.6 replay, cold | 1.26 ms | 3.0 |
| 3.8 engine | **4.65 ms** | 5.32 |
| 3.8 replay, cold | **2.62 ms** | 5.0 |

Same request size, same offsets, same session, same depth: **3.6's engine and its replay
agree to 3% per request, and 3.8's engine takes 1.78x as long as its replay for the same
2.64 MiB read.** That is not queueing, not a pattern, not a destination and not dispatch
-- six candidates have now been priced and removed -- it is the read itself, and it is
specific to the 3.8 install.

- **Disposition:** the destination working set does not account for the deficit
  (+1.5 ms/step of 86), so §2.1's named branch is taken -- but the branch it named, a
  GPU-overlap experiment, is worth at most the device work it could hide (0.87 ms/layer
  of a 4.48 ms read window, §2.1's own sum), while the read window is 1.42-1.46x what a
  cold replay of the identical reads achieves. The next discriminator is the per-read
  service time above: 4.65 against 2.62 ms at equal depth on 3.8, and in agreement on
  3.6. What differs between the installs inside the read is the open question.
- **Lesson:** a replay harness is only as good as its cache state, and a warm one
  flatters itself, not the engine. Quote a replay number with the cache state it was
  measured in, and control it by eviction when the volume will not honour `F_NOCACHE`
  and `purge` is unavailable -- three of the numbers this section used to rest on were
  warm-pass readings of a 4.3 GiB working set on a 16 GB box.

### IO-19: The read window is a per-layer round-trip floor, and caching more of it changes nothing

**IO-18 left a per-read service time with no mechanism under it.** The read window is
40-48 strictly serialized per-layer batches, so the simplest hypothesis that could close
§2.1's read side is that the drive charges per **command** rather than per **byte** -- in
which case cutting the number of reads should leave the window where it is. Three
measurements, then the two predictor families that floor implies.

**1. A ring-free control, to establish that read count is visible at all.** Served from
RAM rather than from the drive (`read_batch` at 22-34 GB/s, so this says nothing about
NAND), a batch's span **does** scale with the reads in it: **0.078 / 0.111 / 0.171 ms for
1 / 2 / 3.33 reads**, in both orderings. The fixed per-batch cost is therefore ~0.05 ms,
and a flat span curve measured against the drive is the drive's flatness and not the
harness's or the barrier's. That control is what licenses reading the next two results as
NAND behaviour.

**2. The engine's own slot count, A/B'd.** Same prompt, same session,
`--expert-cache-slots` the only variable, three rounds, arms interleaved with the order
alternating by round (METH-15):

| slots | misses | MiB/step | `io_read_wall` ms/step | tok/s |
| --- | ---: | ---: | ---: | ---: |
| 16 | 4,179 | 227.5 | 53.10 / 43.22 / 45.85 | 7.417 / 9.324 / 9.003 |
| 32 | **2,934 (-29.8%)** | 159.7 | 50.50 / 45.00 / 51.04 | 8.157 / 9.259 / 8.070 |

**-29.8% reads, +3% wall time** (pooled means 47.39 -> 48.85 ms/step). The read reduction
reproduces exactly, round after round; the time change is noise around it. `io_conc` moves
only 3.14 -> 2.93, and `io_thread_wall` 165.32 -> 147.09 (-11%) -- real, and not the
window. **A 30% cut in transferred bytes and in miss count is worth 0 +/- 5% of the read
window.**

**3. The same, at fixed cache size, across four prompt domains.** 16 slots, one trace each:

| prompt | misses | `io_read_wall` ms/step |
| --- | ---: | ---: |
| a-coastal wetlands | 4,179 | **61.13** |
| b-debt covenants | 4,433 | 53.94 |
| c-protein folding | 4,659 | 59.81 |
| d-counterpoint | 4,829 | **52.57** |

The prompt with the **fewest** misses has the **highest** window and the one with the most
has the lowest -- read count does not order the window even in the direction its own sign
predicts.

**The confound, named and bounded rather than waved away.** The 16-vs-32 arm moves two
things: bytes read, and the destination ring those reads land in -- 48 x 32 x 2,768,896 B
= **2.11 GiB** at 32 slots against **1.05 GiB** at 16. IO-18 priced exactly that
destination shape in the engine's own condition at **+1.5 ms/step**, measured on 3.8,
whose ring is twice 3.6's, so on 3.6 it is an upper bound. A 30% cut in bytes, converting
at the cold rate, would be worth ~14 ms/step of a ~47 ms window. **1.5 ms cannot absorb
14 ms**, so the destination is not the explanation and the flatness belongs to the drive.

**What the floor is.** ~**1.2 ms per batch** on 3.6, flat across batch widths of 2.37 to
3.14 reads: **40 batches x ~1.2 ms = 47.4 ms** of a 47-52 ms window. It is per-layer
round-trip latency, not bytes and not read count -- and it says the lever is **fewer round
trips**, which needs a *prediction*, not a cache.

**Which is what the two prefetch families are, so both were priced.** First, a fact about
the instrument that had been read the wrong way: `FQ_EXPERT_TRACE` is the **miss list**
(`RealForwardRunner.swift:830` -- `[layer, missCount, expert...]`), conditioned on the
16-slot LRU ring's policy, so an expert the ring served never appears. The ring already
takes **48.8%** of 3.8's routed experts (7,263 hits / 7,617 misses of 14,880) and **57.9%**
of 3.6's (5,741 / 4,179 of 9,920), matching the engine's own counters. What is left after
the ring is small and unpredictable:

| family | how it would work | coverage of the remaining reads |
| --- | --- | --- |
| temporal, layer L -> L+1 (the S7 lever) | predicts the next layer's set from this one's | **1.00%** on 3.8, **1.84%** on 3.6 |
| static per-layer hot set (the `8c9b496` family) | pins the most-used experts, no prediction | see the matrix below |

The temporal result is a **negative** rather than a small number: expected overlap under
pure independence is 5.12^2/512 = 0.0512 experts per batch against a measured 0.0497 --
the residual read stream after the ring is statistically independent across layers.

**The static family, leave-one-domain-out.** Four prompts in one register, so domain is the
only variable. The traces are a *deterministic* function of the prompt (byte-identical
across runs -- `cmp` clean against the committed captures), so the in-sample/out-of-sample
gap is pure domain shift with zero run-to-run noise to subtract. Table built from three
prompts, scored on the held-out fourth:

| model | top-8 | top-16 | **top-32** | top-64 |
| --- | ---: | ---: | ---: | ---: |
| 3.6, out-of-sample | 15.3% | 26.4% | **42.3%** | 62.6% |
| 3.6, in-sample ceiling | 24.4% | 42.1% | 67.6% | 93.6% |
| 3.8, out-of-sample | 11.1% | 19.1% | **31.5%** | 48.3% |
| 3.8, in-sample ceiling | -- | -- | 52.4% | -- |

Fold spread is 5.5 pp at top-32 on 3.6 and 3.6 pp on 3.8, so the margin is not one odd
prompt. Coverage is sublinear in pinned size (32.7%/GiB at top-4 falling to 20.0%/GiB at
top-32). **The archived engine's independently measured 39.5% lands within 3 points of
3.6's 42.3%** -- two engines, two corpora, one answer for this family.

**The decision rule was satisfied and then overridden by its own premise.** The rule as
set: >=40% out-of-sample at top-32 build the pinning; <25% fall through to the CB1-overlap
probe. 3.6 reads 42.3% and clears the bar; 3.8 reads 31.5%, between them. But **coverage
does not convert** -- that is measurement 2 above, in the engine's own condition with the
engine's own slot knob: removing 30% of the reads removed none of the time. So the
satisfied bar does not license the build.

**Disposition: close both families, and close the CB1-overlap branch of the rule as well.**
The pinning path would pay **2.11 GiB resident** (top-32 alongside the ring, not replacing
it) plus a shipped table for approximately no speedup, and on 3.8 it is memory-blocked
outright -- top-32 alongside the ring is **5.94 GiB** on a box whose memguard kills 3.8
runs at 1.98 GiB compressed. The rule's own fallback does not survive either: CB1 overlap
could hide at most the 0.87 ms/layer of device work inside a 4.48 ms read window that is
itself 1.42-1.46x a cold replay, so its ceiling is inside the unexplained deficit, not
beside it. **Nothing on the read side is left to build.** What would reopen it is a
mechanism for **fewer round trips** (a deeper per-layer read that the router's readback
does not currently permit) or a drive whose per-command cost is lower than this USB4
bridge's -- neither of which is a cache.

- **Lesson:** two families can have *measurably different* coverage and identical value,
  if the resource being saved is not the resource being spent. Price the saving in the
  currency of the bottleneck *before* setting a coverage bar, or the bar will be met by
  something that does not help.
- **Lesson:** the archived negative (`8c9b496`, prefill, on the C engine) and the S7
  negative (GGUF decode) were both about *other paths*. This is the first verdict on the
  Swift Qwen decode read path, and it agrees with them for a different reason -- not that
  prefetch is subtle, but that these reads are round trips.

### IO-20: The in-engine gap is a throughput deficit, not a latency shape, and the fast reads are not the page cache

IO-18 left the 3.8 engine-side access gap as a **mean**: `io_thread_wall / misses` is 4.65 ms
per read in the engine against 2.62 ms for an offline replay of the identical offsets at the
same depth. A mean cannot say whether every read is slower or a few of them are, so the
question "where does the 4.65 ms sit inside the engine" was not answerable from any counter
in the line. It is now, and the answer moved the question twice.

**The instrument.** A 34-bucket log2 histogram of per-read thread time (2^0 .. 2^33 ns,
clamped at the top so a hang cannot be silently dropped), filled from the `marks` walk that
`PreadExpertStreamer.executeExpertCachePlan` already does to compute `lastReadThreadNanos` --
so the read path pays nothing for it, and `io_pread == io_thread_wall` as before since
`stageReads` and `readSplit` are both at their defaults. It prints as
`io_read_latency_ms=p50=.. p90=.. p99=.. fast=..% n=..` on the `--counters` line, at bucket
**lower** edges: a bucket spans a factor of two, so its lower edge is the only part that is a
measurement rather than an interpolation. Read every percentile as "at least this slow".

The same histogram was added to `tools/read-sweep/replay_dest.py` behind `--hist`, timing the
`preadv` on the worker thread -- the same quantity the engine marks -- with the engine's own
bucket edges and reporting convention. Without it the comparison is mean-to-mean, and IO-17's
1.20-1.33x / 1.31-1.43x ratios could not be told apart from a shape difference.

**Shape: it is a shift, not a tail.** Four engine runs each, `--max-context 2048 --max-new 32
--temperature 0 --expert-cache-slots 16`, the short-explanation prompt, 31 decode steps:

| | misses/step | mean ms | p50 | p90 | p99 | fast | conc |
|---|---|---|---|---|---|---|---|
| 3.6 run 1 | 134.8 | 1.19 | 0.52 | 2.10 | 4.19 | -- | 3.25 |
| 3.6 run 2 | 134.8 | 1.22 | 0.52 | 2.10 | 4.19 | -- | 3.27 |
| 3.6 run 3 | 134.8 | -- | 0.52 | 2.10 | 4.19 | 17.0% | -- |
| 3.6 run 4 | 134.8 | 0.99 | **0.26** | 2.10 | 4.19 | 16.3% | 3.22 |
| 3.8 run 1 | 245.7 | 4.15 | 2.10 | 4.19 | 8.39 | -- | 5.22 |
| 3.8 run 2 | 245.7 | 4.53 | **2.10** | 8.39 | 8.39 | -- | 5.31 |
| 3.8 run 3 | 245.7 | -- | 2.10 | 4.19 | 8.39 | 2.2% | -- |
| 3.8 run 4 | 245.7 | 4.33 | **2.10** | 8.39 | 8.39 | 2.2% | 5.37 |

Every percentile moves about one bucket between the installs, and p50/p90/p99 move *together*
-- the signature of a uniform shift. The 3.7-4.0x gap in the mean is therefore not a tail of
slow reads on top of a normal population. 3.8's p50 is 2.10 ms in all four runs; **3.6's is
not a stable number** -- it read 0.52 in three runs and 0.26 in the fourth, because the
fast/slow boundary sits on its median -- so quote the contrast, not 3.6's p50.

**Correction to my own first reading of this table.** I first decomposed the gap as
`mean = conc x (size / aggregate rate)` using the engine's 4.65 ms thread mean against a
*span-derived* mean, and concluded the engine's tail was 2x its median while the replay's was
below 1 -- an engine-vs-harness shape difference. That was a quantity error: `ss/n` is wall
per read and `io_thread_wall/misses` is thread time per read, and they differ by the
concurrency factor. Comparing thread-time mean to thread-time p50 in all four cells gives
**mean/p50 = 1.7-2.3 everywhere** (engine 2.30/1.98, replay 1.75-2.08/1.51-1.85). There is no
engine-vs-harness shape difference. The replay harness now prints the thread-time mean and the
concurrency it implies, and no longer prints the mixed ratio.

**`fast`, and the control that killed its first explanation.** `fast=` is the share of reads
under 0.262 ms. No cold read can land there: this volume's single-queue cold rate is 3.4-3.6
GB/s, so a cold 1.69 MiB expert read is ~0.48 ms and a cold 2.64 MiB one ~0.75 ms. It is
printed as a share rather than inferred from percentiles because a *half*-cached run puts p50
exactly on the boundary bucket -- the same shape a fully drive-served run produces.

The first reading of `fast` was "the page cache", and the record's own `purge-null` note
supported it. **A 17 GiB eviction control refutes it.** `replay_dest.py --evict-gib 17
--evict-from models/Qwen3.8-.../ple_shards` streams more than RAM from a directory neither
arm's trace reads, before every pass, and the fast population does not move:

| | engine fast | replay fast (4 rounds, evicted) | replay span/step |
|---|---|---|---|
| 3.6 | 16.3-17.0% | 38.5 / 38.8 / 38.8 / 38.7% | 57.39 / 57.63 / 57.43 / 57.53 |
| 3.8 | 2.2% | 11.8 / 11.4 / 11.7 / 10.7% | 136.87 / 136.86 / 143.38 / 143.53 |

It removes only the *wholly* cached round -- 3.6's 98.9%-fast, 11 ms/step round from IO-18
becomes 38.7% -- and leaves the rest stable to 0.3 points across four rounds. Whatever serves
those reads is not pushed out by read pressure, and **the mechanism is unidentified**: a
device-side cache would be flushed by a 17 GiB scan on the same volume, and a host cache would
be evicted by it, and neither happened. What is established is only that it is a cache benefit
which **differs by install (38.7% vs 11.4%), by working-set-to-RAM ratio, and by process (the
engine sees 17.0% and 2.2% where the bare replay sees 38.7% and 11.4%)** -- so no two rates in
this record are comparable until it is accounted for.

**Cold-corrected rates, and what survives.** Taking the fast reads as costing 0.1 ms each and
inverting `mean = f*t_fast + (1-f)*t_slow` gives a cold per-read service time and, with the
measured concurrency, an all-cold rate (`n*t_slow/conc` per step, against `n*stride`):

| | t_slow | conc | all-cold rate | raw rate |
|---|---|---|---|---|
| 3.6 engine | 1.16 ms | 3.22 | **4.68 GB/s** | 5.47 |
| 3.6 replay | 1.45 ms | 2.18 | **2.54 GB/s** | 4.16 |
| 3.8 engine | 4.43 ms | 5.37 | **3.20 GB/s** | 3.27 |
| 3.8 replay | 2.05 ms | 3.26 | **4.20 GB/s** | 4.97 |

The correction is insensitive where it matters -- 3.8's engine arm is 2.2% fast, so `t_fast` is
nearly irrelevant there; doubling it moves 3.6's replay arm by 4%, the most sensitive cell.

The surviving statement is then a single one: **the drive serves 3.8's colder, larger-read
stream 1.65x faster than 3.6's (4.20 against 2.54), and the engine turns that into a 1.46x
deficit (3.20 against 4.68) -- a 2.4x swing.** That is IO-17's sign flip ("the engine flips
the sign of its own traces") with the cache population removed, and it is no longer expressible
as a per-read latency curiosity: it is a **throughput** deficit, in the engine, on 3.8.

**Eliminated, with the numbers that do it.** `fast` = 2.2% on the engine's 3.8 arm, so the page
cache is not it. Depth is not it twice over: IO-14 swept the engine's own `FINCHMOE_IO_READ_WAVE`
over a 2.9x range for a 6% window move at a pinned 2.87-2.97 GB/s, the offline replay is flat
over depth 3-10, and the engine here runs **deeper** than its own replay (conc 5.37 against
3.26) while still losing. ~~**GPU contention is not it**: per unit read-window time 3.6 carries
52.5 ms of GPU work against a 41.59 ms read wall (126%) where 3.8 carries 112.8 against 189.58
(60%), so the install that *matches or beats* its replay is the one with the more heavily
loaded GPU.~~ **Withdrawn -- [IO-21](#io-21) refutes it.** The duty-cycle argument is sound
about how *long* the GPU is busy and blind to what it does to the memory the drive is writing
into; adding a GPU to the replay at the engine's own dose costs the drive 12% on 3.8 and 8% on
3.6, and a ladder spanning 16x in delivered GPU bandwidth shows that price is **flat**. So what a
busy-time ratio cannot see is not a per-byte cost but a fixed one -- the elimination failed for
the same reason, whether the price is charged per byte or once per activation. PLE cannot be it
(IO-19's ordering argument). Destination shape was priced at +1.5 ms/step (IO-18) and destination kind eliminated
(IO-15).

**What this does not say.** The replay's arms are harness-limited -- 3.6's batches average 4.35
reads at pool width 3 and reach conc 2.18, 3.8's average 5.1 at width 5 and reach 3.26, so
neither saturates its width -- which *strengthens* the 3.8 claim (the engine is deeper and still
slower than a shallow replay) and *weakens* the 3.6 one (its 4.68 GB/s is a lower bound, so the
engine's apparent win there is partly the harness). The load-bearing comparison is 3.8, and it
does not depend on the limitation. Also unmeasured: whether `fast` is the drive's own cache,
the bridge's, or a compressor path -- `fast` is a share, not a mechanism, and it should not be
quoted as "cache hits".

**What is left.** One structural difference had never been tested with the GPU *running*: 3.8's
648.8 MiB/step is DMA'd into a 1.98 GiB ring of `makeBuffer(bytesNoCopy:)` shared allocations
**while the GPU reads expert weights back out of those same slots** (`gpu_routed` 38.8 ms/step,
against 3.6's 19.1 over 227.5 MiB/step into 1.05 GiB). IO-15 eliminated the destination *kind*;
if that ran with the GPU idle, "drive writing the page while the GPU reads it" is the cell still
open, and it is the only one left that is both install-correlated and inside the read. **That
cell was tested in [IO-21](#io-21) and it is positive**: +12.0% on the 3.8 replay's read window,
+8.0% on 3.6's, worth ~14% of the engine's gap as a lower bound. It is no longer open, and the
GPU-contention elimination above is withdrawn.

- **Lesson:** the mean was not hiding a tail, it was hiding a *population*. The histogram was
  built to answer "shift or tail" and it answered that, then immediately raised a second
  question the mean had also been hiding -- what fraction of these reads was never the drive's.
- **Lesson:** `fast` was interpreted as the page cache because the interpretation was
  plausible and the memory note agreed. The 17 GiB eviction cost ten minutes and refuted it. A
  threshold that classifies reads needs a control that moves it, or it is a story about a
  number rather than a measurement of one.

<a id="io-21"></a>
### IO-21: Adding the GPU to the replay costs the drive 12%, and the per-read median is not a stable statistic

- **Hypothesis:** IO-20 left exactly one cell open -- the drive DMA-ing bytes into a slot ring
  while the GPU reads expert weights back out of it. 3.8 does 648.8 MiB/step into a 1.98 GiB
  ring of `makeBuffer(bytesNoCopy:)` with `gpu_routed` at 38.8 ms/step; 3.6 does 227.5 MiB into
  1.05 GiB at 19.1. If that interaction is what costs the engine throughput, it is the only
  remaining difference that is both install-correlated and inside the read.
- **Why the engine cannot be asked.** `FQ_EXPERT_TRACE` is capture-only -- there is no playback,
  so nop'ing the routed readback makes the model emit garbage at step 1, routing diverges, and
  the arm stops reading the trace it is being compared against. Worse, a degenerate loop warms
  the ring, so the read *population* changes rather than just its timing.
- **The inversion.** The replay already holds the engine's ring (`makeBuffer(bytesNoCopy:)` over
  the same `posix_memalign`) and issues the engine's own preads, and has no GPU at all. So
  instead of removing the GPU from the engine, add one to the replay: the trace is then fixed by
  construction and only the GPU varies. `tools/read-sweep/gpu_load.swift` reads a shared buffer
  in a burst/period loop; `gpu-contention.sh` interleaves loaded and unloaded arms, alternating
  which goes first.
- **The dose is bytes and burst length, not a rate.** 3.8's 16.7 GB/s is compute-bound -- 648.8
  MiB is all it has to read and it spends 38.8 ms reading it -- so a generator reading the same
  bytes flat out would hit the memory system at 95 GB/s over a 7 ms window, a dose no engine run
  produces. Matched: 648.8 MiB per 147 ms at 16.7 GB/s (a 42 ms window) for 3.8, 227.5 MiB per
  57 ms at 11.9 for 3.6.
- **Evidence.** Five rounds on 3.8, arms alternating in order, every round unambiguous:

  | round | order | alone span | loaded span |
  |---|---|---|---|
  | 1 | loaded first | 147.18 | 165.04 |
  | 2 | alone first | 148.24 | 165.11 |
  | 3 | loaded first | 146.64 | 165.17 |

  **+17.7 ms/step, +12.0%**, achieved rate **4.62 -> 4.12 GB/s**, per-read thread time 1.97 ->
  2.27 ms, `fast` ~12% -> ~10%. The loaded arm reproduced to 0.08% across three rounds while
  the unloaded arm reproduced to 1.1%, so the effect is an order of magnitude larger than the
  scatter. 3.6 under its own (smaller) dose separates 6/6 the same way: 43.7 -> 47.2 ms/step,
  **+8.0%**, 5.46 -> 5.05 GB/s. The alternation is what makes it causal: the state follows the
  arm, not the clock, in every round of both installs.
- **Specificity: one null holds, and the other had to be withdrawn.** `gpu_load --no-dispatch` --
  the Metal ring allocated, page-touched, and never read by the GPU -- produces the **full** fault
  storm (77,993 / 79,429 / 79,422 `minflt`) with **no slowdown at all** (149.93-150.30 ms/step,
  rate 4.53), so the minor-fault population belongs to the Metal *allocation* and the slowdown to
  the GPU *reading*. That one stands.
  **The CPU half does not.** `mem_load.c` was handed `gpu_load`'s flag set, which contains
  `--gbps` -- a flag it does not have -- so it exited 2 on its **first argument** and every
  "loaded" arm of it ran with nothing in it. Span 147-150, `minflt` 15-34, six arms: a null that
  was never a measurement, and the fault count is what gives it away (a loader that holds and
  reads 1.98 GiB cannot fault like a process running alone). Re-run through a harness that builds
  the flag set per program -- and now refuses to report an arm whose loader died at startup --
  the matched doses come back at **+7.2 ms/step at 3.40 GB/s and +9.5 at 13.57**, against the
  GPU's **+15.4 and +15.9** at the same delivered rates. So CPU traffic is **not null**, at
  roughly half the GPU's price.
  It is not a *matched* control either, which is why this re-opens the question rather than
  answering it: `mem_load` burns four CPU cores where `gpu_load` runs at 1.1%, and it ran while
  the box was in a slower state (alone arms 184-194 ms/step against 167-173 in the GPU runs). What
  survives is the allocation/reading split, which the dead loader never touched; what does not
  survive is "DRAM bandwidth is retired".
- **How much of the engine this explains.** Replay alone 1.97 ms per read -> replay + GPU dose
  2.27 -> engine 4.13 (`io_thread_wall/misses` = 1013.62/245.7 this session). The cell accounts
  for **0.30 of the 2.16 ms gap, ~14%**, and that is a *lower bound*: the generator models only
  the routed ring reads, while the engine's GPU also runs `gpu_cb1` (75.2 ms/step of attention
  and GDN) over the same memory system.
- **The dose does not scale, so the price is per-activation rather than per-byte.** A five-rung
  ladder at a fixed 147 ms period with one burst per replay pass, spanning **16x in delivered GPU
  read bandwidth** -- 0.86, 1.72, 3.46, 6.90 and 13.80 GB/s, each rung calibrated by its own
  standalone run, five order-alternated rounds each -- moves the penalty by **+15.59 / +15.28 /
  +15.40 / +15.60 / +15.93 ms/step** (+9.0% to +9.5%), with per-read `thread_mean` going 2.33-2.44
  -> 2.58-2.70 at every rung. A per-byte mechanism would have carried the 0.86 GB/s rung's +15.6
  to roughly **+250 ms** at 13.80; the measured drift across the whole range is **+0.34 ms, 2% of
  the effect**. The engine's unmodelled `gpu_cb1` traffic therefore buys nothing, and **~14% is a
  ceiling, not a floor**.
  What the ladder does *not* settle is the mechanism, because two different fixed costs predict the
  same flatness: a per-activation price for the GPU read engine being active at all, and a
  per-DMA-write price on pages the GPU has touched (the drive writes into slots the GPU reads back,
  so any GPU access could be what marks them). Separating those needs the GPU reading a buffer the
  drive is *not* writing into -- a different experiment, not a bigger dose.
  Two harness notes, since the earlier 2x attempt is superseded: a burst that does not divide the
  ring leaves a sliver burst at the wrap (`min(burstUint4, nUint4 - offset)`), so the *delivered*
  dose is not the commanded one and must be calibrated -- it is why 1297.6 and 2027.5 MiB/period
  delivered the same 7.05 GB/s. And the first rung of the ladder was discarded and re-run: with
  `alone` drifting 180.6 -> 168.6 across its five rounds it read +18.6, which is warm-up, not dose.
- **Final disposition:** Recorded, and it **refutes IO-20's GPU-contention elimination**. That
  elimination argued from duty cycle -- 3.6 carries 126% of GPU work per unit read wall against
  3.8's 60%, and 3.6 is the install that matches its replay -- which is sound about *how long*
  the GPU is busy and silent about *what it is doing to the memory the drive is writing into*.
  A duty-cycle ratio cannot see a coherence or fabric cost per byte DMA'd into a GPU-shared
  page, which is the mechanism at issue. Tools: `tools/read-sweep/gpu_load.swift`,
  `mem_load.c`, `gpu-contention.sh`.
- **What is left.** ~86% of the gap, and the GPU cell is now bounded from *above* as well as below:
  it cannot grow with more GPU traffic, so ~14% is the whole of it. The open question this
  experiment leaves is no longer the dose but the specificity one named above -- whether the price
  is the GPU's or any sufficiently heavy traffic's -- and it is open because the control that would
  have answered it did not run.
- **An unplanned finding, and it changes how IO-20's numbers should be read.** The replay's per-read
  distribution is **bimodal** -- a bulk at 0.13-0.52 ms and another at 2.1-8.4 ms, with the median
  sitting in the empty valley between them (bucket 19, [0.524, 1.049), holds 4.1%): `b17 13.5%,
  b18 25.6%, b19 4.1%, b20 12.6%, b21 34.7%, b22 9.4%`. A few points of mixture change therefore
  hops the reported p50 a full bucket -- 1.05 -> 2.10 -- while the mean moves 3%. IO-20's
  shift-to-p50 was the right instrument for "shift or tail" and remains so, but p50 on this
  volume is a knife-edge statistic and must never be quoted alone; `thread_mean` is the stable
  quantity, and `--hist-dump` (added here) is what shows the mixture. Both the p50 contrast and
  the mean are reported for every arm above for that reason.
- **Lesson:** a ratio can be arithmetically sound and still not speak to the mechanism. The
  duty-cycle argument was true, relevant-looking, and invisible to the thing being asked about.
  When an elimination reasons about *when* a resource is busy, check whether the hypothesis is
  about *what* it does while busy.
- **Lesson:** the control that mattered was not another arm of the same experiment but a second
  program that removes the suspected ingredient while keeping everything else -- and then a
  third that keeps the allocation and removes only the work. The first null would have left
  "Metal allocation" standing; only the pair separates them.

### IO-22: The drive's contention penalty is generic, not GPU-specific, and a third of it is thread pressure

- **Hypothesis:** IO-21 left the *specificity* question open. It had shown that adding a
  dose-matched GPU to the replay costs the drive 12%, but its CPU half could not answer whether
  that price is the GPU's or any heavy concurrent activity's: `mem_load` burned four CPU cores
  where `gpu_load` runs at ~1.1%, and the two halves were measured in different sessions whose
  alone arms differed by more than the effect (184-194 ms/step against 167-173).
- **The control, matched on both axes.** `tools/read-sweep/contention-control.sh` runs every arm
  **interleaved in one session with a rotating start**, so no arm is always measured against the
  same drive state, and no session state is unique to an arm. The CPU arm is dose-matched by
  construction — `--burst-mb` and `--period-ms` are the same flags the GPU arm uses, and only the
  rate flag differs, because `mem_load`'s rate is what the CPUs deliver rather than a throttle.
  Two thread counts separate "how many threads" from "how many bytes", and a fourth arm burns
  four cores with **no memory traffic at all** (`yes`) to separate traffic from the scheduler.
  Statistic is `thread_mean`, not p50: IO-21 found the per-read median hops a whole bucket on a
  few points of mixture.
- **Evidence.** Eight rounds of five arms, alone on the box, this install:

  | arm | span mean | vs alone | thread_mean | vs alone |
  |---|---|---|---|---|
  | alone | 145.67 | — | 1.94 | — |
  | gpu (648.8 MiB / 147 ms / 16.7 GB/s) | 164.10 | **+18.43** | 2.26 | **+0.31** |
  | cpu4 (same dose, 4 threads) | 159.45 | **+13.78** | 2.18 | +0.24 |
  | cpu1 (same dose, 1 thread) | 157.57 | +11.90 | 2.13 | +0.19 |
  | burn (4 cores, no traffic) | 151.53 | **+5.86** | 2.09 | +0.14 |

  The alone arm's span spread is 143.7-148.7, so within one session the drift that wrecked the
  cross-session comparison is 3.5% — smaller than every effect measured.
- **What it says.** The penalty is **not GPU-specific**: CPU memory traffic at the same dose
  costs **75%** of the GPU's price, and thread count is not the driver (cpu4 minus cpu1 is
  1.9 ms). A third of the CPU arm's cost survives with **no traffic at all**, so it is CPU and
  scheduler pressure rather than bytes moved, and only ~4.6 ms separates the GPU from the
  matched CPU arm. That also explains IO-21's flat 16x ladder: the price follows the *presence*
  of concurrent activity, not its volume.
- **Disposition.** Specificity is closed in the direction of generic contention. In the units
  IO-21 used (per-read thread time: replay alone 1.97 -> replay+GPU 2.27 -> engine 4.13, a
  2.16 ms gap), the GPU cell is 0.30 ms and the CPU-traffic cell 0.24, with 0.14 of that
  reachable by thread pressure alone — so the contention family is worth **up to a quarter to a
  third** of the engine's gap, and whether those cells *stack* in the engine is untested (they
  are alternatives here, not additive). The majority remains unexplained, but the reframing
  matters: the drive is slower when the box is **busy**, not when the GPU specifically reads.
- **Lesson:** an unmatched control is not a weak measurement, it is no measurement — and the
  mismatch hid in two places at once (a flag set that killed one arm outright, and a session
  whose drive state differed from its comparison's). Rotating arms *within* a session is what
  makes the comparison causal; IO-21 did that between loaded and unloaded and still lost the
  specificity question to the session boundary.

[Experiment inventory](../EXPERIMENT_INVENTORY.md) |
[Optimization journey](../../OPTIMIZATION_JOURNEY.md) |
[Next: Decode, MoE, INT4, and router](02-decode-moe-int4-and-router.md)
