# FinchMoE Optimization Plan — Prefill, Decode, KV/State Turbo Quant, and Systemic Performance

Grounded in: `docs/SYSTEM_DESIGN.md`, `docs/OPTIMIZATION_JOURNEY.md`, `docs/QWEN36_PORT.md`, `docs/RUNTIME_CONTROLS.md`, `README.md`, and direct inspection of `Sources/FinchMoE/Runtime/{KVCache,Inference,Prefill,Generation}`, `Sources/FinchMoE/Kernels/{LinearAttn,MoE,Quant}`, `Sources/FinchMoE/Metal/LinearAttn/gdn.metal`, `Sources/FinchMoE/Infrastructure/Streaming/PreadExpertStreamer.swift`, `Sources/FinchMoE/Infrastructure/ModelIO/ModelTypes.swift`, `Sources/FinchMoE/Runtime/Configuration/RuntimeConfiguration.swift`.

**Hard constraint carried through every recommendation below:** the runtime's entire raison d'être is out-of-core streaming at ~1.1–1.2 GiB resident (`docs/SYSTEM_DESIGN.md` "Resource split", README "At a glance"). Any recommendation that grows resident memory is explicitly flagged with a `[MEMORY RISK]` tag and a bound on how much it may grow, and none of them propose caching the full expert pool or materializing the whole KV/state set beyond current bounds.

**Already tried and rejected — do not resubmit without new evidence:**
- Packed K4/V4 "TurboQuant" KV cache (`OPTIMIZATION_JOURNEY.md` "The packed KV cache failed two gates") — failed the quality gate on Gemma's FP16-comparison harness *and* grew larger than FP16 at long context because it applied per-token packing overhead to all 30 (Gemma) attention layers rather than a fixed-size state. This finding does **not** directly disqualify Qwen KV/state quantization (Qwen only has 10 full-attention KV layers, not 30, and the GDN state is fixed-size, not growing) but the same two gates (quality regression, and "does it actually shrink asymptotically") must be re-run before any similar attempt ships. See item 3.1 below for how this plan differs from the rejected design.
- `mmap`-based expert streaming (lost to explicit bounded `pread`).
- SIMD-cooperative MoE kernel (lost to persistent-workgroup MoE).
- Fine-grained per-expert-read dispatch overlap (lost to coarse shared/routed overlap).
- Monolithic post-attention/pre-FFN kernel fusion (regressed 2.756→1.811 tok/s).
- Cross-layer expert-selection prefetch (7% predictive accuracy, not viable).
- `F_RDADVISE`/read hints as a default (unstable, workload-dependent) — stays an opt-in experimental control (`RUNTIME_CONTROLS.md`).
- Routing-trace-derived expert file layout (helped one workload 3.6%, regressed another 16%).

---

## 1. Prefill speedup

### 1.1 [HIGH impact, LOW risk] Raise `allowedPrefillChunkTokens` ceiling above 128 and re-benchmark
- **Evidence**: `RuntimeConfiguration.swift:2` hard-caps chunk size at `[32, 64, 128]`. `OPTIMIZATION_JOURNEY.md` shows chunk 32→128 cut a 1,017-token prefill from 92.89s→52.35s (~1.77x) — a strongly monotonic curve that was never pushed past 128 for Qwen specifically (that data point predates the Qwen port; Qwen's chunk-scratch sizing in `PrefillChunkScratch.swift` and MoE tile grouping are different shapes: 256 experts vs 128, `moeIntermediateSize` 512).
- **Action**: measure 192/256-token chunks against the current 705–2,509-token Qwen prompts (README + the 2,509-token 4K soak in `QWEN36_PORT.md` item 4, which measured 21.3 tok/s pure chunked rate). `PrefillChunkScratch.swift` scratch is currently ~15.6 MiB at 128 tokens; watch the linear scratch growth (est. ~23–31 MiB at 192–256) against the resident budget.
- **Expected impact**: 10–25% prefill throughput on long prompts if the amortization curve from the 32→128 experiment continues past 128 (diminishing but plausibly still positive since GEMM/QMM setup cost still needs more rows to amortize at 256-expert scale).
- **Risk**: low — this is a config sweep, not new code; the runtime already supports variable chunk size. Regression risk is in scratch memory growth and MPP tile-size mismatches.
- **Validate**: re-run the prefill-only benchmark protocol in `RUNTIME_CONTROLS.md` ("Run an experiment") at each chunk size on both the 16GB and 24GiB machines; confirm output token-for-token identical to the 128-chunk baseline (prefill math must be exact, not reordered, per the "Correctness and safety invariants" in `SYSTEM_DESIGN.md`).

### 1.2 [DONE 2026-09-11] Default the trusted-install receipt in the CLI, the server and the Mac app
- **Original evidence**: `QWEN36_PORT.md` items 5 and 7 flagged this as an **open gap** — the CLI exposed `--verify trusted-install` (cuts fixed per-run cost from ~8s to <1s per README), but the Mac app's "verification default stays `full-sha256` (no UI setting)" and the server had no `--verify` flag at all, so every server process ate the full layer-SHA256 pass on first expert touch ("8.79s wall, of which ~8s is the first-use layer-SHA pass").
- **Shipped**: `ModelIntegrityPreference` (`.automatic` / `.fullSha256` / `.sizeCheckTrustedReceipt`) is resolved at load by one function that returns the resolved policy, the receipt, *and* the outcome together, so a fallback can never set one half without the other. All four call sites — CLI, server, decode service, Mac app — default to `.automatic`, each keeps an explicit override, and each reports what actually happened through the single `ModelIntegrityOutcome.logDescription`, so no two of them can describe the same load differently.
- **On the original risk note — "a security/robustness knob, not just perf, so it should stay opt-in/configurable, not silently forced"**: automatic mode never verifies *less* than `.fullSha256` would have. It skips only the hashing a **validated** receipt independently covers, and every failure — absent, unreadable, symlinked, wrong manifest, wrong size — falls back to hashing more. An absent receipt stays silent (the normal state of an install made without one); a present-but-unusable receipt warns and falls back. Explicit `trusted-install` keeps the strict semantics: no receipt is an error.
- **Measured** (release CLI, `models/Qwen3.8-Flash-Next-125B.finch`, 19-token prompt, `--max-new 24 --temperature 0`, 2026-09-11; the default row is the current source, debug CLI — prefill at this length is I/O-bound, not compute-bound):

  | `--verify`        | prefill  | peak memory |
  |-------------------|----------|-------------|
  | `full-sha256`     | 63.5 s   | 3.9 GB      |
  | `trusted-install` | 4.6 s    | 2.1 GB      |
  | default (`auto`)  | 4.7 s    | 2.6 GB      |

  All three produced the same text and the same 9 output tokens. On a warm page cache the receipt path measures ~2.2 s; the spread across runs is the file cache, not the policy. **The saving is the hash pass only** — the eager `manifest.json` + `model_weights.bin` + `packed_experts/layout.json` hash runs in *both* modes and is not part of it.
- **Validate**: done — `ModelLoaderTests+IntegrityPreference.swift` discriminates the two paths by flipping a byte in a layer file (a size-preserving change only SHA can catch, so a suppressed hash and a wrong stored policy each fail a different assertion); the CLI smoke tests cover the default end-to-end plus the stderr warning for an unusable receipt; the server logs `model integrity ...` at boot and carries `integrity=...` on the ready line; the app persists the choice and shows the resolved outcome in its diagnostics pane.

### 1.3 [MEDIUM impact, MEDIUM risk] Parallelize expert prefetch across prefill tiles further
- **Evidence**: `SYSTEM_DESIGN.md` "Prefill" section: the runtime already "may fetch the next tile while GPU work for the current tile remains queued, with both tiles fitting in the 16-slot cache" and streams "in tiles of at most eight." `docs/OPTIMIZATION_JOURNEY.md` shows fine-grained overlap failed for decode (regressed 4.799→4.648 tok/s) specifically because per-read launches broke synchronization — but that experiment was against single-token decode granularity, not the larger multi-row prefill tiles where read latency can be hidden behind a bigger GEMM.
- **Action**: audit `PrefillRoutedTileScheduler.swift` (67 lines — small, worth a full read before changing) to confirm tile depth is tunable, and try issuing the *read* for tile N+2 while tile N computes and tile N+1's read is in flight (currently 2-deep per doc; test 3-deep bounded by slot count 16 ÷ 8-per-tile = 2 tiles max resident, so this requires either more slots (`[MEMORY RISK]`, bounded by `allowedExpertCacheSlots` up to 32) or smaller tiles with more overlap depth at the same slot budget).
- **Expected impact**: 5–15% on prefill I/O-bound phases (this is prefill, so GEMM tends to dominate over I/O already per the journey doc — "not every strong isolated result still mattered to the whole prefill" is a real risk here).
- **Risk**: medium — the decode-side lesson (finer overlap can regress) may generalize; must be measured end-to-end, not on the isolated I/O phase.
- **Validate**: full prefill benchmark (README long-prompt protocol) plus output byte-identity check (no reordering of floating point should occur here — this is purely a scheduling change, not a math change).

### 1.4 [LOW impact, LOW risk] Confirm 256-expert MoE prefill batching didn't inherit the Gemma diminishing-returns ceiling
- **Evidence**: `OPTIMIZATION_JOURNEY.md`: "Batched routed MoE reduced its kernel time by about 31%. End-to-end prefill improved by only about 2%" (Gemma, 128 experts). Qwen doubles the expert count per layer (256) and runs a router+MoE tail on **all 40 layers** vs Gemma's routed-MoE-on-30-layers-of-30 — so the routed-MoE fraction of prefill time is structurally larger for Qwen. Re-profile before assuming the old "kernel got faster, e2e barely moved" conclusion still holds.
- **Action**: profile a representative Qwen prefill chunk with Instruments/Metal System Trace to get the current per-phase breakdown (attention vs router vs routed-MoE vs shared-expert vs epilogue) — this number isn't in the docs for Qwen yet and should gate whether further MoE-kernel-only optimization (1.3, act-tile reuse) is worth doing at all, per the journey doc's core lesson ("profile the whole token step first").
- **Expected impact**: N/A (this is a measurement task, prerequisite to prioritizing 1.1/1.3/2.2 correctly).
- **Risk**: none (profiling only).
- **Validate**: N/A — this produces the baseline the other prefill items should be judged against.

---

## 2. Decode speedup

### 2.1 [HIGH impact, MEDIUM risk] Widen or restructure the expert cache for Qwen's 256-expert/40-layer shape
- **Evidence**: `ModelTypes.swift:161-162` — Qwen has `numExperts: 256, topKExperts: 8` on **40** layers (vs Gemma's 128 experts on 30 layers). `RuntimeConfiguration.swift` still defaults `expertCacheSlots: 16` — i.e., 16/256 = 6.25% of the expert pool is cacheable per layer, versus Gemma's 16/128 = 12.5%. `PreadExpertStreamer.swift`'s LFU logic (`shouldEvictSlot`) is architecture-agnostic and correctly implemented, but the *capacity* was tuned against Gemma's cache-hit-rate curves in `OPTIMIZATION_JOURNEY.md` ("cut repeated expert I/O from about 166 to 88 ms/token" at 16 slots on 128 experts) — those numbers have never been re-measured at 256 experts. Decode I/O is the single largest cost per the `cb1`/`io`/`cb2` breakdown in `SYSTEM_DESIGN.md`.
- **Action**:
  1. Instrument and measure Qwen decode-time expert cache hit rate at slot counts 16/24/32 (`allowedExpertCacheSlots` already supports this — pure config sweep, `RUNTIME_CONTROLS.md` "Expert-cache slots" control already exists).
  2. `[MEMORY RISK]`: going from 16→32 slots roughly doubles routed-expert-slot resident memory per opened layer. Using the Gemma resident-memory table's math (`SYSTEM_DESIGN.md` "Resource split": 16 slots × ~3.36 MB/slot × 30 layers ≈ 1.50 GiB reserved capacity, though pages fill lazily), for Qwen's likely-smaller per-expert blob (moeIntermediateSize 512 vs Gemma's larger F) but 40 layers instead of 30, re-derive the actual per-slot byte size from `packed_experts/layout.json` before committing — this must be checked against the stated ~1.1–1.2 GiB peak-resident budget. If the hit-rate curve is flat past 16-24 slots, don't take the memory hit.
  3. Consider a **per-layer-tier** cache policy instead of uniform slots: e.g. give the 10 full-attention layers (which run alongside a KV cache with cost already paid) fewer expert slots and the GDN layers (which are cheaper elsewhere) more, if trace data shows aggregate expert reuse differs by layer kind — but only after the flat cross-layer-transfer result (7% predictability) is re-checked at 256-expert scale, since it may behave differently.
- **Expected impact**: this is speculative until measured, but decode I/O dominated the biggest historical wins (LFU alone: 72.6→64.8 ms/token in the journey doc) — a hit-rate deficit from halved cache coverage could plausibly be costing 10-30% of current Qwen decode throughput. This is the single highest-value *unmeasured* question in the codebase today.
- **Risk**: medium — memory growth must be bounded and verified; LFU/LRU policy itself needs no change, only capacity.
- **Validate**: decode tok/s at fixed prompt/response length across slot counts 16/24/32, RSS via the Mac app's "Peak memory" HUD metric or `sysctl`, and confirm no quality change (expert selection is unaffected by cache size — this only changes I/O cost, not values — so output should be byte-identical across slot counts; use that as a correctness check, not just a benchmark).

- **Measured 2026-09-11 (4.1's counters), Qwen 3.8, 54-token prompt, 64 tokens, `--temperature 0`:**

  | slots | hit rate | misses/step | `io` MB/step | tok/s (interleaved, warm) |
  | ---: | ---: | ---: | ---: | ---: |
  | 16 | 39.5% | 290.3 | 766.6 | 2.738 |
  | 24 | 46.8% | 255.3 | 674.1 | — |
  | 32 | 52.4% | 228.5 | 603.4 | 2.760 |

  For reference, the 3.6 install at 32 slots hits **68.2%** (top-k 8 on 40 layers) and decodes at 8.72 tok/s.

  **The hit-rate curve is not flat — and it does not matter.** Widening 16→32 removes 21% of read bytes and 21% of reads per step for **+0.8% throughput**, which is below the run-to-run spread (individual 64-token runs ranged 2.39-2.95 tok/s; of three interleaved rounds the two warm ones gave 2.707 and 2.768 at 16 slots against 2.744 and 2.775 at 32, means 2.738 and 2.760). The reason is visible in the same counters: `io` is *awaited* read time on a path whose misses are issued in parallel, so removing a fifth of them does not shorten the critical path. This is the shape of METH-07 — a mechanism count that is not an outcome.

  **The memory cost is real, though.** 32 slots is 32 x 2,768,896 B = 88.6 MB per layer against 16 slots' 44.3 MB, and two independent runs put the observed cost at ~15 points of free memory and ~2.5 GB more compressed (16 slots: 36-37% low-water, 2.2-2.9 GB; 32 slots: 22% low-water, 4.8-5.6 GB) for no throughput.

  **Recommendation: keep 16 slots.** Action 2's conditional ("if the hit-rate curve is flat past 16-24 slots, don't take the memory hit") is not met literally — the curve is not flat — but the *decision* it was guarding is the same, and is now made on throughput rather than on hit rate. Action 3 (per-layer-tier policy) has no support here: the sweep shows cache capacity is not a throughput lever on this workload at all, so repartitioning it cannot be either. What would change this is a demonstration that the miss path is byte-serial rather than latency-parallel — e.g. a workload with far more misses per step, or a far slower disk.

  **Correctness check passed**: 16, 24 and 32 slots produced byte-identical generated text (995 bytes each, identical payload), confirming cache capacity changes I/O cost only.

  **Re-measured 2026-09-11 (GPU-split cycle) — the flat reading above does not survive a wider range, and it was flat for the wrong reason.** Six runs, Qwen 3.6 35B, one fixed 54-token prompt, `--max-new 32 --max-context 2048 --temperature 0`, only `--expert-cache-slots` varying. The sweep was run in both orders, because a single ordering cannot separate a slot effect from session drift:

  | slots | bytes/step | io ms/step, order 16-24-32 | io ms/step, order 32-24-16 | tok/s fwd | tok/s rev |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 16 | 268.2 MiB | **45.21** | **49.61** | **9.309** | **8.812** |
  | 24 | 231.2 MiB | 49.07 | 52.59 | 8.876 | 8.508 |
  | 32 | 200.0 MiB | **51.92** | **53.81** | **8.563** | **8.300** |

  Every run in the reverse sweep is 2-4 ms/step slower than its forward counterpart, which is honest session drift and has to be subtracted before anything else is claimed. What is left after that: **io time rises with slots in both orderings, while bytes read falls.** Going 32 -> 16 slots cuts io time 12.9% in the forward ordering and 7.8% in the reverse, for 8.7% and 6.2% more throughput respectively — while reading 34% *more* bytes. The decisive row is the reverse sweep's last run: it was both the final run of the session *and* the fastest, so the slot effect is not the drift — drift pushes the other way.

  A controlled pair on the 3.8 install says the same thing with the confound removed (both runs `--max-context 2048 --max-new 32`, so only slots differ): 16 slots, 773.7 MiB/step in 227.36 ms; 24 slots, 683.6 MiB/step in 224.89 ms. **11.6% fewer bytes bought 1.1% less time.**

  **This refutes the explanation the section above settled on, and the section asked for exactly this test.** "The misses are issued in parallel, so removing a fifth of them does not shorten the critical path" predicts *flat* io in miss count. Measured, io is not flat — it moves the opposite way to the bytes. A latency-parallel model with a fixed per-batch cost has io independent of how many misses are in the batch; this has io *rising* as the batch shrinks. The parsimonious reading is that the miss count is also the **queue depth**: more outstanding `pread`s per layer capture more of the drive per unit of latency, so reading *more* bytes costs *less* time. That is a queue-depth-limited read path, not a byte-limited one.

  **It also appeared to reopen the read side, which three earlier cycles had closed.** The standing conclusion was that decode reads run at 2.1-3.2 GB/s against a ~2.1-2.3 GB/s cold-stripe ceiling, therefore the drive is saturated and only fewer bytes could help. That inference came from dividing bytes by an `io` figure that is mostly latency. At 16 slots this sweep reaches 268.2 MiB in 45.21 ms — 6.2 GB/s apparent, with the true transfer rate higher still, because the window contains fixed cost that the division wrongly attributes to transfer. (The reopening was itself tested and does not survive; see the drive probes below. The refutation of the *old* ceiling reading is what stands from this paragraph.)

  **What this changes, on the interim reading:** item 2.1's recommendation (keep 16 slots) stands — 16 is the minimum legal count for both installs, so it is also the best measured one, and nothing here says to go lower. What does *not* stand is the reason, and with it the claim that capacity is the only axis. If outstanding-request depth is the lever, the experiments that matter are about **how many reads are in flight at once** — issuing layer N's preads from a wider pool, or overlapping layer N+1's shared-expert work — not about cache size. That is a different action list from the one this section was written to evaluate, and it should be re-scoped before item 2.1 is considered closed.

  **That re-scoping was then tested against the drive directly, and it does not hold. The queue is not the lever either.** Two offline probes on the real `layer_00.bin` of the 3.6 install (452,984,832 B, 256 experts, stride 1,769,472), pseudo-random offsets so readahead cannot flatter a level, 256 MiB per cell. First, delivered bandwidth against outstanding-request depth, `F_NOCACHE` set:

  | depth | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | MiB/s | 2564.7 | 4595.8 | 5398.2 | 5404.2 | 5088.0 | 4661.5 | 4046.1 |
  | GB/s | 2.69 | 4.82 | 5.66 | 5.67 | 5.34 | 4.89 | 4.24 |

  Concurrency is worth a great deal at the bottom — depth 1 to depth 2 nearly doubles it, 1 to 4 more than doubles — and then it is over. **The drive tops out at depth 4-8, about 5.7 GB/s, and gets worse past that.** Decode already runs at roughly 3-4 misses per layer on 3.6 and 5-6 on 3.8, so it is at or just past the knee. A wider pool or a prefetch would be pushing on the flat-to-declining part of that curve.

  Second probe, same file, same volume, depth held at 4, varying repetition and page cache:

  | offsets | page cache | MiB/s | GB/s |
  | --- | --- | ---: | ---: |
  | diverse | bypassed | 6776.4 | 7.11 |
  | diverse | allowed | 9325.1 | 9.78 |
  | **repeated** (24-expert pool) | bypassed | 13700.8 | 14.37 |
  | **repeated** (24-expert pool) | allowed | 26884.9 | 28.19 |

  Repetition is worth more than concurrency by a wide margin, and it is worth it through both caches: even with `F_NOCACHE` set — so the OS buffer cache is out of it — re-reading a 24-expert pool runs at 14.4 GB/s, more than twice the diverse rate, because the SSD's own cache holds it. With the buffer cache allowed, it is 28.2 GB/s.

  **So the slot inversion has two mechanisms, and neither is byte count.** A smaller expert cache (a) re-reads a smaller, more repetitive set, which both the buffer cache and the SSD's controller cache absorb — the engine opens its layer files *without* `F_NOCACHE`, so this is fully in play — and (b) concentrates its misses into more per-layer parallelism, whereas a large cache leaves many layers with one or two misses, and depth 1-2 is the steep part of the curve above. Widening the cache moves the workload toward the diverse and shallow regime on both counts. **32 slots reading 34% fewer bytes than 16 and taking longer is exactly what those two mechanisms predict.**

  **Recommendation, revised: do not build the prefetch or the widened pool.** This cycle proposed re-scoping item 2.1 around in-flight depth; the drive measurement refutes it, and it is recorded here rather than quietly dropped. Predicted experts are by construction either already resident — free and pointless to prefetch — or genuinely cold. For the cold ones the prefetch is speculative: an imperfect predictor reads a *superset* of the demand set, so the byte count rises while the time per byte improves only in the sense that the bytes are asked for earlier. This paragraph first argued that the wider set settled it, on the strength of the 2x repetition advantage above. That is a bandwidth-only argument and it is not sufficient — the win a prefetch could actually claim is hidden *latency*, not bytes. It is settled instead by the block below, which measures how much latency there is to hide. The current operating point is a local optimum in which two caches absorb a deliberately repetitive read set, and every axis that looked like a lever is already at its knee.

  **Caveats, stated plainly.** The two probes disagree on the absolute depth-4 figure (5.66 against 7.11 GB/s), and probe 1 ran its levels in increasing depth order on a warming drive, so the absolute numbers are soft — the *shape* is what both agree on. `F_NOCACHE` on APFS is a hint, not a guarantee; probe 2's repeated/bypassed cell at 14.4 GB/s exceeds probe 1's peak, which is the drive's own cache showing through, not 14 GB/s of NAND. And n is one run per cell: this is enough to rule a direction out and not enough to publish a curve.

  **The read window and the device work are serial, and that is now the binding constraint.** Everything above prices the read side on its own terms. It can also be checked against the device side for free, because every term is already on the counters line and the device term is *complete* rather than a sample: `totalGpuRoutedNanos` accumulates the routed, shared and phase-1-hit buffers as they drain (`RealForwardRunner.swift:2775-2780`), and `gpu_samples` covers 5821 of 5883 buffers — the two per forward it misses are the embed and head syncs. So `gpu_cb1 + gpu_routed` is every GPU nanosecond a decode layer spends. Summing that against `io`, `head_wall` and `ple_wall` across all seven captured runs:

  | run | install | io | gpu cb1 | gpu routed | head | PLE | sum | token | sum/token |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 16 slots, 54-tok prompt | 3.8 | 227.36 | 75.37 | 41.59 | 5.52 | 3.74 | 353.6 | 361.5 | **97.8%** |
  | 24 slots, 54-tok prompt | 3.8 | 224.89 | 73.71 | 38.16 | 5.42 | 3.78 | 346.0 | 355.7 | **97.2%** |
  | 32 slots, 1791-tok prompt | 3.6 | 72.05 | 42.79 | 19.53 | 4.21 | 0.00 | 138.6 | 149.1 | **93.0%** |
  | 16 slots, 54-tok prompt | 3.6 | 45.21 | 33.69 | 19.35 | 4.23 | 0.00 | 102.5 | 107.4 | **95.4%** |
  | 24 slots, 54-tok prompt | 3.6 | 49.07 | 34.38 | 19.07 | 4.25 | 0.00 | 106.8 | 112.7 | **94.8%** |
  | 32 slots, 54-tok prompt | 3.6 | 51.92 | 34.94 | 18.00 | 4.21 | 0.00 | 109.1 | 116.8 | **93.4%** |
  | 32 slots, 54-tok prompt | 3.6 | 49.32 | 33.82 | 18.06 | 4.13 | 0.00 | 105.3 | 113.2 | **93.1%** |

  `ms/step`, token from `1000 / tok/s` in the `[stop=` footer.

  The remainder is 2.2-7.0% and it has to hold the CPU encode (1.7-2.6 ms/step, ~1.5-2.3%), sampling, detokenization and any genuine overlap. **Overlap between reading and device execution is therefore bounded at a few percent of the token — on a design whose stated mechanism is that they overlap.** The mechanism exists and is documented in the code (`encodeRoutedTail`'s own comment has the shared expert running on an early-committed CB overlapping the pread, `:2794-2795`); it is undersized. Per layer on Qwen 3.8, the whole routed tail — routed plus shared plus phase-1-hit, *all* of it, overlapped or not — is 41.59/48 = **0.87 ms** of device time against a 227.36/48 = **4.74 ms** read window. Perfect overlap of all three tail buffers would fill 18% of it.

  **Why the order is forced, from the code.** `encodeRoutedTail` opens with a CPU readback of the router's top-k indices (`:2813-2819`), and the blocking `waitForCompletion(cb)` at `:5236` is what makes that readback safe. Layer N's router runs on the GPU, the CPU waits out the whole of `cb1`, the CPU reads the indices, and only then are the preads issued. Neither end can move: layer N+1 cannot be *encoded* before layer N's routed output exists, because that output is layer N+1's input, and no read can be issued before its own layer's router has chosen. This is a data dependency, not a scheduling artifact — which is why the read side is closed a third time, but now for a reason that names the structure instead of inferring a ceiling from a quotient.

  **What it does reopen, on the compute side.** The token is a *serial sum*, so device work pays 1:1 against it. GDN's 61.85 ms/step is 17.1% of a 3.8 token and is a legitimate target in a way the overlap model would have under-priced; item 3.2 (int8 KV) lands on the smaller stack — attention is 13.52 ms/step of a 361.5 ms token at 54 tokens, and 12.81 of 149.1 at 1791 tokens — so it stays a VRAM/context-headroom play rather than a throughput one.

  **And it identifies the largest unexplained quantity left.** 773.7 MiB in 227.36 ms is 3.57 GB/s, against a drive that delivers 5.4-5.7 GB/s at this depth on pseudo-random offsets and *more* on a repetitive set. At the probe's plateau the same bytes take 143-150 ms, so **~84 ms/step — 23% of the token — is fixed per-layer-batch cost rather than transfer.** One batch per layer gives ~1.8 ms of fixed cost per batch, which is issue, queue, first-byte and tail latency plus the CPU plan and the copy into the slot buffer. That is the next thing worth measuring, and it is measurable without touching the runtime: re-run the harness at depth 6 with the engine's *actual* offset sequence and a cold cache, and see whether it reproduces 4.74 ms or the ~3 ms the depth curve predicts.

  See [METH-13](experiments/summaries/09-validation-and-measurement-lessons.md#meth-13) for the method — summing two spans that are claimed to overlap bounds the overlap at the complement, and needs no new instrumentation.

  **That measurement was made, and it splits the ~1.8 ms/batch three ways — none of them the CPU.** The `io` window is now instrumented into its parts (`PreadExpertStreamer.executeExpertCachePlan`, surfaced as four new counters fields and a derived remainder). On the same 3.8 install at 16 slots:

  | field | ms/step | share |
  | --- | ---: | ---: |
  | `io_wall_ms/step` | 224.67 | 100% |
  | `io_read_wall_ms/step` (the `concurrentPerform`) | 224.07 | 99.7% |
  | `io_plan_cpu_ms/step` (expert selection and cache plan, outside the window) | 0.58 | 0.26% |
  | `io_handoff_wall_ms/step` (continuation hops, `streamersQueue.sync`, `ensureLayerOpened`) | 0.33 | 0.15% |
  | `io_dispatch_wall_ms/step` (submit → thread entry) | 0.23 | 0.10% |
  | `io_tail_wall_ms/step` (cache bookkeeping, view construction) | 0.04 | 0.02% |

  So of the ~1.8 ms/batch the runtime accounts for 0.025 ms/batch. **The three-level dispatch chain is not the cost** — `streamersQueue.sync` → `DispatchQueue.global` → `concurrentPerform`, run 48 times a token, is 0.27% of the window between them — and neither is the CPU plan. The copy into the slot buffer is not merely small but *absent*: `readFull` preads straight into `slotPointers[slot]`, which are `posix_memalign`ed pages already wrapped by `makeBuffer(bytesNoCopy:options:.storageModeShared)`, and `expertCachePlanBuffers` returns views over those same pages (`PreadExpertStreamer.swift:213-217, 344-351`). The remaining ~1.8 ms/batch is inside the read: issue, queue, first-byte and tail latency. This closes item 2.2's premise — see §2.2.

  **The requested replay was then run, and what it falsifies is the harness.** The block above asks for the engine's own offset sequence replayed offline at depth 6 with a cold cache. It was run — 9,083 preads captured through a new `FQ_EXPERT_TRACE` writer, 1488 layer-batches, 293 reads/step, 6.10/batch, 773.7 MiB/step, replayed at the engine's own per-layer depth into anonymous zero-copy destinations, interleaved in both orders (METH-06):

  | condition | ms/step | GB/s |
  | --- | ---: | ---: |
  | **the engine's own `io` window** | **225.4** | **3.60** |
  | replay, page cache allowed (the engine's own condition) | 177.9 | 4.56 |
  | replay, page cache bypassed | 130.3 | 6.23 |
  | replay, bypassed, engine-shaped destinations (48×16 = 1.98 GiB) | 149.6 | 5.42 |
  | replay, bypassed, engine-shaped, depth 3 | 147.3 | 5.51 |

  It reproduces neither 4.74 ms/layer nor the ~3 ms the depth curve predicts; it lands *below both*, and **the engine runs slower than its own worst offline condition.** Two of the differences are now measured rather than assumed. Engine-shaped destination pages — 48 layers × 16 slots of rotating anonymous memory, 1.98 GiB, each page reused only once per ~773 MiB of traffic, against the harness's ten resident 2.64 MiB buffers — cost 19.3 ms/step, real and small. Pool width costs nothing at all: depth 3 is 147.3 against depth 10's 149.6, the queue-depth knee again. **The residual ~75 ms/step is unmodelled and is not claimed.**

  **The one lever the replay did appear to find is inverted on the engine.** Bypassing the page cache is worth 42-61 ms/step offline — a fifth to a quarter of the engine's own window — on identical offsets at identical depth, in both orders. The engine opens its layer files without `F_NOCACHE`, so this was the obvious candidate. A/B'd on the engine with the flag added as an off-by-default knob, both orders, token IDs identical across all four runs:

  | `FINCHMOE_IO_NOCACHE` | `io` ms/step | tok/s |
  | --- | ---: | ---: |
  | off | 225.36 / 225.38 | 2.807 / 2.829 |
  | on | 261.24 / 260.97 | 2.484 / 2.483 |

  **The read window grows 15.9% and throughput falls 12%.** The trace says why the harness was wrong to predict a gain: 41.7% of the engine's 9,083 reads repeat a (layer, expert) pair already read during the run, but the *median* reuse distance is 3.97 GiB and **not one of the 3,787 repeats lands within 512 MiB**. The harness's ten recycling destination pages were holding reuse the engine's OS cache cannot, so bypassing threw away a hit rate the engine actually has. This is the same two-cache mechanism the drive probes found — repetition worth 2-3×, absorbed by both the buffer cache and the SSD controller — reappearing from the opposite direction. The knob ships off by default as the falsifier for that claim, with the engine number recorded at the call site so the offline argument is not re-derived.

  **Caveat on that A/B.** `gpu_routed` moved in the same direction as `io` (39.96/38.38 off, 47.50/47.61 on), and `F_NOCACHE` cannot change device work. The io difference is consistent and dominant across both orderings, but the two arms differ in more than the read path, so the mechanism is not pinned and the −12% should be read as the flag's effect in this runtime rather than as a clean one-variable result.

  **Where the fixed per-batch cost now stands.** The bounce copy was the last candidate, and it is priced by a control rather than argued: 181.6 µs per 2.64 MiB memcpy is 1.82 ms per 10-expert batch, 29.3 batches/step, **53.2 ms/step — 24% of the window — had a copy existed.** It does not. So the recoverable part of the ~84 ms/step is bounded by what is inside the transfer itself, and the only remaining axis is the drive's own dynamic range: 3.60 GB/s observed against 5.4-5.7 GB/s at this depth on synthetic offsets. Closing that gap is not a scheduling, dispatch or copy problem, and every offline model built so far has failed to reproduce it — the next discriminator is one the replay cannot reach, namely that the engine's slot pages are GPU-shared `MTLBuffer`s under a live Metal heap, which changes the vm object's reclamation behaviour in ways a Python replay has no way to hold.

  **That named discriminator was tested before that paragraph was written, and it is flat — the plan simply never recorded it.** `FINCHMOE_IO_STAGE` exists to test exactly this and is documented at its call site (`PreadExpertStreamer.swift:159-207`): it reads into a plain staging buffer allocated with the same `posix_memalign(Self.scratchAlignment, …)` and the same padded length as a slot, then copies into the slot, so the only variable left is that the slot is handed to `makeBuffer(bytesNoCopy:options:.storageModeShared)` and the staging buffer is not. **Qwen 3.8 reads at 3.28 GB/s with it off and 3.28 with it on**, two rounds each agreeing to 0.3%; the window grows 2.5% and the added memcpy is 2.4% of it. The control is what makes that a finding rather than an artifact: 3.6 reads at 6.24/6.55 off against 6.90/6.87 on, so the staging path is not itself expensive and its null result on 3.8 belongs to the read, not to the knob. The copy separates the installs by 15% (0.043 ms/MiB against 0.050) while the *read* differs 3.3x (1.68 ms/MiB against 0.51) — same memory system, same bytes, same pages: **symmetric on the copy, asymmetric on the read.** The slot mapping is not the cost, and the destination is exonerated. Recorded here because §2.1 asked for it and then spent a cycle without it.

  **A synthetic harness then reproduced the engine's batch structure exactly, and the result is negative in a way that matters.** `tools/read-sweep/read_batch.c` walks a list of files in order, issues `m` concurrent preads from each behind a `dispatch_apply` barrier — the engine's own primitive, chosen after a hand-rolled spin/atomic handshake was measured 27% slower and would have been the harness's own wake-up cost reported as the drive's — and moves to the next file. It reproduces the structure closely: 133 reads/round against the engine's 134.8 on 3.6 and 249 against 245.7 on 3.8, batch widths 3.33 and 5.19 against `io_conc` 3.26 and 5.30. Cold round 1, against the same-hour single-file continuous reader:

  | condition | 3.6 | 3.8 |
  | --- | ---: | ---: |
  | one file, continuous, no barrier | 3.007 | 2.966 |
  | 40–48 files, engine batch structure | 1.917 / 2.123 | 2.038 / 2.482 |
  | **the engine itself** | **4.736** | **3.167** |

  **It inverts the engine's ordering** — the harness puts 3.8 ahead, the engine puts 3.6 ahead by 1.5x — so per-file batch structure is not the mechanism. Two controls along the way are worth keeping. The barrier is nearly free: 48 batches against one file gives 2.759 GB/s against the continuous reader's 2.966, so a `dispatch_apply` per layer costs about 7% and the earlier suspicion that it was the harness's dominant cost was wrong. And the distinct-file walk is not the cost either: holding reads/round and bytes/round fixed and varying only the file count gives 3.163 (48 files x5), 3.003 (24 x10) and 2.883 (16 x15) — **fewer files is slower, not faster.** Neither branch of the prediction this cycle was built on held.

  **A correction this session owes the section: the previous cycle's headline here was wrong, and it was wrong because of exactly the reuse METH-14 warns about.** An interim reading held that 3.8's engine sits at the device's cold rate for its shape and that §2.1's premise therefore did not survive. That was measured against synthetic uniform-random offsets, which are *slower* than the engine's real access pattern — so the comparison flattered the engine. Against the engine's own trace (below) the premise survives: 3.8's engine reads at 0.7x what its own pattern delivers cold.

  **The measurement that settles it replays both installs' real traces through one harness, interleaved.** `FQ_EXPERT_TRACE` was captured for 3.6 and 3.8 in one session — 4,179 reads on 3.6 (134.8/step, 227.5 MiB/step) and 7,617 on 3.8 (245.7/step, 648.8 MiB/step); note these are a *different* workload from the 9,083-read trace in the table above, which came from a different prompt, so the two are not interchangeable. Both traces were then replayed in the same process, alternating, each at its own engine concurrency, scored on the engine's own metric — the **sum of per-batch read spans**, which is what `io_read_wall` measures and which excludes anything happening between batches:

  | | engine, this session | own trace, replayed | engine ÷ replay |
  | --- | ---: | ---: | ---: |
  | 3.6 | 47.18–52.19 ms/step | **62.68 ms/step** (3.81 GB/s) | **1.20–1.33x faster** |
  | 3.8 | 213.73–233.15 ms/step | **162.95 ms/step** (4.18 GB/s) | **1.31–1.43x slower** |

  **The traces put 3.8 ahead of 3.6 by 1.34–1.49x, and the engine reads them the other way round by 1.57–1.59x. The engine flips the sign** — so the gap is not in the access pattern, not in the stride, not in the file walk and not in the bytes. 3.6's engine *beats* a faithful offline replay of its own reads; 3.8's engine *loses* to one. Both arms of that comparison are generous to the engine (warm reused destinations, where the section above prices engine-shaped destinations at +19.3 ms/step, ~15%), so the 3.8 deficit is a lower bound.

  **The shape of the asymmetry is working-set size, which the traces give directly:**

  | | distinct experts touched | working set | reuse distance, p50 |
  | --- | ---: | ---: | ---: |
  | 3.6 | 2,531 | **4,271 MiB** | 1,333 MiB |
  | 3.8 | 4,195 | **11,077 MiB** | 3,694 MiB |

  3.6's working set fits in 16 GB and warms completely — a fifth replay pass reaches 21.4 GB/s — while 3.8's does not, so 3.8 never gets that relief and pays cold cost on all 31 steps. That is consistent with the `FINCHMOE_IO_NOCACHE` A/B above, whose whole explanation is that 3.6 has reuse the OS cache can serve and 3.8 does not.

  **The two installs, for scale.** 3.6 is a 19 GB install — 17 GB of packed experts, 1.76 GiB of dense weights, no PLE shards — holding 40 layers x 256 experts x 1.69 MiB. 3.8 is 162 GB: 63 GB of packed experts, 3.61 GiB of dense weights and **95 GB of PLE shards**. Per layer that is 432 MiB against 1,352 MiB. Measured end to end on the same prompt: 8.55–8.66 tok/s decode and 13.4–13.7 tok/s prefill on 3.6, against 2.76–2.91 and 5.9–6.0 on 3.8. 3.8 also reads its PLE shards on a **separate path** — 496 opens, `ple_wall_ms/step` 3.77 — which 3.6 does not have at all, so the two installs differ by a second, much larger I/O stream as well as by the expert read this section is about, and that stream is not part of the span-sum metric used above. It is small per step (1.7% of 3.8's read window) but it is a real difference in what the runtime is doing, and it belongs in the list of things to rule out before the deficit is attributed to the expert read alone.

  **The last structural difference between the replay and the engine was tested and is also not it.** The engine's batches are spaced by GPU work (~0.86 ms/layer on 3.6, ~1.53 on 3.8); the replay's are back-to-back, so the drive's queue never drains. Sweeping an inter-batch gap on 3.8 — whose 11 GiB working set cannot be flattered by the page cache — moves nothing: 162.95 / 152.29 / 158.99 / 154.63 ms/step at 0 / 0.5 / 1.0 / 2.0 ms. The 3.6 column of the same sweep *looks* like a large spacing effect (62.68 down to 23.27 ms/step) and is not one: run in reverse order the same monotone climb reappears (2.65 → 4.16 → 5.83 → 8.25 → 21.44 GB/s), so it tracks position in the sweep, not gap width. That is the page cache, and it is a warning about single-order sweeps on a 4 GiB working set.

  **Instrument warning, and it governs every comparison in this section.** The drive drifts *within* a session: 3.6's engine read 4.68 and 5.18 GB/s in paired runs eleven minutes apart, and one identical synthetic cell gave 1.825 and 3.163 GB/s twenty minutes apart. That is wider than the 1.57x gap this section exists to explain, and far wider than the 0.3% by which the engine agrees with itself inside a session. **§2.1's own 5.42 GB/s replay row and its 3.60 GB/s engine row were measured in different sessions and cannot be subtracted.** Every paired number above was taken interleaved in one session for that reason. Spotlight indexing is live on the volume (`mdbulkimport`, `mds_stores`, indexing enabled) and is not excluded as a contributor to the drift.

  **Where the question now stands.** The read side is not closed by "the drive is saturated" — that reading was refuted earlier in this section and nothing here revives it — and it is not closed by the access pattern either, which is the new result. What is left is narrower and better posed: **the engine loses 1.3–1.4x on 3.8 and gains 1.2–1.3x on 3.6, on the same code path, against the same replay.** The destination is exonerated (IO_STAGE), the dispatch chain is exonerated (0.025 ms/batch of the ~1.8), the pool width is exonerated (depth 3 ≈ depth 10), the batch barrier is exonerated (~7%), the file walk is exonerated (fewer files is slower), and the pattern is exonerated (this block). The remaining structural difference between the two installs is the **slot destination working set**: 48x16x2.64 MiB = 2.03 GiB of rotating pages on 3.8 against 40x16x1.69 MiB = 1.08 GiB on 3.6, on a 16 GB box that memguard reports peaking at 3.1 GB compressed. The section above prices that at 15% on the old trace; it has never been priced on the trace that shows the deficit. That is the next measurement, and it is bounded: re-run the four replay conditions against the new traces with the span-sum metric, and ask whether the 2.03 GiB destination accounts for 1.3–1.4x. If it does not, the deficit is in the engine's own runtime and the next step is a GPU-overlap experiment, not a read-side one.


### 2.2 [MEDIUM impact, MEDIUM-HIGH risk] Reduce per-layer command-buffer count in the decode hot path
- **Evidence**: `RealForwardRunner.swift` shows the Qwen decode path issuing multiple `makeCommandBuffer()`/`commit()` pairs per layer per token — a `cb1`-equivalent, a `sharedCB`, one or more `tileCB` per miss-tile (in `executeExpertCachePlan`-driven decode paths near lines 1046-1305, 2135-2271), and a `tailCB`. At 40 layers this is potentially 3-5+ command buffers × 40 = 120-200+ command buffer commits per generated token. `SYSTEM_DESIGN.md`'s own "Metal execution" section says decode already stays on custom GEMV to avoid MPP overhead — but doesn't discuss CB-count overhead itself, and the OPTIMIZATION_JOURNEY.md doesn't record a CB-batching experiment for Qwen (all the historical fusion experiments — QKV, layer-tail, head — reduced *kernel* count within a CB, not CB count across layers).
- **Action**: profile actual CB commit/encode overhead via Instruments (`cb1`/`cb2` counters already exist per `SYSTEM_DESIGN.md`'s phase table — extend them to report raw CB count per token). If encode+commit overhead is a meaningful fraction of the ~55-95ms/token step (16GB Mac mini ~95ms/token at 10.5 tok/s; 24GiB M4 Pro ~55ms/token at 18 tok/s), investigate coalescing the miss-tile CBs into a single CB per layer using multiple encoders/wait-events instead of separate command buffers, since Metal command buffer submission has fixed per-CB CPU-side overhead independent of GPU work size — this would matter most on the *faster* M4 Pro machine where the CB floor is a larger fraction of a per-token budget that's already only ~55ms.
- **Expected impact**: 5-15% decode speedup if CB overhead is currently significant; **could be near-zero** if the existing overlap design (shared-expert branch runs while I/O happens, tile CBs run concurrently with reads) means CB overhead is already hidden — this needs the profiling step first, and note the strong prior in `OPTIMIZATION_JOURNEY.md` that "clean local designs often lost in the full runtime," especially schemes that reduce launch count at the cost of concurrency/overlap (the exact failure mode of the rejected monolithic fusion and the rejected "reusing Metal argument buffers" experiment, which *cut 21,217 allocations to two* and still **slowed** long prefill by 9%).
- **Risk**: medium-high — this directly touches the `cb1`/`io`/`cb2` overlap design that is core to the runtime's decode-speed story; any CB coalescing that removes the ability to start the shared-expert branch early or start cache-hit routed work before misses land could regress throughput, per the explicit "Finer-grained overlap did not help" lesson.
- **Validate**: must be a full end-to-end decode benchmark (not isolated CB-timing microbenchmark, per the journey doc's central lesson), output byte-identical to current path, tested at both short and long context (attention-heavy full-layer cost changes with context depth per the 4096-soak data: 10.2→7.6 tok/s).

- **Measured 2026-09-11 — the count is now instrumented, and the premise is wrong twice over.**

  **The evidence above cites the wrong path.** The `tileCB` machinery it points at (`:1046-1305`, `:2135-2271`) is **prefill**; decode has no `tileCB` at all. Decode's commits are exactly seven sites — the layer's `cb`, `sharedCB`, `routedCB` (three per layer, unconditional), `phase1HitCB` when the hit-split branch runs, and two `runSync` buffers per forward for the embed and the head. That yields a falsifiable prediction, and the counters now test it:

  ```
  CBs = 3 · layers + H + 2        H = layers that took the hit-split branch
  ```

  Qwen 3.8 (48 layers): **146 ≤ n ≤ 194**. Qwen 3.6 (40): **122 ≤ n ≤ 162**. Measured on 3.8 at 16 slots: **`cbs=5883`, `cbs/step=189.8`**, i.e. `H` = 43.8 of 48 layers — in band, high, and the shape is what the prediction describes.

  **The overhead the section was written to reduce does not exist.** The `io` split in §2.1 prices the whole per-layer submission path: `streamersQueue.sync` → `DispatchQueue.global` → `concurrentPerform`, plus the continuation handoff and the `ensureLayerOpened` check, is **0.6 ms/step of 224.67 — 0.27%** — across 48 invocations per token. The `cb1` encode clocks are 2.62 ms/step total. Whatever 190 command buffers cost, it is not in the CPU submission path.

  **Action 1 is done, not deferred**: the CB count is reported as `cbs` / `cbs/step` on every `--counters` line, so the prediction stays checkable at any slot count or install without re-instrumenting. **Recommendation: close this item.** Coalescing miss-tile CBs cannot recover a cost that measures at 0.27%, and the section's own risk note — that removing the early-committed shared-expert buffer could regress the overlap it exists to provide — argues against spending the risk budget on a 0.27% target. The one thing that would reopen it is a machine whose `io` window is far shorter than this one's, where a fixed CPU cost would be a larger share; on this box at this operating point it is not.

### 2.3 [MEDIUM impact, LOW-MEDIUM risk] Fuse the GDN gate + recurrent-step epilogue further, or batch value-head dispatch
- **Evidence**: `Metal/LinearAttn/gdn.metal` currently dispatches `gdn_conv_update`, `gdn_gate`/`gdn_gate_gemv`, `gdn_recurrent` (one threadgroup per value head — 32 threadgroups per GDN layer, 30 layers = 960 threadgroup dispatches per token just for the recurrent step, likely as separate kernel launches per layer given the per-layer state buffer indexing in `RealForwardRunner.swift:1658-1768`), and `gdn_rmsnorm_gated` as **separate kernel dispatches** per layer. Each GDN layer's recurrent-state read+write is only ~2 MiB (per `QWEN36_PORT.md`: "32 heads × 128 × 128 × 4B = 2MiB per GDN layer") — computationally trivial (O(V·D²) ≈ 524K fp32 ops/layer) but currently paying full per-kernel dispatch overhead (PSO bind, argument encode, barrier) for ~4 separate kernels × 30 layers = 120 dispatches/token, on data that's small enough to be dispatch-bound rather than compute- or bandwidth-bound.
- **Action**: fuse `gdn_gate` (or `gdn_gate_gemv`) directly into `gdn_recurrent`'s prologue (both already run per-value-head in threadgroup-parallel form; the gate is a tiny elementwise op computed once and read by all threads in `gdn_recurrent`) to eliminate a barrier+kernel-launch round trip per layer. Similarly examine whether `gdn_rmsnorm_gated` can read directly from `gdn_recurrent`'s output buffer inside the same command encoder without an intervening dispatch boundary (Metal doesn't require separate CBs for sequential dispatches within one encoder — check whether these are currently issued as separate encoders unnecessarily).
- **Expected impact**: 3-8% decode speedup — smaller than 2.1/2.2 because this is pure dispatch-overhead removal on already-tiny kernels, similar in kind to the "LM-head tiling" experiment in the journey doc that saved 1.1ms out of 167.7ms (inconclusive end-to-end) — flag as a **candidate, not a committed win**, exactly per that precedent.
- **Risk**: low-medium — must preserve the exact op ordering (`decay → read → update → read-out`) the port went through significant validation to lock (`QWEN36_PORT.md`: "the recurrence order... is the part most likely to be subtly wrong"); any fusion must be validated bit-for-bit against the existing GDN reference tests (`Tests/FinchMoE/Core/Kernels/LinearAttn/GDNTests.swift`) before being trusted.
- **Validate**: rerun `GDNTests.swift` (fp32 CPU reference comparison within `fp16ChainedReduction` tolerance) plus full end-to-end decode benchmark; this is exactly the kind of change the journey doc says needs "a repeatable gain" bar to ship.

### 2.4 [LOW priority, explicitly NOT recommended now] Speculative decoding
- **Evidence**: no draft model exists in this codebase, and the MTP head mentioned in `QWEN36_PORT.md` ("Extra: MTP head with 1 hidden layer... not needed for greedy decode") is present in the checkpoint but explicitly unused. Speculative decoding needs either a small draft model (adds a second resident model — likely violates the ~1.1-1.2 GiB budget outright, `[MEMORY RISK — HIGH]`) or self-speculation via the MTP head (would require porting and validating a second decode path, a project-sized effort comparable to the GDN port itself).
- **Recommendation**: worth a scoping spike (does the 1-layer MTP head's weights already ship in `model_weights.bin` or were they dropped — `README.md` says routed-expert/common repack "omits the vision tensors" but doesn't mention MTP head handling; `PATH: models/Qwen3.6-35B-A3B-bf16/` presumably has `mtp` tensors that may or may not be repacked) but should be sequenced **after** items 2.1/2.2/2.3 land, since it's the highest-effort, highest-risk item on this list, and its benefit (fewer full 40-layer forward passes per accepted token) is multiplicative with faster decode, not a substitute for it.
- **Impact/Risk**: impact potentially large (1.5-2.5x is typical for speculative decoding in literature) but risk is high given no infrastructure exists yet, and self-speculation via MTP would need its own correctness validation program on the scale of the GDN port.

---

## 3. "Turbo quant": KV cache, GDN recurrent state, and further weight/activation quantization

### 3.1 Current state precisely (from `KVCacheManager.swift`, `RealForwardRunner.swift`, `QWEN36_PORT.md`)

| Store | Count | Precision | Layout | Size |
|---|---|---|---|---|
| Full-attention KV cache | 10 layers | FP16 | linear (append-only), `numFullKVHeads=2 × fullHeadDim=256 × 2B` = 1,024 B/token per K, same for V | 2,048 B/token × 10 layers = ~20 KB/token; ~82 MB at 4K context, scales linearly to `maxContext` |
| GDN recurrent state | 30 layers | **FP32** | v-major `state[(hv*D+v)*D+k]`, `[32][128][128]` per layer | 2 MiB/layer × 30 = 60 MiB fixed (context-independent) |
| GDN conv state | 30 layers | FP16 | `[qkvDim][3]` per layer | trivial (~50 KB total) |
| GDN in_proj_a/b gate weights | 30 layers | int4 or int8 affine (config-dependent, `linearAttnBits`) | group-64 MLX affine | small, resident |
| Router | all 40 layers | int8 affine | — | small, resident |
| Shared/routed experts | all 40 layers | 4-bit affine group-64 | — | ~20 GB on disk, streamed |
| Activations | — | FP16 | — | — |
| Metal accumulators | — | FP32 | — | — |

Note the KV cache is **not the current bottleneck it was for Gemma**: Gemma had 30 attention layers with a growing cache; Qwen has only 10 full-attention layers (the other 30 are the fixed-size GDN state). This changes the cost/benefit math versus the rejected Gemma K4/V4 experiment substantially — there is 3x less KV-cache surface to quantize, and the potential win is smaller in absolute terms but also smaller in risk (less exposure to the "grows past FP16 at long context" failure mode, since Qwen's cache is already 1/3 the layer count of Gemma's). (The rows above are 3.6-specific; 3.8 Flash-Next has 12 full-attention layers of 48 and an extra indexer timeline — see §3.1b.)

### 3.1b Memory vs context length, both Qwen models (derived from the allocation code, not measured)

§3.1's figures are 3.6-specific and stop at 4K. Completed 2026-09-12 for both models across 4K/8K/16K/32K/64K, from `KVCacheManager.swift`, `QSAIndexerState.swift`, `RealForwardRunner.swift` and the per-family `ArchConfig` presets. **Arithmetic from the allocation code — no run above 4K exists anywhere in this repo.** Read it as what the engine *will* allocate, not what has been observed.

Both models are hybrid, and that is the whole story: only every 4th layer is full attention, and the rest are GatedDeltaNet, which holds a **fixed** recurrent state and **no KV cache at all**. The per-token stride is identical for both (`numFullKVHeads=2 × fullHeadDim=256 × 2B`, K and V each = 1,024 B/token/layer), so the entire KV difference between the two models is the layer count, 10 vs 12.

| Store @ 65,536 tokens | 3.6 35B-A3B | 3.8 Flash-Next 125B |
|---|---|---|
| Full-attention layers (of 40 / 48) | 10 | 12 |
| KV cache, FP16 | 1.342 GB | 1.611 GB |
| QSA indexer (rawKeys + pooled + scores + cells) | — | 0.253 GB |
| GDN recurrent state, FP32 — context-independent | 60 MiB | 108 MiB |
| GDN conv state, FP16 — context-independent | 1.41 MiB | 2.11 MiB |
| PLE conv history, FP16 — context-independent | — | 360 KiB |
| **Persistent total @ 64K** | **1.31 GiB** | **1.84 GiB** |

KV alone, per context length — the term that actually scales:

| Context | 3.6 | 3.8 |
|---|---|---|
| 4K | 80 MiB | 96 MiB |
| 8K | 160 MiB | 192 MiB |
| 16K | 320 MiB | 384 MiB |
| 32K | 640 MiB | 768 MiB |
| 64K | 1.25 GiB | 1.50 GiB |

Three consequences for the §3.2 proposal. **(a)** The quantizable surface is 10–12 layers of 40–48, so the KV store is roughly a **quarter** of what a dense model of the same shape would carry — at 64K, 1.34 GB instead of 5.37 GB for 3.6, and 1.61 GB instead of 6.44 GB for 3.8. **(b)** 3.8 pays a **QSA indexer on top** (`QSAIndexerState.swift:136-145`: `rawKeys` = `maxContext × 128` fp16 per full layer, plus pooled/scores/cells), ~13% of its KV at 64K and absent from §3.1's table entirely. Any int8 KV proposal must state which of these two timelines it quantizes. **(c)** The store is small in absolute terms at every context this engine supports, which is the case for treating §3.2 as low priority rather than high.

**Not in the table, and not small:** the streaming working set. The PLE table is 95.4 GiB on disk but is `pread`-gathered 5 KB/token with nothing resident (`PLEHost.swift:11-14`); the experts (18.1 GB / 68.1 GB on disk) stream through rotating slot pages — 2.03 GiB resident for 3.8 vs 1.08 GiB for 3.6 (EXPERIMENT summary 01). At ≤4K the process peaks at ~0.5–0.85 GiB RSS (3.6) and ~3.1 GB compressed (3.8), so a 3.8 64K run should land near 5 GB. That sum is an extrapolation across an unmeasured region; the state figures above are not.

**KV is preallocated at `--max-context`, never grown.** `KVCacheManager.init` sizes every layer at `capacity = maxContext`, and `advance` *traps* rather than growing (`precondition(position + count <= maxContext)`). Asking for 64K pays 64K of KV at init even for a 10-token prompt. Defaults differ per surface: CLI 4096 (no cap), HTTP server 16384 (max 65536), Mac app 4K (max 64K). The Mac app's context-menu byte labels are now computed from the loaded `ArchConfig` rather than hardcoded — before 2026-09-12 they showed 3.6's 1.26 GB delta at 64K for every model, under-reporting 3.8 by 0.25 GB.

### 3.2 [LOW-MEDIUM impact, MEDIUM risk] Full-attention KV cache: int8 per-block-scale quantization, revisit only with a stronger quality gate than before
- **Rationale for revisiting despite the rejected precedent**: the earlier failure (`OPTIMIZATION_JOURNEY.md`) was specifically the **packed K4/V4** (4-bit) scheme across **all 30** Gemma attention layers, where the packing overhead ate the savings at long context because most of those layers used a bounded *ring* buffer already (only 5 full-attention layers grew unbounded). For Qwen, apply this only to the **10 full-attention layers**, at **int8** (not int4 — int4 KV cache has a well-documented larger perplexity risk in the broader literature, and this codebase's own experiment history shows int4-class quantization schemes need the most validation scrutiny — e.g., Bug 9-style silent correctness bugs are exactly the class of risk this project has already been burned by more than once for 4-bit paths).
- **Design**: per-token, per-head int8 with a block-local (e.g., per 32 or 64 token block) FP16/BF16 scale+zero-point, computed at KV-write time (adds one small kernel to the K/V write path in `Attention.swift`/`PrefillAttention.swift`) and dequantized on read inside the attention kernel (extra ALU work, but attention kernels already read K/V through a stride-indexed path in `KVCacheManager.swift`'s `kRange`/`vRange`, so the read-side change is localized).
- **Expected impact**: ~50% KV memory reduction on the 10 full-attention layers only — since this is already a small store (~82 MB at 4K, scaling to hundreds of MB at 32-64K context per the "Mac app offers 4K/8K/16K/32K/64K context" note in `SYSTEM_DESIGN.md`), the win is proportionally larger **at long context** (where `docs/QWEN36_PORT.md`'s 4096-soak already shows decode falling from 10.2→7.6 tok/s due to "expected full-attention KV growth" — bandwidth reduction here directly targets that regression). Estimate 5-15% decode speedup specifically at long context (8K+), near-zero at short context where KV is small relative to expert I/O.
- **Risk**: medium. Must not repeat the earlier failure mode. Concretely:
  1. Gate on the *same* trusted-reference quality comparison used before (the "failed the full quality evaluation" bar in the journey doc) — rerun the EvalPlus HumanEval harness (`quality/humaneval/`, the 90.9%/87.8% baseline from `QWEN36_PORT.md` item 6) with int8 KV enabled and require the pass@1 delta to be within noise (±1-2 problems, matching the existing "within 1-2 problems of the 3090 cell" tolerance already accepted as parity in this project).
  2. Gate on the 4096-context soak reproduction (`QWEN36_PORT.md` item 4: pinned-recall quote-exact answers) — any KV quantization must preserve quote-exact recall at full context depth, since that's the existing acceptance bar.
  3. Confirm asymptotic memory behavior explicitly (the size-crossover bug that killed the last attempt) — model bytes/token at int8+scale-overhead vs FP16 across the full context range (4K→64K) before shipping, not just at one context length.
- **Validate**: EvalPlus HumanEval rerun, 4096-context soak rerun, and the memory-vs-context-length table across 4K/8K/16K/32K/64K. **The derived half of that table now exists in §3.1b** (FP16, both models, arithmetic from the allocation code). What remains owed here is its *measured* counterpart: run the sweep with int8 KV enabled and diff it against §3.1b's FP16 column, since the whole point is the size crossover that killed the last attempt.

### 3.3 [LOW impact, HIGH risk — do not pursue without strong justification] GDN recurrent-state quantization
- **Evidence**: `QWEN36_PORT.md` explicitly notes the recurrent state is fp32 because it mirrors `mamba_ssm_dtype` in the reference implementation, and separately notes "the recurrence order... is the part most likely to be subtly wrong" — this state accumulates via repeated decay-multiply and rank-1-update across the *entire* generation (unlike KV cache, which is read-only after being written once), so quantization error compounds multiplicatively across the whole decode sequence rather than being read-order-independent. This is fundamentally a different risk profile than KV-cache quantization.
- **Bandwidth math**: state is 60 MiB total across 30 layers, read+written once per decode step by `gdn_recurrent` — roughly 120 MiB/token of traffic. At an M-series unified-memory bandwidth on the order of 100-200+ GB/s, this is under 1ms/token — a small fraction of the current ~55-95ms/token step. Even quantizing to bf16/fp16 (halving to 60 MiB/token traffic) saves well under 1ms/token — **immaterial** to end-to-end throughput.
- **Recommendation**: do not pursue. The bandwidth saved is not commensurate with the accuracy risk of quantizing a value that's accumulated via repeated multiply-decay over up to 262,144 positions (`position_limit` in `QWEN36_PORT.md`), where even small per-step rounding error could compound into visible long-context quality drift — exactly the kind of subtle, hard-to-detect regression this project's own GDN port took the most care to avoid ("Kernels are validated against the Swift fp32 reference before any layer is wired in"). If pursued at all, it should only be as a research spike behind the 4096-context soak and EvalPlus gates, and expectations should be set at "near-zero measured speedup" going in.

### 3.4 [MEDIUM impact, MEDIUM risk] Router quantization: int8 → mixed int8/int4 or wider-batch fusion
- **Evidence**: `SYSTEM_DESIGN.md`/README: router is already int8 affine on both Gemma and Qwen. Qwen has **256** experts vs Gemma's 128, so the router GEMV (`logits = int8_affine(scaled_input)` per `SYSTEM_DESIGN.md`'s decode pseudocode) does 2x the output-row work per layer per token, across 40 layers instead of 30 (since Qwen routes on every layer) — this is a larger fraction of `cb1` time for Qwen than it was for Gemma, and hasn't been re-profiled at this scale (same profiling gap as item 1.4).
- **Action**: profile router-GEMV cost specifically as part of the `cb1` breakdown (item 1.4's profiling pass should include this). If material, evaluate whether the existing `DequantInt8GEMV.swift` kernel is already using the same width-optimized load pattern validated for the LM head (`OPTIMIZATION_JOURNEY.md`: "two `ushort` loads... reduced LM-head GPU time from about 21.5 to 16 ms") — if the router GEMV was written before that lesson was captured, apply the same fix.
- **Expected impact**: unknown without profiling; likely small (single GEMV, not the dominant `cb1` cost) but cheap to check since the kernel pattern already exists elsewhere in the codebase to copy from.
- **Risk**: low if this is purely applying an already-validated load-width pattern; the correctness pitfall (2-byte vs 4-byte alignment assumptions, per the journey doc's "A 32-bit packed-load path passed an offset-zero fixture, then produced garbage in real decode") must be re-checked against the router's actual live tensor offsets, not assumed from the LM-head case.
- **Validate**: kernel-level parity test against the existing int8 GEMV reference in `FinchMoEValidation/Support/Reference/Quant/`, plus end-to-end decode benchmark.

### 3.5 [LOW impact, LOW risk] Re-examine 4-bit group size for shared/routed experts (currently group-64)
- **Evidence**: `README.md`/`SYSTEM_DESIGN.md`: "affine 4-bit MoE group 64" is fixed for both Gemma and Qwen. Group size trades accuracy for scale/bias metadata overhead (smaller groups = more accurate but more BF16 scale+bias bytes per weight). This project has never published a group-size sweep in the optimization journey (the journey doc's group-64 discussion is about *load alignment*, not group-size tuning).
- **Action**: this is a genuine open question, not a known-rejected idea — but it's a **repack-level** change (would require `FinchMoERepack` to produce group-32 or group-128 variants) and touches every routed-expert kernel's inner loop (`moe.metal`, `DequantInt4GEMV.swift`), making it one of the highest-blast-radius changes on this list.
- **Expected impact**: group-128 could shrink the ~20GB on-disk footprint slightly and reduce scale/bias read bytes per expert fetch (a genuine I/O-bandwidth win during decode, since expert *reads* are the dominant decode cost) — plausibly 2-5% smaller expert blobs, translating to a proportional I/O time reduction. Group-32 would go the other way for quality headroom at the cost of more I/O.
- **Risk**: low probability of being worth the implementation cost given the small expected win and high blast radius (repack format change, kernel change, needs a fresh EvalPlus run) — **deprioritize below all other items in this document** unless profiling in 1.4/2.1 shows read-bytes-per-expert (not read-*count*) is a meaningful bottleneck, since group size only affects bytes-per-expert, not access pattern.
- **Validate**: EvalPlus HumanEval full rerun (this changes weight values, unlike KV/router changes which are runtime-only) plus on-disk size and decode I/O-per-token measurement.

---

## 4. Other performance opportunities

### 4.1 [DONE 2026-09-11] Extend the `cb1`/`io`/`cb2` diagnostic counters with a Qwen-specific breakdown
- **Evidence**: `SYSTEM_DESIGN.md`'s phase table and `RUNTIME_CONTROLS.md`'s "Advanced" HUD already report `cb1`/`cb2`/output-head time and I/O-per-token — but none of the Qwen-specific docs (`QWEN36_PORT.md`) report a GDN-vs-full-attention-layer cost breakdown, or a routed-MoE-vs-shared-expert-vs-attention breakdown *for Qwen specifically*. Every "expected impact" estimate in this document (1.4, 2.1, 2.2, 3.4) is bounded by not having this data yet.
- **Action**: before implementing 1.1/2.1/2.2/3.2, spend a short cycle extending the existing counters (they're described as already itemized per-phase in `RUNTIME_CONTROLS.md`) to break down `cb1` into router/attention/GDN-recurrent sub-costs, and `io` into hit-vs-miss-count and bytes-per-token, specifically on the Qwen install. This turns every "if material" caveat above into a go/no-go decision made with real numbers instead of the Gemma-era priors this whole engine was tuned against.
- **Risk**: none — pure instrumentation.
- **Landed**: `FinchMoECLI --counters` prints one extra stderr line; the buckets accumulate unconditionally so the measured path is the shipped path. The `cb1` split is cursor-tiled (`identity=exact`), and `--expert-cache-slots` exposes item 2.1's sweep variable on the CLI for the first time. See `SYSTEM_DESIGN.md` "The `cb1` sub-buckets".
- **Measured 2026-09-11**, Qwen 3.8 Flash-Next (48 layers, 12 full / 36 GDN, 512 experts, top-k 10), 32 slots, 54-token prompt, 64 tokens, `--temperature 0`, warm cache:

  | bucket | ms/step | share of cb1 |
  | --- | ---: | ---: |
  | **`cb1` (CPU encode, total)** | **2.82** | — |
  | `other` (prologue, layer tail, drain) | 1.66 | 59% |
  | `attention` (12 full layers) | 0.29 | 10% |
  | `gdn_conv_gate` (36 GDN layers) | 0.33 | 12% |
  | `router` | 0.21 | 7% |
  | `gdn_recurrent` | 0.20 | 7% |
  | `gdn_proj` | 0.12 | 4% |
  | `wait` (pipeline wait, *excluded from `cb1`*) | 121.05 | — |

  Alongside it: `io` 232.91 ms/step, `cb2` 0.85, head 5.39 (wall), PLE gather 3.75, 191.5 command buffers/step, 603 MB/step of routed-expert reads.

- **Added 2026-09-11 — GPU attribution (`gpu_cb1`, `gpu_cb1_fullattn`, `gpu_cb1_gdn`, `gpu_routed`, `gpu_samples`).** The same readout now reports device execution time, read from `MTLCommandBuffer` timestamps after completion — no extra buffer, no extra wait, no change to commit order. This is the piece §4.1's first pass named as missing: the CPU buckets price the dispatch, so they cannot say whether a kernel is slow. `gpu_cb1_fullattn + gpu_cb1_gdn == gpu_cb1` exactly on every run below, because both halves come from one sample branched on the layer kind. Three runs, all `--temperature 0`, 31 decode forwards, all `identity=exact`:

  | install / prompt | stack | layers | ms/step | **ms/layer** | share of `gpu_cb1` |
  | --- | --- | ---: | ---: | ---: | ---: |
  | 3.8 Flash-Next, 54 tok, 16 slots | full attention | 12 | 13.52 | **1.127** | 17.9% |
  | | GDN | 36 | 61.85 | **1.718** | 82.1% |
  | | *`gpu_cb1`* | 48 | 75.37 | 1.570 | |
  | 3.6 35B, 54 tok, 32 slots | full attention | 10 | 5.30 | **0.530** | 15.7% |
  | | GDN | 30 | 28.52 | **0.951** | 84.3% |
  | | *`gpu_cb1`* | 40 | 33.82 | 0.846 | |
  | 3.6 35B, 1791 tok, 32 slots | full attention | 10 | 12.81 | **1.281** | 29.9% |
  | | GDN | 30 | 29.98 | **0.999** | 70.1% |
  | | *`gpu_cb1`* | 40 | 42.79 | 1.070 | |

  Four things follow, and they are the reason this cycle was worth its cost:

  1. **GDN is where the GPU time is.** 1.718 ms/layer against attention's 1.127 on 3.8 (1.6x), 84.3% of `gpu_cb1` on 3.6. On 3.8 that is 16.8% of the whole 367 ms token in one stack.
  2. **Attention scales with context; GDN does not.** On 3.6, going from 54 to 1791 tokens of context moves attention 0.530 -> 1.281 ms/layer (+142%) while GDN moves 0.951 -> 0.999 (+5%). Attention **overtakes GDN per layer** somewhere between those two points. It does not overtake it in aggregate, because three of every four layers are GDN — but the crossing is real and it is the lever item 3.2 (int8 KV) acts on. Two points give a slope, not a curve: 0.43 us per layer per context token, which would put attention at ~2.3 ms/layer at 4K on this install. Treat that as an extrapolation to test, not a projection.
  3. **The `wait`-as-overhead reading was wrong, and the size of the error is the finding.** `wait` minus *both* GPU figures is 10.40 / 5.60 / 8.69 ms/step across the three runs — 0.055 / 0.035 / 0.055 ms per command buffer. That is an order of magnitude *below* the ~0.26 ms a kernel-bearing buffer costs, measured in S6. `wait` looks like a third of the token because layer N+1's `cb1` is committed after layer N's routed tail, so its wait drains that tail. Subtracting only `gpu_cb1` charges the largest kernel block in the model to overhead.
  4. **`gpu_samples` is an exact gate, not a rough one.** In all three runs `cbs - gpu_samples` is exactly 62, i.e. exactly two buffers per forward — buffers sampled while still in flight and therefore skipped. The same value across three different installs, prompts and slot counts means the instrument is deterministic, so a shortfall that is *not* two per forward is a real defect rather than noise.

- **What this decides**:
  - **2.3 (GDN fusion) — reconsidered, and still not yet priced.** The first pass called this a no-go because the whole GDN split was 0.65 ms/step of *encode*. That reasoning was sound but empty on the GPU side, and the GPU side now answers the half that matters: GDN is 82% of `gpu_cb1` and 16.8% of the token, so the stack 2.3 targets is unambiguously where the time is. What is still missing is the split *within* GDN — the 1.718 ms/layer covers `gdn_proj`, `gdn_conv_gate`, `gdn_recurrent`, `gdn_rmsnorm_gated` and `gdn_o_proj` together, and 2.3 fuses a specific pair of them. Per-kernel GPU timestamps inside the GDN stack are the next instrumentation step, not a re-run of this one.
  - **3.4 (router GEMV) — unchanged, and now for a measured reason.** The router bucket is 0.20-0.31 ms/step of encode on both installs. It is not broken out on the GPU side at all, so the kernel question stays open — but the attention curve above says where the GPU headroom is, and it is not the router.
  - **2.2 (CB coalescing) — refuted, not merely unsized.** The first pass read `wait` = 121.05 ms/step as reachable overhead and made it 2.2's target. The corrected residue is 0.055 ms/CB: after both GPU figures are subtracted, per-buffer overhead is already an order of magnitude cheaper than a kernel dispatch. There is no three-figure pool of dispatch tax to reclaim, and the stand-alone rejections of ORCH-15 and ORCH-12 are not overturned by anything here.
  - **3.2 (int8 KV) — promoted by the attention curve.** The earlier note that a CPU encode bucket could not price a KV-format change was correct; the GPU split prices the *stack*, and the stack grows 2.4x per layer over 1.7K tokens of context while everything else in the layer stays flat. That curve is the case for 3.2, and it is measurable with the instrument now in place.
  - **The frame itself is still the finding**: every CPU encode bucket on the machine sums to 0.8% of the token, and the GPU work in `cb1` is 20-30% of it. The token is I/O and pipeline wait. Neither the encode split nor the GPU split touches that, because neither instrument can see a *prediction* — filling the read window with GPU work would need layer N+1's experts before layer N's router finishes, which is the decode-side prefetch question §1.3 does not cover (it is prefill-only) and which no item on this list currently owns. The read side was re-examined this cycle and closed again for a better reason: see the drive probes under 2.1, which rule out both the byte-count model and the in-flight-concurrency replacement.
- **Note for 2.1**: `io` is *awaited* read time, so it overlaps the shared-expert GPU work rather than following it. These buckets are not a serial timeline — see the sweep result under 2.1.
- **Method note**: `gpu_cb1_fullattn`/`gpu_cb1_gdn` are alternatives, so they are read per layer — dividing each by its own layer count (12/36 on 3.8, 10/30 on 3.6) — never per step. Comparing the two ms/step figures directly would just re-derive the layer ratio.

### 4.2 [MEDIUM impact, LOW risk] Batch/streaming decode requests — explicitly out of scope per current design, flag rather than recommend
- **Evidence**: `SYSTEM_DESIGN.md` "Scope and limitations": "server batching... [is] outside the current scope," and the server "serializes generation." This is a legitimate throughput lever (batched decode amortizes weight reads across multiple sequences) but is a substantial architecture change — the entire LFU expert-cache design assumes one active decode stream. **Flagging, not recommending**, since it's out of scope for "accelerate this engine" in its current single-stream form and would require its own design document.

### 4.3 [LOW impact, LOW risk] Verify `posix_madvise(MADV_DONTNEED)` reset cost isn't on the per-turn hot path unexpectedly
- **Evidence**: `KVCacheManager.swift`'s `reset()` calls `posix_madvise` per K/V buffer per layer at the end of a generation — correct for the 8GB-memory design goal (returning pages promptly), but confirm this isn't triggered mid-conversation in multi-turn server usage in a way that forces cold refaults on the next turn's early tokens (a latency, not throughput, concern). Worth a quick trace during the server's "single-prefix KV reuse" scenario already flagged as "inconclusive" in `QWEN36_PORT.md` item 5.
- **Risk**: none if confirmed turn-boundary-only, as the code comments suggest ("a finished generation does not keep its KV resident into the next turn").

---

## Priority-ordered execution list

1. **4.1** — Extend Qwen-specific profiling counters (prerequisite, no risk)
2. **2.1** — Expert-cache sizing/hit-rate study for 256-expert Qwen shape (highest-value unmeasured question)
3. ~~**1.2** — Expose `--verify trusted-install` in app/server~~ — done 2026-09-11; all four call sites default to `auto`
4. **1.1** — Prefill chunk-size sweep past 128
5. **3.2** — Int8 KV cache for the 10 full-attention layers only, with the stronger dual quality gate (EvalPlus + 4096 soak) — **promoted by the GPU split**: attention is the only part of a layer that grows with context (0.530 -> 1.281 ms/layer over 1.7K tokens on 3.6, +142%, while GDN moves +5%)
6. ~~**2.2** — Command-buffer coalescing in decode~~ — **refuted 2026-09-11**; the residual per-buffer overhead is 0.035-0.055 ms against a ~0.26 ms kernel dispatch, so there is no pool to reclaim. Do not re-open without a new mechanism.
7. **2.3** — GDN kernel fusion (gate into recurrent) — the target stack is now identified (82% of `gpu_cb1`, 1.718 ms/layer on 3.8); what is missing is the per-kernel split inside GDN, which is the prerequisite for pricing this item
8. **1.3** — Deeper prefill expert-prefetch pipelining
9. **3.4** — Router GEMV load-width audit
10. **1.4** — (folded into 4.1)
11. **3.5** — Expert group-size sweep — deprioritized, high blast radius for uncertain gain
12. **3.3** — GDN state quantization — not recommended
13. **2.4** — Speculative decoding — scoping spike only, sequence last

### Critical Files for Implementation
- `Sources/FinchMoE/Infrastructure/Streaming/PreadExpertStreamer.swift` - LFU expert cache plan/eviction logic to extend for item 2.1's slot-count study
- `Sources/FinchMoE/Runtime/Configuration/RuntimeConfiguration.swift` - central knob surface (`allowedExpertCacheSlots`, `allowedPrefillChunkTokens`) for items 1.1 and 2.1
- `Sources/FinchMoE/Runtime/KVCache/KVCacheManager.swift` - K/V storage/layout to modify for item 3.2's int8 full-attention KV cache
- `Sources/FinchMoE/Runtime/Inference/RealForwardRunner.swift` - decode/prefill command-buffer orchestration and GDN state wiring for items 2.2 and 2.3
- `Sources/FinchMoE/Metal/LinearAttn/gdn.metal` - GDN kernels (`gdn_gate`, `gdn_recurrent`, `gdn_rmsnorm_gated`) to fuse for item 2.3, and reference for why item 3.3 is high-risk
