# Experiment inventory

This is the curated experiment and decision index behind [Optimizing a 14.3 GB
model for an 8 GB machine](../OPTIMIZATION_JOURNEY.md). It records what shipped,
what worked only under a named condition, what failed, what a stronger gate
later reversed, and what the project never attempted.

Most readers should start with the article above. Use this inventory to find
the variants, controls, measurements, and final disposition of a specific
experiment. The entries are design archaeology, not an independent reproduction
package. Raw logs, traces, working plans, and project-state documents are not
part of this curated record.

The companion [implementation reference ledger](../IMPLEMENTATION_REFERENCES.md)
records the external model implementations, papers, kernels, and systems that
materially shaped the work, including sources that informed rejected paths.

## How to read the inventory

Each experiment has a stable ID and a detailed summary. The summary states the
hypothesis, variants, measured evidence, later reversal when applicable, final
disposition, and reusable lesson.

| Disposition | Meaning |
| --- | --- |
| Production | Survives in the default runtime or frozen M2 reference profile. |
| Conditional | Helps only on a named host, workload, flag, or memory budget. |
| Rejected | Measured and not promoted. |
| Correctness repair | Required for valid output, independent of speed. |
| Reversed rejection | A better gate invalidated an early rejection; the later candidate shipped. |
| Scope decision | Deliberately not attempted. |
| Unexecuted hypothesis | Plausible idea without a completed runtime gate. |

## Terms used in the summaries

Common abbreviations: **cb1/cb2** are the first and second command-buffer
phases; **TG** is a threadgroup; **SIMDgroup** is Metal's lockstep execution
group; **GEMV** is matrix-vector multiplication; **QMM** is quantized matrix
multiplication; **MPP** is Apple's Metal Performance Primitives tensor path;
**GQA** is grouped-query attention; **SWA** is sliding-window attention;
**PSO** is a pipeline state object; **TTFT** is time to first token; **RSS** is
resident set size; and **NLL** is negative log-likelihood. See
[System design](../SYSTEM_DESIGN.md) for how these pieces fit together.

## Summary package

| Area | Summary | Entries |
| --- | --- | ---: |
| Model installation and expert I/O | [Remote repack, `pread`, hints, compression, speculative I/O, and the 3.6-versus-3.8 read gap](summaries/01-model-install-and-expert-io.md) | 17 |
| Decode MoE, INT4, and router | [Persistent MoE, vectorization, router, and rejected geometries](summaries/02-decode-moe-int4-and-router.md) | 18 |
| Expert cache and layout | [Replacement policy, capacity, prediction, and disk layout](summaries/03-expert-cache-prediction-and-layout.md) | 9 |
| RDADVISE | [The complete short-win to long-context-rejection arc](summaries/04-rdadvise.md) | 7 |
| Attention and KV cache | [Split attention, MLX geometry, K4/V4, and FP16 ring](summaries/05-attention-and-kv-cache.md) | 14 |
| Prefill | [Chunking, MPP, routed MoE, overlap, attention, and allocation experiments](summaries/06-prefill.md) | 17 |
| Fusions and orchestration | [Targeted fusions, head variants, queues, and synchronization](summaries/07-fusions-head-and-orchestration.md) | 15 |
| Sampling and output | [Gumbel sampling, tokenizer caching, and detokenization](summaries/08-sampling-tokenization-and-output.md) | 4 |
| Validation methodology | [False rejections, holdouts, thermal state, and benchmark artifacts](summaries/09-validation-and-measurement-lessons.md) | 15 |
| **Total** | | **116** |

## All 116 experiments

### Model installation and expert I/O

| Experiment | Key evidence | Disposition |
| --- | --- | --- |
| [IO-01](summaries/01-model-install-and-expert-io.md#io-01) — `mmap` versus `pread` | Cold expert: 9.88 ms versus 2.79 ms; full simulation 0.50 versus 3.97 tok/s. | Production; parallel bounded `pread`. |
| [IO-02](summaries/01-model-install-and-expert-io.md#io-02) — Parallel reads and coalescing | About 1.13 to 2.08 to 2.13 tok/s through the two stages. | Production. |
| [IO-03](summaries/01-model-install-and-expert-io.md#io-03) — Darwin file hints | `F_RDAHEAD=0` and `F_NOCACHE` neutral; RDADVISE probe promising. | Rejected; RDADVISE gated separately. |
| [IO-04](summaries/01-model-install-and-expert-io.md#io-04) — Dedicated I/O executor | Best executor 8.59 ms versus existing 8.42 ms. | Rejected. |
| [IO-05](summaries/01-model-install-and-expert-io.md#io-05) — Custom I/O worker pool | Repeated four-worker pairs were mixed. | Rejected; normal parallel miss reads retained. |
| [IO-06](summaries/01-model-install-and-expert-io.md#io-06) — `mlock` | Recovered 0 ms; skipping streaming recovered 12-14 ms. | Rejected; contention, not eviction. |
| [IO-07](summaries/01-model-install-and-expert-io.md#io-07) — Expert compression | Zstandard about 10%; LZ4 0.06%. | Rejected. |
| [IO-08](summaries/01-model-install-and-expert-io.md#io-08) — Speculative reads | Probe reached 10.63 GB/s; decode fell 4.937 to 4.742 tok/s. | Rejected. |
| [IO-09](summaries/01-model-install-and-expert-io.md#io-09) — MTLIO | Warm 13.1-13.3 GB/s; only 5.4-7.5% of misses fully warm. | Rejected for runtime. |
| [IO-10](summaries/01-model-install-and-expert-io.md#io-10) — Remote streaming repack | 14,952,958,284 bytes, 229 ranges, 524,288-byte heap bounds. | Production. |
| [IO-11](summaries/01-model-install-and-expert-io.md#io-11) — Decomposing the read window | Fan-out plus drain is 0.29-0.80% of the `io_read_wall` window; 3.8 achieves 5.60 concurrent reads against 3.6's 2.96 and still gets half the throughput. | Rejected; counters kept, submission path closed. |
| [IO-12](summaries/01-model-install-and-expert-io.md#io-12) — Read request size | `FINCHMOE_IO_READ_SPLIT` 1/2/4: a fifth of a percent across a fourfold change in request count; `io_thread_wall` flat to 1%. | Rejected; knob kept off as falsifier. |
| [IO-13](summaries/01-model-install-and-expert-io.md#io-13) — Retracting the file claim | Five layers per install: medians 3.23 against 3.18 GB/s within a 2.81-3.86 range. The 1.37-1.60x gap was an artifact of picking `layer_00` on both. | Claim withdrawn; request-size refutation stands. |
| [IO-14](summaries/01-model-install-and-expert-io.md#io-14) — Capping the read depth | `FINCHMOE_IO_READ_WAVE` 0/2/3/4/6: the window moves 6% across a 2.9x depth range, toward the wider end, at a pinned 2.87-2.97 GB/s. Per-read latency scales linearly with depth. | Rejected; drive saturated at IO-11's own concurrency. |
| [IO-15](summaries/01-model-install-and-expert-io.md#io-15) — Moving the destination out of the Metal heap | 3.8 reads at 3.28 GB/s with staging off and 3.28 with it on, rounds agreeing to 0.3%; the copy separates the installs by 15% where the read separates them 3.3x. | Rejected; knob kept as measurement tool. |
| [IO-16](summaries/01-model-install-and-expert-io.md#io-16) — Reproducing the engine's batch structure offline | The harness reproduces the batch shape (133/249 reads per round against 134.8/245.7) and inverts the ordering (1.917-2.482 GB/s against the engine's 4.736/3.167). Barrier ~7%; fewer files is slower. | Rejected; harness kept out of the runtime. |
| [IO-17](summaries/01-model-install-and-expert-io.md#io-17) — Replaying the engine's own traces | The engine beats its own replay by 1.20-1.33x on 3.6 and loses to it by 1.31-1.43x on 3.8, so the engine flips the sign of a comparison the traces make 1.34-1.49x the other way. | Read side closed as "not the pattern"; deficit is in the 3.8 runtime. |
| [IO-18](summaries/01-model-install-and-expert-io.md#io-18) — Pricing the slot destination working set, and finding the replay must be run cold | The 1.98 GiB engine-shaped destination ring costs **+1.5 ms/step** in the engine's own condition (+19.3 had been measured with the cache bypassed), and the engine/replay ratio is 1.66x on every round. Inter-batch idle, memory pressure and the 3.8-only PLE stream are all negative. Correcting for cache state (eviction; 6.3x on 3.6 warm vs cold) puts 3.6's engine at **parity** with its own cold replay, leaving a 3.8-specific **1.78x per-read service time** (4.65 against 2.62 ms at equal depth) as the only deficit. | Destination exonerated in the engine's own condition; the deficit is inside the read itself on 3.8, and is now a per-request number. |
| [IO-19](summaries/01-model-install-and-expert-io.md#io-19) — The read window is a per-layer round-trip floor, and caching more of it changes nothing | Doubling `--expert-cache-slots` 16 → 32 cuts misses **29.8%** (4,179 → 2,934) and bytes/step 227.5 → 159.7 while `io_read_wall` goes 47.39 → 48.85 ms/step (**+3%, i.e. nothing**); across four prompt domains at fixed size the fewest-miss prompt has the *highest* window (61.13 ms) and the most the lowest (52.57). A ring-free control shows span *does* scale with read count when the drive is not involved (0.078/0.111/0.171 ms for 1/2/3.33 reads), so the flatness is a **~1.2 ms per-layer drive round trip** (40 x 1.2 = 47.4 ms on 3.6). Both predictor families priced: the trace is the **miss list**, the 16-slot LRU ring already takes **48.8% / 57.9%**, leaving the temporal predictor **1.00% / 1.84%** coverage (statistically independent across layers) and the static hot set **42.3% (3.6) / 31.5% (3.8)** out-of-sample at top-32, corroborated by the archived engine's 39.5%. | **Both prefetch families closed, and the CB1-overlap fallback with them** — coverage does not convert, so a satisfied 40% bar does not license the build. Pinning pays 2.11 GiB resident (5.94 GiB on 3.8, which the box cannot afford) for no speedup; the lever is fewer round trips, not fewer bytes. |
| [IO-20](summaries/01-model-install-and-expert-io.md#io-20) — The in-engine gap is a throughput deficit, not a latency shape, and the fast reads are not the page cache | A 34-bucket log2 histogram of per-read thread time (filled from the marks walk `threadNanos` already needs, printed as `io_read_latency_ms=p50/p90/p99/fast/n`, bucket **lower** edges) shows the 3.6→3.8 difference is a **uniform shift, not a tail**: every percentile moves one bucket and they move together (3.8 p50 = **2.10 ms in all four runs**; 3.6's p50 is 0.52 in three runs and 0.26 in one, so only the contrast is quotable). `fast=` — the share of reads under 0.262 ms, below anything a cold read on this volume can reach — is **not the page cache**: a 17 GiB eviction from a directory neither trace reads leaves it at 38.7% (3.6) and 11.4% (3.8), stable to 0.3 points over four rounds, removing only the wholly-cached round. Cold-corrected, the drive serves 3.8's larger-read stream **1.65x faster** than 3.6's (4.20 against 2.54 GB/s) and the engine turns that into a **1.46x deficit** (3.20 against 4.68) — a **2.4x swing**. GPU duty is eliminated (3.6 carries 126% GPU per unit read wall against 3.8's 60%, and 3.6 is the install that matches its replay), depth twice over (IO-14's engine sweep and a deeper engine still losing), and I retracted my own first reading of the table — a thread-time mean compared against a span-derived mean, which is off by the concurrency factor. | The surviving deficit is **throughput, in the engine, on 3.8** — no longer expressible as per-read latency. The replay's arms are harness-limited (conc 2.18/3.26 against pool widths 3/5), which strengthens the 3.8 claim and weakens the 3.6 one. One untested cell left: the drive writing a slot while the GPU reads the same slot (IO-15 eliminated the destination *kind*, perhaps with the GPU idle). |
| [IO-21](summaries/01-model-install-and-expert-io.md#io-21) — Adding the GPU to the replay costs the drive 12%, and the per-read median is not a stable statistic | IO-20's one open cell, tested by **inversion**: the engine cannot be asked (no trace playback, so nop'ing the routed readback changes routing and the read population with it), so a dose-matched GPU ring reader (`gpu_load.swift`: 648.8 MiB per 147 ms at 16.7 GB/s — bytes and burst length, not a rate, because 3.8's 16.7 GB/s is compute-bound) was run **concurrently with the offline replay**, whose trace is then fixed by construction. Five order-alternated rounds on 3.8: **alone 146.6-148.2 ms/step against loaded 165.0-165.2**, +17.7 ms/step (**+12.0%**), rate **4.62 → 4.12 GB/s**, per-read thread time 1.97 → 2.27 ms. 3.6 under its own smaller dose separates 6/6 the same way: 43.7 → 47.1 (+8.0%), 5.46 → 5.03 GB/s. Two nulls dissociate the mechanism: a CPU loop reading the same bytes with **more** bandwidth (4.37 GB/s) moves nothing, and `gpu_load --no-dispatch` — the Metal ring allocated and page-touched but never read — shows the **full fault storm** (78-79k `minflt`) with **no slowdown**, so the faults belong to the allocation and the cost to the GPU reading. Worth **~14%** of the engine's gap (0.30 of 2.16 ms per read) as a **lower bound** — the generator models only the routed ring reads, not `gpu_cb1`. Unplanned: the replay's per-read distribution is **bimodal** (bulk at 0.13-0.52 ms, another at 2.1-8.4, median in the 4.1%-full valley), so p50 hops a full bucket on a few points of mixture while the mean moves 3% — `--hist-dump` added. | **IO-20's GPU-contention elimination is withdrawn.** ~86% of the gap remains, and the dose did not scale between 648.8 and 1297.6 MiB/period (suggestive of a per-activation rather than per-byte cost). |

### Decode MoE, INT4, and router


| Experiment | Key evidence | Disposition |
| --- | --- | --- |
| [DEC-01](summaries/02-decode-moe-int4-and-router.md#dec-01) — One SIMDgroup per row | Bundled row-SIMD and wider-MoE-TG stage: 2.131 to 2.188 tok/s. | Production; contribution not isolated. |
| [DEC-02](summaries/02-decode-moe-int4-and-router.md#dec-02) — Cooperative MoE row | cb2 about 230 to 527 ms. | Rejected. |
| [DEC-03](summaries/02-decode-moe-int4-and-router.md#dec-03) — Persistent multi-TG MoE | cb2 239 to 60 ms; 2.188 to 3.313 tok/s. | Production. |
| [DEC-04](summaries/02-decode-moe-int4-and-router.md#dec-04) — Initial 16-slot cache | I/O 166 to 88 ms; 3.313 to 4.261 tok/s. | Production; 16-slot capacity. |
| [DEC-05](summaries/02-decode-moe-int4-and-router.md#dec-05) — Shared-MLP/I/O overlap | cb2 62 to 48 ms; 4.404 to 4.736 tok/s. | Production. |
| [DEC-06](summaries/02-decode-moe-int4-and-router.md#dec-06) — MoE INT4 vectorization | Routed GPU 36.5 to 31.3 ms. | Production. |
| [DEC-07](summaries/02-decode-moe-int4-and-router.md#dec-07) — Standalone INT4 vectorization | Unsafe `uint` failed live offsets; corrected head 21.5 to 16 ms. | Production; corrected live-offset path. |
| [DEC-08](summaries/02-decode-moe-int4-and-router.md#dec-08) — Multi-TG router | cb1 about 50 to 44 ms. | Production. |
| [DEC-09](summaries/02-decode-moe-int4-and-router.md#dec-09) — Parallel top-k selector | 36.94 versus 18.86 microseconds. | Rejected. |
| [DEC-10](summaries/02-decode-moe-int4-and-router.md#dec-10) — Gate/up fusion | 5.378 versus 5.266 tok/s; eight rows won. | Production. |
| [DEC-11](summaries/02-decode-moe-int4-and-router.md#dec-11) — Phase 2 plus reduce | About 19 microseconds saved; intermediate removed. | Production. |
| [DEC-12](summaries/02-decode-moe-int4-and-router.md#dec-12) — Phase-1 paired rows | 207-212 versus 194-196 microseconds. | Rejected. |
| [DEC-13](summaries/02-decode-moe-int4-and-router.md#dec-13) — Phase-1 `u16` loads | 5.961 to 5.973 tok/s. | Production; small win. |
| [DEC-14](summaries/02-decode-moe-int4-and-router.md#dec-14) — Phase-2 d2/d2p/d4 | Mixed end to end; target only 5.70 ms/token. | Rejected; default unchanged. |
| [DEC-15](summaries/02-decode-moe-int4-and-router.md#dec-15) — Group-sum reuse | An early 454.05 to 420.53 microsecond result reversed to 309.98 versus 348.83 in the wrapper. | Rejected. |
| [DEC-16](summaries/02-decode-moe-int4-and-router.md#dec-16) — Adaptive geometry | One synthetic state won; real rows mixed. | Rejected. |
| [DEC-17](summaries/02-decode-moe-int4-and-router.md#dec-17) — Progressive execution | Corrected 1K/256 fell 4.799 to 4.648 tok/s. | Rejected. |
| [DEC-18](summaries/02-decode-moe-int4-and-router.md#dec-18) — Hit-first split | 5.169 versus 4.518 tok/s with forced identical IDs. | Production. |

### Expert cache, prediction, and layout

| Experiment | Key evidence | Disposition |
| --- | --- | --- |
| [CACHE-01](summaries/03-expert-cache-prediction-and-layout.md#cache-01) — LRU to LFU | I/O 72.6 to 64.8 ms; 5.476 to 5.631 tok/s. | Production. |
| [CACHE-02](summaries/03-expert-cache-prediction-and-layout.md#cache-02) — Windowed LFU | Slightly fewer misses; two real-decode prompts neutral or mixed. | Rejected. |
| [CACHE-03](summaries/03-expert-cache-prediction-and-layout.md#cache-03) — 24/32 slots | Higher replay hit rate for about +769 MiB/+1.54 GiB; live 32-slot run overloaded the host. | Conditional; additional memory. |
| [CACHE-04](summaries/03-expert-cache-prediction-and-layout.md#cache-04) — Heterogeneous allocation | Exact DP lost on holdout. | Rejected. |
| [CACHE-05](summaries/03-expert-cache-prediction-and-layout.md#cache-05) — Cross-layer predictor | Jaccard 0.039; copy hit 7%. | Rejected; pre-runtime evidence. |
| [CACHE-06](summaries/03-expert-cache-prediction-and-layout.md#cache-06) — Markov prefetch | Flat/negative; later row collapsed and was not correctness-safe. | Rejected. |
| [CACHE-07](summaries/03-expert-cache-prediction-and-layout.md#cache-07) — Offset read order | First win failed to repeat. | Rejected. |
| [CACHE-08](summaries/03-expert-cache-prediction-and-layout.md#cache-08) — Packed layout | Natural text +3.61%; near-4K decode -16.1% and prefill worse. | Rejected. |
| [CACHE-09](summaries/03-expert-cache-prediction-and-layout.md#cache-09) — APFS preallocation | Fragmentation observed; candidate not built. | Unexecuted hypothesis. |

### RDADVISE

| Experiment | Key evidence | Disposition |
| --- | --- | --- |
| [RAD-01](summaries/04-rdadvise.md#rad-01) — Initial miss-only signal | Plausible first real-decode improvement. | Conditional; preliminary signal. |
| [RAD-02](summaries/04-rdadvise.md#rad-02) — Repeated short gate | 5.176 to 5.449 tok/s; I/O 87.4 to 72.2 ms. | Conditional; temporary promotion later removed. |
| [RAD-03](summaries/04-rdadvise.md#rad-03) — Advice cap | Fewer advice calls, more demand I/O. | Rejected. |
| [RAD-04](summaries/04-rdadvise.md#rad-04) — Advice in I/O path | 6.432 to 6.093 tok/s. | Rejected. |
| [RAD-05](summaries/04-rdadvise.md#rad-05) — Long-context policy | Unbounded 2.334; bounded 4.966 tok/s, but bounded lost at 512. | Rejected; static default policy. |
| [RAD-06](summaries/04-rdadvise.md#rad-06) — Adaptive policy | Some state wins, no stable rule. | Rejected; opt-in retained. |
| [RAD-07](summaries/04-rdadvise.md#rad-07) — Async overlap | 4.863 to 5.300 tok/s on 1K/256. | Conditional; production off. |

### Attention and KV cache

| Experiment | Key evidence | Disposition |
| --- | --- | --- |
| [KV-01](summaries/05-attention-and-kv-cache.md#kv-01) — Split-KV attention | SWA about 3.3x; full 4K about 4.1x isolated. | Production. |
| [KV-02](summaries/05-attention-and-kv-cache.md#kv-02) — GQA-aware SWA | Isolated SWA win. | Production. |
| [KV-03](summaries/05-attention-and-kv-cache.md#kv-03) — Full-GQA A3 | +21.2% at 1024, -54.6% at 4096. | Rejected. |
| [KV-04](summaries/05-attention-and-kv-cache.md#kv-04) — A4 local variants | Three 1024 wins became neutral/slower at 4096; TG widths lost. | Rejected. |
| [KV-05](summaries/05-attention-and-kv-cache.md#kv-05) — MLX geometry v1 | 65-68% isolated win, then forced-prefix failure and removal. | Rejected; exact split remains production. |
| [KV-06](summaries/05-attention-and-kv-cache.md#kv-06) — MLX geometry v2 | 71-74% isolated and +1.30-1.59% on the earlier path; the instruction-checkpoint quality gate failed. | Rejected for the instruction checkpoint; exact split is production. |
| [KV-07](summaries/05-attention-and-kv-cache.md#kv-07) — Packed K4/V4 attention | Split beat packed single-pass but stayed slower than FP16. | Rejected and removed. |
| [KV-08](summaries/05-attention-and-kv-cache.md#kv-08) — Alternative codecs | Faster writers, much slower attention. | Rejected. |
| [KV-09](summaries/05-attention-and-kv-cache.md#kv-09) — K4/V4 quality | Mean delta-NLL +0.015197; top-1 -5.0781 points. | Rejected and removed; FP16 remains production. |
| [KV-10](summaries/05-attention-and-kv-cache.md#kv-10) — FP16 full-attention island | Small sample looked better; 256-row top-k gates failed. | Rejected. |
| [KV-11](summaries/05-attention-and-kv-cache.md#kv-11) — Packed chunk 32 | 5.42-6.30% isolated win; holdout quality failed. | Rejected. |
| [KV-12](summaries/05-attention-and-kv-cache.md#kv-12) — FP16 KV ring | About 575-591 MiB saved; speed neutral/mixed; parity retained. | Production. |
| [KV-13](summaries/05-attention-and-kv-cache.md#kv-13) — Ring kernel follow-up | Terminal 10.9% isolated median implied only about 0.34% whole-step opportunity. | Rejected; below action gate. |
| [KV-14](summaries/05-attention-and-kv-cache.md#kv-14) — Prefill attention race | Third barrier fixed one bank but cost 5.1-14.2%; two banks recovered 2.23-6.43%. | Correctness repair. |

### Prefill

| Experiment | Key evidence | Disposition |
| --- | --- | --- |
| [PF-01](summaries/06-prefill.md#pf-01) — Mixed QMM/GEMV | QMM lost for Q, helped selected projection families. | Production; shape policy. |
| [PF-02](summaries/06-prefill.md#pf-02) — Chunk 128 | 121: 15.80 to 9.34 s; 1017: 92.89 to 52.35 s. | Production. |
| [PF-03](summaries/06-prefill.md#pf-03) — Deeper lookahead | 527: 26.61 to 33.38 s; 1017: 52.35 to 73.88 s. | Rejected. |
| [PF-04](summaries/06-prefill.md#pf-04) — Routed SIMD | 1409.666 to 4618.845 microseconds. | Rejected. |
| [PF-05](summaries/06-prefill.md#pf-05) — Shared INT8 QMM | Medium wins; 1017 regressed 45.59 to 52.01 s. | Rejected on M2. |
| [PF-06](summaries/06-prefill.md#pf-06) — Metadata reduction | Allocation counts cut by two-thirds. | Production; allocation hygiene. |
| [PF-07](summaries/06-prefill.md#pf-07) — Argument-buffer reuse v1 | Runtime/lifetime gate failed. | Rejected. |
| [PF-08](summaries/06-prefill.md#pf-08) — Argument-buffer rings | 21,217 allocations to two; long M2 about 9% slower. | Rejected. |
| [PF-09](summaries/06-prefill.md#pf-09) — Shared/fetch overlap M2 | Short/mid wins; long regression. | Rejected; M2. |
| [PF-10](summaries/06-prefill.md#pf-10) — Shared/fetch overlap M5 | Stable about 2.2-2.3% at 527/1017; M2 was order-sensitive and regressed long prefill. | Rejected and removed. |
| [PF-11](summaries/06-prefill.md#pf-11) — Trusted receipt | Avoided 6.741 s hashing in one row; changes trust/cache state. | Conditional; explicit trust policy. |
| [PF-12](summaries/06-prefill.md#pf-12) — Staged affine MPP | M128 about 73.8%; 512 prefill +11.421%; quality passed. | Reversed rejection; production. |
| [PF-13](summaries/06-prefill.md#pf-13) — Direct UInt4 | Beat retired path, lost to staged MPP by 20.40% at M128. | Rejected. |
| [PF-14](summaries/06-prefill.md#pf-14) — QMM TG reuse | Families +3.2-9.7%; current opportunity about 0.41%. | Rejected. |
| [PF-15](summaries/06-prefill.md#pf-15) — Batched routed MoE | Isolated +30.91%; balanced end to end about +2%. | Reversed rejection; production. |
| [PF-16](summaries/06-prefill.md#pf-16) — Long endpoint gate | Delta-NLL +0.002588; top-1 16/16; RSS 888.3 MiB. | Production; validation result. |
| [PF-17](summaries/06-prefill.md#pf-17) — Apple10 TensorOps full attention | Isolated 11.24x at 16K and 11.63x at 64K; 32K end to end 2.404x. | Production on Apple10; tiled fallback elsewhere. |

### Fusions, head, and orchestration

| Experiment | Key evidence | Disposition |
| --- | --- | --- |
| [ORCH-01](summaries/07-fusions-head-and-orchestration.md#orch-01) — QKV epilogue | Joint epilogue-plus-tail rollback gained about +0.076 tok/s/-2.9 ms. | Production; contribution not isolated. |
| [ORCH-02](summaries/07-fusions-head-and-orchestration.md#orch-02) — Layer tail | Parity and paired dispatch win. | Production. |
| [ORCH-03](summaries/07-fusions-head-and-orchestration.md#orch-03) — Monolithic middle fusion | 1.811 versus 2.756 tok/s. | Rejected. |
| [ORCH-04](summaries/07-fusions-head-and-orchestration.md#orch-04) — Fused QKV GEMV | Initial rejection reversed; rollback measured 6.215 versus 6.110 tok/s. | Reversed rejection; production. |
| [ORCH-05](summaries/07-fusions-head-and-orchestration.md#orch-05) — Fusion D head family | Row head won short/long gates; tiled GPU improved while end to end fell 5.962 to 5.926. | Production; row head only. |
| [ORCH-06](summaries/07-fusions-head-and-orchestration.md#orch-06) — Shader sweep | Eleven families; first-run wins disappeared. | Rejected; sweep. |
| [ORCH-07](summaries/07-fusions-head-and-orchestration.md#orch-07) — RoPE tables | Neutral/slower. | Rejected. |
| [ORCH-08](summaries/07-fusions-head-and-orchestration.md#orch-08) — LM-head width/tiles | Small local movement, no whole-step win. | Rejected. |
| [ORCH-09](summaries/07-fusions-head-and-orchestration.md#orch-09) — Single-wait pipeline | Rollback 6.416 to 5.545 tok/s. | Production. |
| [ORCH-10](summaries/07-fusions-head-and-orchestration.md#orch-10) — Second Metal queue | Slower at about 5.059 tok/s. | Rejected. |
| [ORCH-11](summaries/07-fusions-head-and-orchestration.md#orch-11) — O3 token chain | Short favored candidate; long favored rollback. | Rejected. |
| [ORCH-12](summaries/07-fusions-head-and-orchestration.md#orch-12) — CB1 encode-ahead | Smoke +; 1K/256 4.613 versus 4.801. | Rejected. |
| [ORCH-13](summaries/07-fusions-head-and-orchestration.md#orch-13) — Decode arg-buffer reuse | 4.754 to 3.998 tok/s plus divergence. | Rejected. |
| [ORCH-14](summaries/07-fusions-head-and-orchestration.md#orch-14) — All-hit fast path | Only 2.761 ms total sync across 351 states. | Rejected. |
| [ORCH-15](summaries/07-fusions-head-and-orchestration.md#orch-15) — Merge shared/hit CB | Fewer submissions; 4.013 to 3.707 tok/s. | Rejected. |

### Sampling, tokenization, and output

| Experiment | Key evidence | Disposition |
| --- | --- | --- |
| [OUT-01](summaries/08-sampling-tokenization-and-output.md#out-01) — One-pass sampling and Top-64 | 0.377 to about 5.86-5.89 tok/s; later Top-64 route 0.733 ms/sample. | Production. |
| [OUT-02](summaries/08-sampling-tokenization-and-output.md#out-02) — Fused Gumbel head | 5.124 versus 5.192 tok/s standalone. | Conditional; default off. |
| [OUT-03](summaries/08-sampling-tokenization-and-output.md#out-03) — Tokenizer process cache | Suite body 29.002 to 4.425 s; command wall 36.54 to 6.67 s; RSS 3.86 GB to 397 MB. | Production; support path. |
| [OUT-04](summaries/08-sampling-tokenization-and-output.md#out-04) — Bounded detokenizer tail | ASCII and byte fallback neutral; mixed Unicode 17.51% faster. | Production; allocation hygiene. |

### Validation and measurement

| ID | Lesson | Consequence |
| --- | --- | --- |
| [METH-01](summaries/09-validation-and-measurement-lessons.md#meth-01) | Reordered floating-point work needs a distributional quality oracle. | Corrected false rejections of MLX attention, staged MPP, batched MoE, and TensorOps prefill attention. |
| [METH-02](summaries/09-validation-and-measurement-lessons.md#meth-02) | Claimed-lossless work still needs exact identity. | Retained strict gates for cache, storage, and load-width changes. |
| [METH-03](summaries/09-validation-and-measurement-lessons.md#meth-03) | Test production-shaped offsets. | Found the live 2-byte INT4 alignment defect. |
| [METH-04](summaries/09-validation-and-measurement-lessons.md#meth-04) | Prove synchronization scope. | Repaired one reused scratch bank; audited other paths by ownership and queue order. |
| [METH-05](summaries/09-validation-and-measurement-lessons.md#meth-05) | Profiler tok/s is diagnostic. | Use GPU spans for attribution and clean rows for promotion. |
| [METH-06](summaries/09-validation-and-measurement-lessons.md#meth-06) | Interleave warm, thermally balanced rounds. | Removed false first-run shader wins. |
| [METH-07](summaries/09-validation-and-measurement-lessons.md#meth-07) | Mechanism counts are not outcomes. | Allocation and submission reductions still lost end to end. |
| [METH-08](summaries/09-validation-and-measurement-lessons.md#meth-08) | Trace-trained policies need holdouts. | Rejected packed layout and heterogeneous cache allocation. |
| [METH-09](summaries/09-validation-and-measurement-lessons.md#meth-09) | Detect greedy repetition loops. | Reclassified apparent cache decay as period-44 cyclic thrash. |
| [METH-10](summaries/09-validation-and-measurement-lessons.md#meth-10) | A bucket named for a kernel prices the dispatch around it. | Qwen `cb1` split and the slot sweep: the counters moved, end-to-end did not. Kernel claims need GPU spans. |
| [METH-11](summaries/09-validation-and-measurement-lessons.md#meth-11) | A GPU span does not license the subtraction you build from it. | Device timestamps landed and identified the GDN stack at 1.718 ms/layer against attention's 1.127; the `wait` subtraction built on them read 46.8 ms/step of dispatch tax that does not exist. |
| [METH-12](summaries/09-validation-and-measurement-lessons.md#meth-12) | An awaited I/O window is a latency measurement, not a bandwidth one. | Six slot-sweep runs in both orders: 32 -> 16 slots cuts io time 12.9%/7.8% while reading 34% more bytes. Direct drive probes then priced both candidate mechanisms: bandwidth saturates by depth 4-8 (~5.7 GB/s) and declines past it, while a repeated 24-expert pool runs 2-3x the rate of diverse offsets. The queue is at its knee and repetition is what pays; prefetch is not built. |
| [METH-13](summaries/09-validation-and-measurement-lessons.md#meth-13) | A near-complete serial sum bounds the overlap. | Re-reading seven captured runs: `io` plus all device time plus head and PLE is 93.0-97.8% of the token, so the pipeline's claimed read/compute overlap is a few percent at most. The mechanism is documented in the code and five times too small to matter; the read order is forced by the router readback. Device work pays 1:1, and ~84 ms/step of the read window is fixed per-batch cost rather than transfer. |
| [METH-14](summaries/09-validation-and-measurement-lessons.md#meth-14) | A replay is only as faithful as the reuse it cannot see. | The engine's exact 9,083-pread sequence replayed offline lands at 130.3-177.9 ms/step against the engine's 225.4, below both numbers it was built to choose between. The one large offline win -- bypassing the page cache, 42-61 ms/step -- inverts on the engine to -12% tok/s, because 41.7% of the reads repeat a (layer, expert) pair at a median reuse distance of 3.97 GiB that ten recycling 26 MB destination buffers were holding and the engine's cache cannot. |
| [METH-15](summaries/09-validation-and-measurement-lessons.md#meth-15) | A rate is only comparable inside the session that measured it. | One identical synthetic cell gave 1.825 and 3.163 GB/s twenty minutes apart while the engine agrees with itself to 0.3% inside a run; across sessions the same engine figure spans 3.60-6.90 GB/s. The drift is wider than the 1.57x gap the read-gap investigation exists to explain, so its published engine-vs-replay table is not a subtraction -- which is how a named discriminator was spent refuting an artifact. Every compared arm is now paired and interleaved inside one session, with a reverse-order run. |

## Important non-experiments

These boundaries are not failed performance work:

- ANE/Core ML offload was excluded by architecture and platform scope.
- Weights below INT4 were cancelled by the quality floor.
- Draft-model speculative decoding did not reach a complete runtime candidate.
- Production routing was never cache-conditional.
- Fresh APFS preallocation remains unexecuted.
- The iPhone port is deferred; these are Mac measurements.

## How these summaries were prepared

Later revalidation overrides stale intermediate conclusions. These summaries
keep the evidence and final disposition readable without reproducing every raw
capture.

[Optimization journey](../OPTIMIZATION_JOURNEY.md) |
[Benchmarks](../../README.md#at-a-glance) |
[System design](../SYSTEM_DESIGN.md) |
[Implementation references](../IMPLEMENTATION_REFERENCES.md)
