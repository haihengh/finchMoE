# Prefill and decode speedup plan

Two targets, named the way the counters name them: **prefill** (prompt
processing, the `scope=prefill` counters line) and **decode** (token generation,
`tok/s`).

Narrow by design. [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md) already holds
~20 items with measured verdicts and
[`experiments/EXPERIMENT_INVENTORY.md`](experiments/EXPERIMENT_INVENTORY.md) all
116 experiments; most obvious levers in this runtime have been tested and closed.
Nothing here re-opens a closed item, and every claim that a lever is still open
carries the citation that shows it has not been tried.

---

## 1. Baseline (16 GB M4 Mac mini, release CLI, greedy, warm page cache)

| install | prefill | decode | source |
| --- | ---: | ---: | --- |
| Qwen 3.6 35B-A3B, ~19 GB `.finch` | ~42 tok/s (2,940-token prompt, 69.3 s) | ~8.4 tok/s | README "At a glance" |
| Qwen 3.8 Flash-Next 125B, quantized-PLE | ~19 tok/s (157.7 s) | ~2.8 tok/s | same |

Where a decode token goes — Qwen 3.6, 54-token prompt, 16 slots, ms/step
(`OPTIMIZATION_PLAN.md` §2.1, §4.1):

| term | ms/step | share |
| --- | ---: | ---: |
| `io` (routed-expert reads, awaited) | 45.2 | **42%** |
| `gpu_cb1` (attention + GDN device time) | 33.7 | **31%** |
| `gpu_routed` (routed + shared MoE device time) | 19.4 | 18% |
| head | 4.2 | 4% |
| **accounted** | **102.5 of 107.4** | **95%** |

Inside `gpu_cb1`: **84% GDN, 16% full attention** on 3.6 (28.52 vs 5.30 ms/step).
GDN is *flat* with context (0.951 → 0.999 ms/layer from 54 to 1,791 tokens) while
attention grows 2.4x over the same span. On the 2,940-token run the split is
`gpu_cb1` 16.82 (attn 6.01 / GDN 10.81) with `io` 22.30, `wait` 23.89, head 1.32,
`gpu_routed` 7.72 ms/step
(`benchmark-results/qwen36-35b-20260914-112931/long-synthesis.stderr:3`).

Where prefill goes — Qwen 3.8, 426 tokens, chunk 512 (and **3.8 only**: the 3.6
prefill body has no GPU attribution at all, see D1): **GDN 44%, expert I/O 36%**
(10.68 s of 29.65 s at 3.13 GB/s, the same rate the decode stream sustains),
routed MoE 21%, full attention 3%. GPU plus I/O exceeds the wall time, i.e. the
prefill is essentially serialized (`OPTIMIZATION_PLAN.md` §1.4).

Three levers follow: **how much prompt is prefilled at all**, the **expert read
path** (42% of decode, 36% of prefill), and the **GDN projection stack** (82% of
decode's device time, ~44%×95% of prefill).

---

## 2. Track A — prefilling less (highest value)

### A1 [DO THIS FIRST] Give the Mac app the prefix reuse the server already has

- **What the app does today**: every chat turn sends the whole conversation, and
  `RealInferenceClient.run` calls `runner.reset()` before prefill
  (`Sources/FinchMoEApp/Core/Inference/RealInferenceClient.swift:365`). So the
  entire history is re-prefilled on every message. Turn *n* prefills ~*n*
  messages' worth of tokens: **conversation prompt processing is O(n²)**, and on
  3.6 at 42 tok/s a 2,500-token conversation costs **~60 s of prefill per turn**.
  This cost was introduced by chat history, which is new.
- **The mechanism already exists and is shipped elsewhere.**
  `RawCompletion.Start.resume(cachedPromptTokens:)` with
  `ContinuableLogitProducer.prepareForContinuation(expectedPosition:)` skips the
  computed prefix (`Sources/FinchMoE/Runtime/Generation/RawCompletion.swift:163-200`),
  and the HTTP server implements the whole thing in `ServerPromptCache`
  (`single-prefix` mode, `Sources/FinchMoEServer/Core/ServerPromptCache.swift`),
  including the non-obvious part: the assistant's reply is in the KV *by
  construction* because decode wrote it, and the entry carries
  `kvBackedTokenIDs` / `uncommittedBoundaryTokenIDs` / `kvPosition` so the next
  turn resumes exactly.
- **Action**: hold the same cache in the long-lived decode service and let
  `DecodeServiceInferenceClient` resume. Smallest useful version: the service
  keeps the last request's `(domain, tokenIDs, kvPosition)`; when the new
  request's token IDs strictly extend it, prefill only the suffix and report
  `cachedPromptTokens` on the prefill event, so the HUD can read
  `Prefill (37/2,540, 2,503 cached)`.
- **Invalidation** — each one a test: model unload/reload, any load-time setting
  change, session switch, an edit to an earlier message, *Regenerate* (which
  deliberately removes the last exchange), cancellation mid-generation, and
  context trimming.
- **Stability requirement, and it is ours to keep**: the app trims history from
  the oldest exchange when the prompt outgrows the window
  (`RealInferenceClient.trimmedHistory`). If the trim boundary moves by one turn
  per message, the prefix changes every turn and the cache never hits. Trim in
  **fixed-size steps** (drop two turns at a time, only when the prompt no longer
  fits) and the prefix stays stable between trims.
- **Risk**: a resumed state that is subtly wrong degrades chat *silently*. This
  engine also has a known long-context non-reproducibility (one mechanism fixed
  — the `encodeQKPost` double-offset — and one intermittent dense-attention
  divergence still unexplained; `OPTIMIZATION_PLAN.md` item 16). Gate in three
  steps: token-identical output between resumed and full-prefill for the same
  conversation, then EvalPlus, then the 4,096-token quote-exact soak.
- **Measure**: TTFT per turn and prompt tokens actually computed per turn
  (`scope=prefill` counters plus the new `cached=` field).
- **Expected**: per-turn prompt processing O(conversation) → O(new message), a
  10–50x reduction across a chat. Decode untouched. This is the largest
  user-visible latency win available in this plan, and the cheapest.

### A2 [TRIVIAL] App context default

The saved settings here carry `contextTokens: 65536` — 1.34 GiB of FP16 KV
preallocated for 3.6 (1.50 GiB for 3.8, plus ~0.25 GB of QSA indexer), paid even
for a ten-token prompt, because KV is sized at `maxContext` and `advance` traps
rather than grows (`OPTIMIZATION_PLAN.md` §3.1b). With A1 a long context is cheap
to *use*; without it, it multiplies per-turn prefill. Make the app default 8K.

---

## 3. Track B — the expert read path

**The read side is closed, except for one measured anomaly.** Rejected with data,
do not resubmit: `mmap` (IO-01); dedicated I/O executor (IO-04, 8.59 vs 8.42 ms);
custom worker pool (IO-05); `mlock` (IO-06); compression (IO-07); speculative
reads (IO-08); MTLIO (IO-09); request size (`READ_SPLIT`, IO-12); read depth
(`READ_WAVE`, IO-14); staging vs slot destination (IO-15); `F_NOCACHE`
(engine −12% tok/s); RDADVISE by default; offset-sorted reads; both decode
prefetch families (IO-19 — temporal coverage 1.00%/1.84%, static hot set
42.3%/31.5%, pinning 2.11/5.94 GiB for no gain); cache sizing 16→32 slots
(−30% bytes, +3% wall time).

### B1 [BOUNDED RESEARCH] The 3.8 read-throughput deficit

The engine sustains **3.20 GB/s** on 3.8's cold stream where a bare process
sustains **4.20**, and the same engine does **4.68** on 3.6 (IO-20). Contention
explains ~14% (IO-21) and, per IO-22, is **generic rather than GPU-specific** —
a four-core `yes` with no memory traffic costs a third of what a dose-matched
GPU load costs — which reads as scheduler pressure rather than bytes. **~68% is
unexplained**, and this is the largest single quantity open in the codebase: it
is 42% of decode *and* 36% of prefill, on the larger model.

Two cheap things before any engineering:

- **B1a — remove the known confound.** Spotlight indexing is live on the model
  volume (`mdbulkimport`, `mds_stores`) and the record leaves it unexcluded as a
  contributor to drift (one identical synthetic cell moved 1.8 → 3.2 GB/s in
  twenty minutes). Exclude the volume, re-run the paired engine-vs-replay A/B,
  and see whether 3.20/4.20 survives. If it does not, most of Track B is noise
  management.
- **B1b — the one structural cell never tested.** IO-20 names it: the drive
  writing a slot while the GPU reads *that same slot*. IO-21 priced GPU reads of
  slot pages at +12% with the GPU otherwise idle; the engine's real case is the
  slot being both rewritten and read. Verify the engine cannot read a slot under
  an in-flight write into it (per-layer ring partitioning if it can).

If those two do not move it, close B1 as a hardware/environment ceiling and spend
the budget in Track C — that is what the record supports, and the previous ~20
negative results in this area are the reason to bound it now rather than later.

### B2 [FREE FOR 3.6, THEN CHEAP] The prefill chunk size is set for the wrong install

A layer's routed-expert pool is re-read **once per chunk**
(`Sources/FinchMoE/Runtime/Prefill/PrefillRuntimeConfig.swift:183-200`), so chunk
size is literally the prefill I/O divisor. At chunk 512 a 2,940-token prompt is 6
chunks; with 4,096 route draws over 256 experts the pool is essentially fully
touched per layer-chunk, so the volume is

    6 chunks × 40 layers × 256 experts × 1.6875 MiB ≈ 96.6 GiB (103.7 GB)

and **51.8 GB at chunk 1024** — half. At the read rate the project documents
(6.12 GB/s, `PreadExpertStreamer.swift:100-101`) that is ~16.9 s → ~8.5 s, and
because prefill is not overlapped (PF-18: GPU + I/O sums to *more* than the wall)
the saving is nearly additive: **+7–15% on the 69.3 s run**.

Two things make this the cheapest real win in the plan:

- **1024 is already legal** (`allowedPrefillChunkTokens = [32 … 1024]`,
  `RuntimeConfiguration.swift:31`), so the first step is
  `--prefill-chunk-tokens 1024` with **no code change**.
- The default is 512 because it was chosen on the **125B's** scratch curve. The
  scratch is 154 KiB per chunk-token (`PrefillChunkScratch.swift:213-256`), i.e.
  +75 MiB going 512 → 1024 against a ~1.1 GiB resident budget. On this install
  that is not a real constraint.

Second step: raise the ceiling so 2048 can be swept too (a 2,940-token prompt is
then 2 chunks, ~34.5 GB).

**Measure**: `FQ_PREFILL_COUNTERS=1` prints `bytes=` and `MB_per_chunk`
(`Sources/FinchMoECLI/Run.swift:222-237`) — expect ≈96.6 GiB / ≈16,470 MiB per
chunk at 512 and exactly half at 1024; cross-check `io_read_wall_ms` on the
`scope=prefill` line. **Falsified** if `bytes` does not halve (that would mean
re-reads inside a layer — a slot-eviction bug) or if the byte term is a small
fraction of prefill in both arms.

### B3 [CHEAP, NEVER MEASURED] Expert-cache slots below 16

The sweeps started at 16 (`OPTIMIZATION_PLAN.md` §2.1: 16/24/32). **8 is legal
for 3.6** (top-k 8; `allowedExpertCacheSlots = [8, 16, 24, 32]`, and a count
below top-k is refused), and the measured mechanism — repetition pays twice
through the SSD's own cache and the buffer cache, while a wide ring spreads the
working set and leaves many layers with one or two misses at the shallow end of
the depth curve — predicts 8 may beat 16 on 3.6 the same way 16 beat 32. One
config sweep, no code, byte-identical output expected.

---

## 4. Track C — decode device work (the 31% that is not I/O)

Shapes and file references below come from a decode-path audit against the
shipping 3.6 install, whose linear-attention weights are int8
(`models/Qwen3.6-35B-A3B-4bit.finch/manifest.json` →
`linearAttention.weightBits: 8`).

### C1 [HIGH, NEW] Full attention pays two threadgroup barriers per key position

- **Now**: the full-attention decode branch is split-KV flash decoding
  (`Kernels/Attention/Attention.swift:331-414`) over `numQHeads × numChunks`
  threadgroups with `numChunks = min(16, seqLen)`. The kernel's key loop is **one
  position per iteration** and calls `block_reduce_sum`
  (`Sources/FinchMoE/Metal/Attention/attention.metal:95-111, 189-214`), which
  contains **two** `threadgroup_barrier` calls — so a 2,940-position layer pays
  ~184 iterations × 2 barriers per threadgroup, each with a full K and V load
  serialized around it.
- **Why it costs time**: a barrier flushes the memory pipeline, so the next
  position's K/V load cannot be issued across it and every position exposes DRAM
  latency twice. Measured per layer: 0.530 ms at 54 tokens → 1.281 ms at 1,791,
  i.e. ~0.43 µs per position (~600 cycles for a 256-element dot product plus a
  softmax update) — latency-bound. Raising the split factor cannot help, because
  the cost is per position rather than per chunk; the q/k/v/o GEMVs in the same
  stack are only ~1.7 ms/step of it.
- **Change**: tile the key loop — load T (8–32) consecutive K and V rows
  cooperatively into registers, compute T partial dots per lane, and take **one**
  online-softmax update per tile, dividing the barrier count by T and issuing T
  independent loads per lane. Keep split-KV pass 2 so the byte-identical merge
  property documented at `attention.metal:127-131` survives, and route
  hd=256/nq=16/nkv=2 through a specialized PSO — the table at
  `Attention.swift:490-499` special-cases only hd=512/nkv=2, so Qwen's
  full-attention partials currently run the generic kernel.
- **Counter**: `gpu_cb1_fullattn_wall_ms/step` (`attention_cpu_ms/step` prices the
  encode only and must not move).
- **Falsifier**: sweep `chunkCount` 1 → 16 → 64 with everything else fixed. If
  per-layer `gpu_cb1_fullattn` is flat, the per-position critical path is the
  limit and only the tiling helps.
- **Expected**: ~1.3–1.8 → ~0.4 ms/layer at 2,940 tokens ≈ **8–12% tok/s** there,
  more at 8K+. **This also de-prices D1**: if attention is latency-bound rather
  than bandwidth-bound, int8 KV cannot buy more than the latency-bound fraction.
  Do C1, then re-price D1.

### C2 [HIGH] The int8 GEMV that carries every GDN projection is un-blocked

- **Now**: each 3.6 GDN layer dispatches `int8GEMV.encode` per token for qkv
  (M=8192, N=2048), z (4096, 2048) and out_proj (2048, 4096)
  (`RealForwardRunner.swift:3716-3746, 3828-3845`) — **33.6 MB of int8 weights
  per layer, ~1.01 GB per token** across 30 layers, plus ~63 MB of group
  scales/biases. `dequant_int8_gemv_simd` gives one SIMDgroup per row and, per
  64-weight group, each lane does **two scalar `uint8` weight loads and two
  scalar `half` x loads** (`Metal/Quant/dequant_int8.metal:79-92`); the 2048-half
  `x` vector is re-read from global memory by all 14,336 rows per layer.
- **Why it costs time**: ~5 instructions per weight per lane, so it is
  issue-bound rather than bandwidth-bound, and ~40% of the kernel's load
  instructions are re-loads of the same 4 KB `x` vector.
- **The internal control, on the same counter line**: on this box the LM head
  reads 286 MB/token in 4.21 ms (**68 GB/s**) while the int8 GDN GEMVs process
  1.07 GB in 29.98 ms (**36 GB/s**) — a **1.9x per-byte gap between two kernels
  reading the same class of weights in the same token**.
- **Why it is a gap and not a done deal**: the repo contains the fixed version of
  this kernel twice — `dequant_int4_gemv_simd` (`Metal/Quant/dequant_int4.metal:124-155`)
  and `lm_head_greedy_int4_rows_chunk_raw` (`Metal/Sampling/logit.metal:650-675`)
  both use four-group blocks (one 4-byte ushort-paired weight load and two `half4`
  x loads per lane per 128-byte block, 8 weights/lane/iteration).
  `dequant_int8_gemv_simd`'s doc comment claims "same trick applied here"
  (`:52-55`) — the multi-row-per-TG part was, the inner-loop blocking was not.
- **Change**: port that blocked inner loop into the int8 kernel, and add the three
  GDN shapes to `DequantInt8GEMV.realDecodeShapes`
  (`Kernels/Quant/DequantInt8GEMV.swift:24-28`) so `int8_fc_n(N)`
  (`dequant_int8.metal:31-35`) is a function constant and the group loop fully
  unrolls — today every GDN shape misses the specialization table and takes the
  generic PSO.
- **Do the cheap half first**: adding the shapes alone is the falsifier. If full
  unrolling moves `gpu_cb1_gdn_wall_ms/step` by <3%, the kernel is latency-bound
  rather than issue-bound and wants the x-in-threadgroup-memory / multi-row
  rewrite instead.
- **Counter**: `gpu_cb1_gdn_wall_ms/step`; `gdn_proj_cpu_ms/step` prices only the
  encode and must not move.
- **Expected**: GDN device time ~30 → ~16 ms/step on this box, ~**10–12% tok/s**,
  medium confidence on magnitude. Same live-offset alignment caveat as ever — the
  packed-load path that "passed an offset-zero fixture, then produced garbage in
  real decode" is exactly this kernel's family.

### C3 [SMALL] Routed MoE reads at 29 GB/s against the same 68 GB/s control

`gpu_routed_wall_ms/step` (13–15% of the token) carries ~566 MB of expert weights
per token in 19.5 ms = 29 GB/s, against the LM head's 68 GB/s for the same class
of read. Bytes are read once and load width is already settled (DEC-13 `u16`
packed loads 30.65 → 30.30 ms; DEC-12 paired rows slower), so two cheap
experiments before any kernel edit: (a) bind phase 1 to resident dummy weights
instead of the freshly-`pread` slot pages — if `gpu_routed` moves, the cost is the
destination's cache state, not the kernel (this is IO-21's untested read-out
side); (b) if it does not, a d2/d4 phase-2 geometry (2–4 output rows per
threadgroup, one `acts` read). DEC-14 already priced d2 at 0.6–1.1 ms of a 149 ms
step, so expect ≤5% and treat it as a 1–2% item.

### C4 [SMALL] Per-layer allocation and index churn on the critical path

When the hit-split branch runs (~90% of layers), `moe.makeRoutedArgumentBuffer`
allocates a **fresh `MTLBuffer` per layer per token**
(`Kernels/MoE/MoE.swift:192-202`) while the merge path uses the preallocated
reusable one; each layer also allocates the read-stamp array
(`PreadExpertStreamer.swift:565-567`), rebuilds Swift arrays/Sets
(`RealForwardRunner.swift:3305-3310, 3363-3366`), and does nine dictionary lookups
in `routedExpertOffsets(layer:)` (`:3312`). These sit between `totalIoPlanNanos`
and `tIoStart`, so they are in *none* of `cb1`/`io`/`cb2` — the plan's 2.2–7%
unexplained remainder. A Metal buffer allocation takes the device heap lock and
creates a GPU resource 40–48 times per token on the critical path. Fix: mirror
the reusable argument buffer for the split case, hoist the offsets to init, cache
the per-layer arrays, and add this window to `io_plan_cpu_ms/step` so it becomes
attributable. ≤1–2%.

Also worth a free cleanup alongside C1/C2: two extra full GPU syncs per token
(embedding lookup `:6098-6121`, fused head `:6443-6455`, plus one for non-greedy
sampling) — the embed could be encoded at the head of `cb1`, and the measured
per-buffer non-GPU overhead (0.035–0.055 ms) bounds the whole cleanup at ~1.5%.

### C5 — CLOSED, do not resubmit

**int4 GDN projections**: already rejected — the decoder supports them, but
recurrent-state error rose substantially
(`docs/SESSION_HANDOFF_3BIT_EXPERTS.md:188`). **3-bit routed experts**: built,
numerically verified (dequant error 20.8%, cosine 0.979) and still failed
qualitatively — HumanEval 28/164 (17.1%) against 4-bit's 149/164 (90.9%)
(`:113-138`); weights below int4 are cancelled by the quality floor. **4-bit GDN**
and **sub-4-bit PLE** are in the same family. Nothing here is worth re-opening
without a new precision scheme rather than a smaller group size.

## 5. Track D — prefill device work

Ranked by a prefill-path audit against the 3.6 install (40 layers = 30 GDN + 10
full, 256 experts top-8, `moeIntermediateSize` 512, hidden 2048,
`linearAttention.weightBits = 8`, `sharedExpert.weightBits = 4`, expert stride
1,769,472 B). Shares are estimates until D1 lands.

### D1 [PREREQUISITE, ~1 HOUR] Fix the prefill measurement first

Two gaps, both small, both blocking any ranking:

- **The 3.6 prefill body has no GPU attribution.** `encodeQwenPrefillLayer`
  (`RealForwardRunner.swift:2409-3223`) contains no `recordPrefillLayerGpuTime`
  and no `recordGpuTime` on its tile/shared/tail command buffers; those calls
  exist only in `encodeQwen38PrefillLayer` (`:4111` onward, at
  `:4956/5048/5068/5238`). So on a 3.6 `scope=prefill` line **every `gpu_*` field
  is structurally zero**, and the 44/36/21/3 split quoted in §1 is 3.8-only —
  `docs/SYSTEM_DESIGN.md:428-430` says as much. Add the five calls.
- **`misses=`/`io_mb=` on a prefill line are the *decode* counters.**
  `totalExpertMisses` is structurally decode-only on the chunked path
  (`RealForwardRunner.swift:1102-1110`) and `RunnerCounters.line` derives
  `io_mb` from it (`Sources/FinchMoECLI/RunnerCounters.swift:494-495, 520-528`);
  the prefill's own volume appears only on the separate `FQ_PREFILL_COUNTERS`
  line. Fold `totalPrefillExpertMisses × expertStride` into `scope=prefill`.

Already real for prefill and usable today: `io_read_wall_ms`, `io_conc`,
`io_read_identity` (`ModelExpertIO.swift:126-151` fills the box), plus
`FQ_INT8_GEMM=0`, `--prefill-chunk-tokens`, `--prefill-tile-depth/-experts`,
`--expert-cache-slots`, `FQ_GDN_SPLIT=1`.

### D2 [HIGH] The "batched" routed-MoE prefill kernels have no weight reuse

`prefill_grouped_routed_moe_batched_phase1` spends one thread per `(f, pair)` and
calls a full scalar row dot product per thread —
`prefill_moe_int4_gemv_row_dev(gate_W, …)` then `(up_W, …)`
(`Sources/FinchMoE/Metal/Prefill/prefill.metal:646-647`), with `_batched_down`
the same at `:684`. The helper (`:423-457`) walks all K=2048 with scalar byte
loads, per-element dequant, no threadgroup tile, no reuse of a dequantized weight
tile across the microbatch, and uncoalesced loads (a warp spans 4 pairs × 8 f-rows,
so every lane touches a different 1 KB weight row). "Batched" here means one
dispatch per 32 pairs (`PrefillGroupedRoutedMoE.swift:413-465`), not a tiled GEMM.

Size: routed MoE is 8 experts × 3 projections × 512 × 2048 × 2 × 40 layers ×
2,940 tok ≈ **5.9 TFLOP** per run; the 125B's measured 6.19 s at 426 tokens
implies ~450–500 GFLOP/s, ~10% of this GPU's fp16 peak. At that rate the term is
**~13 s of the 69.3 s**. The accepted §1.5 fix is the exact analogue: int8 GDN
projections went from one GEMV per token to a 64-row × 32-token tile with the
weight tile dequantized once per K-step into threadgroup memory and reused across
tokens (`PrefillInt8Gemm.swift:14-17`), winning 3.5–3.95x on the kernel and 1.44x
end to end.

**Change**: one threadgroup per (expert, f-block) with the token block set to that
expert's pair range — the grouping already makes tiles expert-homogeneous
(`PrefillMoEGrouping.swift:156-184`) — and the same for `down`. **Expected +8–15%
prefill.** Gate with token identity plus an oracle against the current kernel's
own arithmetic (no weights change, only accumulation order).

### D3 [CHEAP] The shared expert ignores its own dispatch policy

`PrefillProjectionDispatchPolicy` declares `.shared → .qmm` at ≥32 tokens
(`RealForwardRunner.swift:126-128`), but **no call site ever passes
`family: .shared`** — the 27 `encodeInt4Projection` calls pass `.q/.kv/.o` only —
and `PrefillSharedExpert.encodeBlock` loops rows instead
(`PrefillSharedExpert.swift:44-60`). Per layer-chunk that re-reads
gate(512×2048) + up(512×2048) + down(2048×512) int4 = **1.5 MB of weights once
per token — 768 MB of requested traffic for a 1.5 MB weight set** — and 3.2 GFLOP
at the ~150 GFLOP/s GEMV rate ≈ 21 ms/layer ≈ **5 s per prompt**. It is also ~40%
of the dispatch count in D4. Wire the declared policy; note
`prefill_dequant_int4_qmm_f16_block` (`prefill.metal:689-726`) is itself
scalar-gather and uncoalesced, so `MPPPrefillInt4QMM` (M=64 tokens × N=32 rows)
is the better target for these shapes. **Expected +3–6%.**

### D4 [MEDIUM] ~1.23 M dispatches per prompt from six per-token loops

At t=512 the 3.6 body issues: shared expert 4×512×40 = 81,920; tail `vecAdd`
2×512×40 = 40,960 (`RealForwardRunner.swift:3122-3131`); a/b int8 GEMVs
2×512×30 = 30,720 (`:2640-2659`); shared-expert sigmoid gate 512×40 = 20,480
**on a 32-thread threadgroup** (`QwenDecodeFusions.swift:122-123`); post-attention
norm 512×40 = 20,480 as a `1×1×1` threadgroup of 256 threads (`:54-55`); full-attn
epilogue 2×512×10 = 10,240 (`:2501-2517, 2558-2565`). Total ≈ **204,800 dispatches
per chunk, ~1.23 M per prompt — 99.98% of every dispatch in the prefill**, against
~10–15 GEMM dispatches per layer.

Two costs, both on the critical path: CPU encode at ~1–2 µs per
encoder/setBuffer/dispatch/endEncoding ≈ 1.2–2.5 s, which **cannot be hidden
because the layer's command buffer is committed only at the very end of the layer**
(`:2802`), so the GPU has nothing queued while the CPU encodes; and per-dispatch
launch overhead plus kernels doing a 2,048-element dot product on 32 threads.
Each loop batches trivially with the token on the grid's y dimension, as
`PrefillPerHeadNorm.swift:46` and `PrefillLayerTail.swift:57` already do.
**Expected +4–8%.** Add a per-layer dispatch counter before claiming it.

### D5 [MEDIUM, BOUNDED] The prefill never got the decode path's overlap

Every layer commits and **waits** its non-MoE half to read 4,096 route IDs
(`RealForwardRunner.swift:2802-2803`, readback `:2887-2904`), then commits **and
waits** the shared expert (`:2958-2959`) before issuing the first tile fetch
(`:2992+`). The router readback is unavoidable — route IDs gate expert selection
— but the shared expert is routing-independent, and decode documents the opposite
intent for exactly this stage: "commit it without waiting so its GPU work overlaps
the routed-expert pread" (`:3431-3434`, commit at `:3450`). 240 barriers per
prompt. **Change**: wait the shared buffer with the tail, as decode does; the
existing `splitGdnSubStage` mechanism (`:1330-1340`, wired for the 3.8 body only)
is the other half. **Expected +2–5%**, and bounded — do **not** re-tune tile depth
on the strength of this alone: §1.3's sweep was one 426-token chunk, and the I/O
share at 6 chunks is untested.

### D6 [THE BIG ONE, LARGER JOB] An MPP tensor-ops path for the int8 projections

`OPTIMIZATION_PLAN.md` §1.5's own "remaining headroom": after the tiled int8 GEMM
shipped, the GDN projections run at ~537 GFLOP/s and stay compute-bound on
something other than weight traffic. Hand-vectorizing the inner loop was a
measured no-op, so the limit is the dequant-and-accumulate instruction stream —
what tensor ops replace. Precedent in-tree (`MPPPrefillInt4QMM`; staged affine MPP
already won +11.4% on a 512-token prefill), but no int8 variant. Projections are
~95% of a stack that is 44% of prefill: the largest single prefill lever left.

### D7 [LOW] The full-attention prefill can never take the tensor-ops path on 3.6

`PrefillAttention.swift:71-77` admits the MPP prefill kernel only for
`headDim == 512 && numQHeads == 16 && numKVHeads == 2 && scale == 1.0`. 3.6's full
attention is headDim 256 with `scale = 1/sqrt(256)`
(`RealForwardRunner.swift:2545`), so all 10 full layers fall to
`attention_prefill_causal_tiled` (`:88`). The gate is a shape check, not a
capability check. **+1–3%**; low priority because full attention is ~3% of
prefill.

### Checked, so they do not get re-audited

`PrefillPerHeadNorm`, `PrefillRoPE`, `PrefillLayerTail`, `PrefillPostAttentionSetup`
and `PrefillFinalRowHead` are all one dispatch per layer; GDN conv/gate/rmsnorm are
already batched (`GDNPrefill.swift:67-69, 163-165, 204-205`); the expert read order
is already ascending physical offset (grouping sorts by
`routedExpertPhysicalOffsets`, `RealForwardRunner.swift:2922`), so there is no
seek-ordering win left; the GDN chunked recurrent scan is only 32 threadgroups and
~6% of the GDN stack; PLE and the QSA dump are 3.8-only. One residual: the tile
loop allocates a fresh `MTLBuffer` per tile
(`PrefillGroupedRoutedMoE.swift:345-360`, called at `:3069`) — ~7,680 allocations
per prompt; a slot-indexed ring removes them.

## 6. Track E — long context

### E1 [MEDIUM] int8 KV for the full-attention layers only

The plan's item 3.2, promoted by the attention curve: attention is the only layer
term that grows with context (0.530 → 1.281 ms/layer from 54 → 1,791 tokens,
+142%, vs +5% for GDN). The store is small — 10 (3.6) / 12 (3.8) full-attention
layers, 80 MiB at 4K, 1.25 GiB at 64K — so read it as a VRAM/context-headroom
play with a long-context bonus, not a short-context win. **Do C1 first and
re-price this**: if that growth is the per-position barrier latency C1 describes,
it is not KV bandwidth, and halving KV bytes buys only the bandwidth-bound
remainder. Gates are already written and non-negotiable: EvalPlus within ±1–2
problems of 0.909/0.878, the 4,096 soak quote-exact, and a **measured**
int8-vs-FP16 memory sweep across 4K–64K, because the size crossover is what
killed the Gemma attempt (packed K4/V4: 82 MiB saved at 4K, larger than FP16 at
long context, delta-NLL +0.015).

---

## 7. Track F — the only route to a multiple on decode

### F1 [TIMEBOXED SPIKE] Multi-token prediction / self-speculative decode

Every other item here is a percentage; speculative decoding is the only multiple.
The checkpoint's MTP head is present upstream and explicitly unused, and
draft-model speculation "was never completed" (`EXPERIMENT_INVENTORY.md`). Answer
three questions in a day before writing code: (1) do `mtp.*` tensors actually ship
in the repacked `.finch` (`manifest.json`, `packed_experts/layout.json`)? (2) what
does a 1-layer MTP head cost in resident memory against the ~1.1 GiB budget?
(3) what acceptance rate does it show on a real prompt at greedy? Then go/no-go.
Batched decode (plan item 4.2) is the other multiplier and stays out of scope for
a single-user app.

---

## 8. Measurement discipline (non-negotiable, from this project's own record)

1. **Re-baseline before optimizing.** The drive drifts faster than most effects
   here. Run `--counters` on today's configuration first: if `io` is ≥60 ms/step
   on a run that used to read 45–72, the day's token time is drive state and no
   GPU-side change will show more than its own share.
2. **One interleaved session per comparison, arms alternating order.** 3.6 read
   4.68 and 5.18 GB/s eleven minutes apart; cross-session subtraction is invalid
   (METH-13/14/15).
3. **Cold-cache controls.** `F_NOCACHE` is not honoured on this volume and
   `purge` is unavailable, so warm readings of "cold" conditions are 6.3x off
   (IO-18). State the cache state for every read number.
4. **Attribute the 3.6 prefill before trusting prefill estimates** (D1) — and
   remember `scope=prefill`'s `gpu_*` fields read zero on 3.6 by construction
   today.
5. **Byte/token identity** for anything that must not change the math (cache
   size, scheduling, chunk size, a kernel re-tiling with a fixed accumulation
   order). **EvalPlus (baseline 0.909/0.878 on 3.6) + the 4,096 soak** for
   anything touching weights or KV, plus the measured size-vs-context sweep for
   KV.
6. **Do not A/B long prompts on single pairs**: past ~2,051 tokens the engine is
   not bit-reproducible, and a second intermittent dense-attention divergence is
   unexplained.
7. **Kernel wins must be re-priced end to end.** The record's recurring lesson:
   31% kernel → 2% e2e; LM head 14.2 → 13.1 ms inside a 167.7 ms step; 21,217
   allocations → 2 yet 9% slower.
8. **Machine protocol**: `-j 3`, `tools/memguard.sh` (kills below 12% free,
   >4 GB compressor, or any swap), `memory_pressure -Q` ≥ ~60% free before heavy
   runs, abort near ~13 GB RSS. Three watchdog panics are on record.
9. **Toolchain**: `swift test -c release` discovers 0 tests under `-O`; use
   `-c debug` or `-c release -Xswiftc -Onone`.

---

## 9. Priority order

| # | Item | Cost | Expected | Confidence |
| --- | --- | --- | --- | --- |
| 1 | **A1** prefix reuse in the app | medium | per-turn prompt processing 10–50x less | high — mechanism shipped in the server |
| 2 | **D1** fix prefill attribution (1 h), then **B2** chunk 512 → 1024 | free | prefill **+7–15%** (3.6) | high — byte arithmetic, no code change |
| 3 | **D3 + D4** shared-expert policy, then batch the six per-token loops | low | prefill **+7–14%** combined | medium-high |
| 4 | **D2** tile the routed-MoE prefill GEMM | medium | prefill **+8–15%** | medium-high — same defect §1.5's fix cured |
| 5 | **C2** block the int8 GEMV inner loop + specialize the GDN shapes | low | decode **~10–12%** | medium-high — 1.9x behind the LM head's own rate |
| 6 | **C1** tile the attention key loop | medium | decode **8–12%** at 2,940, more at 8K | medium-high |
| 7 | **B1a/B1b** Spotlight exclusion, then the one untested slot cell | low | 3.8 read path, up to ~14–30% if real | low per experiment, high leverage |
| 8 | **D5**, **A2**, **B3**, **C3**, **C4**, **D7** | low | 2–5% each | medium |
| 9 | **E1** int8 KV (after C1 re-prices it) | medium | decode 5–15% at 8K+, if bandwidth-bound | medium |
| 10 | **D6** MPP int8 projection path | high | prefill, potentially large | medium |
| 11 | **F1** MTP spike | timeboxed | decode, possibly a multiple | unknown until scoped |

**First three actions, concretely.**

1. **Ten minutes, no code**: `mdutil -i off` the model volume, then re-run
   `--counters` on today's configuration. This either removes the 3.8 read anomaly
   or removes the doubt, and it re-baselines everything else — `io` has been
   recorded at 45–72 ms/step on this box, so if today's run reads ≥60 the day's
   token time is drive state, not code.
2. **An hour, then a config change**: land D1 (the five `recordGpuTime` calls plus
   the prefill miss-byte fold) so prefill can be ranked by measurement, then run
   3.6 at `--prefill-chunk-tokens 1024` and confirm `bytes=` halves on the
   `FQ_PREFILL_COUNTERS` line.
3. **A day**: spike A1 — point the decode service at `ServerPromptCache`'s
   matching rule for a fixed two-turn conversation and measure turn-2 TTFT against
   today's full re-prefill.
