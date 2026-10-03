# Speedup execution plan — the five next items

This is the **execution runbook** for the next round of work. It is deliberately
not a new strategy: four of the five items already have full entries in
[`PREFILL_DECODE_SPEEDUP_PLAN.md`](PREFILL_DECODE_SPEEDUP_PLAN.md), and the
per-item work there is better than anything restated here. What this document
adds is the **order** (with dependencies), the **code-level touchpoints verified
against the current checkout**, and the **one item that is genuinely new**.

Where an item already exists, this document says so and points at it rather than
duplicating it. The measurement discipline in
[§8 of the speedup plan](PREFILL_DECODE_SPEEDUP_PLAN.md) is non-negotiable and is
not repeated here.

> **Line numbers below were verified against the current checkout.** Several
> references inside `OPTIMIZATION_PLAN.md` are stale — for example it cites the
> GDN GEMV dispatch sites as `RealForwardRunner.swift:3716-3746, 3828-3845`, but
> `encodeQwenDecodeLayer` now starts at `:3624` and those dispatches live at
> `:3824-3942`.

## Where the five items came from

| # | Item | Existing plan ID | Status |
| --- | --- | --- | --- |
| 1 | Prefill chunk 512 → 1024 | **B2** (`:152-186`) | specified, no code needed |
| 2 | Prefix reuse in the Mac app | **A1** (`:55-100`) | specified, ranked #1 in that plan |
| 3 | The per-layer CPU wait | *(new)* | **partly closed** — see below |
| 4 | Block the int8 GDN GEMV | **C2** (`:241-280`) | specified, ranked #5 |
| 5 | Tensor cores for decode | **D6/D7** (`:434-452`) | **CLOSED — do not do this** |

---

## Pre-work (do this before any of the five)

Both steps are from the speedup plan's own "first three actions"
(`PREFILL_DECODE_SPEEDUP_PLAN.md:552-565`), and every number below depends on
them.

**P1 — Kill Spotlight on the model volume.** `mdutil -i off "/Volumes/samsung 2t"`,
then re-run `--counters` on today's configuration. The record shows one identical
synthetic cell moving **1.8 → 3.2 GB/s in twenty minutes** while indexing was live
(`B1a`, `:130-136`). This either removes the 3.8 read anomaly or removes the
doubt, and it re-baselines everything else.

**P2 — Check today's `io` before believing any GPU-side win.** `io` has been
recorded at 45–72 ms/step on this box and reads **64.65 ms/step** in the
2026-09-30 M6 run. If today's baseline is ≥60 ms/step, the day's token time is
drive state and no kernel change will show more than its own share.

---

## 1. Prefill chunk 512 → 1024  (plan B2)

**The single cheapest real win in the list: a flag, not a patch.** Chunk size is
literally the prefill I/O divisor, because a layer's routed-expert pool is
re-read once per chunk (`PrefillRuntimeConfig.swift:183-200`). At 512 a 2,940-token
prompt is 6 chunks and touches essentially all 256 experts per layer-chunk —
≈96.6 GiB moved; at 1024 it is half that.

**Verified: nothing blocks 1024 today.**

- `RuntimeConfiguration.swift:31` — `allowedPrefillChunkTokens = [32, 64, 128, 256, 512, 1024]`
- `PrefillRuntimeConfig.swift:173` — `maxChunkTokens = 1024`; the planner clamps at `:102`
- `Args.swift:180` (default 512), flag validation `:207-218`, wired at `Run.swift:90-97`
- App: `AppRuntimeOptions.swift:106` / `:116` / `:146-152`
- Scratch is the only real cost: ~154 KiB per chunk-token → 512 ≈ 79 MiB,
  1024 ≈ 158 MiB, against a ~1.1 GiB resident budget (`RuntimeConfiguration.swift:24-30`)

**Touchpoints, in order:**

1. **No code** — CLI: `--prefill-chunk-tokens 1024`. Measure and confirm the win.
2. **App default** — `AppRuntimeOptions.swift:116` 512 → 1024, once step 1 confirms.
3. **Server has no flag** — `ServerInference.swift:410` hardcodes
   `RuntimeConfiguration(forceLogitsHead: true)`, so the HTTP server is pinned at
   512/16 regardless. This is the real code change if the server matters.
4. **Optional second step** — to sweep 2048 you must raise *both* ceilings
   (`RuntimeConfiguration.swift:31` **and** `PrefillRuntimeConfig.swift:173`), or
   the planner silently clamps to 1024. No slot or tile change is needed.

**Not a constraint:** the `(depth+1)*tileExperts <= slots` rule
(`RuntimeConfiguration.swift:107-111`, enforced at `RealForwardRunner.swift:1885-1891`,
`:2993-2999`, `:5100-5106`) governs tiles in flight *within* one layer's MoE loop,
not chunk size. Defaults are depth 1 / 8 experts against 16 slots → 16 ≤ 16, fits.
The sliding-window ring guard at `:1548-1559` is Gemma-specific and does not bind
on the Qwen installs.

**Counter / falsifier:** `FQ_PREFILL_COUNTERS=1` prints `bytes=` and
`MB_per_chunk` (`FinchMoECLI/Run.swift:222-237`) — expect ≈96.6 GiB / ≈16,470 MiB
per chunk at 512 and **exactly half** at 1024; cross-check `io_read_wall_ms` on the
`scope=prefill` line. **Falsified if `bytes` does not halve** — that would mean
re-reads inside a layer, i.e. a slot-eviction bug, and is worth knowing on its own.

**Expected:** +7–15% prefill on 3.6. Confidence high — this is byte arithmetic,
not a kernel claim.

**Quality risk: low.** Scheduling-only; no weight math changes. The chunked-prefill
math is exact — at 256/512/1024 outputs agree with each other and repeats are
identical; the one token of 32 that diverges at 2,940 tokens is the engine's known
long-context non-reproducibility past ~2,051 tokens, not chunk size
(`OPTIMIZATION_PLAN.md:60-67`). Note that the config-layer tests are parameterised
on `[32, 64, 128]` only (`PrefillRuntimeConfigTests.swift:18-24`), so 512 and 1024
are legal but untested at that layer; the real coverage is
`Qwen38EngineLoadTests.swift:1000-1075` and
`QwenLayer0DebugTests.swift:3076-3077` (the latter gated on the install existing).

---

## 2. Prefix reuse in the Mac app  (plan A1 — ranked #1 there)

> **Status (2026-10-02): implemented** in `RealInferenceClient`
> (`SessionPromptCache`): the session retains the last turn's KV and resumes
> by the strict token-prefix check or the server's structured continuation,
> falling back to a reset on any doubt; `cachedPromptTokens` is reported on
> `AppDiagnostics`, over the decode-service protocol, and the HUD reads
> `Prefill (37/2540, 2503 cached)`. The continuation needed a family-aware
> `GFTokenizer.encodeTextContinuation` — the Gemma-shaped bridge was wrong
> for Qwen ChatML (and the server now gets the Qwen-correct one too).
> Measured on the 16 GB M4 mini (Qwen 3.6, the install-gated
> `PrefixReuseInstallTests`):
>
> - 493-token first turn prefills in 10.7 s; the follow-up reuses 494 tokens
>   and prefills in 2.3 s against 10.7 s with reuse off — same output.
> - Soak at 3,840 tokens (past the ~2,051-token non-reproducibility zone):
>   3,814 tokens reused, prefill 2.8 s against 100.1 s — **36×** — and both
>   arms quote the planted sentence exactly.
> - **Identity caveat, measured:** resumed and full-prefill answers are
>   byte-identical in short conversations but can differ in wording at soak
>   length ("…73-19-4." vs the same answer in quotes). A repeat of the full
>   prefill was identical, so this is the prompts genuinely differing — the
>   resumed stream continues what decode wrote (including the template's
>   empty think block), while a re-rendered history does not — not cache
>   error. Closing it entirely would mean rendering historical assistant
>   turns the way the generation prompt is rendered.
>
> Still open from this item: the trimming stability proof (fixed-step trim
> is in place; no long-conversation trim case has been exercised).

**The largest user-visible latency win available, and the mechanism already
ships.** Today every chat turn sends the whole conversation and
`RealInferenceClient.swift:365` calls `runner.reset()` immediately before prefill,
with no `start:` argument at `:368-371` so it defaults to `.reset`
(`RawCompletion.swift:149`). Turn *n* prefills ~*n* messages: conversation prompt
processing is O(n²), and on 3.6 a 2,500-token conversation costs **~60 s of prefill
per turn**.

The server already does this correctly end to end —
`ServerPromptCache.match` (`ServerPromptCache.swift:86-131`) → `.resume(cachedPromptTokens:)`
(`ServerInference.swift:512-530`) → `prepareForContinuation(expectedPosition:)`
(`RawCompletion.swift:163-200`, `RealForwardRunner.swift:1036-1046`).

**The engine API is not the blocker.** `runner` is already a `RealForwardRunner`,
`runRawCompletion` already accepts `start:`, and `RawDecodeResult` already carries
`kvPosition` / `kvBackedTokenIDs` / `uncommittedBoundaryTokenIDs`
(`RawCompletion.swift:330-339`). The app simply discards them
(`RealInferenceClient.swift:398-401`).

**The real blocker, and the reason this is a day and not an hour:**

> **The app's history is lossy.** `AppChatTurn` stores only `role` + `text`
> (`AppGenerationRequest.swift:5-10`). The server can match because it keeps the
> *generated token IDs* and the raw stop reason and can prove that a re-rendered
> assistant message corresponds to what decode actually wrote
> (`assistantMatches`, `ServerPromptCache.swift:133-149`; bridge logic `:166-174`).
> The app re-renders assistant text through the chat template from visible text
> only — so with stop-string filtering, a detokenizer flush tail, or a `maxTokens`
> cut, the re-rendered prefix silently differs from `kvBackedTokenIDs`.

Prefix equality must therefore be a **test, not an assumption**. The safe subset
to implement first is the server's fast path (`ServerPromptCache.swift:101-107`):
strict extension plus `prefix(kvPosition).elementsEqual(kvBackedTokenIDs)`.

**Work items:**

1. **Session state on the actor** (`RealInferenceClient.swift:164-170`): keep the
   prior turn's `RawDecodeResult` + rendered `promptIds` + `SessionLoadKey`
   (`:323-327`). Nothing equivalent to `ServerPromptCacheEntry` exists in
   `FinchMoEAppCore`.
2. **A match step** before the reset at `:365`, emitting `.resume` instead of the
   default `.reset` at `:368`. Implement the fast path only; fall back to `.reset`
   on any doubt.
3. **Fix the trim to fixed-size steps.** `trimmedHistory` currently drops the
   oldest *pair* whenever the render does not fit (`RealInferenceClient.swift:290-300`,
   `:340-346`). If the boundary moves one turn per message the prefix changes every
   turn and the cache never hits. Drop two turns at a time, only when the prompt no
   longer fits (`PREFILL_DECODE_SPEEDUP_PLAN.md:84-89`).
4. **Protocol additions** (`DecodeProtocol.swift`): a cached-token field on
   `DecodeGenerationRequest` (`:62-99`) and a cached count on the prefill event
   (`:120`), so the HUD can read `Prefill (37/2,540, 2,503 cached)`.
5. **Invalidation**, each one a test: model unload/reload, any `SessionLoadKey`
   change, session switch, edit of an earlier message, Regenerate, cancel
   mid-generation, and trim. The server models the domain as
   `ServerPromptCacheDomain` fields (`ServerPromptCache.swift:9-17`).

**Hazards that make naive reuse unsafe** (all already reasoned in the plan at
`:66-95`, listed here so they are not rediscovered):

- The token that ends generation never passes through `produce`
  (`RawCompletion.swift:296-309`), so it is in `uncommittedBoundaryTokenIDs` but
  **not** in the KV. Drop it or bridge it exactly as `ServerPromptCache.swift:166-169`.
- `publish` refuses when the generation was stop-string filtered
  (`ServerPromptCache.swift:57`) — the re-rendered assistant turn cannot equal the
  cached tokens.
- `kvPosition` is only meaningful with the ring configuration (`fp16RingEnabled` is
  a domain field, `:15`). GDN conv/recurrent, PLE window and QSA `pooledBlocks`
  carry positional state too (`RealForwardRunner.swift:1011-1023`) — fine on a true
  continuation, fatal if the caller resets and then tries to resume.
- Long-context non-reproducibility is real (one fixed bug plus one unexplained
  intermittent dense-attention divergence, `OPTIMIZATION_PLAN.md` item 16), so
  resumed and full-prefill runs are not guaranteed token-identical. Gate in three
  steps: **token-identity vs full prefill → EvalPlus → the 4,096-token soak.**

**Measure:** TTFT per turn, and prompt tokens actually computed per turn
(`scope=prefill` counters plus the new `cached=` field).
**Expected:** per-turn prompt processing O(conversation) → O(new message), **10–50×**
across a chat. Decode untouched.

---

## 3. The per-layer CPU wait  (new — and mostly closed)

**This is the one where the original framing was wrong, so read this before
spending time.** I said "40 CPU syncs per token on a single serial queue" as if it
were reclaimable. Your own record says it is not, and measures why:

- There is one blocking `waitForCompletion(cb)` per layer per decode token
  (`RealForwardRunner.swift:3985-3991`; 3.8 at `:6045-6051`) — 40/token on 3.6, 48
  on 3.8, plus ~2 full syncs from `runSync` (`:6568-6576`). The wait is mandatory
  because the router writes top-k indices into a `.storageModeShared` buffer that
  the CPU reads back (`:3380-3388`) to plan the expert preads
  (`planRoutedExperts`, `:3400-3410`).
- **But `wait_cpu_ms/step` is not mostly overhead.** "`wait` minus *both* GPU
  figures is 10.40 / 5.60 / 8.69 ms/step — 0.055 / 0.035 / 0.055 ms per command
  buffer … `wait` looks like a third of the token because layer N+1's `cb1` is
  committed after layer N's routed tail, so its wait drains that tail"
  (`OPTIMIZATION_PLAN.md:608`). Per-buffer overhead is an order of magnitude below
  a kernel dispatch, which is why CB coalescing (item 2.2) is closed (`:420-422`).
- And the overlap is bounded by a **data dependency, not a scheduling artifact**:
  "layer N+1 cannot be *encoded* before layer N's routed output exists, because
  that output is layer N+1's input, and no read can be issued before its own
  layer's router has chosen" (`OPTIMIZATION_PLAN.md:290-292`).

**So do not resubmit:** readback removal, CB coalescing, decode-side prefetch,
static hot-set pinning, per-expert-read overlap, deeper prefill tiles — all
rejected with data (`OPTIMIZATION_PLAN.md:463-484`, `OPTIMIZATION_JOURNEY.md:119-136`).

**Two things in this area are legitimately open:**

**(a) NEW — split the router into its own earlier command buffer.** The blocking
wait spans the *entire* layer cb1 (attention/GDN **plus** router), so the CPU
blocks on the full mixer stack before it can even see the indices. The router is
the last work encoded into that buffer (`:3964-3967`). Committing a small
router-only CB *before* the rest of the mixer would let the readback and the
`planRoutedExperts` planning overlap the mixer's tail without touching the
readback itself. **No experiment in `docs/experiments/` or `OPTIMIZATION_PLAN.md`
tests this** — the recorded reasoning above does not distinguish "the whole stack
must finish" from "only the router must finish". This is a genuine gap, it is
bounded, and it is the only new decode idea in this document.

- *Falsifier:* if `wait_cpu_ms/step` does not move, the tail-drain explanation at
  `OPTIMIZATION_PLAN.md:608` is complete and this closes.
- *Risk:* the router's inputs are the mixer's outputs, so "router early" may not be
  expressible. Check that before building anything.

**(b) SANCTIONED, OPEN — prefill D5.** The prefill shared-expert wait is
routing-independent and could be committed without waiting exactly as decode does
(`PREFILL_DECODE_SPEEDUP_PLAN.md:419-432`, citing `RealForwardRunner.swift:3431-3434`
and the commit at `:3450`). **+2–5% prefill**, low cost. Do this one regardless of
whether (a) pans out.

---

## 4. Block the int8 GDN GEMV  (plan C2 — ranked #5 there)

**A measured 1.9× per-byte gap inside a single token, with the fixed version of the
kernel already in the repo twice.** Verified against source:

- Kernel: `dequant_int8_gemv_simd`, `Metal/Quant/dequant_int8.metal:57`; the
  un-blocked inner loop is the flat `for g in 0..<n_groups` at `:79-92`, doing two
  scalar `uint8` weight loads and two scalar `half` x loads per 64-weight group.
  Its own doc comment claims "same trick applied here" (`:52-55`) — the
  multi-row-per-threadgroup part was applied, the inner-loop blocking was not.
- Control on the same counter line: the LM head reads 286 MB/token at **68 GB/s**
  while the int8 GDN GEMVs process 1.07 GB in 29.98 ms at **36 GB/s**.
- The fix exists twice: `dequant_int4_gemv_simd` (`dequant_int4.metal:121-150`) and
  `lm_head_greedy_int4_rows_chunk_raw` (`logit.metal:650-675`) both use four-group
  blocks — one 4-byte ushort-paired weight load and two `half4` x loads per lane
  per 128-byte block, 8 weights/lane/iteration.
- Decode dispatch sites (all in `encodeQwenDecodeLayer`, `RealForwardRunner.swift:3624`):
  qkv `:3824-3829` (8192×2048), z `:3831-3836` (4096×2048), a/b gate `:3870-3895`,
  out_proj `:3937-3942` (2048×4096). 33.6 MB of int8 weights per layer, ~1.01 GB
  per token across 30 layers.
- The specialisation table `DequantInt8GEMV.realDecodeShapes`
  (`Kernels/Quant/DequantInt8GEMV.swift:24-28`) holds only 128×2816, 2112×2816 and
  2816×2112 — **none of the GDN shapes**, so every GDN dispatch takes the generic
  PSO via `specializedPSOs[Shape(m:n:)] ?? pso` at `:64`.

**Do the cheap half first — it is the falsifier.** Add the three GDN shapes to
`realDecodeShapes` so `int8_fc_n(N)` (`dequant_int8.metal:31-35`, constants 70/71/72
set at `DequantInt8GEMV.swift:37-47`) becomes a function constant and the group
loop fully unrolls. **If that alone moves `gpu_cb1_gdn_wall_ms/step` by <3%, the
kernel is latency-bound rather than issue-bound** and wants the
x-in-threadgroup-memory / multi-row rewrite instead — stop and re-plan rather than
porting the blocked loop.

**Counter:** `gpu_cb1_gdn_wall_ms/step` (`RunnerCounters.swift:509`, accumulated
`RealForwardRunner.swift:3994-3999`). `gdn_proj_cpu_ms/step` prices only the encode
and **must not move**. Current M6 baseline: 25.79 (2,940 tok) / 31.28 (426) / 32.66 (62).

**Expected:** GDN device time ~30 → ~16 ms/step, **~10–12% tok/s**, medium
confidence on magnitude.

**Risk to respect:** the alignment family. The plan's own warning — "the
packed-load path that 'passed an offset-zero fixture, then produced garbage in
real decode' is exactly this kernel's family"
(`PREFILL_DECODE_SPEEDUP_PLAN.md:278-280`). Byte-identity against the current
kernel on a real prompt is the gate, not a fixture.

---

## 5. Tensor cores for decode  —  CLOSED, do not do this

**This was my suggestion and it is wrong. The route is closed by measurement, and
the docs say so as design, not as an open question.**

- `SYSTEM_DESIGN.md:529-530`: "MPP handles prefill projections with enough rows to
  benefit from matrix operations. **Single-token decode stays on custom GEMV
  kernels.**"
- The crossover is measured, not assumed. PF-12 shipped staged affine MPP at
  ~73.8% better weighed M128 work and 11.4% faster 512-token prefill, while PF-13's
  competing direct-UInt4 path *still lost* to staged MPP **at M32 by 38.6%** and at
  M128 by 20.4% (`docs/experiments/summaries/06-prefill.md:188-216`). **M=32 is the
  measured crossover.** Decode is M=1, where the 64-row tile shape
  (`MPPPrefillInt4QMM.swift:9-12`, `tileM=64`) leaves 63 of 64 M-lanes idle and the
  staged tile dequant is pure fixed cost. The gate at `RealForwardRunner.swift:2299-2320`
  exists precisely to keep M=1 off this path.
- There is **no MPP-shaped GEMV** to adopt: the only two `matmul2d` sites are
  `tensorops.metal:25-28` (prefill-only call site) and `prefill.metal:962-973`
  (prefill attention). `simdgroup_matrix` appears **nowhere** in the repo. The
  reason is not "MPP overhead" in the abstract — at M=1 there is nothing for the
  tensor core to do that a GEMV does not already do better.

**Do not write this step.** The only version of "use more of the M6's silicon on
decode" that the record supports is **C1** (below), and even that is a latency fix,
not a FLOP fix.

**What the record says is actually untested in this neighbourhood** — worth one
look, because it is closer to the real bottleneck than tensor cores are: the
**command-buffer count**, 120–200 commits per token at 40 layers
(`OPTIMIZATION_PLAN.md:404`). The same line notes CB coalescing is closed *as
overhead* (per-buffer cost is an order of magnitude below a dispatch), so the open
question is not the commit cost but whether the **commit boundary** constrains
overlap — which is exactly what item **3(a)** above tests.

---

## Order

Sequenced by dependency and by what invalidates what. Items in the same row can be
worked in either order.

| Order | Item | Effort | Expected | Why here |
| --- | --- | --- | --- | --- |
| 0 | **P1/P2** re-baseline, Spotlight off | 10 min | — | every number below depends on it |
| 1 | **1 / B2** chunk 1024 (CLI first) | 10 min + a sweep | prefill +7–15% | no code, highest confidence |
| 2 | **4 / C2** shape specialisation (cheap half) | hours | decode ~3–12%, *decides* C2 | cheap half is itself the falsifier |
| 3 | **3(b) / D5** prefill shared-expert wait | low | prefill +2–5% | sanctioned, independent of C2 |
| 4 | **1 / B2** app + server defaults | low | same win, user-visible | only after the sweep confirms |
| 5 | **2 / A1** prefix reuse | ~a day | per-turn prefill **10–50×** | largest win; the lossy-history test is the work |
| 6 | **4 / C2** blocked inner loop | medium | decode ~10–12% | only if step 2 passed 3% |
| 7 | **3(a)** router-early CB *(new)* | timeboxed | unknown | only genuinely new decode idea here |
| 8 | **C1** attention key-loop tiling | medium | decode 8–12% @2,940 | highest-ranked remaining decode item |

**Free riders, take while passing:** **B3** (expert-cache slots 16 → 8, legal for
3.6's top-k 8, one config sweep, byte-identical output expected) and **A2**
(app context default 65536 → 8K; 1.34 GiB of FP16 KV preallocated even for a
ten-token prompt).

**Explicitly out of scope for this round:** D6/D7 (MPP int8 projections — prefill,
high cost, and step 5 above explains why the decode version is closed), E1 (int8
KV — re-price after C1), and F1 (MTP / self-speculative decode — the only route to
a *multiple* on decode, but it needs a day of scoping before it deserves a slot).

---

## Gate for every item

Taken from the project's own record (`PREFILL_DECODE_SPEEDUP_PLAN.md:502-535`) —
listed here only so nothing is skipped:

- Byte/token identity for anything that must not change the math (cache size,
  scheduling, **chunk size**, a re-tiling with fixed accumulation order).
- EvalPlus (baseline 0.909 / 0.878 on 3.6) plus the 4,096-token soak for anything
  touching weights or KV.
- Kernel wins must be re-priced **end to end**. The recurring lesson: 31% kernel →
  2% e2e.
- One interleaved session per comparison, arms alternating order. Cross-session
  subtraction is invalid — 3.6 read 4.68 and 5.18 GB/s eleven minutes apart.
