# PLE Table Quantization Plan — Cutting Qwen 3.8's SSD Footprint and Per-Token I/O

Grounded in: `docs/QWEN38_PORT.md`, `docs/SYSTEM_DESIGN.md`,
`Sources/FinchMoERepack/Core/Planning/QwenRepackPlanner.swift`,
`Sources/FinchMoERepack/Core/Writing/QwenQuantizedWriter.swift`,
`Sources/FinchMoERepack/Core/Workflow/LocalQwenRepacker.swift`,
`Sources/FinchMoEFormat/FinchQuantization.swift`,
`Sources/FinchMoE/Infrastructure/Streaming/PLEPartStreamer.swift`,
`Sources/FinchMoE/Runtime/Inference/PLEHost.swift`,
`Sources/FinchMoE/Runtime/Inference/Model.swift`,
`Sources/FinchMoE/Runtime/Inference/RealForwardRunner.swift`, and direct
measurement of the installed
`models/Qwen3.8-Flash-Next-125B.finch/` tree.

## 0. Problem statement, measured

The install is 162 GiB, split three ways:

| Path | Size | Quant state |
| --- | --- | --- |
| `model_weights.bin` | 3.6 GB | already int4/int8 affine (embed, attention, hyper-connection) |
| `packed_experts/` | 63 GB | already int4 (routed/shared experts) + int8 (router) |
| `ple_shards/` | 95 GB | **raw BF16, unquantized** — 128 parts × 2,500,012 × 160 rows |

The rest of the model is already at the group-64 affine int4/int8 scheme
this codebase uses everywhere else (`docs/SYSTEM_DESIGN.md` "group-64 affine
quantization"). The PLE n-gram hash-embedding table is the one tensor class
still shipped as raw BF16 — `docs/QWEN38_PORT.md` lists "PLE table quant"
under **Deferred**, and the 2026-09-09 user directive explicitly chose "PLE
table ships as raw BF16 part files (no PLE quant this pass)" to keep M4
scoped. This plan is that deferred work.

This is not only an install-time SSD-copy cost. `RealForwardRunner.swift`
(prefill PLE routing comment) and `PLEHost.gather` read the table **on every
generated token**, not once at load: 16 rows × 320 B = 5 KB per token,
`pread` at hash-random offsets into a 102.4 GB file, deliberately uncached
(`docs/QWEN38_PORT.md`: "reads are hash-random over a 102.4 GB table" —
nothing is cacheable at that access pattern). So quantizing this table cuts
two different costs:

1. **One-time**: ~95 GB → ~27 GB of SSD bytes copied at install (int4,
   group-64-class overhead), bringing the whole install from 162 GiB to
   roughly **~94 GiB**.
2. **Per-token, forever**: the steady-state random-read decode path drops
   from 5 KB/token to roughly 1.4 KB/token (packed nibbles + scale/bias
   overhead) — this is the actual "I/O burden" a long generation run pays
   repeatedly, and it is the harder number to get right because it lives on
   the hot decode path, not a one-time repack.

Quantizing `model_weights.bin` and `packed_experts/` further (e.g. below
int4) is **out of scope**: they are already at this codebase's floor
(int4/group-64), the routed-expert kernels are written against that exact
bit width, and going below it is a different, much larger project (new
kernel dtype, new quality bar) that this plan does not attempt. The 70 GB
estimate in the original ask assumed the *whole* install was 8-bit; it
is not — only the PLE table is unquantized, and quantizing it alone lands
around 94 GB, not 70 GB.

## 1. Hard constraints carried through every phase

- **No accuracy regression tolerance to spend carelessly.** `docs/QWEN38_PORT.md`
  item under "Status" already reports the whole-vocab logit cosine at 0.88,
  under the plan's own 0.95 bar (cross-quantization vs the llama.cpp oracle,
  not same-weights). Adding a second lossy quantization on top of that
  needs its own explicit measurement, not an assumption that "int4 elsewhere
  was fine so this is fine" — the PLE table feeds directly into the residual
  plane additively (`ple.metal` `ple_plane_add`), so its error does not
  wash out the way a routed expert's minor share of one FFN might.
- **Row-independent reads are non-negotiable.** The decode/prefill gather
  reads exactly one row at a time at a hash-derived random offset
  (`PLEHost.gather`, `PLEPartStreamer.readRows`). Any quantization scheme
  must let one row be decoded from its own bytes alone — no scheme that
  needs neighboring rows' data (e.g. a shared scale computed across many
  rows, or entropy coding across rows) is admissible.
- **160 is not a multiple of 64.** `FinchQuantization.quantizeInt4Affine`
  hard-preconditions `count % groupSize(64) == 0`
  (`FinchQuantization.swift:70-71`). PLE rows are 160-wide
  (`ngramRowDim`). The existing group-64 affine code cannot be reused
  as-is — this is a new group-size variant, not a drop-in call. See Phase 1.
- **16 GB machine, panic history.** The repack that produces this install
  reads the full 352 GB BF16 checkpoint and is the same class of job that
  has triggered three watchdog kernel panics on this box
  (`docs/QWEN38_PORT.md` "Machine" section). Any new PLE-quantizing repack
  path must run under `tools/memguard.sh` and follow the same
  staged/resumable/fsync-per-file discipline `LocalQwenRepacker.swift`
  already uses for the rest of the model — not a new one-off script.
- **Old install stays installable until the new one is validated.** Do not
  delete or overwrite `models/Qwen3.8-Flash-Next-125B.finch/` until the
  quantized-PLE install passes its own oracle/EvalPlus gates. This is a new
  install directory, not an in-place edit.

## 2. Open design decisions to settle before writing code

These need explicit answers (Phase 1 produces them) rather than being
discovered mid-implementation:

1. **Group size for the affine scheme.** 160 factors as `32 × 5`, `40 × 4`,
   `16 × 10`, `80 × 2`, or one group of 160. Candidates, smallest scale/bias
   overhead first:
   - Group 160 (1 group/row): 1 BF16 scale + 1 BF16 bias per row = 4 bytes
     overhead on top of 80 packed bytes → 84 B/row (~1.68 KB/token for 16
     rows) but coarsest precision (worst-case quality).
   - Group 40 (4 groups/row): 16 B overhead → 96 B/row (~1.92 KB/token).
   - Group 32 (5 groups/row): 20 B overhead → 100 B/row (~2.0 KB/token) —
     closest in *relative* granularity to the existing group-64 scheme
     (64-wide groups elsewhere average 1 scale+bias per 64 weights; 32-wide
     here is actually *finer* per weight than the rest of the model).
   All are far below the current 320 B/row (BF16). The choice is an
   accuracy/size trade avoided by measurement, not guessed — see Phase 1
   step 1.
2. **New format code path vs. generalizing `FinchQuantization`.** Either
   add a `groupSize` parameter to the existing `quantizeInt4Affine`/
   `dequantizeInt4Affine` (touches the shared format module every other
   quant slot depends on) or add a small PLE-only sibling function that
   hardcodes the chosen group size (isolates blast radius to PLE code).
   Given the "no accuracy regression" constraint above and that this is a
   one-off row shape (160), the sibling-function route is lower-risk and is
   what Phase 1 assumes; revisit only if a second 160-wide tensor class
   shows up.
3. **On-disk row layout.** Packed nibbles must stay contiguous per row for
   a single `pread` to fetch one row's full quantized state (packed +
   scales + biases), matching the current single-`pread`-per-row shape in
   `PLEPartStreamer.readRows`. Layout: `[packed nibbles: cols/2 bytes][scale
   BF16 × nGroups][bias BF16 × nGroups]`, fixed stride, computed once from
   `cols` and the chosen group size — no per-row variable-length metadata.
4. **Manifest/schema versioning.** `manifest.json`'s qwen3_8 PLE section
   currently has no quantization slot (raw BF16, verbatim copy). Adding one
   is a schema change gated the same way the existing "Missing or
   incompatible quantization metadata is rejected" rule works
   (`docs/SYSTEM_DESIGN.md`): older `.finch` installs with no PLE quant slot
   must keep loading as raw-BF16 PLE (backward compatible), and a new
   install must declare its PLE quant slot explicitly so the runtime never
   guesses.

## 3. Step-by-step plan

### Phase 1 — Offline quantization-error measurement (no runtime/writer changes)

**Goal:** pick the group size from evidence, before it is load-bearing in
any format.

1. Write a standalone, throwaway measurement script (not shipped) that
   reads a sample of real PLE rows straight from the existing bf16
   `ple_shards/*.bin` (or the original safetensors part tensors), quantizes
   each candidate group size (160/40/32) with a prototype of the affine
   scheme, dequantizes, and reports per-row max-abs-error and RMS-error
   distributions across a few thousand sampled rows (uniform across parts,
   since rows are hash-selected uniformly at inference time — no part is
   more "important" than another).
2. Compare those errors against the existing int4/group-64 error profile
   already characterized for attention/expert weights
   (`docs/QWEN36_PORT.md` "quant-noise amplification" / regression tests in
   `Tests/FinchMoEFormat/FinchQuantizationTests.swift`) as a sanity floor —
   PLE error should not be materially worse per-element than what the rest
   of the model already tolerates.
3. Pick the smallest group size (finest granularity, most overhead) whose
   error is in that same band; fall back to a coarser group only if error is
   already well under the floor with room to spare. Record the chosen group
   size and the measured error numbers in this document before Phase 2
   starts.
4. **Exit gate:** a chosen group size with recorded error numbers. Do not
   proceed to format/writer work without this.

#### Phase 1 result (measured 2026-09-11) — CHOOSE GROUP 32

**Decision: group 32 (5 groups per 160-wide row).**

Method: standalone throwaway script sampled 16,384 real rows (128 parts ×
128 rows, uniform per-part row indices), read straight from the raw BF16
`ple_shards/*.bin` (128 parts × 2,500,012 rows × 160 cols, 800,003,840 B
each). Each row was quantized/dequantized with an exact replica of
`FinchQuantization.quantizeInt4Affine` (per-group min/max,
`scale=(max−min)/15`, `bias=min`, constant-group special-case, BF16
round-half-to-even of scale/bias, quantize against the BF16-rounded scale)
for each candidate group size, plus a group-64 reference on the largest
128-wide slice as the rest-of-model floor **on the same data distribution**.

Data profile (16,384 sampled rows): 0 non-finite elements, 0 all-zero rows
(no degenerate unused-hash buckets in the sample),
`mean|w| = 6.1e-3`, `p99|w| = 2.0e-2`, `max|w| = 4.3e-2`. The PLE table is a
**small additive residual term** — per-element error is therefore tiny in
absolute terms, and the operative comparison is relative to the element's
own magnitude and to the rest-of-model int4 floor.

Per-element abs-error (int4, 16 levels):

| Group | groups/row | bytes/row | KB/token | max | p50 | p90 | p99 | p99.9 | RMS | rel p99 | rel p99.9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | 1 | 84 | 1.31 | 2.289e-3 | 6.56e-4 | 1.221e-3 | 1.541e-3 | 1.770e-3 | 7.81e-4 | 6.33% | 6.56% |
| 40 | 4 | 96 | 1.50 | 2.205e-3 | 5.04e-4 | 9.92e-4 | 1.312e-3 | 1.549e-3 | 6.25e-4 | 5.76% | 6.29% |
| **32** | **5** | **100** | **1.56** | **2.167e-3** | **4.73e-4** | **9.46e-4** | **1.266e-3** | **1.511e-3** | **5.95e-4** | **5.63%** | **6.21%** |
| 64 (ref, 128-wide slice) | 2 | — | — | 2.197e-3 | — | — | 1.389e-3 | 1.633e-3 | 6.81e-4 | — | — |

Reading: every candidate sits **inside** the rest-of-model group-64 floor
(RMS 5.9e-4–7.8e-4 all ≤ the 6.8e-4 group-64 reference; group-160 is only
~15% higher RMS, i.e. same band, not materially worse). The relative p99
error is ~5.6–6.3% for all group sizes — that is the int4 16-level floor,
not a group-size effect; finer groups buy a modest RMS reduction (group 32
is ~24% lower RMS than group 160) at the cost of ~16 more bytes per row.
No candidate is *well under* the floor with room to spare (the relative
error is the int4 floor, independent of group size), so per the decision
rule the **smallest group size (finest granularity) is selected**: group 32.

Cost deltas vs group 160: +16 B/row → +5 GiB total (84→100 B/row) and
+0.25 KB/token decode I/O. That is a small, acceptable price for the
finest relative granularity, consistent with the "finest group within
band" rule and with group 32 being *finer per-weight* than the group-64
rest of the model.

**Exit gate: MET.** Chosen group size = 32; error numbers recorded above.
Phase 2 may proceed.

### Phase 2 — Format: extend `FinchQuantization` for the PLE row shape

**Files:** `Sources/FinchMoEFormat/FinchQuantization.swift`,
`Tests/FinchMoEFormat/FinchQuantizationTests.swift`.

1. Add `quantizeInt4AffinePLE` / `dequantizeInt4AffinePLE` (naming TBD at
   implementation time) mirroring the existing group-64 functions but
   parameterized on the Phase 1 group size, with the on-disk layout from
   design decision 3 above (packed nibbles, then per-group BF16 scale
   array, then per-group BF16 bias array, fixed stride from `cols` and
   group size — no dynamic-length fields).
2. Handle the "constant group" and "near-zero range" edge cases the
   existing code already special-cases (`FinchQuantization.swift:86,
   106`) — the PLE table is a hash embedding, so degenerate
   all-same-value or all-zero rows are plausible (e.g. unused hash
   buckets) and must round-trip exactly like the existing scheme requires.
3. Unit tests: round-trip a synthetic 160-wide row through
   quantize→dequantize, assert the same constant/near-zero/normal-range
   cases the existing `FinchQuantizationTests.swift` covers for group-64,
   plus a fixed-stride/byte-layout assertion (given `cols` and group size,
   the packed byte count is exactly what the runtime reader will expect).
4. **Exit gate:** new unit tests pass; `swift test --filter FinchQuantizationTests`
   green.

#### Phase 2 result (implemented 2026-09-14) — DONE

- Added `FinchQuantization.pleGroupSize = 32`, the `Int4AffinePLERow` struct
  (fields `packed`/`scales`/`biases` in on-disk order per design decision 3),
  `quantizeInt4AffinePLE(_:_:)` (array + `UnsafeBufferPointer` buffer forms,
  parameterized on `groupSize`), and `dequantizeInt4AffinePLE(_:_:)`.
- The affine math, constant-group (`scaleF=1, biasF=wmin`) and near-zero
  (`effectiveScale` guard) handling mirror the group-64 codec exactly; only
  the group size (32 vs 64) and the row struct differ.
- The decoder derives the group size from the row itself
  (`n / r.scales.count`), so a decoded row is self-describing — no separate
  group-size field to keep in sync, and it decodes other group widths too.
- 6 new unit tests in `FinchQuantizationTests.swift`: 160-wide round-trip
  within one affine step, fixed-stride byte-layout (100 bytes/row), constant
  group exact round trip, subnormal-residue no-trap, a self-describing decode
  built by hand, and a non-32 group-size decode.
- **Exit gate: MET.** `swift test --filter FinchQuantizationTests` → 9 tests,
  0 failures. Phase 3 may proceed.

### Phase 3 — Writer: quantize PLE parts during repack

**Files:** `Sources/FinchMoERepack/Core/Planning/QwenRepackPlanner.swift`,
`Sources/FinchMoERepack/Core/Writing/QwenQuantizedWriter.swift`,
`Sources/FinchMoERepack/Core/Workflow/LocalQwenRepacker.swift`,
`Sources/FinchMoERepack/Core/Format/ArchInfo.swift`, manifest schema code
under `Sources/FinchMoEFormat/`.

1. `QwenPLEPartFilePlan` currently carries only `rows`/`cols`/`source`
   (verbatim-copy plan). Extend it with the transform info Phase 2 defined
   (group size, computed byte stride per row, total file size under the new
   layout) — parallel to how `QwenResidentEntry`/`LayerFilePlan` already
   carry `.int4Affine(rows:cols:)` transforms for other tensor classes
   (`QwenRepackPlanner.swift:22-55`).
2. Replace `QwenQuantizedWriter.writePLEPart`'s current verbatim
   chunked-copy body (`QwenQuantizedWriter.swift:403-437`, currently an
   8 MiB-chunk `pread`→`pwrite` loop with `adviseDontNeed`) with a row-batch
   quantizing writer: read a bounded batch of BF16 rows from the mapped
   source, quantize each row independently (Phase 2 function), write the
   packed batch — same bounded-scratch discipline the resident/expert
   writers already use (`QwenQuantizedWriter.maxRowElements`,
   `QwenRowScratch`), same per-part `fsync` + SHA-256 the current code does
   at `QwenQuantizedWriter.swift:432-436`.
3. `ArchInfo`: add whatever quant-descriptor fields the manifest needs for
   PLE (group size, bits — mirroring how `arch.ngramPartCount`/
   `ngramPartRows` already exist as census-corrected fields,
   `QwenRepackPlanner.swift:118-124` and the guard block around line 230).
4. Manifest write path: add a PLE quantization slot (group size, bits,
   scheme identifier) to `manifest.json`, following the "Missing or
   incompatible quantization metadata is rejected" convention already in
   force for every other tensor class (`docs/SYSTEM_DESIGN.md`). This is
   additive — installs without the slot are the current raw-BF16 PLE
   installs and must keep loading (Phase 4 backward-compat requirement).
5. `LocalQwenRepacker.swift`'s size-accounting (`pleBytes` at line 144) and
   its `writePLEPart` call site (line 208) update to the new sizes/call
   signature.
6. **Exit gate:** a repack against a small synthetic qwen3_8 snapshot (the
   existing test fixtures already used for `QwenRepackPlanner`/
   `LocalQwenRepacker` tests) produces quantized PLE part files of the
   expected byte-exact size and passes existing repack-plan tests plus new
   ones asserting the PLE transform is `.int4AffineNarrow`-class (whatever
   it's named) with the chosen group size.

#### Phase 3 result (implemented 2026-09-14) — DONE

- **Planner** (`QwenRepackPlanner.swift`): `QwenPLEPartFilePlan` now carries
  the Phase-2 transform info — `groupSize` (locked to `FinchQuantization.pleGroupSize`
  = 32), plus computed `rowByteStride` (`cols/2 + 2 * nGroups * 2`) and
  `quantizedByteCount` (`rows × rowByteStride`). A plan-time guard rejects a
  row width that is not a whole multiple of the group size (160 / 32 = 5)
  instead of letting the quantizer's precondition trap mid-repack.
- **Writer** (`QwenQuantizedWriter.writePLEPart`): replaced the verbatim
  chunked copy with a row-batch quantizer. It reads a bounded batch of BF16
  rows from the mmap slice, decodes each to `Float`, calls
  `quantizeInt4AffinePLE` per row, and lays each row out as
  `[packed nibbles][scale BF16 × nGroups][bias BF16 × nGroups]` at the fixed
  `rowByteStride` — one `pwrite` per batch, then `adviseDontNeed` on the
  source batch, same bounded-scratch / fsync / SHA-256 discipline as the
  resident and expert writers.
- **Manifest slot** (`FinchManifestQuantV1.pleNgram`): an additive optional
  quant slot. `LocalQwenRepacker` sets it (int4 / "affine" / group 32) for a
  qwen3_8 install that has PLE parts and leaves it `nil` otherwise; the
  codec validates it when present and tolerates its absence, so the existing
  raw-BF16 install keeps decoding unchanged (Phase 4 backward-compat).
- **Size accounting** (`LocalQwenRepacker`): `pleBytes` now sums
  `quantizedByteCount` instead of `rows × cols × 2`.
- **Tests:**
  - `qwen38RepackWritesPLEPartsAndFamilyManifest` now asserts the part files
    are the quantized stride size (`rows × (cols/2 + 2·nGroups·2)`), and
    byte-compares each written part against a canonical re-quantization of its
    source rows (proving the repack is the canonical transform, not a copy);
    it also asserts the manifest `pleNgram` slot is present, int4, group 32,
    "affine".
  - `plan38CoversEveryEntryWithExpectedTransformsAndParts` now asserts each
    part's `groupSize`, `rowByteStride`, and `quantizedByteCount`.
  - `FinchManifestCodecTests.pleNgramSlotIsOptionalAndRoundTrips` proves the
    slot is optional: a slot-less quant block still decodes (`pleNgram == nil`,
    old raw-BF16 install), a slot-carrying one round-trips, and stripping the
    slot from the wire still decodes.
- **Exit gate: MET.** `swift test --filter FinchManifestCodecTests
  --filter FinchQuantizationTests --filter QwenRepackPlannerTests
  --filter LocalQwenRepackerTests` → 30 tests in 4 suites, 0 failures. Phase 4
  may proceed.

### Phase 4 — Runtime: read and dequantize PLE rows

**Files:** `Sources/FinchMoE/Infrastructure/Streaming/PLEPartStreamer.swift`,
`Sources/FinchMoE/Runtime/Inference/PLEHost.swift`,
`Sources/FinchMoE/Runtime/Inference/Model.swift` (`openPLEPart`,
`ngramPartCount`/PLE config surface).

1. `PLEPartStreamer`: `byteStride` currently hardcodes `columns * 2` (raw
   BF16, `PLEPartStreamer.swift` doc comment + `byteStride`). Change to the
   new fixed stride from Phase 2/3 (packed nibbles + scale/bias block);
   `readRows` itself needs no logic change since it is already a
   byte-range `pread` keyed off `byteStride` — only the stride computation
   and the file-size validation (`expected` in `init`) change.
2. `PLEHost.gather` (`PLEHost.swift:246-251`) currently does
   `bits = raw.bindMemory(to: UInt16.self)` then
   `Float16(Quantization.bf16ToFloat(bits[d]))` per element. Replace with:
   unpack nibbles, read the row's scale/bias block, call the Phase 2
   dequantize function, convert to `Float16` for the GPU buffer — same
   output type/shape the rest of the pipeline expects, so nothing downstream
   of `gather` (the GPU gate/conv/plane-add kernels in `ple.metal`) changes.
3. Add a runtime load-time check mirroring the existing "Missing or
   incompatible quantization metadata is rejected" rule: if a qwen3_8
   install's manifest carries no PLE quant slot, keep using the current raw
   BF16 path unchanged (backward compat for the already-built 162 GiB
   install); if it carries a slot with an unrecognized group size/scheme,
   reject the load rather than silently misreading bytes.
4. **Exit gate:** engine unit/integration tests around `PLEPartStreamer`/
   `PLEHost` (existing + new) pass against both a raw-BF16 fixture (old
   format, unchanged behavior) and a new quantized fixture.

#### Phase 4 result (implemented 2026-09-14) — DONE

- **Streamer** (`PLEPartStreamer`): new `Layout` enum
  (`rawBF16` / `quantized(groupSize:)`). `byteStride` is now computed per
  layout (`columns × 2` vs `columns/2 + 2 · nGroups · 2`) and `init`
  validates the file's size against *that* stride, so a part opened under
  the wrong layout is rejected at open rather than decoded into garbage.
  `readRows` is unchanged — it was already a byte-range `pread` keyed off
  `byteStride`, and the stride is the only thing that had to learn the new
  layout.
- **Gather** (`PLEHost.gather`): decodes per `streamer.layout`, leaving the
  raw-BF16 path byte-for-byte as it was. The output is the same
  `[Float16]` shape/type for both layouts, so nothing downstream (the GPU
  gate/conv/plane-add kernels in `ple.metal`) changed.
- **Layout decision** (`Model.plePartLayout(quant:)`): the single place the
  layout is decided, from the manifest's `pleNgram` slot. `openPLEPart` and
  the load-time part-size check both call it, so they cannot disagree.
- **Load-time validation** (`ManifestReader`): `pleNgram` is optional.
  Absent → the legacy raw-BF16 path, so the already-built 162 GiB install
  keeps loading unchanged. Present → it must be int4 / "affine" / BF16
  scales and biases / group `Quantization.pleGroupSize`; anything else
  rejects the load rather than misreading bytes.
- **Tests:** `PLEHostTests` gains two — the gather decodes int4-affine rows
  (hand-checkable group size 2 over 4-wide toy rows, multi-group), and the
  raw-BF16 default layout rejects a quantized file's size (the wrong-layout
  guard). `Qwen38EngineLoadTests` now byte-compares the written parts
  against a canonical re-quantization of the source rows and asserts the
  gathered vector equals `dequantize(quantize(sourceRow))`.
  `Qwen38ToyReplayTests`' reference decodes per layout, so it runs on
  exactly the engine's gathered vector.
- **Exit gate: MET.** 54 tests in 8 suites, 0 failures
  (`FinchQuantizationTests`, `FinchManifestCodecTests`,
  `QwenRepackPlannerTests`, `LocalQwenRepackerTests`, `PLEHostTests`,
  `Qwen38EngineLoadTests`, `Qwen38DecodeWiringTests`,
  `Qwen38ToyReplayTests`). Run with `--no-parallel`: parallel runs flake on
  an unrelated prefill-cancel test.

##### Collateral: the toy replay's prefill ceilings were re-measured

Quantizing the toy install's PLE parts changes the engine's residual plane,
and `Qwen38ToyReplayTests.prefillLastRowMatchesFP32Replay` holds a
*deliberately* non-convergent chain (layer 3's sparse selection amplifies
unanchorable KV residue) at *empirically measured* ceilings — so those
ceilings moved, and all seven of the documented stages roughly doubled
(`logits` 0.0221 → 0.1387). That is a data change, not an arithmetic one,
and it was confirmed as such before the ceilings were touched:

- The movement tracks quantization error monotonically. Sweeping the group
  size (one constant, `FinchQuantization.pleGroupSize`) moves every ceiling
  with it — `3|attnBlockOut` 0.0298 / 0.1423 / 0.5267 at groups 8 / 32 /
  160, `logits` 0.0081 / 0.1387 / 0.1801 — while layer 1, the PLE layer
  itself, never leaves `tolerance` (≤ 5.5e-3) at any group size.
- The engine's own two paths (12 `produce` steps vs one `prefillChunked`
  chunk) moved *with* the replay, staying the same distance apart:
  0.142604 vs the replay's 0.14233708 at `3|attnBlockOut` (was 0.09765625
  vs 0.07423675). The control that argues "this is not a replay artifact"
  still holds.
- The PLE data path itself is pinned byte-for-byte elsewhere
  (`plePartsStreamVerifiedRows`,
  `pleHashMetadataAndGatherMatchTheCheckpoint`, `PLEHostTests`).

The new ceilings are recorded in `T38.prefillAmplified` with the sweep in
its doc comment, alongside a note that the toy's PLE rows are ~100x the real
table's `mean|w| = 6.1e-3` (Phase 1) at the same ~5.6% relative int4 error,
so those ceilings are calibrated to the toy's own amplification and are not
a statement about the shipped model's quality. `3|hc.mid` re-enters the
amplified set at 0.0128 (it had dropped out under the `hc_norm` correction
at 4.9e-3).

### Phase 5 — Correctness validation (the part that actually matters)

This mirrors the M4 oracle-cross-check discipline `docs/QWEN38_PORT.md`
already used, scoped to isolate the PLE change:

1. **Layer-level isolation test first.** Before any full repack, build a
   small synthetic/sliced checkpoint (or reuse existing test fixtures) with
   a real-shaped PLE table, run the PLE gather+gate+conv path through both
   the old raw-BF16 code path and the new quantized path with the *same*
   input rows, and diff the resulting plane contributions numerically
   (fp32 comparison, not just argmax) — this isolates PLE quantization
   error from every other source of error in the model.
2. **Full install + oracle re-check.** Once Phase 5.1 passes, run the full
   quantizing repack (Phase 3) into a *new* install directory (e.g.
   `models/Qwen3.8-Flash-Next-125B-ple4bit.finch/`, never overwriting the
   validated 162 GiB one), then re-run the same M4 oracle cross-check
   protocol from `docs/QWEN38_PORT.md` (tokenization byte-exactness, argmax
   agreement, top-10/100/whole-vocab logit cosine against the llama.cpp
   GGUF oracle) on this new install. Given the existing whole-vocab cosine
   is already 0.88 (under the 0.95 bar) on a cross-quantization comparison,
   the acceptance criterion here is **no further regression** on
   argmax/top-k agreement versus the current 162 GiB install — not a fresh
   0.95 bar, since that bar was already established as unreachable for
   cross-quantization comparisons in this port.
3. **EvalPlus HumanEval re-run**, same protocol as
   `docs/QWEN38_PORT.md` item 8 (`archive/humaneval_evalplus/run_server_cell.sh`,
   greedy T=0), scored on the new install. Acceptance: base pass@1 and
   HumanEval+ within noise of the existing 0.945/0.921 (a handful of
   flips on borderline problems is expected register noise; a broad drop
   is not).
4. **Exit gate:** Phase 5.1 numeric diff within the Phase 1-recorded error
   band, Phase 5.2 shows no oracle regression, Phase 5.3 shows no EvalPlus
   regression. Any failure here means back to Phase 1 (coarser or finer
   group size), not shipping anyway.

#### Phase 5.1 result (measured 2026-09-15) — PASS, on the real install

**Venue changed, deliberately, from the step-1 sketch.** The sketch proposed a
synthetic/sliced checkpoint "with a real-shaped PLE table". The two fixtures
available were both wrong for the question: the toy's PLE rows are uniform in
`[-2, 2)` against the real table's `mean|w| = 6.1e-3` — ~100× the magnitude at
the same ~5.6% relative int4 error, a far noisier regime (see the Phase 4
collateral below, where that toy's own amplification doubles every ceiling) —
and a sliced real checkpoint is a purpose-built fixture that would still only
approximate the real one. Instead the isolation is done **in memory, on the
real install**:

- `FQ_PLE_QUANT_SIM=<groupSize>` (`RealForwardRunner` → `PLEHost`) decodes
  every **raw-BF16** PLE row as `dequantize(quantize(row))` before it reaches
  the GPU. The installed table is not touched, every other weight is
  bit-identical between the two runs, and the transform is the writer's own
  canonical one — so the A/B differs in exactly one thing: PLE row precision.
- No repack, no new install, no sliced fixture: the Phase 6 multi-hour job is
  not on the critical path of this measurement, and 5.2/5.3 still get to run
  against the real thing.
- `PLEHostTests.gatherSimulatesQuantization` pins the knob itself: the
  simulated decode must equal `dequantize(quantize(row))` of the same on-disk
  row, and the test fails if no element moved (so a silent no-op cannot pass).

Three prompts from `docs/benchmark-prompts/real-generation-v1/`, greedy T=0,
32 generated tokens, `--max-context` 2048 (4096 for the 2940-token prompt),
each pair run under `tools/memguard.sh` one at a time. Logits are the
final-prefill row (`FQ_DUMP_PREFILL_LOGITS`), scored by rank band:

| prompt (prefill tokens) | argmax | top-10 cos (overlap) | top-100 cos | whole-vocab cos | top-1 margin / perturbation | greedy text |
| --- | --- | --- | --- | --- | --- | --- |
| short-explanation (62) | MATCH | 0.999953 (9/10) | 0.999865 | 0.996408 | 5.56 / 0.047 | identical, 32 tokens |
| medium-review (426) | MATCH | 0.999940 (10/10) | 0.999880 | 0.997876 | 0.47 / 0.219 | diverges ~token 25 |
| long-synthesis (2940) | MATCH | 0.999897 (10/10) | 0.999806 | 0.996318 | 1.11 / 0.031 | diverges ~token 20 |

**Reading.** Argmax holds on all three, top-10 overlap is 9-10/10, and the
top-band cosine is 0.9998-0.99995. For scale — not as an apples-to-apples
comparison, since those runs differ in both weights and engine — the
cross-quantization oracle comparison already accepted in
`docs/QWEN38_PORT.md` sits at top-10 cos 0.995 with whole-vocab 0.880, so the
PLE term is roughly an order of magnitude below the error the port already
carries in the bands that decide behaviour. Whole-vocab (0.9963-0.9979) is
*not* the number to read here: 98% of the reference vector's squared magnitude
lives in the near-uniform tail, exactly the trap `docs/QWEN38_PORT.md`
documents.

**The honest caveat, and it is the one that matters downstream.** The
perturbation is a *fraction of the top-1 margin* (0.031 of 1.11, 0.047 of
5.56) — but on the one prompt with a thin margin (0.47 nats) it is ~half of
it, and there the greedy trajectory diverges after ~25 tokens, as it does
after ~20 on the third. Both continuations stay fluent English; this is not a
quality collapse, it is a low-margin argmax flip, and it is precisely the
mechanism by which a downstream pass@1 could move. Phase 5.1's own gate is
about the numeric diff and it is met — the diff is inside the Phase 1 band
(RMS 5.95e-4 per element, rel p99 5.63%, amplified to a ~0.03-0.22 nat
logit perturbation) — but a greedy-trajectory flip on thin margins is a real
behavioural difference that **5.3 must bound on the scored benchmark**, not
something this phase can clear on its own.

**Exit gate: MET** for the numeric-diff criterion. 5.2 and 5.3 remain, and
5.3 is now the load-bearing one: the flips above are the reason to run it
rather than assume.

### Phase 6 — Full repack on the real machine, sizing, and rollout

1. Run the real 352 GB→quantized-PLE repack under
   `tools/memguard.sh`, following the same staged/`--resume`/fsync-per-file
   discipline as the existing M4 repack (`docs/QWEN38_PORT.md` "the repack
   reads 360 GB ... writes 174 GB ... staged checkpoints with fsync per
   file class, `--resume`"). Expect a similarly multi-hour job; budget for
   at least one interrupted/resumed run given the box's panic history.
2. Measure and record the actual resulting install size (expected ~94 GiB:
   3.6 + 63 + ~27 GB PLE) and confirm `verified-install.json` promotes
   cleanly.
3. Measure decode-time I/O: repeat a representative generation run (same
   shape as the existing benchmark protocol in `docs/BENCHMARKS.md`/
   `docs/RUNTIME_CONTROLS.md`) and record actual bytes read per token for
   the PLE path specifically (expected ~1.4–2.0 KB/token depending on the
   Phase 1 group size, down from 5 KB/token), plus overall tokens/sec impact
   (expected neutral-to-positive, since less random-read I/O per token
   should help on a disk-bound box, but this must be measured, not assumed).
4. Update `docs/QWEN38_PORT.md`'s "Deferred" list to mark PLE table quant
   done, with a summary and links into this document's evidence, and update
   `docs/SYSTEM_DESIGN.md` if the general quantization description needs a
   qwen3_8-specific caveat.
5. Only after Phase 5 and 6.2/6.3 gates pass: update
   `AppModelLocation`/default-install pointers (the same kind of switch
   `docs/QWEN36_PORT.md` item lists for `Qwen3.6-35B-A3B-4bit.finch`) to
   prefer the new install, leaving the old 162 GiB install in place and
   documented as the fallback until there is real-world confidence in the
   new one.

## 4. What this plan explicitly does not do

- Does not touch `model_weights.bin` or `packed_experts/` — already at this
  codebase's int4/int8 floor.
- Does not attempt sub-4-bit PLE quantization (e.g. int2/int3) — the group
  size problem in §2 is hard enough at int4; going lower is a follow-up to
  consider only after this lands and is measured.
- Does not change the hash/n-gram routing logic (`PLEHost` position/token
  hashing) — only how a selected row's bytes are stored and decoded.
- Does not change the Metal-side gate/conv/plane-add kernels in
  `ple.metal` — the dequantized row still lands as `Float16` in the same
  buffer shape those kernels already consume.
