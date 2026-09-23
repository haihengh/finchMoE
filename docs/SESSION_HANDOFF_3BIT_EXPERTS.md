# Session Handoff: 3-Bit Routed-Expert Experiment

Date: 2026-09-21 (real-model gate resolved 2026-09-23 — see Verdict)

## Goal

Transfer the archived first-generation engine's most promising speed advantage into the current correct Swift/Metal runtime: affine 3-bit routed experts, while preserving the current 4-bit path as the default.

The archived 38-42 tok/s result was measured on a 24 GB M4 Pro with a warm page cache and 3-bit experts. It was not an apples-to-apples runtime comparison. The archived engine's HumanEval failure was shared forward-pass divergence: native 3-bit, native 4-bit, and the exact reference Q4_K_M all converged near 13%. This means the old failure does not specifically disqualify 3-bit experts, but the new format still needs current-runtime EvalPlus validation.

## Implemented

### Canonical format codec

- Added `FinchQuantization.Int3AffineRow`.
- Added group-64 affine int3 quantization and dequantization.
- Packing matches the archived format: eight unsigned 3-bit values in three little-endian bytes.
- Scale and bias are rounded to BF16 before quantization, matching current format conventions.
- Handles subnormal scales without reciprocal overflow.

Files:

- `Sources/FinchMoEFormat/FinchQuantization.swift`
- `Sources/FinchMoE/Infrastructure/ModelIO/Quantization.swift`
- `Tests/FinchMoEFormat/FinchQuantizationTests.swift`

### Local repacker

- Added `--routed-expert-bits 3|4` for local bf16 Qwen snapshot conversion.
- Default remains 4-bit.
- Planner computes packed sizes and expert offsets from the selected bit count.
- Writer emits int3 rows through the canonical codec.
- Manifest routed-expert width is derived from the planned layout.
- Resume fingerprints now include routed-expert width; journal version was raised to 2 so 3-bit and 4-bit partial outputs cannot be mixed.

Files:

- `Sources/FinchMoERepack/Command/main.swift`
- `Sources/FinchMoERepack/Core/Planning/QwenRepackPlanner.swift`
- `Sources/FinchMoERepack/Core/Writing/QwenQuantizedWriter.swift`
- `Sources/FinchMoERepack/Core/Workflow/LocalQwenRepacker.swift`
- `Sources/FinchMoERepack/Core/Workflow/LocalRepackJournal.swift`
- `Tests/FinchMoERepack/Core/Planning/QwenRepackPlannerTests.swift`
- `Tests/FinchMoERepack/Core/Workflow/LocalQwenRepackResumeTests.swift`

### Runtime and Metal

- Runtime schema accepts routed experts at 3 or 4 bits; other quantization slots retain their existing restrictions.
- `Model.routedExpertWeightBits` exposes the manifest value.
- Decode and grouped-prefill pipelines select int3 Metal variants once at model/runner construction through function constants.
- Existing int4 kernels and defaults remain unchanged.
- Added int3 unpack/dot implementations to both decode and grouped-prefill Metal modules.

Files:

- `Sources/FinchMoE/Infrastructure/ModelIO/ManifestReader.swift`
- `Sources/FinchMoE/Runtime/Inference/Model.swift`
- `Sources/FinchMoE/Runtime/Inference/RealForwardRunner.swift`
- `Sources/FinchMoE/Kernels/MoE/MoE.swift`
- `Sources/FinchMoE/Kernels/Prefill/MoE/PrefillGroupedRoutedMoE.swift`
- `Sources/FinchMoE/Metal/MoE/moe.metal`
- `Sources/FinchMoE/Metal/Prefill/prefill.metal`

### Tests added

- Exact int3 byte pattern and round trip.
- Int3 subnormal-scale regression.
- Planner verifies 3-bit packed size, metadata, and page alignment.
- Decode Metal pipeline compares against an independently dequantized CPU reference.
- Grouped-prefill int3 Metal pipelines compile.
- Synthetic bf16 snapshot repack/load/runner integration test was added.

Files:

- `Tests/FinchMoE/Core/Kernels/MoE/MoEFusedFFNTests.swift`
- `Tests/FinchMoE/Core/Kernels/Prefill/PrefillGroupedRoutedMoETests.swift`
- `Tests/FinchMoE/Core/Infrastructure/ModelIO/QwenRepackEngineLoadTests.swift`

## Validation Status (updated 2026-09-21)

All focused int3 suites are green. Re-verified this session:

```bash
swift test --filter FinchQuantizationTests                          # 3 tests, pass (incl. int3 round-trip + subnormal)
swift test --filter QwenRepackPlannerTests                          # pass
swift test --filter MoEFusedFFNTests                                # 11 tests, pass (int3 decode numerical + int4)
swift test --filter QwenRepackEngineLoadTests.threeBitRoutedExpertInstallLoadsAndBuildsRunner  # 1 test, pass (0.301s)
swift test --filter 'LocalQwenRepackerTests|LocalQwenRepackResumeTests|RepackCLITests'  # 14 tests, 3 suites, pass
swift test --filter PrefillGroupedRoutedMoETests                    # 15 tests, pass (incl. two NEW int3 execution tests)
```

The previously-cancelled integration test now passes (the `writeManifest` hard-coded-`routedExpert: 4` defect is fixed).

The grouped-prefill int3 execution coverage gap is now closed: two new tests run the Metal int3 pipeline against an independently dequantized CPU reference, mirroring the existing int4 execution tests:

- `PrefillGroupedRoutedMoETests.int3BatchedMatchesReferenceAcrossPartialMicrobatch` (silu, the Qwen 3.6 routed-expert activation)
- `PrefillGroupedRoutedMoETests.int3BatchedGeluMatchesReferenceAcrossPartialMicrobatch` (gelu)

The existing grouped-prefill helpers were generalized to accept `bits: Int = 4`: `makeSyntheticExpertPool`, `appendProjection`, `cpuSyntheticRoutePartials`, and `cpuInt4Dot` → `cpuAffineDot` (reads int3 octets as 8 unsigned 3-bit lanes per 3 bytes, matching the canonical codec). `runStreamedBatched` builds `PrefillGroupedRoutedMoE(context:routedExpertWeightBits: bits)`. Int4 callers are unchanged (defaults preserve prior behavior).

### Full suite

```bash
swift test   # 180 tests, 23 suites
```

Exactly one test fails, the same one flagged below. No new failures.

### Known unrelated failure

`AppModelTests.cancelDuringPrefillKeepsPromptSnapshotUntilClear` fails (3 issues, AppModelTests.swift:303–305: `outputText`/`outputResponsePlainText`/`outputConversationPlainText` not empty after cancel-during-prefill). This is a pre-existing app-state test unrelated to int3 — it does not touch the quantization format, repacker, or Metal kernels. Do not treat it as an int3 failure.

## Real-Model Verdict (2026-09-23)

**The 3-bit expert path is correct. It loses too much quality on Qwen 3.6 to ship. 4-bit remains the default. Decision (2026-09-23): the 3-bit changes are archived on the `3bit-experts` branch and removed from main; `main` stays pure 4-bit.**

**Where the code lives:** the `3bit-experts` branch (off `main`) holds every 3-bit source change, test, and the HumanEval evidence files. `main` has none of it. The `Qwen3.6-35B-A3B-3bit.finch` install and the bf16 source snapshot remain on disk under `models/` (untracked); delete the ~16 GB install if the disk is needed and the experiment is closed.

The real-model gate was run end-to-end and the hypothesis — that 3-bit experts are a good tradeoff on this model — is **refuted by the data**.

**The pipeline is verified correct, not buggy:**
- Dequantizing the *actual bytes* in `Qwen3.6-35B-A3B-3bit.finch` (expert 0, gate) against the bf16 source gives **20.8% per-element error, cosine 0.979** — exactly the expected 3-bit reconstruction error. Writer, canonical codec, and Metal kernel all agree.
- The 3-bit manifest correctly declares `routedExpert.weightBits: 3` (every other slot identical to the 4-bit install), and the stride math is exact (1,376,256 = 4/3 × the 4-bit stride).
- A 231% error figure seen during diagnosis was a bug in a throwaway Python check (NumPy `uint8 << 8` overflow); the Swift `UInt32` / Metal `uint` casts are correct.

**Quality cliff (greedy, HumanEval 164 tasks, same prompts/ordering as the 4-bit run):**

| | 4-bit | 3-bit |
|---|---|---|
| HumanEval pass@1 | 149/164 (90.9%) | 28/164 (17.1%) |
| HumanEval+ pass@1 | 144/164 (87.8%) | 27/164 (16.5%) |
| Per-element error | ~9–12% | ~19–25% (down_proj worst, ~25%) |

0 tasks gained, 121 lost. 3-bit outputs are fluent but degraded — HumanEval/0 stops after the docstring (no body), HumanEval/5 burns tokens on a preamble and cuts off mid-function. That is the signature of noisy expert activations, not a broken kernel.

**Why:** Qwen 3.6's routed-expert weights are tiny (`|w|max` ≈ 0.08, `mean|w|` ≈ 0.003). 8-level (3-bit) affine quantization can't represent them — the error more than doubles int4's (~20% vs ~10%), and that is enough to collapse code generation.

**Action taken:** no code change needed. `FinchMoERepack` defaults to `--routed-expert-bits 4` (main.swift:34); 3-bit requires an explicit `--routed-expert-bits 3` flag, so 4-bit is the default by construction. The correct 3-bit codec/writer/kernels/tests stay in the tree (useful for models with larger expert weights), but it is not a shippable path for Qwen 3.6.

## Immediate Next Steps (resolved 2026-09-23)

1. ~~Rerun the synthetic integration test~~ — DONE, passes.
2. ~~Run the local repacker suite (resume + CLI)~~ — DONE, 14 tests pass.
3. ~~Add a numerical grouped-prefill int3 execution test~~ — DONE, two tests added and green.
4. ~~Run the full suite after focused tests are green~~ — DONE; only the known unrelated AppModel cancellation test fails.
5. ~~Real-model gate (convert, load, HumanEval vs 4-bit)~~ — DONE, 2026-09-23. 3-bit scores 17% vs 4-bit's 91% → **parked**, 4-bit stays the default.

Remaining work (optional follow-ups): see the next section.

## Real Model Conversion

`models/` currently holds `Qwen3.6-35B-A3B-4bit.finch` (the 4-bit comparison target) and `Qwen3.8-Flash-Next-125B-ple4bit.finch`; the bf16 `Qwen3.6-35B-A3B` source snapshot is still not present. Disk has ~227 GB free (926 GB volume, 75% used), so there is ample room for the bf16 source plus the ~16 GB int3 `.finch` install. Download the bf16 source before converting.

Expected command once the source snapshot exists:

```bash
swift run -c release FinchMoERepack \
  --input-snapshot models/Qwen3.6-35B-A3B-bf16 \
  --output models/Qwen3.6-35B-A3B-3bit.finch \
  --routed-expert-bits 3 \
  --overwrite
```

The repository's existing `download_fast.py` and `download_model.py` target the already-quantized Qwen 3.8 install, not the Qwen 3.6 bf16 source. Use `huggingface_hub.snapshot_download` for `Qwen/Qwen3.6-35B-A3B`, with resume support and an explicit local directory.

## Required Real-Model Gates

Compare the 3-bit install against `models/Qwen3.6-35B-A3B-4bit.finch` using identical prompts and ordering.

1. Verify install and load.
2. Run short greedy generation and compare for coherence.
3. Benchmark 16 GB and, ideally, 24 GB hardware with `--counters`.
4. Record decode tok/s, `io` ms/step, expert MB/step, GPU routed time, and peak resident memory.
5. Run the current oracle/long-context checks.
6. Run full EvalPlus HumanEval and HumanEval+ before considering 3-bit shippable.

The archived data suggests roughly 22% smaller expert blobs and a possible page-cache threshold effect on a 24 GB machine. Do not promise 40 tok/s: the archived result had a fully warm expert working set and different dense-weight precision.

## Potential Follow-Up

If all-3-bit loses too much quality, test mixed precision:

- gate/up at 3-bit
- down projection at 4-bit

That requires per-role bit metadata; the current manifest has one routed-expert bit width for all three roles, so this is a format extension rather than a small follow-up.

A separate archived idea, 4-bit GDN projections, is already supported by the current decoder but was previously rejected for production because recurrent-state error increased substantially. It should remain behind the 3-bit expert experiment in priority.
